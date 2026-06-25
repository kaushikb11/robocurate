"""Tests for the SimPhysicsValidity signal: per-check detection, skip-on-real, integration."""

from __future__ import annotations

import numpy as np
import pytest

from robocurate import signals
from robocurate.curator import Budget, Curator
from robocurate.signals.sim_validity import NONFINITE_PENALTY, SimPhysicsValidity
from robocurate.trajectory import (
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)
from tests.synthetic import make_trajectory
from tests.test_jerk_signal import _ListReader

# A toy *sim* embodiment carrying sim-state features under the `sim.` convention.
SIM_EMBODIMENT = EmbodimentSpec(
    embodiment_id="toy_sim",
    control_hz=10.0,
    features=(
        FeatureSpec("timestamp", FeatureRole.TIME, shape=(), dtype="float32", units="s"),
        FeatureSpec("action", FeatureRole.ACTION, shape=(2,), dtype="float32"),
        FeatureSpec(
            "sim.penetration_depth", FeatureRole.EXTRA, shape=(), dtype="float32", units="m"
        ),
        FeatureSpec("sim.object_pos", FeatureRole.EXTRA, shape=(3,), dtype="float32", units="m"),
    ),
)


def _sim_traj(
    episode_index: int,
    *,
    num_steps: int = 12,
    penetration: float = 0.0,
    teleport: bool = False,
    nonfinite: bool = False,
) -> Trajectory:
    t = (np.arange(num_steps, dtype=np.float32) * 0.1).astype(np.float32)
    action = np.zeros((num_steps, 2), dtype=np.float32)
    pen = np.full(num_steps, penetration, dtype=np.float32)
    pos = np.cumsum(np.full((num_steps, 3), 0.01, dtype=np.float32), axis=0)
    if teleport:
        pos[num_steps // 2] += 5.0  # a 5 m jump in one step
    if nonfinite:
        action[2, 0] = np.inf
    columns = {
        "timestamp": t,
        "action": action,
        "sim.penetration_depth": pen,
        "sim.object_pos": pos,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/sim",
        episode_index=episode_index,
        embodiment=SIM_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_sim_v0",
        success=SuccessLabel(value=True, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _score(traj: Trajectory) -> dict:  # type: ignore[type-arg]
    from tests.synthetic import make_signal_context

    sig = SimPhysicsValidity()
    [s] = sig.score([traj], make_signal_context())
    return {"value": s.value, "skipped": s.skipped, "diag": s.diagnostics, "score": s}


def test_clean_sim_trajectory_is_valid() -> None:
    out = _score(_sim_traj(0, penetration=0.0))
    assert not out["skipped"]
    assert out["diag"]["is_valid"] is True
    assert out["value"] == 0.0
    assert out["score"].per_transition is not None  # per-step penetration emitted


def test_penetration_violation_detected() -> None:
    out = _score(_sim_traj(0, penetration=0.05))  # 50 mm >> 5 mm threshold
    assert out["diag"]["is_valid"] is False
    assert out["diag"]["max_penetration"] == pytest.approx(0.05, abs=1e-6)
    assert out["value"] > 0.0


def test_teleportation_violation_detected() -> None:
    out = _score(_sim_traj(0, teleport=True))
    assert out["diag"]["is_valid"] is False
    assert out["diag"]["max_position_jump"] > 1.0
    assert out["diag"]["jump_excess"] > 0.0


def test_nonfinite_is_catastrophic() -> None:
    out = _score(_sim_traj(0, nonfinite=True))
    assert out["diag"]["has_nonfinite"] is True
    assert out["diag"]["is_valid"] is False
    assert out["value"] >= NONFINITE_PENALTY


def test_skips_on_real_data() -> None:
    # A real-data trajectory (toy embodiment, no sim.* features) -> recorded skip.
    real = make_trajectory(0)
    from tests.synthetic import make_signal_context

    [score] = SimPhysicsValidity().score([real], make_signal_context())
    assert score.skipped
    assert "requires sim state" in (score.skip_reason or "")


def test_known_answer_removes_invalid_sim_trajectories() -> None:
    trajs = [_sim_traj(i, penetration=0.0) for i in range(4)]  # valid
    trajs.append(_sim_traj(4, penetration=0.08))  # penetration violation
    trajs.append(_sim_traj(5, nonfinite=True))  # blown-up sim
    reader = _ListReader(trajs)
    result = Curator([SimPhysicsValidity()], budget=Budget.count(4), seed=0).run(reader)
    assert {4, 5}.issubset(set(result.removed_episode_indices))
    assert set(result.kept_episode_indices) == {0, 1, 2, 3}


def test_score_is_deterministic() -> None:
    from tests.synthetic import make_signal_context

    sig = SimPhysicsValidity()
    ctx = make_signal_context()
    trajs = [_sim_traj(i, penetration=0.01 * i, teleport=(i == 2)) for i in range(4)]
    first = [s.value for s in sig.score(trajs, ctx)]
    second = [s.value for s in sig.score(trajs, ctx)]
    assert first == second


def test_requires_sim_state_token() -> None:
    assert "sim_state" in SimPhysicsValidity().spec.requires


def test_registered_as_builtin_entry_point() -> None:
    assert "sim_physics_validity" in signals.available()
    assert isinstance(signals.get("sim_physics_validity"), SimPhysicsValidity)
