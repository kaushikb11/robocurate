"""Tests for the Jerk signal: known-answer, determinism, skips, and integration.

The known-answer test is the one the blueprint asks for: a tiny synthetic dataset where the
bad (jerky) trajectory is known in advance, and we assert it gets flagged / removed.
"""

from __future__ import annotations

import numpy as np

from robocurate import signals
from robocurate.curator import Budget, Curator
from robocurate.signals.jerk import Jerk
from robocurate.trajectory import (
    Array,
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)
from tests.synthetic import TOY_EMBODIMENT, make_signal_context, make_trajectory


def _traj_with_action(episode_index: int, action: Array, *, dt: float = 0.1) -> Trajectory:
    """Build a toy trajectory carrying a specific (T, 2) action sequence."""
    num_steps = action.shape[0]
    action = action.astype(np.float32)
    t = (np.arange(num_steps, dtype=np.float32) * dt).astype(np.float32)
    state = np.cumsum(action, axis=0).astype(np.float32)
    reward = np.zeros(num_steps, dtype=np.float32)
    columns = {
        "timestamp": t,
        "action": action,
        "observation.state": state,
        "reward": reward,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/jerk",
        episode_index=episode_index,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
        success=SuccessLabel(value=True, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _smooth_action(num_steps: int = 16) -> Array:
    # Linear ramp -> near-zero second derivative -> low jerk.
    ramp = np.linspace(-1.0, 1.0, num_steps)
    return np.stack([ramp, ramp], axis=-1)


def _jerky_action(num_steps: int = 16) -> Array:
    # Alternating +1/-1 -> large second difference -> high jerk.
    alt = np.where(np.arange(num_steps) % 2 == 0, 1.0, -1.0)
    return np.stack([alt, alt], axis=-1)


def test_jerky_scores_higher_than_smooth() -> None:
    sig = Jerk()
    ctx = make_signal_context()
    smooth = _traj_with_action(0, _smooth_action())
    jerky = _traj_with_action(1, _jerky_action())

    [s_smooth, s_jerky] = sig.score([smooth, jerky], ctx)
    assert not s_smooth.skipped and not s_jerky.skipped
    assert s_jerky.value > s_smooth.value  # jerky is rougher
    assert s_smooth.higher_is_better is False  # lower jerk = better
    # Per-transition jerk is emitted with one value per timestep.
    assert s_jerky.per_transition is not None
    assert s_jerky.per_transition.shape == (16,)


def test_known_answer_curation_removes_the_jerky_trajectory() -> None:
    # Three smooth + one known-jerky; keep 3 of 4 -> the jerky one must be dropped.
    trajs = [_traj_with_action(i, _smooth_action()) for i in range(3)]
    trajs.append(_traj_with_action(3, _jerky_action()))
    reader = _ListReader(trajs)

    result = Curator([Jerk()], budget=Budget.count(3), seed=0).run(reader)
    assert 3 in result.removed_episode_indices
    assert set(result.kept_episode_indices) == {0, 1, 2}
    # The decision explains why.
    jerky_decision = next(d for d in result.decisions if d.episode_index == 3)
    assert not jerky_decision.kept
    assert "jerk" in jerky_decision.signal_values


def test_score_is_deterministic() -> None:
    sig = Jerk()
    ctx = make_signal_context(seed=42)
    trajs = [_traj_with_action(i, _jerky_action()) for i in range(3)]
    first = [s.value for s in sig.score(trajs, ctx)]
    second = [s.value for s in sig.score(trajs, ctx)]
    assert first == second


def test_skips_when_source_missing() -> None:
    # A trajectory with no action feature.
    store = InMemoryFeatureStore(
        {
            "timestamp": np.arange(4, dtype=np.float32) * 0.1,
            "observation.state": np.zeros((4, 2), dtype=np.float32),
        }
    )
    traj = Trajectory(make_trajectory(0).meta, store)
    [score] = Jerk().score([traj], make_signal_context())
    assert score.skipped and "action" in (score.skip_reason or "")


def test_skips_when_too_short() -> None:
    # deriv_order 2 needs >= 3 steps; give 2.
    traj = _traj_with_action(0, _jerky_action(num_steps=2))
    [score] = Jerk(deriv_order=2).score([traj], make_signal_context())
    assert score.skipped and "too short" in (score.skip_reason or "")


def test_configurable_source_and_order() -> None:
    sig = Jerk(source="observation.state", deriv_order=3)
    assert sig.spec.requires == frozenset({"observation.state"})
    [score] = sig.score([_traj_with_action(0, _jerky_action())], make_signal_context())
    assert not score.skipped
    assert score.diagnostics["source"] == "observation.state"
    assert score.diagnostics["deriv_order"] == 3


def test_registered_as_builtin_entry_point() -> None:
    assert "jerk" in signals.available()
    assert isinstance(signals.get("jerk"), Jerk)


class _ListReader:
    """Minimal in-memory DatasetReader over a fixed list of trajectories (test support)."""

    def __init__(self, trajs: list[Trajectory]) -> None:
        self._trajs = trajs
        from robocurate.metadata import DatasetFingerprint, DatasetMeta

        self.meta = DatasetMeta(
            fingerprint=DatasetFingerprint("synthetic/jerk", "synthetic_v0", "0" * 64, len(trajs)),
            embodiment_ids=(TOY_EMBODIMENT.embodiment_id,),
            feature_keys=tuple(s.key for s in TOY_EMBODIMENT.features),
        )

    def __len__(self) -> int:
        return len(self._trajs)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._trajs)

    def read_episode(self, index: int) -> Trajectory:
        return self._trajs[index]

    def fingerprint(self):  # type: ignore[no-untyped-def]
        return self.meta.fingerprint
