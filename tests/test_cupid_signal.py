"""Tests for the CUPID-inspired proxy-influence signal (requires torch).

Marked ``ml``: skipped in the core-only CI run.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from robocurate.curator import Budget, Curator
from robocurate.signals.cupid import Cupid
from robocurate.trajectory import (
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)
from tests.synthetic import TOY_EMBODIMENT, make_signal_context
from tests.test_jerk_signal import _ListReader

pytestmark = pytest.mark.ml


def _bc_traj(idx: int, *, sign: float, seed: int, num_steps: int = 16) -> Trajectory:
    """A trajectory whose action is ``sign * state`` — 'helpful' (+1) or 'harmful' (-1)."""
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 1.0, size=(num_steps, 2)).astype(np.float32)
    action = (sign * state + rng.normal(0.0, 0.01, size=(num_steps, 2))).astype(np.float32)
    columns = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) * 0.1),
        "action": action,
        "observation.state": state,
        "reward": np.zeros(num_steps, dtype=np.float32),
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/cupid",
        episode_index=idx,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
        success=SuccessLabel(value=True, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _dataset() -> _ListReader:
    # Majority 'helpful' (action = +state) + a few contradictory 'harmful' (action = -state).
    helpful = [_bc_traj(i, sign=1.0, seed=i) for i in range(10)]
    harmful = [_bc_traj(100 + i, sign=-1.0, seed=100 + i) for i in range(3)]
    return _ListReader(helpful + harmful)


def _influence(reader: _ListReader, sig: Cupid) -> dict[str, float]:
    ctx = make_signal_context(seed=0)
    sig.fit(iter(reader), ctx)
    scored = sig.score(list(reader), ctx)
    return {s.trajectory_fingerprint: s.value for s in scored if not s.skipped}


def test_tracin_harmful_below_helpful() -> None:
    reader = _dataset()
    values = _influence(reader, Cupid(mode="tracin"))
    helpful = [values[reader.read_episode(i).meta.fingerprint] for i in range(10)]
    harmful = [values[reader.read_episode(10 + i).meta.fingerprint] for i in range(3)]
    # Contradictory trajectories have lower (often negative) influence on the val objective.
    assert max(harmful) < min(helpful)


def test_known_answer_curation_drops_harmful() -> None:
    reader = _dataset()
    result = Curator([Cupid(mode="tracin")], budget=Budget.count(10), seed=0).run(reader)
    # The three contradictory trajectories (episode_index 100..102) are the ones removed.
    assert set(result.removed_episode_indices) == {100, 101, 102}


def test_self_influence_mode_orientation_and_values() -> None:
    sig = Cupid(mode="self_influence")
    assert sig.spec.deterministic
    reader = _dataset()
    ctx = make_signal_context(seed=0)
    sig.fit(iter(reader), ctx)
    scores = [s for s in sig.score(list(reader), ctx) if not s.skipped]
    assert scores and all(s.higher_is_better is False for s in scores)
    assert all(s.value >= 0.0 for s in scores)  # gradient magnitudes are non-negative


def test_score_is_deterministic() -> None:
    reader = _dataset()
    first = _influence(reader, Cupid(mode="tracin"))
    second = _influence(reader, Cupid(mode="tracin"))
    assert first == second


def test_untrainable_too_few_skips() -> None:
    reader = _ListReader([_bc_traj(0, sign=1.0, seed=0)])
    ctx = make_signal_context(seed=0)
    sig = Cupid()
    sig.fit(iter(reader), ctx)
    scores = sig.score(list(reader), ctx)
    assert all(s.skipped for s in scores)
    assert "too few" in (scores[0].skip_reason or "")


def test_skips_trajectory_without_state_or_action() -> None:
    store = InMemoryFeatureStore(
        {"timestamp": np.arange(8, dtype=np.float32) * 0.1, "reward": np.zeros(8, np.float32)}
    )
    bad = Trajectory(_bc_traj(0, sign=1.0, seed=0).meta, store)
    reader = _ListReader([*[_bc_traj(i, sign=1.0, seed=i) for i in range(1, 6)], bad])
    ctx = make_signal_context(seed=0)
    sig = Cupid()
    sig.fit(iter(reader), ctx)
    by_fp = {s.trajectory_fingerprint: s for s in sig.score(list(reader), ctx)}
    assert by_fp[bad.meta.fingerprint].skipped


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        Cupid(mode="nonsense")


def test_registered_as_builtin_entry_point() -> None:
    from robocurate import signals

    assert "cupid" in signals.available()
    assert isinstance(signals.get("cupid"), Cupid)
