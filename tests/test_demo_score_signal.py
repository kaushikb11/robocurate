"""Tests for the Demo-SCORE-inspired learned quality classifier (requires torch).

Marked ``ml``: skipped in the core-only CI run that does not install the demo-score extra.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # skip the whole module if the demo-score extra is absent

from robocurate.curator import Budget, Curator
from robocurate.signals.demo_score import DemoScore
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


def _traj(idx: int, *, good: bool, success: bool | None, seed: int) -> Trajectory:
    """A 'good' trajectory is smooth; a 'bad' one is noisy. `success` is the (maybe wrong) label."""
    rng = np.random.default_rng(seed)
    num_steps = 20
    base = np.linspace(-2.0, 2.0, num_steps)
    noise = 0.02 if good else 0.6
    action = (np.stack([base, base], axis=-1) + rng.normal(0, noise, (num_steps, 2))).astype(
        np.float32
    )
    columns = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) * 0.1),
        "action": action,
        "observation.state": np.cumsum(action, axis=0).astype(np.float32),
        "reward": np.zeros(num_steps, dtype=np.float32),
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/demo_score",
        episode_index=idx,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
        success=SuccessLabel(value=success, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _labeled_dataset() -> _ListReader:
    good = [_traj(i, good=True, success=True, seed=i) for i in range(12)]
    bad = [_traj(100 + i, good=False, success=False, seed=100 + i) for i in range(12)]
    return _ListReader(good + bad)


def _scores(reader: _ListReader, sig: DemoScore) -> dict[str, float]:
    ctx = make_signal_context(seed=0)
    sig.fit(iter(reader), ctx)
    scored = sig.score(list(reader), ctx)
    return {s.trajectory_fingerprint: s.value for s in scored if not s.skipped}


def test_classifier_separates_good_from_bad() -> None:
    reader = _labeled_dataset()
    values = _scores(reader, DemoScore())
    good_p = [values[reader.read_episode(i).meta.fingerprint] for i in range(12)]
    bad_p = [values[reader.read_episode(12 + i).meta.fingerprint] for i in range(12)]
    # Out-of-fold predicted P(good) is higher for the genuinely-good trajectories.
    assert np.mean(good_p) > np.mean(bad_p)


def test_flags_suspected_mislabel() -> None:
    # 12 good(success) + 11 bad(fail) + one noisy trajectory mislabelled as success.
    trajs = [_traj(i, good=True, success=True, seed=i) for i in range(12)]
    trajs += [_traj(100 + i, good=False, success=False, seed=100 + i) for i in range(11)]
    mislabeled = _traj(999, good=False, success=True, seed=999)
    trajs.append(mislabeled)
    reader = _ListReader(trajs)

    ctx = make_signal_context(seed=0)
    sig = DemoScore()
    sig.fit(iter(reader), ctx)
    by_fp = {s.trajectory_fingerprint: s for s in sig.score(list(reader), ctx)}
    flagged = by_fp[mislabeled.meta.fingerprint]
    assert flagged.diagnostics["label"] is True
    assert flagged.diagnostics["p_good"] < 0.5
    assert flagged.diagnostics["suspected_mislabel"] is True


def test_known_answer_curation_drops_low_quality() -> None:
    reader = _labeled_dataset()
    result = Curator([DemoScore()], budget=Budget.fraction(0.5), seed=0).run(reader)
    # Keeping the better half should retain mostly 'good' (low-index) trajectories.
    kept_good = sum(1 for i in result.kept_episode_indices if i < 12)
    assert kept_good >= 9  # strong majority of the kept set are the genuinely-good ones


def test_score_is_deterministic() -> None:
    reader = _labeled_dataset()
    first = _scores(reader, DemoScore())
    second = _scores(reader, DemoScore())
    assert first == second  # torch CPU + fixed seed => byte-identical


def test_untrainable_single_class_skips_with_reason() -> None:
    # Only successes -> classifier cannot be trained.
    reader = _ListReader([_traj(i, good=True, success=True, seed=i) for i in range(8)])
    ctx = make_signal_context(seed=0)
    sig = DemoScore()
    sig.fit(iter(reader), ctx)
    scores = sig.score(list(reader), ctx)
    assert all(s.skipped for s in scores)
    assert "one class" in (scores[0].skip_reason or "")


def test_uses_default_statistical_embedding() -> None:
    from robocurate.signals.redundancy import statistical_embedding

    assert DemoScore().embedding is statistical_embedding
