"""Tests for the ActionNoise signal: jitter, outlier, fit seam, known-answer, skips."""

from __future__ import annotations

import numpy as np

from robocurate import signals
from robocurate.curator import Budget, Curator
from robocurate.signals.action_noise import ActionNoise
from robocurate.signals.base import InMemoryCache, NamespacedCache
from robocurate.trajectory import (
    Array,
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)
from tests.synthetic import TOY_EMBODIMENT, make_signal_context, make_trajectory
from tests.test_jerk_signal import _ListReader


def _traj(episode_index: int, action: Array, *, dt: float = 0.1) -> Trajectory:
    num_steps = action.shape[0]
    action = action.astype(np.float32)
    t = (np.arange(num_steps, dtype=np.float32) * dt).astype(np.float32)
    columns = {
        "timestamp": t,
        "action": action,
        "observation.state": np.cumsum(action, axis=0).astype(np.float32),
        "reward": np.zeros(num_steps, dtype=np.float32),
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/noise",
        episode_index=episode_index,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
        success=SuccessLabel(value=True, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _ramp(num_steps: int = 24, amp: float = 1.0, noise: float = 0.0, seed: int = 0) -> Array:
    rng = np.random.default_rng(seed)
    base = np.linspace(-amp, amp, num_steps)
    jitter = rng.normal(0.0, noise, size=(num_steps, 2)) if noise > 0 else 0.0
    return np.stack([base, base], axis=-1) + jitter


def _dataset() -> _ListReader:
    # Five clean ramps with mild, varied jitter; one heavily noisy (the known-bad one).
    trajs = [_traj(i, _ramp(amp=1.0 + 0.1 * i, noise=0.01, seed=i)) for i in range(5)]
    trajs.append(_traj(5, _ramp(amp=1.0, noise=0.5, seed=99)))  # known-bad
    return _ListReader(trajs)


def test_noisy_scores_worse_than_clean() -> None:
    sig = ActionNoise()
    reader = _dataset()
    ctx = make_signal_context()
    sig.fit(iter(reader), ctx)
    scores = {s.trajectory_fingerprint: s for s in sig.score(list(reader), ctx)}

    clean = scores[reader.read_episode(0).meta.fingerprint]
    noisy = scores[reader.read_episode(5).meta.fingerprint]
    assert noisy.value > clean.value  # higher badness for the noisy one
    assert noisy.diagnostics["jitter"] > clean.diagnostics["jitter"]
    assert noisy.per_transition is not None
    assert noisy.per_transition.shape == (24,)


def test_known_answer_removes_the_noisy_trajectory() -> None:
    reader = _dataset()
    result = Curator([ActionNoise()], budget=Budget.count(5), seed=0).run(reader)
    assert 5 in result.removed_episode_indices
    assert set(result.kept_episode_indices) == {0, 1, 2, 3, 4}


def test_fit_populates_namespaced_cache() -> None:
    sig = ActionNoise()
    ctx = make_signal_context()
    sig.fit(iter(_dataset()), ctx)
    # fit() stored dataset statistics that score() reads back.
    assert ctx.cache.has("stats")
    stats = ctx.cache.get("stats")
    assert stats is not None and "jitter_median" in stats


def test_namespaced_cache_isolates_signals() -> None:
    backing = InMemoryCache()
    a = NamespacedCache(backing, "action_noise@0.1.0")
    b = NamespacedCache(backing, "other@0.1.0")
    a.put("stats", 1)
    b.put("stats", 2)
    assert a.get("stats") == 1  # no collision despite identical logical key
    assert b.get("stats") == 2


def test_outlier_component_flags_a_length_outlier() -> None:
    # All low-jitter, but one trajectory is far longer -> a num_steps outlier.
    trajs = [_traj(i, _ramp(num_steps=20, noise=0.01, seed=i)) for i in range(6)]
    trajs.append(_traj(6, _ramp(num_steps=120, noise=0.01, seed=7)))
    reader = _ListReader(trajs)
    sig = ActionNoise()
    ctx = make_signal_context()
    sig.fit(iter(reader), ctx)
    scores = {s.trajectory_fingerprint: s for s in sig.score(list(reader), ctx)}
    odd = scores[reader.read_episode(6).meta.fingerprint]
    assert odd.diagnostics["is_outlier"] is True
    assert odd.diagnostics["outlier_z"] > 3.5


def test_score_is_deterministic() -> None:
    sig = ActionNoise()
    reader = _dataset()
    ctx = make_signal_context(seed=3)
    sig.fit(iter(reader), ctx)
    first = [s.value for s in sig.score(list(reader), ctx)]
    second = [s.value for s in sig.score(list(reader), ctx)]
    assert first == second


def test_skips_when_action_missing() -> None:
    store = InMemoryFeatureStore(
        {"timestamp": np.arange(5, dtype=np.float32) * 0.1, "reward": np.zeros(5, np.float32)}
    )
    traj = Trajectory(make_trajectory(0).meta, store)
    sig = ActionNoise()
    ctx = make_signal_context()
    sig.fit(iter([traj]), ctx)
    [score] = sig.score([traj], ctx)
    assert score.skipped and "action" in (score.skip_reason or "")


def test_skips_when_too_short() -> None:
    traj = _traj(0, _ramp(num_steps=2))
    sig = ActionNoise()
    ctx = make_signal_context()
    sig.fit(iter([traj]), ctx)
    [score] = sig.score([traj], ctx)
    assert score.skipped and "too short" in (score.skip_reason or "")


def test_registered_as_builtin_entry_point() -> None:
    assert "action_noise" in signals.available()
    assert isinstance(signals.get("action_noise"), ActionNoise)
