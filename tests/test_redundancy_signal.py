"""Tests for the Redundancy signal: known-answer dedup, embedding seam, skips."""

from __future__ import annotations

import numpy as np

from robocurate import signals
from robocurate.curator import Budget, Curator
from robocurate.signals.redundancy import Redundancy, statistical_embedding
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


def _traj(episode_index: int, action: Array) -> Trajectory:
    num_steps = action.shape[0]
    action = action.astype(np.float32)
    columns = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) * 0.1),
        "action": action,
        "observation.state": np.cumsum(action, axis=0).astype(np.float32),
        "reward": np.zeros(num_steps, dtype=np.float32),
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/redundancy",
        episode_index=episode_index,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
        success=SuccessLabel(value=True, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _ramp(num_steps: int, amp: float, *, noise: float = 0.0, seed: int = 0) -> Array:
    rng = np.random.default_rng(seed)
    base = np.linspace(-amp, amp, num_steps)
    jitter = rng.normal(0.0, noise, size=(num_steps, 2)) if noise > 0 else 0.0
    return np.stack([base, base], axis=-1) + jitter


def _dataset_with_cluster() -> _ListReader:
    # Three well-separated distinct trajectories, plus a tight cluster of three near-copies.
    distinct = [
        _traj(0, _ramp(24, amp=1.0)),
        _traj(1, _ramp(48, amp=8.0)),  # long + large
        _traj(2, _ramp(24, amp=0.1)),  # tiny magnitude
    ]
    cluster = [_traj(3 + i, _ramp(24, amp=4.0, noise=1e-4, seed=100 + i)) for i in range(3)]
    return _ListReader(distinct + cluster)


def _run(sig: Redundancy, reader: _ListReader) -> dict[str, float]:
    ctx = make_signal_context()
    sig.fit(iter(reader), ctx)
    return {s.trajectory_fingerprint: s.value for s in sig.score(list(reader), ctx)}


def test_near_duplicates_score_less_unique_than_distinct() -> None:
    reader = _dataset_with_cluster()
    values = _run(Redundancy(), reader)
    cluster_vals = [values[reader.read_episode(i).meta.fingerprint] for i in (3, 4, 5)]
    distinct_vals = [values[reader.read_episode(i).meta.fingerprint] for i in (0, 1, 2)]
    # Every cluster member is closer to its neighbours (less unique) than any distinct one.
    assert max(cluster_vals) < min(distinct_vals)


def test_known_answer_keeps_distinct_drops_cluster_bloat() -> None:
    reader = _dataset_with_cluster()
    # Keep 4 of 6: the 3 distinct trajectories survive; only one cluster member is retained.
    result = Curator([Redundancy()], budget=Budget.count(4), seed=0).run(reader)
    assert {0, 1, 2}.issubset(set(result.kept_episode_indices))
    cluster_removed = {3, 4, 5} & set(result.removed_episode_indices)
    assert len(cluster_removed) == 2


def test_k_is_configurable_and_recorded() -> None:
    reader = _dataset_with_cluster()
    ctx = make_signal_context()
    sig = Redundancy(k=2)
    sig.fit(iter(reader), ctx)
    score = sig.score([reader.read_episode(3)], ctx)[0]
    assert score.diagnostics["k"] == 2


def test_custom_embedding_seam() -> None:
    # A trivial custom embedding: just the episode length. Pluggable with no core change.
    def length_only(traj: Trajectory) -> Array:
        return np.array([float(traj.num_steps)], dtype=np.float64)

    reader = _dataset_with_cluster()
    ctx = make_signal_context()
    sig = Redundancy(embedding=length_only)
    sig.fit(iter(reader), ctx)
    score = sig.score([reader.read_episode(0)], ctx)[0]
    assert not score.skipped
    assert score.diagnostics["embedding_dim"] == 1


def test_score_is_deterministic() -> None:
    reader = _dataset_with_cluster()
    first = _run(Redundancy(), reader)
    second = _run(Redundancy(), reader)
    assert first == second


def test_skips_when_no_embeddable_features() -> None:
    store = InMemoryFeatureStore(
        {"timestamp": np.arange(4, dtype=np.float32) * 0.1, "reward": np.zeros(4, np.float32)}
    )
    traj = Trajectory(make_trajectory(0).meta, store)
    sig = Redundancy()
    ctx = make_signal_context()
    sig.fit(iter([traj]), ctx)
    [score] = sig.score([traj], ctx)
    assert score.skipped and "no embeddable" in (score.skip_reason or "")


def test_skips_single_trajectory_dataset() -> None:
    reader = _ListReader([_traj(0, _ramp(24, amp=1.0))])
    sig = Redundancy()
    ctx = make_signal_context()
    sig.fit(iter(reader), ctx)
    [score] = sig.score(list(reader), ctx)
    assert score.skipped and "no other trajectories" in (score.skip_reason or "")


def test_statistical_embedding_is_fixed_length() -> None:
    emb = statistical_embedding(_traj(0, _ramp(24, amp=1.0)))
    assert emb is not None and emb.shape == (7,)


def test_registered_as_builtin_entry_point() -> None:
    assert "redundancy" in signals.available()
    assert isinstance(signals.get("redundancy"), Redundancy)
