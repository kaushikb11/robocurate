"""Tests for the SpectralSmoothness (SPARC) signal: known-answer, skips, curation.

Known answer: a smooth single-stroke motion has a compact low-frequency speed spectrum (SPARC
near 0), while a jerky/high-frequency motion adds spectral content (more negative SPARC). We
assert the smooth trajectory scores higher (closer to 0) than the jerky one, that it is
``higher_is_better``, that the curator drops the jerky trajectory, and that degenerate inputs
are skipped rather than scored.
"""

from __future__ import annotations

import numpy as np

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.curator import Budget, Curator
from robocurate.signals.spectral_smoothness import SpectralSmoothness
from robocurate.trajectory import (
    Array,
    InMemoryFeatureStore,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)
from tests.synthetic import TOY_EMBODIMENT, make_signal_context

_FS = 20.0  # control rate -> dt = 0.05


def _traj(episode_index: int, action: Array) -> Trajectory:
    """Build a toy trajectory with a specific (T, D) action sequence and real timestamps."""
    action = action.astype(np.float32)
    num_steps = action.shape[0]
    columns = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) / _FS),
        "action": action,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/sparc",
        episode_index=episode_index,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _smooth(num_steps: int = 64) -> Array:
    # A slow half-sine speed in x: compact low-frequency spectrum -> smooth.
    t = np.linspace(0.0, np.pi, num_steps)
    speed = np.sin(t)
    return np.stack([speed, np.zeros(num_steps), np.zeros(num_steps)], axis=-1)


def _jerky(num_steps: int = 64) -> Array:
    # The same gross motion plus a strong high-frequency wobble -> less smooth.
    t = np.linspace(0.0, np.pi, num_steps)
    speed = np.sin(t) + 0.5 * np.sin(np.arange(num_steps) * 2.2)
    return np.stack([speed, np.zeros(num_steps), np.zeros(num_steps)], axis=-1)


def test_smooth_scores_higher_than_jerky() -> None:
    sig = SpectralSmoothness(motion="increments")
    ctx = make_signal_context()
    smooth = _traj(0, _smooth())
    jerky = _traj(1, _jerky())

    [s_smooth, s_jerky] = sig.score([smooth, jerky], ctx)
    assert not s_smooth.skipped and not s_jerky.skipped
    assert s_smooth.higher_is_better is True
    assert s_smooth.value <= 0.0  # SPARC is non-positive
    assert s_smooth.value > s_jerky.value  # smoother is closer to 0


def test_known_answer_curation_removes_the_jerky_trajectory() -> None:
    trajs = [_traj(i, _smooth()) for i in range(3)]
    trajs.append(_traj(3, _jerky()))
    reader = InMemoryDatasetReader(trajs)
    result = Curator([SpectralSmoothness()], budget=Budget.count(3), seed=0).run(reader)
    assert 3 in result.removed_episode_indices
    assert set(result.kept_episode_indices) == {0, 1, 2}
    decision = next(d for d in result.decisions if d.episode_index == 3)
    assert not decision.kept and "spectral_smoothness" in decision.signal_values


def test_skips_when_source_missing() -> None:
    store = InMemoryFeatureStore({"timestamp": np.arange(8, dtype=np.float32) / _FS})
    traj = Trajectory(_traj(0, _smooth()).meta, store)
    [score] = SpectralSmoothness().score([traj], make_signal_context())
    assert score.skipped and "action" in (score.skip_reason or "")


def test_skips_on_zero_motion() -> None:
    [score] = SpectralSmoothness().score(
        [_traj(0, np.zeros((16, 3), dtype=np.float32))], make_signal_context()
    )
    assert score.skipped and "no motion" in (score.skip_reason or "")


def test_skips_when_too_short() -> None:
    [score] = SpectralSmoothness().score(
        [_traj(0, np.ones((3, 3), dtype=np.float32))], make_signal_context()
    )
    assert score.skipped and "too short" in (score.skip_reason or "")


def test_score_is_deterministic() -> None:
    sig = SpectralSmoothness()
    ctx = make_signal_context(seed=3)
    trajs = [_traj(i, _jerky()) for i in range(3)]
    first = [s.value for s in sig.score(trajs, ctx)]
    second = [s.value for s in sig.score(trajs, ctx)]
    assert first == second
