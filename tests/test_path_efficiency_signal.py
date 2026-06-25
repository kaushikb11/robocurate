"""Tests for the PathEfficiency (directness) signal: known-answer, skips, dims, curation.

Known answer (what the blueprint asks for): a straight, committed motion is maximally direct
(directness -> 1); a back-and-forth motion covering the same ground is inefficient
(directness -> 0). We assert the ordering, the curator drops the meandering trajectory, the
translational-dims default ignores rotation/gripper noise, and degenerate inputs are skipped
rather than scored.
"""

from __future__ import annotations

import numpy as np

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.curator import Budget, Curator
from robocurate.signals.path_efficiency import PathEfficiency
from robocurate.trajectory import (
    Array,
    InMemoryFeatureStore,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)
from tests.synthetic import TOY_EMBODIMENT, make_signal_context


def _traj(episode_index: int, action: Array) -> Trajectory:
    """Build a toy trajectory carrying a specific (T, D) action sequence."""
    action = action.astype(np.float32)
    num_steps = action.shape[0]
    columns = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) * 0.1),
        "action": action,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/path_efficiency",
        episode_index=episode_index,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _straight(num_steps: int = 12) -> Array:
    # Constant-direction step: every increment aligned -> net == path -> directness 1.
    return np.tile(np.array([0.1, 0.0, 0.0]), (num_steps, 1))


def _meandering(num_steps: int = 12) -> Array:
    # Alternating +/- along x: net displacement ~ 0, path large -> directness ~ 0.
    alt = np.where(np.arange(num_steps) % 2 == 0, 0.1, -0.1)
    return np.stack([alt, np.zeros(num_steps), np.zeros(num_steps)], axis=-1)


def test_straight_is_more_direct_than_meandering() -> None:
    sig = PathEfficiency()
    ctx = make_signal_context()
    straight = _traj(0, _straight())
    meandering = _traj(1, _meandering())

    [s_straight, s_meander] = sig.score([straight, meandering], ctx)
    assert not s_straight.skipped and not s_meander.skipped
    assert s_straight.higher_is_better is True
    assert s_straight.value > 0.99  # essentially perfectly direct
    assert s_meander.value < 0.2  # covers ground but goes nowhere
    assert s_straight.value > s_meander.value


def test_translational_dims_default_ignores_rotation_noise() -> None:
    # A clean straight translation (dims 0:3) with heavy noise in extra (rotation/gripper)
    # dims must still score ~1 under the default dims=(0, 3).
    rng = np.random.default_rng(0)
    straight_xyz = _straight()
    noise = rng.standard_normal((straight_xyz.shape[0], 4)).astype(np.float32)
    action = np.concatenate([straight_xyz, noise], axis=1)
    [score] = PathEfficiency().score([_traj(0, action)], make_signal_context())
    assert score.value > 0.99
    # Using ALL dims instead would be dominated by the noise -> much less "direct".
    [all_dims] = PathEfficiency(dims=None).score([_traj(0, action)], make_signal_context())
    assert all_dims.value < score.value


def test_positions_mode_diffs_absolute_path() -> None:
    # Absolute positions tracing a straight ramp -> directness 1; a zig-zag path -> low.
    ramp = np.cumsum(_straight(), axis=0)  # straight absolute path
    zig = np.cumsum(_meandering(), axis=0)  # oscillating absolute path
    sig = PathEfficiency(motion="positions")
    [s_ramp, s_zig] = sig.score([_traj(0, ramp), _traj(1, zig)], make_signal_context())
    assert s_ramp.value > 0.99
    assert s_zig.value < s_ramp.value


def test_known_answer_curation_removes_the_meandering_trajectory() -> None:
    trajs = [_traj(i, _straight()) for i in range(3)]
    trajs.append(_traj(3, _meandering()))
    reader = InMemoryDatasetReader(trajs)
    result = Curator([PathEfficiency()], budget=Budget.count(3), seed=0).run(reader)
    assert 3 in result.removed_episode_indices
    assert set(result.kept_episode_indices) == {0, 1, 2}
    decision = next(d for d in result.decisions if d.episode_index == 3)
    assert not decision.kept and "path_efficiency" in decision.signal_values


def test_skips_when_source_missing() -> None:
    store = InMemoryFeatureStore({"timestamp": np.arange(4, dtype=np.float32) * 0.1})
    traj = Trajectory(_traj(0, _straight()).meta, store)
    [score] = PathEfficiency().score([traj], make_signal_context())
    assert score.skipped and "action" in (score.skip_reason or "")


def test_skips_on_zero_motion() -> None:
    [score] = PathEfficiency().score(
        [_traj(0, np.zeros((8, 3), dtype=np.float32))], make_signal_context()
    )
    assert score.skipped and "no motion" in (score.skip_reason or "")


def test_skips_when_too_short() -> None:
    [score] = PathEfficiency().score(
        [_traj(0, np.ones((1, 3), dtype=np.float32))], make_signal_context()
    )
    assert score.skipped and "too short" in (score.skip_reason or "")


def test_score_is_deterministic() -> None:
    sig = PathEfficiency()
    ctx = make_signal_context(seed=7)
    trajs = [_traj(i, _meandering()) for i in range(3)]
    first = [s.value for s in sig.score(trajs, ctx)]
    second = [s.value for s in sig.score(trajs, ctx)]
    assert first == second
