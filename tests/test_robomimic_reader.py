"""Tests for the robomimic demonstration reader (needs h5py; the `robomimic` extra).

Builds a synthetic robomimic-format HDF5 with h5py (no robomimic package needed) and checks
the conversion to canonical trajectories: the flat low-dim state is concatenated in a
deterministic key order, the per-demo proficiency tier from the ``mask/`` filter keys is
carried through on ``meta.extra``, the source file is never mutated, and the trajectories flow
through the curator. This is the known-answer test required for data I/O: a tiny dataset where
the bad (jerky) demo is known, asserting it gets the worst jerk score.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

import h5py

from robocurate.adapters import RoboMimicReader
from robocurate.curator import Budget, Curator
from robocurate.signals.jerk import Jerk
from robocurate.trajectory import FeatureRole

pytestmark = pytest.mark.robomimic

# obs keys deliberately out of lexical order on disk to prove the reader sorts them.
_OBS_KEYS = ("robot0_eef_pos", "object", "robot0_gripper_qpos")
_OBS_DIMS = {"object": 4, "robot0_eef_pos": 3, "robot0_gripper_qpos": 2}
_ACTION_DIM = 7


def _write_robomimic(path: Path, *, horizon: int = 12) -> None:
    """Write 3 demos: demo_0 'better' (smooth), demo_1/demo_2 'worse' (demo_2 is jerky)."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["total"] = 3 * horizon
        for i in range(3):
            g = data.create_group(f"demo_{i}")
            g.attrs["num_samples"] = horizon
            t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
            # demo_2 gets a high-frequency jitter added to the action -> high jerk.
            jitter = 0.6 * np.sin(np.arange(horizon) * 2.5) if i == 2 else 0.0
            actions = np.stack(
                [np.sin(t) + (jitter if d == 0 else 0.0) for d in range(_ACTION_DIM)], axis=1
            ).astype(np.float32)
            g.create_dataset("actions", data=actions)
            g.create_dataset("rewards", data=np.append(np.zeros(horizon - 1), 1.0).astype("f4"))
            g.create_dataset("dones", data=np.append(np.zeros(horizon - 1), 1).astype("i8"))
            obs = g.create_group("obs")
            for key in _OBS_KEYS:
                obs.create_dataset(key, data=rng.standard_normal((horizon, _OBS_DIMS[key])))
        mask = f.create_group("mask")
        mask.create_dataset("better", data=np.array([b"demo_0"]))
        mask.create_dataset("worse", data=np.array([b"demo_1", b"demo_2"]))
        # an orthogonal split that must NOT be read as a tier
        mask.create_dataset("train", data=np.array([b"demo_0", b"demo_1", b"demo_2"]))


def test_reads_demos_with_flat_state_and_tiers(tmp_path: Path) -> None:
    path = tmp_path / "low_dim.hdf5"
    _write_robomimic(path)
    reader = RoboMimicReader(path)

    assert len(reader) == 3
    trajs = list(reader)
    assert [t.meta.episode_index for t in trajs] == [0, 1, 2]

    first = trajs[0]
    # flat state = concat of sorted obs keys: object(4) + robot0_eef_pos(3) + gripper(2) = 9
    assert first.feature("observation.state").shape == (12, sum(_OBS_DIMS.values()))
    assert first.feature("action").shape == (12, _ACTION_DIM)
    assert first.embodiment.feature("action").role is FeatureRole.ACTION  # type: ignore[union-attr]
    assert first.embodiment.feature("observation.state").role is FeatureRole.PROPRIO  # type: ignore[union-attr]

    tiers = [t.meta.extra.get("operator_tier") for t in trajs]
    assert tiers == ["better", "worse", "worse"]  # 'train' split is not a tier
    assert [t.meta.extra["source_demo"] for t in trajs] == ["demo_0", "demo_1", "demo_2"]
    assert first.meta.success is not None and first.meta.success.value is True


def test_source_file_is_not_mutated(tmp_path: Path) -> None:
    path = tmp_path / "low_dim.hdf5"
    _write_robomimic(path)
    before = path.read_bytes()
    mtime = os.path.getmtime(path)
    reader = RoboMimicReader(path)
    _ = list(reader)
    assert path.read_bytes() == before  # invariant 1: source is read-only
    assert os.path.getmtime(path) == mtime


def test_jerky_demo_is_flagged_and_curation_runs(tmp_path: Path) -> None:
    path = tmp_path / "low_dim.hdf5"
    _write_robomimic(path)
    reader = RoboMimicReader(path)

    # Known answer: demo_2 carries the injected high-frequency jitter -> highest jerk, and it
    # flows through the curator end to end (drop the single worst-jerk demo).
    result = Curator([Jerk()], budget=Budget.fraction(2 / 3), seed=0).run(reader)
    jerk_by_episode = {
        ref.episode_index: value
        for ref, value in zip(
            result.score_matrix.refs, result.score_matrix.signal_values("jerk"), strict=True
        )
    }
    assert jerk_by_episode[2] == max(jerk_by_episode.values())
    assert result.num_kept == 2
    removed = [d.episode_index for d in result.decisions if not d.kept]
    assert removed == [2]


def test_explicit_obs_keys_order_controls_state_layout(tmp_path: Path) -> None:
    path = tmp_path / "low_dim.hdf5"
    _write_robomimic(path)
    reader = RoboMimicReader(path, obs_keys=["robot0_gripper_qpos", "object"])
    state = reader.read_episode(0).feature("observation.state")
    # only the two selected keys, in the given order: gripper(2) + object(4) = 6
    assert state.shape == (12, _OBS_DIMS["robot0_gripper_qpos"] + _OBS_DIMS["object"])
