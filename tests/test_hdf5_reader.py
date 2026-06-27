"""Tests for the generic configurable HDF5 reader (needs h5py; the `hdf5` extra).

Builds synthetic HDF5 files with h5py in BOTH supported layouts — a root-level ``traj_*``
file with a flat ``obs`` array (read with :meth:`HDF5Schema.maniskill_like`) and a nested
``data/demo_*`` file with an ``obs`` group of named low-dim keys (read with
:meth:`HDF5Schema.robomimic_like`) — and checks the conversion to canonical trajectories:
correct feature shapes/roles/values, deterministic episode order + fingerprint, the
image-hint guard, that the source file is never mutated (invariant 1), and that the
trajectories flow end-to-end through the curator.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

import h5py

from robocurate.adapters import GenericHDF5Reader, HDF5Schema
from robocurate.curator import Budget, Curator
from robocurate.signals.jerk import Jerk
from robocurate.trajectory import FeatureRole

pytestmark = pytest.mark.hdf5

_FLAT_OBS_DIM = 5
_ACTION_DIM = 2
# group obs keys deliberately out of lexical order on disk to prove the reader sorts them.
_GROUP_OBS_KEYS = ("robot0_eef_pos", "object", "robot0_gripper_qpos")
_GROUP_OBS_DIMS = {"object": 4, "robot0_eef_pos": 3, "robot0_gripper_qpos": 2}


def _write_maniskill_like(path: Path, *, num_traj: int = 3, horizon: int = 8) -> None:
    """Root-level ``traj_*`` groups with a flat ``obs`` array, plus rewards + success."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for i in range(num_traj):
            g = f.create_group(f"traj_{i}")
            # T+1 observations for T actions, like ManiSkill.
            g.create_dataset("obs", data=rng.normal(size=(horizon + 1, _FLAT_OBS_DIM)).astype("f4"))
            g.create_dataset("actions", data=rng.normal(size=(horizon, _ACTION_DIM)).astype("f4"))
            g.create_dataset("rewards", data=rng.normal(size=(horizon,)).astype("f4"))
            success = np.zeros(horizon, dtype=bool)
            success[-1] = i % 2 == 0  # even trajectories succeed
            g.create_dataset("success", data=success)


def _write_robomimic_like(path: Path, *, num_demos: int = 3, horizon: int = 12) -> None:
    """Nested ``data/demo_*`` groups with an ``obs`` group of named low-dim keys + rewards."""
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        for i in range(num_demos):
            g = data.create_group(f"demo_{i}")
            t = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
            # demo 2 gets high-frequency jitter on the action -> high jerk (known answer).
            jitter = 0.6 * np.sin(np.arange(horizon) * 2.5) if i == 2 else 0.0
            actions = np.stack(
                [np.sin(t) + (jitter if d == 0 else 0.0) for d in range(_ACTION_DIM)], axis=1
            ).astype(np.float32)
            g.create_dataset("actions", data=actions)
            g.create_dataset("rewards", data=np.append(np.zeros(horizon - 1), 1.0).astype("f4"))
            obs = g.create_group("obs")
            for key in _GROUP_OBS_KEYS:
                obs.create_dataset(
                    key, data=rng.standard_normal((horizon, _GROUP_OBS_DIMS[key])).astype("f4")
                )


def test_maniskill_like_flat_array_layout(tmp_path: Path) -> None:
    path = tmp_path / "traj.state.h5"
    _write_maniskill_like(path, num_traj=3, horizon=8)
    reader = GenericHDF5Reader(path, schema=HDF5Schema.maniskill_like())

    assert len(reader) == 3
    trajs = list(reader)
    assert [t.meta.episode_index for t in trajs] == [0, 1, 2]
    assert [t.meta.extra["source_group"] for t in trajs] == ["traj_0", "traj_1", "traj_2"]

    first = trajs[0]
    assert first.num_steps == 8  # aligned to the number of actions
    assert first.feature("action").shape == (8, _ACTION_DIM)
    assert first.feature("observation.state").shape == (8, _FLAT_OBS_DIM)  # T+1th obs dropped
    assert first.feature("reward").shape == (8,)
    assert first.timestamps().shape == (8,)  # type: ignore[union-attr]

    emb = first.embodiment
    assert emb.feature("action").role is FeatureRole.ACTION  # type: ignore[union-attr]
    assert emb.feature("observation.state").role is FeatureRole.PROPRIO  # type: ignore[union-attr]
    assert emb.feature("reward").role is FeatureRole.REWARD  # type: ignore[union-attr]
    assert emb.feature("timestamp").role is FeatureRole.TIME  # type: ignore[union-attr]

    # success: even traj succeeds, odd fails.
    assert reader.read_episode(0).success().value is True  # type: ignore[union-attr]
    assert reader.read_episode(1).success().value is False  # type: ignore[union-attr]

    # values: obs/action/reward survive the conversion intact.
    with h5py.File(path, "r") as f:
        np.testing.assert_allclose(
            np.asarray(first.feature("action"), dtype=float), np.asarray(f["traj_0"]["actions"])
        )
        np.testing.assert_allclose(
            np.asarray(first.feature("observation.state"), dtype=float),
            np.asarray(f["traj_0"]["obs"])[:8],
        )
        np.testing.assert_allclose(
            np.asarray(first.feature("reward"), dtype=float),
            np.asarray(f["traj_0"]["rewards"]).reshape(-1),
        )


def test_robomimic_like_group_obs_layout(tmp_path: Path) -> None:
    path = tmp_path / "low_dim.hdf5"
    _write_robomimic_like(path, num_demos=3, horizon=12)
    reader = GenericHDF5Reader(path, schema=HDF5Schema.robomimic_like())

    assert len(reader) == 3
    trajs = list(reader)
    assert [t.meta.extra["source_group"] for t in trajs] == ["demo_0", "demo_1", "demo_2"]

    first = trajs[0]
    # flat state = concat of sorted obs keys: object(4)+robot0_eef_pos(3)+gripper(2)=9
    assert first.feature("observation.state").shape == (12, sum(_GROUP_OBS_DIMS.values()))
    assert first.feature("action").shape == (12, _ACTION_DIM)
    assert first.feature("reward").shape == (12,)
    assert first.embodiment.feature("observation.state").role is FeatureRole.PROPRIO  # type: ignore[union-attr]

    # state is the sorted-key concatenation (object, robot0_eef_pos, robot0_gripper_qpos).
    with h5py.File(path, "r") as f:
        obs = f["data"]["demo_0"]["obs"]
        expected = np.concatenate(
            [np.asarray(obs[k]) for k in sorted(_GROUP_OBS_KEYS)], axis=1
        ).astype("f4")
    np.testing.assert_allclose(
        np.asarray(first.feature("observation.state"), dtype=float), expected
    )


def test_deterministic_order_and_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "traj.h5"
    _write_maniskill_like(path)
    a = GenericHDF5Reader(path, schema=HDF5Schema.maniskill_like())
    b = GenericHDF5Reader(path, schema=HDF5Schema.maniskill_like())
    assert a.fingerprint() == b.fingerprint()
    assert [t.meta.extra["source_group"] for t in a] == [t.meta.extra["source_group"] for t in b]
    # natural sort by trailing integer: traj_0 < traj_1 < traj_2 (not lexical traj_0,1,2 anyway).
    assert [t.meta.extra["source_group"] for t in a] == ["traj_0", "traj_1", "traj_2"]


def test_image_hint_obs_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "with_image.hdf5"
    with h5py.File(path, "w") as f:
        g = f.create_group("data").create_group("demo_0")
        g.create_dataset("actions", data=np.zeros((6, _ACTION_DIM), dtype="f4"))
        obs = g.create_group("obs")
        obs.create_dataset("state", data=np.zeros((6, 4), dtype="f4"))
        obs.create_dataset("rgb_camera", data=np.zeros((6, 8), dtype="f4"))  # image hint
    with pytest.raises(NotImplementedError, match="image observation"):
        GenericHDF5Reader(path, schema=HDF5Schema.robomimic_like())


def test_source_file_is_not_mutated(tmp_path: Path) -> None:
    path = tmp_path / "traj.h5"
    _write_maniskill_like(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = GenericHDF5Reader(path, schema=HDF5Schema.maniskill_like())
    _ = list(reader)
    _ = reader.fingerprint()
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert after == before  # invariant 1: source is read-only


def test_flow_through_curator_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "low_dim.hdf5"
    _write_robomimic_like(path, num_demos=3, horizon=12)
    reader = GenericHDF5Reader(path, schema=HDF5Schema.robomimic_like())

    result = Curator([Jerk()], budget=Budget.fraction(0.67), seed=0).run(reader)
    assert result.num_kept == 2
    assert result.baseline is not None
    card = result.scorecard()
    assert card is not None
    # known answer: demo_2 carries the injected jitter -> worst jerk -> removed.
    removed = [d.episode_index for d in result.decisions if not d.kept]
    assert removed == [2]
