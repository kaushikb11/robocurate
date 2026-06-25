"""Tests for the ManiSkill demonstration reader (needs h5py; rlds-style optional extra).

Builds a synthetic ManiSkill-format HDF5 with h5py (no mani_skill needed) and checks the
conversion to canonical trajectories and that they flow through the curator + config.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

import h5py

from robocurate.adapters import ManiSkillDemoReader
from robocurate.curator import Budget, Curator
from robocurate.experiment import ExperimentConfig, run_config
from robocurate.signals.jerk import Jerk
from robocurate.trajectory import FeatureRole

pytestmark = pytest.mark.maniskill


def _write_demo(path: Path, *, num_traj: int = 4, horizon: int = 8, obs_dim: int = 5) -> None:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for i in range(num_traj):
            g = f.create_group(f"traj_{i}")
            # ManiSkill stores T+1 observations for T actions.
            g.create_dataset("obs", data=rng.normal(size=(horizon + 1, obs_dim)).astype("f4"))
            g.create_dataset("actions", data=rng.normal(size=(horizon, 2)).astype("f4"))
            g.create_dataset("rewards", data=rng.normal(size=(horizon,)).astype("f4"))
            success = np.zeros(horizon, dtype=bool)
            success[-1] = i % 2 == 0  # even trajectories succeed
            g.create_dataset("success", data=success)


def test_reads_maniskill_demo_h5(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.state.h5"
    _write_demo(path, num_traj=4, horizon=8, obs_dim=5)
    reader = ManiSkillDemoReader(path)

    assert len(reader) == 4
    traj = reader.read_episode(0)
    assert traj.num_steps == 8  # aligned to the number of actions
    assert traj.feature("action").shape == (8, 2)
    assert traj.feature("observation.state").shape == (8, 5)  # the T+1th obs is dropped
    assert traj.feature("reward").shape == (8,)


def test_success_labels_and_roles(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo(path)
    reader = ManiSkillDemoReader(path)

    assert reader.read_episode(0).success().value is True  # type: ignore[union-attr]
    assert reader.read_episode(1).success().value is False  # type: ignore[union-attr]

    emb = reader.read_episode(0).embodiment
    assert emb.feature("observation.state").role is FeatureRole.PROPRIO  # type: ignore[union-attr]
    assert emb.feature("action").role is FeatureRole.ACTION  # type: ignore[union-attr]


def test_demos_flow_through_curator(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo(path, num_traj=6, horizon=10)
    reader = ManiSkillDemoReader(path)
    result = Curator([Jerk()], budget=Budget.fraction(0.5), seed=0).run(reader)
    assert result.num_kept == 3
    assert result.baseline is not None


def test_config_builds_maniskill_dataset(tmp_path: Path) -> None:
    path = tmp_path / "demo.h5"
    _write_demo(path, num_traj=6, horizon=10)
    # The full-vertical config shape, but with the fake env/policy so no GPU is needed here.
    config = ExperimentConfig(
        dataset={"kind": "maniskill_demos", "params": {"path": str(path)}},
        signals=[{"name": "jerk", "params": {}}],
        budget={"kind": "fraction", "value": 0.5},
        policy={"name": "fake", "params": {}},
        environment={"name": "fake", "params": {}},
        seeds=[0, 1],
        eval_episodes=50,
    )
    report = run_config(config)
    assert report.total_episodes == 6
    assert report.curated_vs_equal_n is not None


def test_dict_obs_raises(tmp_path: Path) -> None:
    path = tmp_path / "dict_obs.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("traj_0")
        obs = g.create_group("obs")  # a dict/visual observation, not supported in v1
        obs.create_dataset("state", data=np.zeros((9, 3), dtype="f4"))
        g.create_dataset("actions", data=np.zeros((8, 2), dtype="f4"))
    with pytest.raises(NotImplementedError, match="dict/visual"):
        ManiSkillDemoReader(path)
