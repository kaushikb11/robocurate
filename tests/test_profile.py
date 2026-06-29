"""Tests for the dataset-profile module and the ``profile`` CLI subcommand.

``dataset_profile`` is read-only EDA: episode counts and length distribution, per-feature
shape + value summaries, embodiment ids, success rate, task balance, and a cheap diversity
estimate. These tests pin its fields on a synthetic dataset and assert the source is never
touched (Invariant 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robocurate.adapters import LeRobotReader
from robocurate.cli import main
from robocurate.profile import Distribution, dataset_profile
from tests.synthetic import write_synthetic_lerobot_dataset


def test_distribution_from_values() -> None:
    d = Distribution.from_values([1.0, 2.0, 3.0])
    assert d.count == 3
    assert d.minimum == 1.0
    assert d.median == 2.0
    assert d.maximum == 3.0


def test_distribution_empty() -> None:
    d = Distribution.from_values([])
    assert d.count == 0
    assert d.minimum is None and d.median is None and d.maximum is None


def test_profile_basic_fields(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(
        tmp_path / "src", num_episodes=5, num_steps=8, success=[True, True, False, None, True]
    )
    report = dataset_profile(LeRobotReader(src))

    assert report.num_episodes == 5
    assert report.embodiment_ids == ("toy2dof",)
    # All episodes have 8 steps in this fixture.
    assert report.episode_lengths.minimum == 8.0
    assert report.episode_lengths.maximum == 8.0
    assert report.episode_lengths.median == 8.0

    # Feature summaries cover the declared low-dim features with correct dims.
    by_key = {f.key: f for f in report.features}
    assert by_key["action"].dim == 2
    assert by_key["observation.state"].dim == 2
    assert by_key["timestamp"].dim == 1
    assert by_key["action"].values is not None
    assert by_key["action"].values.count > 0


def test_profile_success_rate(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(
        tmp_path / "src", num_episodes=4, success=[True, True, False, None]
    )
    report = dataset_profile(LeRobotReader(src))
    assert report.num_success == 2
    assert report.num_failure == 1
    assert report.num_success_unknown == 1
    assert report.success_rate == pytest.approx(2 / 3)


def test_profile_no_success_labels(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=3)
    report = dataset_profile(LeRobotReader(src))
    # The fixture writes empty "tasks" but no "success" field, and the toy embodiment has no
    # SUCCESS-role column, so every episode is unlabelled.
    assert report.num_success_unknown == 3
    assert report.success_rate is None


def test_profile_diversity_estimate(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    report = dataset_profile(LeRobotReader(src))
    # Four distinct episodes are all embeddable; the mean NN distance is a finite positive.
    assert report.num_embedded == 4
    assert report.mean_nn_distance is not None
    assert report.mean_nn_distance > 0.0


def test_profile_source_unchanged(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    before = _tree_snapshot(src)
    dataset_profile(LeRobotReader(src))
    assert _tree_snapshot(src) == before


def test_profile_to_dict_roundtrips_to_json(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=3)
    report = dataset_profile(LeRobotReader(src))
    encoded = json.dumps(report.to_dict())  # must be JSON-serializable
    decoded = json.loads(encoded)
    assert decoded["num_episodes"] == 3
    assert decoded["diversity"]["num_embedded"] == 3


def test_profile_cli_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    rc = main(["profile", str(src), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_episodes"] == 4
    assert "episode_lengths" in data
    assert "features" in data


def test_profile_cli_markdown(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    rc = main(["profile", str(src)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dataset profile" in out
    assert "Episode length" in out
    assert "Features" in out
    assert "Diversity" in out


def _tree_snapshot(root: Path) -> dict[str, int]:
    return {
        str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()
    }
