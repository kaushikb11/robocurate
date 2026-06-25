"""Tests for the read-only dataset health module (``robocurate.health``).

These exercise the diagnosis on a tiny synthetic LeRobot dataset and assert it never
mutates the source (Invariant 1): a clean dataset reports OK with full feature coverage,
and the report round-trips through ``to_dict`` / ``to_markdown``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from robocurate.adapters import LeRobotReader
from robocurate.adapters.lerobot import _trajectory_to_table
from robocurate.health import HealthReport, dataset_health
from tests.synthetic import make_trajectory, write_synthetic_lerobot_dataset


def _checksum_tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def test_health_clean_dataset_is_ok(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=5, num_steps=8)
    report = dataset_health(LeRobotReader(src))

    assert isinstance(report, HealthReport)
    assert report.num_episodes == 5
    assert report.ok
    assert report.structural.num_valid == 5
    assert report.structural.num_invalid == 0
    assert report.structural.num_truncated == 0
    assert report.structural.num_nonfinite == 0
    assert report.structural.defect_episode_indices == ()


def test_health_reports_feature_coverage(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    report = dataset_health(LeRobotReader(src))

    by_key = {c.key: c for c in report.coverage}
    # The toy embodiment's declared features are all present on every episode.
    for key in ("timestamp", "action", "observation.state", "reward"):
        assert key in by_key
        assert by_key[key].present_episodes == 4
        assert by_key[key].coverage == 1.0
        assert by_key[key].finite_fraction == 1.0
    # Finite stats are populated for a numeric feature.
    assert by_key["action"].mean is not None


def test_health_flags_truncated_episode(tmp_path: Path) -> None:
    # One episode far shorter than the others -> truncation defect.
    src = tmp_path / "src"
    write_synthetic_lerobot_dataset(src, num_episodes=5, num_steps=40)
    # Overwrite a single episode parquet with a much shorter one and update episodes.jsonl.
    short = make_trajectory(0, num_steps=3)
    table, _ = _trajectory_to_table(short, 0, 0)
    pq.write_table(table, src / "data" / "chunk-000" / "episode_000000.parquet")  # type: ignore[no-untyped-call]
    ep_path = src / "meta" / "episodes.jsonl"
    records = [json.loads(line) for line in ep_path.read_text().splitlines() if line.strip()]
    for r in records:
        if r["episode_index"] == 0:
            r["length"] = 3
    ep_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")

    report = dataset_health(LeRobotReader(src))
    assert report.structural.num_truncated >= 1
    assert 0 in report.structural.defect_episode_indices
    assert not report.ok


def test_health_is_read_only(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    before = _checksum_tree(src)
    dataset_health(LeRobotReader(src))
    after = _checksum_tree(src)
    assert before == after  # source is byte-for-byte untouched (Invariant 1)


def test_health_report_serializes(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=3)
    report = dataset_health(LeRobotReader(src))

    d = report.to_dict()
    assert d["num_episodes"] == 3
    assert d["ok"] is True
    assert d["structural"]["num_valid"] == 3
    assert isinstance(d["coverage"], list) and d["coverage"]

    md = report.to_markdown()
    assert "Dataset health" in md
    assert "Structural validity" in md
    assert "Feature coverage" in md
