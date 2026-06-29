"""End-to-end tests for the ``compare`` CLI subcommand.

Two curation runs at different budgets are saved (each producing a manifest), then
``compare`` diffs them: kept-set sizes, the Jaccard overlap of kept episode indices, how many
episodes flipped kept<->removed, and per-signal summary deltas. It only reads the manifests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robocurate.adapters import LeRobotReader
from robocurate.cli import main
from robocurate.curator import Budget, Curator
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


def _curate(src: Path, dest: Path, *, fraction: float) -> Path:
    result = Curator(
        [FakeActionMagnitudeSignal()],
        budget=Budget.fraction(fraction),
        seed=7,
    ).run(LeRobotReader(src))
    result.save(dest)
    return dest


def _two_runs(tmp_path: Path) -> tuple[Path, Path]:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    a = _curate(src, tmp_path / "a", fraction=0.5)  # keeps 3
    b = _curate(src, tmp_path / "b", fraction=1.0)  # keeps 6 (superset)
    return a, b


def test_compare_reports_jaccard_and_flips(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    a, b = _two_runs(tmp_path)
    rc = main(["compare", str(a / "manifest.json"), str(b / "manifest.json"), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)

    assert data["a"]["num_kept"] == 3
    assert data["b"]["num_kept"] == 6
    # B keeps everything A keeps plus 3 more: intersection 3, union 6 -> Jaccard 0.5.
    assert data["num_intersection"] == 3
    assert data["num_union"] == 6
    assert data["jaccard"] == pytest.approx(0.5)
    # The 3 extra episodes flipped removed->kept; none went the other way.
    assert data["num_flipped"] == 3
    assert data["kept_in_a_only"] == []
    assert len(data["kept_in_b_only"]) == 3


def test_compare_identical_runs_jaccard_one(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    a = _curate(src, tmp_path / "a", fraction=0.5)
    b = _curate(src, tmp_path / "b", fraction=0.5)
    rc = main(["compare", str(a), str(b), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["jaccard"] == pytest.approx(1.0)
    assert data["num_flipped"] == 0


def test_compare_signal_deltas_present(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    a, b = _two_runs(tmp_path)
    rc = main(["compare", str(a / "manifest.json"), str(b / "manifest.json"), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    deltas = data["signal_deltas"]
    assert "fake_action_magnitude" in deltas
    assert set(deltas["fake_action_magnitude"]) == {"min", "median", "max"}


def test_compare_markdown(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    a, b = _two_runs(tmp_path)
    rc = main(["compare", str(a / "manifest.json"), str(b / "manifest.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Curation comparison" in out
    assert "Jaccard overlap" in out
    assert "flipped" in out


def test_compare_missing_manifest_errors(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    a = _curate(src, tmp_path / "a", fraction=0.5)
    with pytest.raises(SystemExit):
        main(["compare", str(a / "manifest.json"), str(tmp_path / "nope")])
