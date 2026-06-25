"""End-to-end tests for the ``report`` and ``diff`` CLI subcommands.

Both run against a real curation: a tiny synthetic LeRobot dataset is curated and saved
(producing a manifest + curated dataset), then ``report`` re-renders that manifest and
``diff`` compares the source against the curated output. The source is never written
through either command (Invariant 1).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from robocurate import signals
from robocurate.adapters import LeRobotReader
from robocurate.cli import main
from robocurate.curator import Budget, Curator
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


@pytest.fixture
def fake_signal_registered() -> Iterator[None]:
    signals.register("fake_action_magnitude", FakeActionMagnitudeSignal, overwrite=True)
    try:
        yield
    finally:
        signals.unregister("fake_action_magnitude")


def _curate(tmp_path: Path, *, num_episodes: int = 6) -> tuple[Path, Path]:
    """Curate a synthetic dataset and return ``(source_dir, curated_dir)``."""
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=num_episodes)
    result = Curator(
        [FakeActionMagnitudeSignal()],
        budget=Budget.fraction(0.5),
        seed=7,
    ).run(LeRobotReader(src))
    result.save(tmp_path / "curated")
    return src, tmp_path / "curated"


def test_report_markdown_explains_every_removal(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _, curated = _curate(tmp_path)
    rc = main(["report", str(curated / "manifest.json")])
    assert rc == 0
    md = capsys.readouterr().out

    assert "Curation scorecard" in md
    assert "Removed episodes" in md
    # Each removed episode's reason is surfaced (invariant 6: why-removed is explicit).
    manifest = json.loads((curated / "manifest.json").read_text())
    removed = [d for d in manifest["decisions"] if not d["kept"]]
    assert removed
    for d in removed:
        assert f"episode {d['episode_index']}" in md
    assert "baseline" in md.lower()


def test_report_accepts_directory_path(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _, curated = _curate(tmp_path)
    rc = main(["report", str(curated)])
    assert rc == 0
    assert "Curation scorecard" in capsys.readouterr().out


def test_report_json_is_machine_readable(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    _, curated = _curate(tmp_path)
    rc = main(["report", str(curated / "manifest.json"), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)

    assert data["summary"]["num_episodes"] == 6
    assert data["summary"]["num_removed"] == 3
    assert data["baseline"]["n"] == 3
    assert len(data["flags"]) == 6
    assert data["per_signal"][0]["name"] == "fake_action_magnitude"
    # Reconstructed from a manifest => no downstream-eval claim.
    assert data["effects"] is None


def test_report_missing_manifest_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["report", str(tmp_path / "nope")])


def test_diff_reports_exactly_the_removed_episodes(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src, curated = _curate(tmp_path)
    manifest = json.loads((curated / "manifest.json").read_text())
    expected_removed = {d["episode_index"] for d in manifest["decisions"] if not d["kept"]}

    rc = main(["diff", str(src), str(curated), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)

    assert data["n_source"] == 6
    assert data["n_curated"] == 3
    assert data["n_removed"] == 3
    assert {r["episode_index"] for r in data["removed"]} == expected_removed


def test_diff_markdown_lists_removed_episodes(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src, curated = _curate(tmp_path)
    manifest = json.loads((curated / "manifest.json").read_text())
    expected_removed = [d["episode_index"] for d in manifest["decisions"] if not d["kept"]]

    rc = main(["diff", str(src), str(curated)])
    assert rc == 0
    out = capsys.readouterr().out

    assert "removed: 3" in out
    for index in expected_removed:
        assert f"episode {index}" in out


def test_diff_identical_datasets_reports_no_removals(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    rc = main(["diff", str(src), str(src), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_removed"] == 0
    assert data["removed"] == []
