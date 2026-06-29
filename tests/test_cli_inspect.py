"""End-to-end tests for the ``inspect`` CLI subcommand.

``inspect`` is ``score`` zoomed into a single episode: it runs the requested signals on one
trajectory and reports each signal's value, orientation, skip status, diagnostics, and — for
signals that produce one — a compact per-transition summary. It only reads the source.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from robocurate import signals
from robocurate.cli import main
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


@pytest.fixture
def fake_signal_registered() -> Iterator[None]:
    signals.register("fake_action_magnitude", FakeActionMagnitudeSignal, overwrite=True)
    try:
        yield
    finally:
        signals.unregister("fake_action_magnitude")


def _dataset(tmp_path: Path, *, num_episodes: int = 4) -> Path:
    return write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=num_episodes)


def test_inspect_default_signals_report_a_value(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src = _dataset(tmp_path)
    rc = main(["inspect", str(src), "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Episode 1" in out
    # The default Tier-0 signals score the low-dim toy episode.
    assert "jerk" in out
    assert "value:" in out


def test_inspect_json_has_signal_value_and_per_transition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_signal_registered: None
) -> None:
    src = _dataset(tmp_path)
    rc = main(["inspect", str(src), "0", "--signals", "fake_action_magnitude", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["episode_index"] == 0
    assert len(data["signals"]) == 1
    entry = data["signals"][0]
    assert entry["signal"] == "fake_action_magnitude"
    assert entry["value"] is not None
    assert entry["higher_is_better"] is False
    assert entry["skipped"] is False
    # FakeActionMagnitudeSignal produces a per-transition trace; the summary is present.
    pt = entry["per_transition"]
    assert pt["length"] == 8
    assert pt["min"] is not None and pt["median"] is not None and pt["max"] is not None
    assert pt["worst"]  # the worst step indices are surfaced
    assert all("step" in w and "value" in w for w in pt["worst"])


def test_inspect_markdown_shows_per_transition_for_episode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], fake_signal_registered: None
) -> None:
    src = _dataset(tmp_path)
    rc = main(["inspect", str(src), "2", "--signals", "fake_action_magnitude"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fake_action_magnitude" in out
    assert "per-transition" in out
    assert "worst steps" in out


def test_inspect_source_unchanged(tmp_path: Path) -> None:
    src = _dataset(tmp_path)
    before = _tree_snapshot(src)
    rc = main(["inspect", str(src), "0", "--json"])
    assert rc == 0
    assert _tree_snapshot(src) == before


def test_inspect_unknown_episode_errors(tmp_path: Path) -> None:
    src = _dataset(tmp_path)
    with pytest.raises(SystemExit):
        main(["inspect", str(src), "9999"])


def _tree_snapshot(root: Path) -> dict[str, int]:
    return {
        str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()
    }
