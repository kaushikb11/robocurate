"""End-to-end tests for the ``explain`` CLI subcommand.

A tiny synthetic dataset is curated and saved (producing a manifest), then ``explain`` is
asked about specific episodes. The command surfaces the kept/removed status, the recorded
reason, and the per-signal values — never mutating anything (it only reads the manifest).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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


def _curate(tmp_path: Path, *, num_episodes: int = 6) -> Path:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=num_episodes)
    result = Curator(
        [FakeActionMagnitudeSignal()],
        budget=Budget.fraction(0.5),
        seed=7,
    ).run(LeRobotReader(src))
    result.save(tmp_path / "curated")
    return tmp_path / "curated"


def _decisions(curated: Path) -> list[dict[str, Any]]:
    manifest = json.loads((curated / "manifest.json").read_text())
    decisions: list[dict[str, Any]] = manifest["decisions"]
    return decisions


def test_explain_shows_reason_for_removed_episode(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    curated = _curate(tmp_path)
    removed = next(d for d in _decisions(curated) if not d["kept"])

    rc = main(["explain", str(curated / "manifest.json"), str(removed["episode_index"])])
    assert rc == 0
    out = capsys.readouterr().out

    assert f"Episode {removed['episode_index']}" in out
    assert "REMOVED" in out
    assert removed["reason"] in out
    # Per-signal values are surfaced.
    assert "fake_action_magnitude" in out


def test_explain_shows_kept_episode(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    curated = _curate(tmp_path)
    kept = next(d for d in _decisions(curated) if d["kept"])

    rc = main(["explain", str(curated), str(kept["episode_index"])])
    assert rc == 0
    out = capsys.readouterr().out
    assert "KEPT" in out
    assert kept["reason"] in out


def test_explain_json_is_machine_readable(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    curated = _curate(tmp_path)
    decision = _decisions(curated)[0]

    rc = main(["explain", str(curated / "manifest.json"), str(decision["episode_index"]), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["episode_index"] == decision["episode_index"]
    assert data["kept"] == decision["kept"]
    assert data["reason"] == decision["reason"]


def test_explain_unknown_episode_errors(tmp_path: Path) -> None:
    curated = _curate(tmp_path)
    with pytest.raises(SystemExit):
        main(["explain", str(curated / "manifest.json"), "9999"])


def test_explain_missing_manifest_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["explain", str(tmp_path / "nope"), "0"])
