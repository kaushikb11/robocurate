"""Smoke tests for the CLI surface and the high-level Python API (Dataset facade)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from robocurate import Budget, Curator, Dataset, signals
from robocurate.cli import main
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


@pytest.fixture
def fake_signal_registered() -> Iterator[None]:
    signals.register("fake_action_magnitude", FakeActionMagnitudeSignal, overwrite=True)
    try:
        yield
    finally:
        signals.unregister("fake_action_magnitude")


def test_quickstart_shape(tmp_path: Path) -> None:
    # Mirrors the documented 5-line quickstart against a local dataset.
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    ds = Dataset.from_lerobot(src)
    result = Curator([FakeActionMagnitudeSignal()], budget=Budget.fraction(0.8)).run(ds)
    receipt = result.save(tmp_path / "curated")

    assert receipt.validation is not None and receipt.validation.ok
    assert "Curation scorecard" in result.scorecard().to_markdown()
    # The Dataset facade exposes only read access.
    assert not hasattr(Dataset, "write")


def test_cli_curate_end_to_end(tmp_path: Path, fake_signal_registered: None) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    rc = main(
        [
            "curate",
            str(src),
            "--out",
            str(tmp_path / "curated"),
            "--signals",
            "fake_action_magnitude",
            "--budget",
            "0.5",
            "--seed",
            "1",
            "--json",
        ]
    )
    assert rc == 0
    assert (tmp_path / "curated" / "manifest.json").is_file()


def test_cli_score_json(tmp_path: Path, fake_signal_registered: None, capsys) -> None:  # type: ignore[no-untyped-def]
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    rc = main(["score", str(src), "--signals", "fake_action_magnitude", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["summary"]["num_episodes"] == 4


def test_cli_score_without_signals_errors(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=2)
    with pytest.raises(SystemExit):
        main(["score", str(src)])
