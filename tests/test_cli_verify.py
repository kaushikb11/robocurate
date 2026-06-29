"""End-to-end tests for the ``verify`` CLI subcommand.

``verify`` makes Invariant 3 (determinism) user-facing: it rebuilds the curator from a saved
manifest's config, re-runs it on the dataset, and asserts the recomputed selection (and each
episode's reason) matches what the manifest recorded. A faithful re-run verifies; a tampered
manifest does not (and the command exits non-zero).
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


@pytest.fixture(autouse=True)
def fake_signal_registered() -> Iterator[None]:
    # verify rebuilds the curator from the manifest config, instantiating signals by name from
    # the registry, so the fake signal must be registered for the round-trip to reconstruct it.
    signals.register("fake_action_magnitude", FakeActionMagnitudeSignal, overwrite=True)
    try:
        yield
    finally:
        signals.unregister("fake_action_magnitude")


def _curate(tmp_path: Path, *, num_episodes: int = 6) -> tuple[Path, Path]:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=num_episodes)
    result = Curator(
        [FakeActionMagnitudeSignal()],
        budget=Budget.fraction(0.5),
        seed=7,
    ).run(LeRobotReader(src))
    result.save(tmp_path / "curated")
    return src, tmp_path / "curated"


def test_verify_true_for_faithful_manifest(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src, curated = _curate(tmp_path)
    rc = main(["verify", str(src), str(curated / "manifest.json"), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verified"] is True
    assert data["has_recorded_decisions"] is True
    assert data["mismatches"] == []


def test_verify_markdown_says_verified(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src, curated = _curate(tmp_path)
    rc = main(["verify", str(src), str(curated)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "verified: true" in out


def test_verify_false_for_tampered_manifest(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src, curated = _curate(tmp_path)
    manifest_path = curated / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Tamper: flip the kept flag on the first kept episode so the recorded selection no longer
    # matches what a faithful re-run produces.
    kept = next(d for d in manifest["decisions"] if d["kept"])
    kept["kept"] = False
    manifest_path.write_text(json.dumps(manifest))

    rc = main(["verify", str(src), str(manifest_path), "--json"])
    assert rc == 1  # non-zero exit on mismatch
    data = json.loads(capsys.readouterr().out)
    assert data["verified"] is False
    assert data["mismatches"]  # a precise mismatch summary is reported


def test_verify_detects_reason_tampering(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    src, curated = _curate(tmp_path)
    manifest_path = curated / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    # Keep the selection intact but corrupt a recorded reason string.
    manifest["decisions"][0]["reason"] = "this is not the real reason"
    manifest_path.write_text(json.dumps(manifest))

    rc = main(["verify", str(src), str(manifest_path), "--json"])
    assert rc == 1
    data = json.loads(capsys.readouterr().out)
    assert data["verified"] is False
    assert any("reason changed" in m for m in data["mismatches"])


def test_verify_missing_spec_errors(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    with pytest.raises(SystemExit):
        main(["verify", str(src), str(tmp_path / "nope")])
