"""Contract tests for the scorecard / report output."""

from __future__ import annotations

import json
from pathlib import Path

from robocurate.adapters import LeRobotReader
from robocurate.curator import Budget, Curator, WeightedSum
from robocurate.scorecard import Scorecard
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


def _run(tmp_path: Path) -> Scorecard:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = Curator(
        [FakeActionMagnitudeSignal()],
        combiner=WeightedSum(),
        budget=Budget.fraction(0.5),
        seed=5,
    ).run(LeRobotReader(src))
    return result.scorecard()


def test_scorecard_json_is_machine_readable(tmp_path: Path) -> None:
    card = _run(tmp_path)
    data = json.loads(card.to_json())

    assert data["summary"]["num_episodes"] == 6
    assert data["summary"]["num_removed"] == 3
    assert data["baseline"]["n"] == 3
    assert len(data["flags"]) == 6
    assert data["per_signal"][0]["name"] == "fake_action_magnitude"
    # No eval attached => no downstream-gain claim.
    assert data["effects"] is None


def test_scorecard_markdown_explains_every_removal(tmp_path: Path) -> None:
    card = _run(tmp_path)
    md = card.to_markdown()

    assert "Curation scorecard" in md
    assert "Removed episodes" in md
    # Each removed episode appears with its reason (invariant 6: why-removed is explicit).
    removed = [f for f in card.flags if not f.kept]
    assert removed
    for flag in removed:
        assert f"episode {flag.episode_index}" in md
        assert flag.reason  # non-empty justification


def test_scorecard_without_eval_makes_no_gain_claim(tmp_path: Path) -> None:
    md = _run(tmp_path).to_markdown()
    assert "makes no" in md and "claim" in md


def test_hf_dataset_card_fragment(tmp_path: Path) -> None:
    card = _run(tmp_path)
    frag = card.to_hf_dataset_card()
    assert "RoboCurate curation summary" in frag
    assert "fake_action_magnitude" in frag
