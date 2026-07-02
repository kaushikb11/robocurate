"""End-to-end tests for the ``rank`` CLI subcommand.

``rank`` is the "worst N episodes" report (lerobot#3760: turn "watch 200 episodes" into
"watch these 8"): it scores every episode with cheap signals, combines them with the same
machinery the curator uses, and lists the lowest keep-score episodes worst-first with the
signals responsible. It is read-only, and episodes no requested signal could score are
reported as unscored rather than silently ranked neutral.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robocurate.cli import main
from robocurate.curator import Curator, WeightedSum
from robocurate.dataset import Dataset
from robocurate.examples import write_demo_dataset
from robocurate.signals.jerk import Jerk

# The demo dataset interleaves good/bad by construction: odd episode indices are the
# jittery, wandering ("bad") ones (see robocurate.examples).
_DEMO_EPISODES = 8
_BAD_EPISODES = {1, 3, 5, 7}


@pytest.fixture
def demo(tmp_path: Path) -> Path:
    return write_demo_dataset(tmp_path / "demo", num_episodes=_DEMO_EPISODES, seed=0)


def test_rank_flags_the_known_bad_episodes_worst(
    demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["rank", str(demo), "--worst", "4", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_episodes"] == _DEMO_EPISODES
    assert data["num_shown"] == 4
    # The construction-known bad episodes occupy exactly the worst 4 slots.
    assert {e["episode_index"] for e in data["worst"]} == _BAD_EPISODES
    # Worst-first: keep-scores are non-decreasing down the list.
    scores = [e["keep_score"] for e in data["worst"]]
    assert scores == sorted(scores)


def test_rank_worst_two_shows_exactly_two(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["rank", str(demo), "--worst", "2", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_shown"] == 2
    assert len(data["worst"]) == 2
    assert {e["episode_index"] for e in data["worst"]} <= _BAD_EPISODES


def test_rank_worst_is_capped_at_dataset_size(
    demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["rank", str(demo), "--worst", "100", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_shown"] == _DEMO_EPISODES
    assert len(data["worst"]) == _DEMO_EPISODES


def test_rank_json_payload_fields(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["rank", str(demo), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    for key in (
        "dataset",
        "signals",
        "seed",
        "num_episodes",
        "num_ranked",
        "num_shown",
        "worst",
        "unscored",
    ):
        assert key in data
    assert data["signals"] == ["jerk", "action_noise", "path_efficiency", "spectral_smoothness"]
    entry = data["worst"][0]
    for key in (
        "rank",
        "episode_index",
        "fingerprint",
        "num_steps",
        "keep_score",
        "worst_signals",
        "reason",
        "signals",
    ):
        assert key in entry
    assert 0.0 <= entry["keep_score"] <= 1.0
    # Per-signal breakdown carries the raw value AND the keep-oriented normalized score.
    per_signal = entry["signals"][0]
    for key in ("signal", "value", "normalized", "higher_is_better", "skipped", "skip_reason"):
        assert key in per_signal


def test_rank_every_entry_names_a_responsible_signal(
    demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Invariant 6: the combined keep-score is never a black box — each ranked line names
    # the signal(s) that drove it, in both the JSON and the human reason string.
    rc = main(["rank", str(demo), "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    for entry in data["worst"]:
        assert entry["worst_signals"], "each ranked entry must name at least one signal"
        named = entry["worst_signals"][0]["signal"]
        assert named in data["signals"]
        assert named in entry["reason"]
        assert entry["worst_signals"][0]["value"] is not None


def test_rank_output_is_byte_identical_across_runs(
    demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Invariant 3: same dataset, config, and seed -> byte-identical report (both formats).
    for flags in ([], ["--json"]):
        assert main(["rank", str(demo), "--seed", "7", *flags]) == 0
        first = capsys.readouterr().out
        assert main(["rank", str(demo), "--seed", "7", *flags]) == 0
        assert capsys.readouterr().out == first


def test_rank_markdown_header_and_caveat(demo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["rank", str(demo), "--worst", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"Showing the 3 lowest-scoring of {_DEMO_EPISODES} episodes" in out
    assert "diagnostics, not proof" in out
    assert "equal-N random baseline" in out
    # Each line names a signal and shows the keep-score.
    assert "worst on " in out
    assert "keep-score" in out


def test_rank_all_skipped_episodes_are_unscored_not_ranked(
    demo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The demo dataset has no sim-state features, so the sim-only signal skips every
    # episode: nothing may be ranked (the combiner's neutral 0.5 imputation must not
    # silently place them mid-pack), and every episode is reported with its skip reason.
    rc = main(["rank", str(demo), "--signals", "sim_physics_validity", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["num_ranked"] == 0
    assert data["worst"] == []
    assert len(data["unscored"]) == _DEMO_EPISODES
    assert all("sim state" in u["skip_reason"] for u in data["unscored"])

    rc = main(["rank", str(demo), "--signals", "sim_physics_validity"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Unscored episodes (excluded from the ranking)" in out
    assert "(no episode could be ranked)" in out


def test_rank_is_read_only(demo: Path) -> None:
    before = _tree_snapshot(demo)
    assert main(["rank", str(demo), "--json"]) == 0
    assert _tree_snapshot(demo) == before


def test_rank_rejects_worst_below_one(demo: Path) -> None:
    with pytest.raises(SystemExit):
        main(["rank", str(demo), "--worst", "0"])


def test_curation_result_exposes_the_combiners_keep_scores(demo: Path) -> None:
    # The rank command reuses the run's combined keep-score rather than recombining;
    # the exposed scores must be exactly what the WeightedSum combiner computes.
    reader = Dataset.from_lerobot(demo).reader
    result = Curator([Jerk()], seed=0, emit_baseline=False).run(reader)
    expected = WeightedSum().combined_score(result.score_matrix)
    assert len(result.keep_scores) == result.score_matrix.num_trajectories
    np.testing.assert_allclose(np.asarray(result.keep_scores), expected, rtol=0, atol=0)


def _tree_snapshot(root: Path) -> dict[str, int]:
    return {
        str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()
    }
