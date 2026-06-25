"""Contract tests for the Curator: budget selection, determinism, equal-N baseline."""

from __future__ import annotations

from pathlib import Path

from robocurate.adapters import LeRobotReader
from robocurate.curator import Budget, Curator, WeightedSum
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


def _curator(seed: int = 0, emit_baseline: bool = True) -> Curator:
    return Curator(
        [FakeActionMagnitudeSignal()],
        combiner=WeightedSum(),
        budget=Budget.fraction(0.5),
        seed=seed,
        emit_baseline=emit_baseline,
    )


def test_budget_selects_expected_count(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = _curator().run(LeRobotReader(src))

    assert result.num_kept == 3  # 50% of 6
    assert result.num_removed == 3
    assert set(result.kept_episode_indices) | set(result.removed_episode_indices) == set(range(6))
    # Every episode has a decision with an explanation.
    assert len(result.decisions) == 6
    assert all(d.reason for d in result.decisions)


def test_selection_is_deterministic(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    a = _curator(seed=7).run(LeRobotReader(src))
    b = _curator(seed=7).run(LeRobotReader(src))
    assert a.kept_episode_indices == b.kept_episode_indices
    assert a.removed_episode_indices == b.removed_episode_indices


def test_equal_n_baseline_matches_curated_size_and_is_seeded(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = _curator(seed=11).run(LeRobotReader(src))

    assert result.baseline is not None
    # Invariant 5: the baseline keeps the SAME number of episodes as the curated selection.
    assert result.baseline.n == result.num_kept
    assert len(result.baseline.kept_episode_indices) == result.num_kept
    assert result.baseline.method == "equal_n_random"

    # Same master seed -> same baseline draw (deterministic).
    again = _curator(seed=11).run(LeRobotReader(src))
    assert again.baseline is not None
    assert again.baseline.kept_episode_indices == result.baseline.kept_episode_indices


def test_baseline_can_be_disabled(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    result = _curator(emit_baseline=False).run(LeRobotReader(src))
    assert result.baseline is None


def test_result_saves_curated_dataset_and_manifest(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = _curator(seed=3).run(LeRobotReader(src))

    receipt = result.save(tmp_path / "curated", created_utc="2026-06-23T00:00:00Z")
    assert receipt.validation is not None and receipt.validation.ok
    assert receipt.manifest_path.is_file()

    curated = LeRobotReader(tmp_path / "curated")
    assert len(curated) == result.num_kept

    # Manifest records a decision per source episode and the baseline.
    import json

    manifest = json.loads(receipt.manifest_path.read_text())
    assert len(manifest["decisions"]) == 6
    assert manifest["baseline"]["n"] == result.num_kept
    assert manifest["seed"] == 3
