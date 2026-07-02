"""Failure-tolerant reading: ``Curator(on_error="quarantine")`` and the abort default.

One corrupt episode must never abort a whole scoring run — but only when the user opts in:
the default stays ``"abort"`` because a data-integrity tool must not silently tolerate
corruption. Under ``"quarantine"`` an unreadable episode is recorded as an unconditional
removal (never a silent drop), excluded from the valid + equal-N baseline pools, and
summarized in one warning; determinism and the recipe round-trip are preserved.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from robocurate import signals
from robocurate.adapters import LeRobotReader
from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.curator import Budget, Curator, WeightedSum
from robocurate.manifest import EpisodeDecision
from robocurate.recipe import load_recipe, save_recipe
from tests.synthetic import FakeActionMagnitudeSignal, make_trajectory

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from robocurate.metadata import DatasetFingerprint
    from robocurate.trajectory import Trajectory


class FlakyReader:
    """A reader whose episode at ``bad_index`` raises on read (a simulated corrupt shard).

    Wraps :class:`InMemoryDatasetReader` and fails in both read paths — ``read_episode``
    and streaming iteration — the way a corrupt parquet file would.
    """

    def __init__(self, inner: InMemoryDatasetReader, bad_index: int) -> None:
        self._inner = inner
        self._bad_index = bad_index
        self.meta = inner.meta

    def __len__(self) -> int:
        return len(self._inner)

    def __iter__(self) -> Iterator[Trajectory]:
        for index in range(len(self._inner)):
            yield self.read_episode(index)

    def read_episode(self, index: int) -> Trajectory:
        if index == self._bad_index:
            raise RuntimeError("simulated corrupt parquet shard")
        return self._inner.read_episode(index)

    def fingerprint(self) -> DatasetFingerprint:
        return self._inner.fingerprint()


def _flaky_reader(num_episodes: int = 6, bad_index: int = 2) -> FlakyReader:
    inner = InMemoryDatasetReader([make_trajectory(i) for i in range(num_episodes)])
    return FlakyReader(inner, bad_index)


def _decision_tuples(decisions: tuple[EpisodeDecision, ...]) -> list[tuple[int, str, bool, str]]:
    return [(d.episode_index, d.fingerprint, d.kept, d.reason) for d in decisions]


def test_default_abort_propagates_reading_error() -> None:
    curator = Curator([FakeActionMagnitudeSignal()], budget=Budget.fraction(0.5))
    with pytest.raises(RuntimeError, match="simulated corrupt parquet shard"):
        curator.run(_flaky_reader())


def test_invalid_on_error_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="on_error"):
        Curator([FakeActionMagnitudeSignal()], on_error="ignore")


def test_quarantine_completes_and_records_unconditional_removal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    curator = Curator(
        [FakeActionMagnitudeSignal()],
        budget=Budget.count(4),
        seed=0,
        on_error="quarantine",
    )
    with caplog.at_level(logging.WARNING, logger="robocurate.curator"):
        result = curator.run(_flaky_reader(num_episodes=6, bad_index=2))

    # The run completed and every source episode has a decision (never a silent drop).
    assert len(result.decisions) == 6
    by_index = {d.episode_index: d for d in result.decisions}
    bad = by_index[2]
    assert not bad.kept
    assert bad.reason == (
        "quarantined: unreadable episode (RuntimeError: simulated corrupt parquet shard)"
    )
    assert 2 in result.removed_episode_indices
    assert 2 not in result.kept_episode_indices

    # Budget applies to the readable pool: count(4) of the 5 readable episodes.
    assert result.num_kept == 4

    # The equal-N baseline pool excludes the quarantined episode (like the validity gate).
    assert result.baseline is not None
    assert result.baseline.n == 4
    assert 2 not in result.baseline.kept_episode_indices

    # ONE summary warning was logged.
    assert "quarantined 1 unreadable episode(s) of 6" in caplog.text


def test_quarantine_is_deterministic() -> None:
    def run_once() -> tuple[list[tuple[int, str, bool, str]], tuple[int, ...]]:
        curator = Curator(
            [FakeActionMagnitudeSignal()],
            budget=Budget.fraction(0.5),
            seed=7,
            on_error="quarantine",
        )
        result = curator.run(_flaky_reader(num_episodes=6, bad_index=1))
        assert result.baseline is not None
        return _decision_tuples(result.decisions), result.baseline.kept_episode_indices

    decisions_a, baseline_a = run_once()
    decisions_b, baseline_b = run_once()
    # Byte-identical decisions and baseline: same input -> same exceptions -> same output.
    assert decisions_a == decisions_b
    assert baseline_a == baseline_b


def test_quarantine_survives_corrupt_v21_parquet_on_disk(tmp_path: Path) -> None:
    from tests.synthetic import write_synthetic_lerobot_dataset

    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=5)
    # Open first (LeRobotReader fingerprints all episodes at construction), then corrupt
    # exactly one episode's parquet bytes — v2.1 keeps one file per episode, so precisely
    # one episode becomes unreadable at scoring time.
    reader = LeRobotReader(src)
    (src / "data" / "chunk-000" / "episode_000002.parquet").write_bytes(b"\x00not a parquet")

    curator = Curator(
        [FakeActionMagnitudeSignal()], budget=Budget.fraction(1.0), on_error="quarantine"
    )
    result = curator.run(reader)

    assert len(result.decisions) == 5
    by_index = {d.episode_index: d for d in result.decisions}
    assert not by_index[2].kept
    assert by_index[2].reason.startswith("quarantined: unreadable episode (")
    assert result.num_kept == 4  # the whole readable pool
    assert sorted(result.kept_episode_indices) == [0, 1, 3, 4]


@pytest.fixture
def fake_signal_registered() -> Iterator[None]:
    signals.register("fake_action_magnitude", FakeActionMagnitudeSignal, overwrite=True)
    try:
        yield
    finally:
        signals.unregister("fake_action_magnitude")


def test_recipe_roundtrip_preserves_on_error_and_decisions(
    tmp_path: Path, fake_signal_registered: None
) -> None:
    original = Curator(
        [signals.get("fake_action_magnitude")],
        combiner=WeightedSum(weights={"fake_action_magnitude": 1.0}),
        budget=Budget.fraction(0.5),
        seed=3,
        on_error="quarantine",
    )
    recipe_path = tmp_path / "recipe.json"
    save_recipe(original, recipe_path)
    reloaded = load_recipe(recipe_path)
    assert reloaded.on_error == "quarantine"

    before = _decision_tuples(original.run(_flaky_reader(bad_index=1)).decisions)
    after = _decision_tuples(reloaded.run(_flaky_reader(bad_index=1)).decisions)
    # Byte-identical decisions through the recipe round-trip (Invariant 3), including the
    # quarantine removal.
    assert before == after
    assert any(reason.startswith("quarantined: unreadable episode") for *_, reason in after)


def test_cli_curate_records_on_error_in_manifest(
    tmp_path: Path, fake_signal_registered: None
) -> None:
    import json

    from robocurate.cli import main
    from tests.synthetic import write_synthetic_lerobot_dataset

    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
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
            "--on-error",
            "quarantine",
        ]
    )
    assert rc == 0
    manifest = json.loads((tmp_path / "curated" / "manifest.json").read_text())
    assert manifest["config"]["on_error"] == "quarantine"
