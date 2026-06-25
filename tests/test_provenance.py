"""Provenance: curating a curated dataset records a link to the parent manifest.

When a curated dataset (which carries its own ``manifest.json``) is itself curated again, the
new run's manifest must reference the parent manifest so the lineage is auditable end-to-end
(Invariant 6). A first-generation curation of un-curated source data records ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path

from robocurate.adapters import LeRobotReader
from robocurate.curator import Budget, Curator, WeightedSum
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


def _curator(seed: int = 0) -> Curator:
    return Curator(
        [FakeActionMagnitudeSignal()],
        combiner=WeightedSum(),
        budget=Budget.fraction(0.5),
        seed=seed,
    )


def test_first_generation_curation_has_no_parent(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = _curator().run(LeRobotReader(src))

    receipt = result.save(tmp_path / "curated", created_utc="2026-06-25T00:00:00Z")
    manifest = json.loads(receipt.manifest_path.read_text())
    # The source is plain (un-curated) data, so there is no parent manifest.
    assert manifest["parent_manifest_path"] is None


def test_curating_a_curated_dataset_links_parent_manifest(tmp_path: Path) -> None:
    # A: curate the raw source -> dataset A (carries A's manifest.json).
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=8)
    result_a = _curator(seed=1).run(LeRobotReader(src))
    receipt_a = result_a.save(tmp_path / "curated_a", created_utc="2026-06-25T00:00:00Z")
    assert receipt_a.manifest_path.is_file()

    # B: curate A's output -> dataset B; B's manifest must point back at A's manifest.
    result_b = _curator(seed=2).run(LeRobotReader(tmp_path / "curated_a"))
    receipt_b = result_b.save(tmp_path / "curated_b", created_utc="2026-06-25T00:00:00Z")

    manifest_b = json.loads(receipt_b.manifest_path.read_text())
    parent = manifest_b["parent_manifest_path"]
    assert parent is not None
    assert Path(parent) == receipt_a.manifest_path
    assert Path(parent).is_file()


def test_build_manifest_records_parent_before_save(tmp_path: Path) -> None:
    # The provenance link is computed from the (read-only) source reader, so it is present on
    # the in-memory manifest before any write occurs.
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result_a = _curator().run(LeRobotReader(src))
    result_a.save(tmp_path / "curated_a", created_utc="2026-06-25T00:00:00Z")

    result_b = _curator().run(LeRobotReader(tmp_path / "curated_a"))
    manifest_b = result_b.build_manifest()
    assert manifest_b.parent_manifest_path == str(tmp_path / "curated_a" / "manifest.json")


def test_card_written_on_save(tmp_path: Path) -> None:
    # A README dataset card is written into the curated output (and only there; source stays
    # read-only).
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = _curator().run(LeRobotReader(src))
    receipt = result.save(tmp_path / "curated", created_utc="2026-06-25T00:00:00Z")

    card = (Path(receipt.path) / "README.md").read_text()
    assert "RoboCurate curation summary" in card
    assert not (src / "README.md").exists()  # source untouched (Invariant 1)


def test_card_can_be_disabled(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = _curator().run(LeRobotReader(src))
    receipt = result.save(tmp_path / "curated", write_card=False)
    assert not (Path(receipt.path) / "README.md").exists()
