"""Contract tests for the dataset adapter layer.

Covers the read-only guarantee (invariant 1), the write-new + round-trip + checksum
validation path (invariant 2), and that a written dataset reloads to identical content.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from robocurate.adapters import (
    DatasetReader,
    LeRobotReader,
    LeRobotVersion,
    LeRobotWriter,
    SourceWriteError,
)
from robocurate.manifest import EpisodeDecision, Manifest, code_version
from robocurate.metadata import DatasetFingerprint
from tests.synthetic import write_synthetic_lerobot_dataset


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _minimal_manifest(source_fp: DatasetFingerprint, kept: list[int]) -> Manifest:
    decisions = tuple(
        EpisodeDecision(
            episode_index=i,
            fingerprint=f"ep{i}",
            kept=i in kept,
            reason="kept" if i in kept else "removed: synthetic test drop",
        )
        for i in range(source_fp.num_episodes)
    )
    return Manifest(
        schema_version="1",
        source=source_fp,
        output=source_fp,  # placeholder; writer computes the true output fingerprint
        config_dict={"combiner": "test"},
        seed=0,
        code_version=code_version(),
        signals=(),
        decisions=decisions,
        baseline=None,
    )


def test_reader_has_no_write_method() -> None:
    # The read-only guarantee is structural: the reader type exposes no way to write.
    assert not hasattr(LeRobotReader, "write")
    assert not hasattr(LeRobotReader, "save")
    assert not hasattr(DatasetReader, "write")


def test_read_roundtrips_content(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=3)
    reader = LeRobotReader(src)

    assert len(reader) == 3
    trajs = list(reader)
    assert len(trajs) == 3
    # Feature content survives the parquet round-trip with correct shapes.
    a0 = trajs[0].feature("action")
    assert a0.shape == (8, 2)
    assert reader.read_episode(1).meta.episode_index == 1


def test_success_labels_round_trip(tmp_path: Path) -> None:
    labels: list[bool | None] = [True, False, None]
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=3, success=labels)

    # Read: episode-level success labels are reconstructed (incl. None = unknown).
    reader = LeRobotReader(src)
    read_back = [reader.read_episode(i).success() for i in range(3)]
    assert [s.value if s else "missing" for s in read_back] == [True, False, None]

    # Write a curated subset and confirm the labels survive into the new dataset.
    source_fp = reader.fingerprint()
    keep = [reader.read_episode(i) for i in (0, 2)]
    LeRobotWriter(tmp_path / "out", source_root=src).write(
        keep, _minimal_manifest(source_fp, [0, 2])
    )
    out = LeRobotReader(tmp_path / "out")
    assert [t.success().value for t in out] == [True, None]  # type: ignore[union-attr]


def test_no_success_field_means_no_label(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=2)  # no success arg
    reader = LeRobotReader(src)
    assert reader.read_episode(0).success() is None


def test_curation_write_leaves_source_untouched(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=4)
    before = _hash_tree(src)

    reader = LeRobotReader(src)
    source_fp = reader.fingerprint()
    kept = [0, 2, 3]  # drop episode 1
    keep_trajs = [reader.read_episode(i) for i in kept]
    manifest = _minimal_manifest(source_fp, kept)

    writer = LeRobotWriter(tmp_path / "curated", source_root=src)
    receipt = writer.write(keep_trajs, manifest)

    # Invariant 1: every source file is byte-for-byte unchanged.
    assert _hash_tree(src) == before
    # Invariant 2: the written dataset validated (schema + checksum + round-trip).
    assert receipt.validation is not None and receipt.validation.ok
    assert receipt.manifest_path.is_file()
    assert receipt.file_checksums  # non-empty checksum map

    # The curated dataset reloads to exactly the kept episodes, in order.
    curated = LeRobotReader(tmp_path / "curated")
    assert len(curated) == 3
    reloaded_fps = [t.meta.fingerprint for t in curated]
    assert reloaded_fps == [t.meta.fingerprint for t in keep_trajs]


def test_writer_refuses_existing_destination(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=2)
    (tmp_path / "exists").mkdir()
    with pytest.raises(SourceWriteError, match="already exists"):
        LeRobotWriter(tmp_path / "exists", source_root=src)


def test_writer_refuses_writing_into_source(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=2)
    with pytest.raises(SourceWriteError, match=r"overlaps source|equals the source"):
        LeRobotWriter(src / "nested_out", source_root=src)


def test_v2_1_reader_redirects_v3_to_the_v3_reader(tmp_path: Path) -> None:
    # v3 is now implemented (LeRobotReaderV3); the v2.1 reader rejects a v3 version with a
    # message pointing there, rather than silently mis-handling the different layout.
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=1)
    with pytest.raises(NotImplementedError, match="LeRobotReaderV3"):
        LeRobotReader(src, version=LeRobotVersion.V3)
