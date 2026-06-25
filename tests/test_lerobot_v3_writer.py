"""Tests for the LeRobotDataset v3.0 write adapter.

Builds real :class:`Trajectory` objects (via the v3 reader over a tiny hand-written v3 source),
re-writes them with :class:`LeRobotWriterV3`, reloads with :class:`LeRobotReaderV3`, and asserts the
low-dim content round-trips, the source is byte-for-byte untouched (invariant 1), unsafe
destinations are refused (:class:`SourceWriteError`), and the curated output is itself readable and
curatable. Needs only pyarrow (a core dependency) — no extra/marker.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from robocurate.adapters.base import SourceWriteError
from robocurate.adapters.lerobot_v3 import LeRobotReaderV3
from robocurate.adapters.lerobot_v3_writer import LeRobotWriterV3
from robocurate.manifest import EpisodeDecision, Manifest, code_version
from robocurate.metadata import DatasetFingerprint
from robocurate.trajectory import Trajectory
from tests.test_lerobot_v3_reader import _write_synthetic_v3


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _minimal_manifest(source_fp: DatasetFingerprint, kept: list[int], total: int) -> Manifest:
    decisions = tuple(
        EpisodeDecision(
            episode_index=i,
            fingerprint=f"ep{i}",
            kept=i in kept,
            reason="kept" if i in kept else "removed: synthetic test drop",
        )
        for i in range(total)
    )
    return Manifest(
        schema_version="1",
        source=source_fp,
        output=source_fp,  # placeholder; the writer computes the true output fingerprint
        config_dict={"combiner": "test"},
        seed=0,
        code_version=code_version(),
        signals=(),
        decisions=decisions,
        baseline=None,
    )


def _source_trajectories(src: Path) -> tuple[LeRobotReaderV3, list[Trajectory]]:
    reader = LeRobotReaderV3(src)
    return reader, list(reader)


def test_writer_has_no_v2_downgrade_roundtrips_content(tmp_path: Path) -> None:
    src = tmp_path / "src"
    written = _write_synthetic_v3(src, lengths=[5, 7, 6])
    reader, trajs = _source_trajectories(src)
    source_fp = reader.fingerprint()

    kept_idx = [0, 2]  # drop episode 1
    keep_trajs = [reader.read_episode(i) for i in kept_idx]
    writer = LeRobotWriterV3(tmp_path / "out", source_root=src)
    receipt = writer.write(keep_trajs, _minimal_manifest(source_fp, kept_idx, total=len(trajs)))

    # Output is genuinely v3 (not downgraded to v2.1).
    out = LeRobotReaderV3(tmp_path / "out")
    info_version = (tmp_path / "out" / "meta" / "info.json").read_text()
    assert '"codebase_version": "v3.0"' in info_version
    assert receipt.validation is not None and receipt.validation.ok

    # Re-indexed 0..k-1 and the low-dim content round-trips for each kept episode.
    assert len(out) == 2
    out_trajs = list(out)
    assert [t.meta.episode_index for t in out_trajs] == [0, 1]
    np.testing.assert_allclose(
        np.asarray(out_trajs[0].feature("action"), dtype=np.float32),
        written[0]["action"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(out_trajs[1].feature("observation.state"), dtype=np.float32),
        written[2]["observation.state"],
        rtol=1e-6,
    )


def test_episode_slicing_and_lengths_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_synthetic_v3(src, lengths=[4, 9, 6, 3])
    reader = LeRobotReaderV3(src)
    source_fp = reader.fingerprint()

    kept_idx = [1, 3]
    keep_trajs = [reader.read_episode(i) for i in kept_idx]
    LeRobotWriterV3(tmp_path / "out", source_root=src).write(
        keep_trajs, _minimal_manifest(source_fp, kept_idx, total=4)
    )

    out = LeRobotReaderV3(tmp_path / "out")
    assert [t.meta.num_steps for t in out] == [9, 3]  # sliced correctly by episode_index
    assert out.read_episode(0).feature("action").shape == (9, 2)
    assert out.read_episode(1).feature("action").shape == (3, 2)


def test_fingerprint_stable_across_reloads(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_synthetic_v3(src, lengths=[5, 6])
    reader = LeRobotReaderV3(src)
    keep = list(reader)
    LeRobotWriterV3(tmp_path / "out", source_root=src).write(
        keep, _minimal_manifest(reader.fingerprint(), [0, 1], total=2)
    )
    a = LeRobotReaderV3(tmp_path / "out").fingerprint().content_hash
    b = LeRobotReaderV3(tmp_path / "out").fingerprint().content_hash
    assert a == b


def test_curation_write_leaves_source_untouched(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_synthetic_v3(src, lengths=[6, 6, 6, 6])
    before = _hash_tree(src)

    reader = LeRobotReaderV3(src)
    source_fp = reader.fingerprint()
    kept_idx = [0, 2, 3]
    keep_trajs = [reader.read_episode(i) for i in kept_idx]
    receipt = LeRobotWriterV3(tmp_path / "curated", source_root=src).write(
        keep_trajs, _minimal_manifest(source_fp, kept_idx, total=4)
    )

    # Invariant 1: every source file is byte-for-byte unchanged.
    assert _hash_tree(src) == before
    # Invariant 2: validated (schema + checksum + round-trip) with a manifest + checksums emitted.
    assert receipt.validation is not None and receipt.validation.ok
    assert receipt.manifest_path.is_file()
    assert receipt.file_checksums


def test_writer_refuses_existing_destination(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_synthetic_v3(src, lengths=[3, 3])
    (tmp_path / "exists").mkdir()
    with pytest.raises(SourceWriteError, match="already exists"):
        LeRobotWriterV3(tmp_path / "exists", source_root=src)


def test_writer_refuses_writing_into_or_equal_to_source(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_synthetic_v3(src, lengths=[3, 3])
    with pytest.raises(SourceWriteError, match=r"overlaps source|equals the source"):
        LeRobotWriterV3(src / "nested_out", source_root=src)
    # A dest equal to the source is also refused (here the existence guard fires first).
    with pytest.raises(SourceWriteError):
        LeRobotWriterV3(src, source_root=src)


def test_output_is_readable_and_curatable(tmp_path: Path) -> None:
    from robocurate.curator import Budget, Curator
    from robocurate.signals.jerk import Jerk

    src = tmp_path / "src"
    _write_synthetic_v3(src, lengths=[8, 8, 8, 8])
    reader = LeRobotReaderV3(src)
    keep = list(reader)
    LeRobotWriterV3(tmp_path / "out", source_root=src).write(
        keep, _minimal_manifest(reader.fingerprint(), [0, 1, 2, 3], total=4)
    )

    # The curated v3 output drives end-to-end through the curator just like a source v3 dataset.
    out = LeRobotReaderV3(tmp_path / "out")
    result = Curator([Jerk()], budget=Budget.fraction(0.5), seed=0).run(out)
    assert result.num_kept == 2
