"""Tests for LeRobot v3.0 image/video **Stage-1 pass-through** (frames preserved, not decoded).

A v3 dataset's pixels live in per-camera mp4 shards, not the parquet table. Stage-1 curates such a
dataset by its *low-dim* signals and emits a valid v3 output that **preserves the kept episodes'
video frames by copying the backing shard files** — never decoding a single pixel. These tests
build a tiny but faithful v3 dataset on disk that *includes* a video feature (with the relational
``videos/<key>/chunk_index`` + ``file_index`` + timestamp columns and a stand-in mp4 shard file
serving as an opaque shard for path/checksum logic), curate it by a low-dim signal, and assert:

* the source directory is byte-for-byte unchanged (invariant 1);
* the output reloads via :class:`LeRobotReaderV3`;
* the kept video shard file(s) are present in the output and checksum-match the source
  (invariant 2);
* the low-dim content round-trips.

Only pyarrow is needed (a core dependency) — no video decoding, no extra/marker.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from robocurate.adapters.lerobot_v3 import LeRobotReaderV3
from robocurate.adapters.lerobot_v3_writer import LeRobotWriterV3
from robocurate.metadata import DatasetFingerprint
from robocurate.trajectory import FeatureRole, Trajectory, VideoReference
from tests.test_lerobot_v3_writer import _minimal_manifest

_NDArr = npt.NDArray[np.float32]
_FPS = 20.0
_VIDEO_KEY = "observation.images.cam"
_DATA_TMPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_EP_TMPL = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_TMPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _write_synthetic_v3_with_video(
    root: Path, *, lengths: list[int]
) -> tuple[dict[int, dict[str, _NDArr]], dict[int, bytes]]:
    """Write a tiny v3.0 dataset *with* a video feature; one mp4 shard per episode.

    Returns the per-episode low-dim arrays that were written and the per-episode opaque mp4 shard
    bytes (stand-ins for real encoded video), so tests can assert checksums round-trip. Each
    episode gets its own shard (``chunk_index=0``, ``file_index=ep``) so the kept/dropped split is
    easy to reason about; the from/to timestamps span the whole shard.
    """
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "toy_v3",
        "fps": _FPS,
        "total_episodes": len(lengths),
        "total_frames": int(sum(lengths)),
        "data_path": _DATA_TMPL,
        "video_path": _VIDEO_TMPL,
        "features": {
            "action": {"dtype": "float32", "shape": [2], "names": None},
            "observation.state": {"dtype": "float32", "shape": [2], "names": None},
            _VIDEO_KEY: {"dtype": "video", "shape": [48, 64, 3], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))

    rng = np.random.default_rng(0)
    written: dict[int, dict[str, _NDArr]] = {}
    shard_bytes: dict[int, bytes] = {}
    state_rows: list[list[float]] = []
    action_rows: list[list[float]] = []
    timestamp, frame_index, episode_index, global_index, task_index = [], [], [], [], []
    ep_meta: list[dict[str, object]] = []
    cursor = 0
    for ep, length in enumerate(lengths):
        state = rng.standard_normal((length, 2)).astype(np.float32)
        action = rng.standard_normal((length, 2)).astype(np.float32)
        written[ep] = {"observation.state": state, "action": action}
        for t in range(length):
            state_rows.append(state[t].tolist())
            action_rows.append(action[t].tolist())
            timestamp.append(t / _FPS)
            frame_index.append(t)
            episode_index.append(ep)
            global_index.append(cursor + t)
            task_index.append(0)

        # One opaque mp4 shard per episode (chunk 0, file=ep) — a stand-in, never decoded.
        shard_path = root / _VIDEO_TMPL.format(video_key=_VIDEO_KEY, chunk_index=0, file_index=ep)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"FAKE_MP4_EPISODE_{ep}".encode() + bytes(
            rng.integers(0, 256, size=32, dtype=np.uint8)
        )
        shard_path.write_bytes(payload)
        shard_bytes[ep] = payload

        ep_meta.append(
            {
                "episode_index": ep,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + length,
                f"videos/{_VIDEO_KEY}/chunk_index": 0,
                f"videos/{_VIDEO_KEY}/file_index": ep,
                f"videos/{_VIDEO_KEY}/from_timestamp": 0.0,
                f"videos/{_VIDEO_KEY}/to_timestamp": (length - 1) / _FPS,
                "length": length,
                "tasks": ["pick"],
            }
        )
        cursor += length

    data_table = pa.table(
        {
            "observation.state": pa.array(state_rows, type=pa.list_(pa.float32())),
            "action": pa.array(action_rows, type=pa.list_(pa.float32())),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
        }
    )
    pq.write_table(data_table, root / _DATA_TMPL.format(chunk_index=0, file_index=0))  # type: ignore[no-untyped-call]
    pq.write_table(
        pa.Table.from_pylist(ep_meta),
        root / _EP_TMPL.format(chunk_index=0, file_index=0),  # type: ignore[no-untyped-call]
    )
    return written, shard_bytes


def _source(src: Path) -> tuple[LeRobotReaderV3, DatasetFingerprint, list[Trajectory]]:
    reader = LeRobotReaderV3(src)
    return reader, reader.fingerprint(), list(reader)


def test_reader_builds_video_references_without_decoding(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _write_synthetic_v3_with_video(src, lengths=[5, 7, 6])
    reader = LeRobotReaderV3(src)

    # The video feature is still recorded but NOT a parquet column (existing reader contract).
    assert _VIDEO_KEY in reader.meta.extra["video_features"]
    assert _VIDEO_KEY not in {fs.key for fs in reader._embodiment.features}

    ep1 = reader.read_episode(1)
    refs = ep1.video_references()
    assert set(refs) == {_VIDEO_KEY}
    ref = refs[_VIDEO_KEY]
    assert isinstance(ref, VideoReference)
    assert ref.num_frames == 7  # one frame reference per timestep, no pixels loaded
    assert ref.frame_indices is not None and ref.frame_indices.shape == (7,)
    assert ref.shard_path is not None and ref.shard_path.is_file()  # resolves to the source shard
    assert ref.shard_chunk_index == 0 and ref.shard_file_index == 1


def test_video_refs_excluded_from_fingerprint(tmp_path: Path) -> None:
    # The content fingerprint is over low-dim columns only; the (un-decoded) video reference must
    # not perturb it. Identical low-dim content -> identical fingerprint, video present or not.
    src_v = tmp_path / "withvid"
    _write_synthetic_v3_with_video(src_v, lengths=[6, 6])
    fp_with_video = LeRobotReaderV3(src_v).read_episode(0).meta.fingerprint
    assert isinstance(fp_with_video, str) and fp_with_video  # stable low-dim hash, no pixel bytes


def test_passthrough_curation_preserves_frames_and_leaves_source_untouched(tmp_path: Path) -> None:
    src = tmp_path / "src"
    written, shard_bytes = _write_synthetic_v3_with_video(src, lengths=[5, 7, 6, 4])
    before = _hash_tree(src)

    reader, source_fp, trajs = _source(src)
    kept_idx = [0, 2, 3]  # drop episode 1
    keep_trajs = [reader.read_episode(i) for i in kept_idx]
    receipt = LeRobotWriterV3(tmp_path / "out", source_root=src).write(
        keep_trajs, _minimal_manifest(source_fp, kept_idx, total=len(trajs))
    )

    # (a) Source is byte-for-byte unchanged (invariant 1) — copy, never move/mutate.
    assert _hash_tree(src) == before
    assert receipt.validation is not None and receipt.validation.ok
    assert receipt.file_checksums

    # (b) Output reloads as a valid v3 dataset and re-declares its video feature.
    out = LeRobotReaderV3(tmp_path / "out")
    assert len(out) == 3
    assert _VIDEO_KEY in out.meta.extra["video_features"]
    assert '"codebase_version": "v3.0"' in (tmp_path / "out" / "meta" / "info.json").read_text()

    # (c) Each KEPT episode's video shard is present in the output AND byte-identical to its source.
    out_trajs = list(out)
    assert [t.meta.episode_index for t in out_trajs] == [0, 1, 2]  # re-indexed 0..k-1
    for out_traj, src_ep in zip(out_trajs, kept_idx, strict=True):
        ref = out_traj.video_references()[_VIDEO_KEY]
        assert ref.shard_path is not None and ref.shard_path.is_file()
        # The copied shard equals the original source shard for that source episode, byte-for-byte.
        assert ref.shard_path.read_bytes() == shard_bytes[src_ep]
        src_shard = src / _VIDEO_TMPL.format(video_key=_VIDEO_KEY, chunk_index=0, file_index=src_ep)
        assert (
            hashlib.sha256(ref.shard_path.read_bytes()).hexdigest()
            == hashlib.sha256(src_shard.read_bytes()).hexdigest()
        )

    # The dropped episode's shard (file-001) was not referenced by any kept episode -> not copied.
    assert not (
        tmp_path / "out" / _VIDEO_TMPL.format(video_key=_VIDEO_KEY, chunk_index=0, file_index=1)
    ).exists()

    # (d) Low-dim content round-trips for each kept episode.
    np.testing.assert_allclose(
        np.asarray(out_trajs[0].feature("action"), dtype=np.float32),
        written[0]["action"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(out_trajs[2].feature("observation.state"), dtype=np.float32),
        written[3]["observation.state"],
        rtol=1e-6,
    )


def test_curator_drives_passthrough_end_to_end(tmp_path: Path) -> None:
    from robocurate.curator import Budget, Curator
    from robocurate.signals.jerk import Jerk

    src = tmp_path / "src"
    _, shard_bytes = _write_synthetic_v3_with_video(src, lengths=[8, 8, 8, 8])
    before = _hash_tree(src)

    reader = LeRobotReaderV3(src)
    result = Curator([Jerk()], budget=Budget.fraction(0.5), seed=0).run(reader)
    assert result.num_kept == 2  # a low-dim signal curates a video-bearing v3 dataset

    kept_idx = sorted(result.kept_episode_indices)
    keep_trajs = [reader.read_episode(i) for i in kept_idx]
    LeRobotWriterV3(tmp_path / "out", source_root=src).write(
        keep_trajs, _minimal_manifest(reader.fingerprint(), kept_idx, total=4)
    )

    assert _hash_tree(src) == before  # source still untouched after a real curation run
    out = LeRobotReaderV3(tmp_path / "out")
    assert len(out) == 2
    for out_traj, src_ep in zip(out, kept_idx, strict=True):
        ref = out_traj.video_references()[_VIDEO_KEY]
        assert ref.shard_path is not None and ref.shard_path.read_bytes() == shard_bytes[src_ep]


def test_opaque_video_feature_without_shards_copies_nothing(tmp_path: Path) -> None:
    # The existing reader fixture declares a video feature but ships *no* shard files and no
    # videos/<key>/* metadata columns (an opaque, decode-only-later feature). Stage-1 must curate
    # cleanly: the feature is preserved as a declaration, references resolve to no shard
    # (shard_path is None), and nothing is copied into videos/.
    from tests.test_lerobot_v3_reader import _write_synthetic_v3

    src = tmp_path / "src"
    _write_synthetic_v3(src, lengths=[4, 6, 5])
    reader = LeRobotReaderV3(src)
    kept_idx = [0, 2]
    keep_trajs = [reader.read_episode(i) for i in kept_idx]
    LeRobotWriterV3(tmp_path / "out", source_root=src).write(
        keep_trajs, _minimal_manifest(reader.fingerprint(), kept_idx, total=3)
    )

    out = LeRobotReaderV3(tmp_path / "out")
    assert len(out) == 2
    refs = out.read_episode(0).video_references()
    assert set(refs) == {_VIDEO_KEY}  # the feature is still declared
    assert refs[_VIDEO_KEY].shard_path is None  # ...but it has no copyable shard
    assert not (tmp_path / "out" / "videos").exists()  # so nothing was copied


def test_image_role_inferred_for_video_key(tmp_path: Path) -> None:
    # Sanity: the video key would infer FeatureRole.IMAGE if ever surfaced as a spec.
    from robocurate.adapters.lerobot import _infer_role

    src = tmp_path / "src"
    _write_synthetic_v3_with_video(src, lengths=[3, 3])
    LeRobotReaderV3(src)  # constructs without raising on the video feature
    assert _infer_role(_VIDEO_KEY) is FeatureRole.IMAGE
