"""Tests for the LeRobotDataset v3.0 read adapter.

Writes a tiny but faithful v3.0 layout directly with pyarrow (multi-episode data shard + the
relational ``meta/episodes`` table + ``meta/info.json`` with path templates and a video feature),
then asserts the reader round-trips the low-dim content, slices episodes correctly by
``episode_index``, records (but does not load) the video feature, leaves the source untouched, and
that ``Dataset.from_lerobot`` auto-detects the version. Needs only pyarrow (a core dependency).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from robocurate.adapters.base import LeRobotVersion
from robocurate.adapters.lerobot_v3 import LeRobotReaderV3, detect_lerobot_version
from robocurate.dataset import Dataset
from robocurate.trajectory import FeatureRole

_NDArr = npt.NDArray[np.float32]
_FPS = 20.0
_DATA_TMPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_EP_TMPL = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _write_synthetic_v3(root: Path, *, lengths: list[int]) -> dict[int, dict[str, _NDArr]]:
    """Write a tiny v3.0 dataset; return the per-episode state/action arrays that were written."""
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    info = {
        "codebase_version": "v3.0",
        "robot_type": "toy_v3",
        "fps": _FPS,
        "total_episodes": len(lengths),
        "total_frames": int(sum(lengths)),
        "data_path": _DATA_TMPL,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [2], "names": None},
            "observation.state": {"dtype": "float32", "shape": [2], "names": None},
            "observation.images.cam": {"dtype": "video", "shape": [48, 64, 3], "names": None},
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
        ep_meta.append(
            {
                "episode_index": ep,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": cursor,
                "dataset_to_index": cursor + length,
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
    return written


def test_detect_version_and_dispatch(tmp_path: Path) -> None:
    _write_synthetic_v3(tmp_path, lengths=[5, 7, 6])
    assert detect_lerobot_version(tmp_path) is LeRobotVersion.V3
    ds = Dataset.from_lerobot(tmp_path)  # auto-detect
    assert isinstance(ds.reader, LeRobotReaderV3)
    assert len(ds) == 3


def test_reads_episodes_with_correct_shapes_and_slicing(tmp_path: Path) -> None:
    written = _write_synthetic_v3(tmp_path, lengths=[5, 7, 6])
    reader = LeRobotReaderV3(tmp_path)
    assert len(reader) == 3

    trajs = list(reader)
    assert [t.meta.episode_index for t in trajs] == [0, 1, 2]
    assert [t.meta.num_steps for t in trajs] == [5, 7, 6]  # each episode sliced by episode_index

    ep1 = reader.read_episode(1)
    assert ep1.feature("observation.state").shape == (7, 2)
    assert ep1.feature("action").shape == (7, 2)
    # round-trip: the values match what was written for episode 1
    np.testing.assert_allclose(
        np.asarray(ep1.feature("action"), dtype=np.float32), written[1]["action"], rtol=1e-6
    )
    roles = {fs.key: fs.role for fs in ep1.embodiment.features}
    assert roles["action"] is FeatureRole.ACTION
    assert roles["observation.state"] is FeatureRole.PROPRIO


def test_timestamps_are_1d_so_dt_signals_dont_skip(tmp_path: Path) -> None:
    # v3 declares timestamp shape [1] -> stored as (T, 1); timestamps() must flatten to (T,) so
    # dt-dependent signals (jerk) see 1-D time and actually score instead of skipping.
    from robocurate.signals.jerk import Jerk
    from tests.synthetic import make_signal_context

    _write_synthetic_v3(tmp_path, lengths=[12, 12])
    ep = LeRobotReaderV3(tmp_path).read_episode(0)
    ts = ep.timestamps()
    assert ts is not None and ts.ndim == 1 and ts.shape == (12,)
    [score] = Jerk().score([ep], make_signal_context())
    assert not score.skipped  # jerk computes on v3 data (regression: it used to skip)


def test_video_feature_recorded_not_loaded(tmp_path: Path) -> None:
    _write_synthetic_v3(tmp_path, lengths=[4, 4])
    reader = LeRobotReaderV3(tmp_path)
    feature_keys = {fs.key for fs in reader._embodiment.features}
    assert "observation.images.cam" not in feature_keys  # video is not a parquet column
    assert "observation.images.cam" in reader.meta.extra["video_features"]
    assert "observation.state" in feature_keys  # low-dim features are present


def test_reader_has_no_write_method(tmp_path: Path) -> None:
    _write_synthetic_v3(tmp_path, lengths=[3, 3])
    reader = LeRobotReaderV3(tmp_path)
    assert not hasattr(reader, "write")  # read-only by construction (invariant 1)


def test_fingerprint_is_deterministic(tmp_path: Path) -> None:
    _write_synthetic_v3(tmp_path, lengths=[5, 6])
    a = LeRobotReaderV3(tmp_path).fingerprint().content_hash
    b = LeRobotReaderV3(tmp_path).fingerprint().content_hash
    assert a == b


def test_curation_runs_on_v3(tmp_path: Path) -> None:
    from robocurate.curator import Budget, Curator
    from robocurate.signals.jerk import Jerk

    _write_synthetic_v3(tmp_path, lengths=[8, 8, 8, 8])
    reader = LeRobotReaderV3(tmp_path)
    result = Curator([Jerk()], budget=Budget.fraction(0.5), seed=0).run(reader)
    assert result.num_kept == 2  # the curator drives end-to-end on a v3 source


@pytest.mark.lerobot
def test_reads_real_v3_hub_dataset() -> None:
    """Read a real v3.0 Hub dataset (low-dim only). Opt-in: needs huggingface_hub + network."""
    pytest.importorskip("huggingface_hub")
    from huggingface_hub import snapshot_download

    root = snapshot_download(
        repo_id="lerobot/svla_so101_pickplace",
        repo_type="dataset",
        allow_patterns=["meta/*", "meta/**", "data/**"],  # low-dim only; skip the mp4 shards
    )
    reader = LeRobotReaderV3(root)
    assert len(reader) == 50  # known episode count for this dataset
    ep0 = reader.read_episode(0)
    assert ep0.feature("action").shape[1] == 6  # SO-100 6-DoF action
    assert ep0.meta.num_steps > 0
    # the two camera streams are recorded as video features, not loaded as columns
    assert set(reader.meta.extra["video_features"]) == {
        "observation.images.up",
        "observation.images.side",
    }
