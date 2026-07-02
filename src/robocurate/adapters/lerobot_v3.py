"""LeRobotDataset **v3.0** read adapter (the current Hub default, ``lerobot >= 0.4.0``).

v3.0 changed the on-disk layout from v2.1 (see :mod:`robocurate.adapters.lerobot`): episodes are
no longer one-parquet-each but packed into **multi-episode shard files**, and the per-episode
metadata moved from ``episodes.jsonl`` to a relational parquet table. Layout (paths come from
``meta/info.json`` templates, not hardcoded)::

    <root>/meta/info.json                       # codebase_version "v3.0", fps, features, templates
    <root>/meta/episodes/chunk-*/file-*.parquet # one ROW per episode (the relational index)
    <root>/meta/tasks.parquet                   # task_index -> task string
    <root>/data/chunk-*/file-*.parquet          # one ROW per FRAME, many episodes per file
    <root>/videos/<camera>/chunk-*/file-*.mp4   # per-camera video shards (not read here)

The episode-metadata row carries ``data/chunk_index`` + ``data/file_index`` (which data shard
holds this episode's frames) and ``length``. To read one episode we open its data shard and
**filter rows by ``episode_index``** — robust against the v3 quirk that ``dataset_from/to_index``
are *global* (concatenation) offsets, not per-file ones.

Scope (v1): low-dim features (state/action/reward/timestamp + bookkeeping) read with **pyarrow +
json only** — no torch, no ``lerobot``, no GPU. Video features (``dtype: "video"``) are recorded
in ``meta.extra["video_features"]`` but their pixels are not decoded.

Stage-1 image/video pass-through: for each episode the reader also builds per-camera shard
**references** (:class:`~robocurate.trajectory.VideoReference`: frame indices + the mp4 shard path
+ the episode's timestamp slice) and carries them in ``meta.extra["video_references"]`` so the v3
writer can *copy* the kept episodes' video shard files without ever decoding pixels. Video features
are deliberately kept out of the embodiment/parquet feature table (their pixels live in mp4 shards);
full frame decode lands behind an optional extra later. The reader has no write method, so the
source is read-only by construction (invariant 1).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from robocurate.adapters.base import LeRobotVersion
from robocurate.adapters.lerobot import (
    _arrow_column_to_array,
    _content_fingerprint,
    _infer_role,
    _is_image_key,
)
from robocurate.metadata import DatasetFingerprint, DatasetMeta
from robocurate.trajectory import (
    Array,
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    VideoReference,
)

CODEBASE_VERSION = "v3.0"
_VIDEO_DTYPES = ("video", "image")


def _is_video_feature(key: str, spec: dict[str, Any]) -> bool:
    """Whether a feature is pixel data (a video shard / image), not a parquet column."""
    return spec.get("dtype") in _VIDEO_DTYPES or _is_image_key(key)


class LeRobotReaderV3:
    """Read-only :class:`~robocurate.adapters.base.DatasetReader` for a LeRobot **v3.0** dataset.

    Has no write/save/mutate method by construction — the source can never be written through
    this object (Invariant 1).

    Args:
        root: Path to the dataset directory (containing ``meta/info.json``).
        dataset_id: Identifier recorded in fingerprints/metadata (defaults to ``root``).
    """

    def __init__(self, root: str | Path, *, dataset_id: str | None = None) -> None:
        self.root = Path(root)
        self.version = LeRobotVersion.V3
        self.dataset_id = dataset_id or str(self.root)
        self._info = self._load_info()
        self._data_path_tmpl: str = self._info["data_path"]
        # The video shard path template (e.g. "videos/{video_key}/chunk-{chunk_index:03d}/
        # file-{file_index:03d}.mp4"). May be absent on a low-dim-only dataset; then video shard
        # files cannot be located and references carry shard_path=None.
        self._video_path_tmpl: str | None = self._info.get("video_path")
        self._video_keys = tuple(
            k for k, s in self._info.get("features", {}).items() if _is_video_feature(k, s)
        )
        # The raw v3 feature-spec dict for each video key (dtype/shape/names/video metadata),
        # preserved verbatim so the writer can re-emit faithful video feature specs in the
        # curated output's info.json without re-reading the source.
        self._video_feature_specs: dict[str, dict[str, Any]] = {
            k: dict(self._info["features"][k]) for k in self._video_keys
        }
        self._episodes = self._load_episodes()
        self._embodiment = self._build_embodiment()
        self.meta = self._build_meta()

    # -- construction helpers --------------------------------------------------------

    def _load_info(self) -> dict[str, Any]:
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"not a LeRobot dataset: missing {info_path}")
        data: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
        version = str(data.get("codebase_version", ""))
        if not version.startswith("v3"):
            raise ValueError(
                f"{info_path} is codebase_version {version!r}, not v3.0; use LeRobotReader for "
                "v2.x datasets (or Dataset.from_lerobot, which auto-detects the version)."
            )
        if "data_path" not in data:
            raise ValueError(f"{info_path} is missing the 'data_path' template (not a v3 dataset)")
        return data

    def _load_episodes(self) -> list[dict[str, Any]]:
        """Load the relational per-episode metadata table (concat all episode shards)."""
        ep_files = sorted((self.root / "meta" / "episodes").glob("*/*.parquet"))
        if not ep_files:
            raise FileNotFoundError(
                f"no episode metadata under {self.root / 'meta' / 'episodes'} (not a v3 dataset)"
            )
        records: list[dict[str, Any]] = []
        for path in ep_files:
            records.extend(pq.read_table(path).to_pylist())  # type: ignore[no-untyped-call]
        records.sort(key=lambda r: r["episode_index"])
        return records

    def _build_embodiment(self) -> EmbodimentSpec:
        # Only non-video features are actual parquet columns; video features are recorded in
        # meta.extra (their pixels live in mp4 shards, not the data table).
        features: list[FeatureSpec] = []
        for key, spec in self._info["features"].items():
            if _is_video_feature(key, spec):
                continue
            names = spec.get("names")
            features.append(
                FeatureSpec(
                    key=key,
                    role=_infer_role(key),
                    shape=tuple(spec["shape"]),
                    dtype=spec["dtype"],
                    names=tuple(names) if names else None,
                )
            )
        return EmbodimentSpec(
            embodiment_id=self._info.get("robot_type", "unknown"),
            features=tuple(features),
            control_hz=float(self._info["fps"]) if self._info.get("fps") else None,
        )

    def _build_meta(self) -> DatasetMeta:
        return DatasetMeta(
            fingerprint=self.fingerprint(),
            embodiment_ids=(self._embodiment.embodiment_id,),
            feature_keys=tuple(spec.key for spec in self._embodiment.features),
            extra={"video_features": list(self._video_keys)},
        )

    # -- DatasetReader protocol ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._episodes)

    def __iter__(self) -> Iterator[Trajectory]:
        for record in self._episodes:
            yield self._read_record(record)

    def read_episode(self, index: int) -> Trajectory:
        for record in self._episodes:
            if record["episode_index"] == index:
                return self._read_record(record)
        raise IndexError(f"episode {index} not found in {self.root}")

    def fingerprint(self) -> DatasetFingerprint:
        per_ep = sorted(self._read_record(r).meta.fingerprint for r in self._episodes)
        roll = hashlib.sha256()
        for fp in per_ep:
            roll.update(fp.encode("utf-8"))
        return DatasetFingerprint(
            dataset_id=self.dataset_id,
            source_format="lerobot_v3",
            content_hash=roll.hexdigest(),
            num_episodes=len(self._episodes),
        )

    # -- internals -------------------------------------------------------------------

    def _read_record(self, record: dict[str, Any]) -> Trajectory:
        index = int(record["episode_index"])
        data_file = self.root / self._data_path_tmpl.format(
            chunk_index=record["data/chunk_index"], file_index=record["data/file_index"]
        )
        table = pq.read_table(data_file)  # type: ignore[no-untyped-call]
        # Filter to this episode's frames by the per-frame episode_index column (robust against
        # the global dataset_from/to_index quirk). Rows stay in stored (frame) order.
        mask = pc.equal(table.column("episode_index"), index)  # type: ignore[attr-defined]
        rows = table.filter(mask)

        columns: dict[str, Array] = {}
        for spec in self._embodiment.features:
            columns[spec.key] = _arrow_column_to_array(rows.column(spec.key), spec)

        # Stage-1 image/video pass-through: build per-camera shard *references* (frame indices +
        # mp4 shard path, never pixels) from the relational episode row, and carry them in
        # meta.extra so the writer can copy the kept episodes' video shard files. Video features
        # stay out of the embodiment/parquet feature table (v3 keeps pixels in mp4 shards), and
        # references are excluded from the content fingerprint (no pixels are loaded or modified).
        video_refs = self._build_video_references(record, num_frames=rows.num_rows)

        meta = TrajectoryMeta(
            source_dataset_id=self.dataset_id,
            episode_index=index,
            embodiment=self._embodiment,
            fingerprint=_content_fingerprint(columns),
            num_steps=rows.num_rows,
            source_format="lerobot_v3",
            success=self._read_success(columns),
            extra={
                "tasks": record.get("tasks", []),
                "video_features": list(self._video_keys),
                "video_feature_specs": self._video_feature_specs,
                "video_references": video_refs,
            },
        )
        return Trajectory(meta, InMemoryFeatureStore(columns))

    def _build_video_references(
        self, record: dict[str, Any], *, num_frames: int
    ) -> dict[str, VideoReference]:
        """Build a ``{video_key: VideoReference}`` mapping for one episode (no pixels decoded).

        Each reference records the per-camera mp4 shard the episode's frames live in — resolved
        from the relational episode row's ``videos/<key>/chunk_index`` + ``file_index`` columns
        and the ``video_path`` template — plus the episode-local frame indices ``0..T-1`` and the
        episode's ``from_timestamp``/``to_timestamp`` slice within that (possibly multi-episode)
        shard. ``shard_path`` is ``None`` when the dataset exposes no ``video_path`` template or
        the row omits the shard indices (the reference is then opaque but still preserved).
        """
        refs: dict[str, VideoReference] = {}
        frame_indices = np.arange(num_frames, dtype=np.int64)
        for key in self._video_keys:
            chunk = record.get(f"videos/{key}/chunk_index")
            file = record.get(f"videos/{key}/file_index")
            shard_path: Path | None = None
            chunk_index: int | None = None if chunk is None else int(chunk)
            file_index: int | None = None if file is None else int(file)
            have_indices = chunk_index is not None and file_index is not None
            if self._video_path_tmpl is not None and have_indices:
                shard_path = self.root / self._video_path_tmpl.format(
                    video_key=key, chunk_index=chunk_index, file_index=file_index
                )
            refs[key] = VideoReference(
                video_key=key,
                num_frames=num_frames,
                frame_indices=frame_indices,
                shard_path=shard_path,
                shard_chunk_index=chunk_index,
                shard_file_index=file_index,
                from_timestamp=record.get(f"videos/{key}/from_timestamp"),
                to_timestamp=record.get(f"videos/{key}/to_timestamp"),
            )
        return refs

    def _read_success(self, columns: dict[str, Array]) -> SuccessLabel | None:
        """Reconstruct an episode :class:`SuccessLabel` from a SUCCESS-role column, if any.

        v3 datasets usually carry no success notion; when a ``success``/``done``-role feature is
        present, the final step's value is used. ``None`` otherwise (never coerced to ``False``).
        """
        for spec in self._embodiment.features:
            if spec.role is FeatureRole.SUCCESS and spec.key in columns:
                per_step = columns[spec.key]
                if per_step.size:
                    last = per_step.reshape(per_step.shape[0], -1)[-1]
                    return SuccessLabel(
                        value=bool(last.max() > 0.5), source="dataset", per_step=per_step
                    )
        return None


def detect_lerobot_version(root: str | Path) -> LeRobotVersion:
    """Read ``meta/info.json`` and return the on-disk LeRobot format version.

    ``v3.x`` -> :attr:`LeRobotVersion.V3`; anything else (v2.0/v2.1) -> :attr:`LeRobotVersion.V2_1`.
    """
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"not a LeRobot dataset: missing {info_path}")
    version = str(json.loads(info_path.read_text(encoding="utf-8")).get("codebase_version", ""))
    return LeRobotVersion.V3 if version.startswith("v3") else LeRobotVersion.V2_1


__all__ = ["CODEBASE_VERSION", "LeRobotReaderV3", "detect_lerobot_version"]
