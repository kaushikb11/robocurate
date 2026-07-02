"""LeRobotDataset **v3.0** write adapter — emit a curated subset *as v3* (not downgraded to v2.1).

The v2.1 writer (:class:`robocurate.adapters.lerobot.LeRobotWriter`) is the template for every
guarantee here; this writer produces the v3.0 on-disk layout instead so curating a v3 source no
longer silently downgrades it. Layout written (single-shard is valid — the v3 reader globs
``chunk-*/file-*``)::

    <dest>/meta/info.json                            # codebase_version "v3.0", fps, features
    <dest>/meta/episodes/chunk-000/file-000.parquet  # one ROW per episode (relational index)
    <dest>/meta/tasks.parquet                        # task_index -> task string
    <dest>/data/chunk-000/file-000.parquet           # one ROW per FRAME, all episodes concatenated
    <dest>/manifest.json                             # auditable kept/removed record

Scope (v1, **low-dim curation + Stage-1 image/video pass-through**): scalar and 1-D features
(action, state/proprio, reward, timestamp, ...) plus the five v3 bookkeeping columns the
reader/spec expect (``timestamp``/``frame_index``/``episode_index``/``index``/``task_index``) are
written as parquet exactly as before. **Image/video pixels are never decoded.** Instead, for the
kept episodes the writer **copies** the backing per-camera mp4 shard files (located via the source
:class:`~robocurate.trajectory.VideoReference`\\ s on each trajectory) into the output ``videos/``
tree, checksums each copied file against its source (invariant 2), and re-emits the video feature
specs + ``video_path`` template in ``info.json`` so the output reloads as a valid v3 dataset that
*preserves the kept frames* (Stage 2 will decode/curate pixels).

**What gets copied (shard granularity):** a v3 mp4 shard may bundle several episodes. Stage-1 copies
the *whole* shard file that any kept episode references, byte-for-byte, and preserves the source
``chunk_index``/``file_index`` and ``from_timestamp``/``to_timestamp`` in the output episode
metadata so frame/timestamp indexing stays consistent. The output is therefore not re-sharded:
copied shards may still contain frames from episodes that were dropped (those frames are simply not
indexed by any output episode). The low-dim round-trip guarantee is unchanged (over the persisted
parquet columns); the video guarantee is that each referenced source shard is present in the output,
byte-identical.

Like the v2.1 writer: the destination must not exist and must not overlap the source (invariant 1);
every write finishes with schema + checksum + round-trip validation and any failure quarantines the
partial output and re-raises (invariant 2); kept episodes are re-indexed ``0..k-1`` because the
output is a fresh dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robocurate.adapters.base import (
    DatasetWriter,
    SourceWriteError,
    ValidationError,
    ValidationReport,
    WriteReceipt,
)
from robocurate.adapters.lerobot import (
    _array_to_arrow_column,
    _checksum_tree,
    _content_fingerprint,
    _is_image_key,
)
from robocurate.adapters.lerobot_v3 import CODEBASE_VERSION, LeRobotReaderV3, _is_video_feature
from robocurate.manifest import Manifest
from robocurate.trajectory import (
    Array,
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    Trajectory,
    VideoReference,
)

_CHUNK = "chunk-000"
_FILE = "file-000.parquet"
_DATA_TMPL = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_TMPL = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
# The five bookkeeping columns a v3 data shard carries alongside the declared features. The v3
# reader reconstructs these as low-dim feature columns, so they are part of the round-trip.
_BOOKKEEPING_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def _sha256_file(path: Path) -> str:
    """Return the sha256 of a file's bytes (chunked, so large mp4 shards stream)."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _data_file(root: Path) -> Path:
    return root / "data" / _CHUNK / _FILE


def _episodes_meta_file(root: Path) -> Path:
    return root / "meta" / "episodes" / _CHUNK / _FILE


def _is_persisted(spec: FeatureSpec) -> bool:
    """Whether a feature is persisted as a parquet column (low-dim; not video/image pixels).

    Video pixels live in mp4 shards we never write, so IMAGE-role features, ``video``/``image``
    dtypes, and ``observation.image*`` keys are all dropped — the writer emits a low-dim dataset.
    """
    return not (
        spec.role is FeatureRole.IMAGE
        or spec.dtype in ("video", "image")
        or _is_image_key(spec.key)
    )


def _persisted_specs(embodiment: EmbodimentSpec) -> list[FeatureSpec]:
    """The feature specs the writer persists as parquet columns (low-dim, no video/image)."""
    return [spec for spec in embodiment.features if _is_persisted(spec)]


class LeRobotWriterV3(DatasetWriter):
    """Writes a *new* LeRobot **v3.0** dataset; refuses to write to the source (invariant 1).

    The destination must not already exist and must not overlap the source directory. Every write
    ends with schema + checksum + round-trip validation; any failure quarantines the partial output
    and raises (invariant 2). Kept episodes are re-indexed ``0..k-1`` (the output is fresh).

    Video / image-role features are never decoded; the kept episodes' mp4 shard files are copied
    byte-for-byte (Stage-1 pass-through — see the module docstring). The round-trip is asserted
    over the low-dim parquet columns plus the copied shards' checksums.
    """

    def __init__(self, dest: str | Path, *, source_root: str | Path | None = None) -> None:
        self.dest = Path(dest)
        self._source_root = Path(source_root).resolve() if source_root else None
        self._guard_destination()

    def _guard_destination(self) -> None:
        dest = self.dest.resolve()
        if self.dest.exists():
            raise SourceWriteError(
                f"refusing to write: destination {self.dest} already exists "
                "(curation never overwrites; choose a fresh path)"
            )
        if self._source_root is not None:
            if dest == self._source_root:
                raise SourceWriteError(
                    "refusing to write: destination equals the source dataset (source is read-only)"
                )
            if dest.is_relative_to(self._source_root) or self._source_root.is_relative_to(dest):
                raise SourceWriteError(
                    f"refusing to write: destination {dest} overlaps source "
                    f"{self._source_root} (source is read-only)"
                )

    # -- DatasetWriter protocol ------------------------------------------------------

    def write(self, trajectories: Iterable[Trajectory], manifest: Manifest) -> WriteReceipt:
        self._guard_destination()  # re-check at write time (TOCTOU safety)
        (self.dest / "meta" / "episodes" / _CHUNK).mkdir(parents=True, exist_ok=False)
        (self.dest / "data" / _CHUNK).mkdir(parents=True, exist_ok=False)

        expected_fingerprints: list[str] = []
        try:
            embodiment, episode_records, data_table, tasks, video_specs, episode_refs = self._build(
                trajectories, expected_fingerprints
            )
            self._write_data(data_table)
            # Stage-1 image/video pass-through: copy each referenced source mp4 shard into the
            # output videos/ tree and checksum it against the source (invariant 2). This mutates
            # episode_records in place to carry the per-key shard indices + timestamp slice.
            copied_video_checksums = self._copy_videos(episode_records, episode_refs)
            self._write_meta(embodiment, episode_records, tasks, video_specs)
            manifest_path = self.dest / "manifest.json"

            report = self.validate(self.dest)
            report.raise_if_invalid()
            self._assert_roundtrip(expected_fingerprints, copied_video_checksums)

            checksums = _checksum_tree(self.dest)
            fingerprint = LeRobotReaderV3(self.dest).fingerprint()
            manifest.write(manifest_path)
        except BaseException:
            self._quarantine()
            raise

        return WriteReceipt(
            path=self.dest,
            fingerprint=fingerprint,
            manifest_path=manifest_path,
            file_checksums=checksums,
            validation=report,
        )

    def validate(self, path: Path) -> ValidationReport:
        errors: list[str] = []
        checked: list[str] = []
        info_path = path / "meta" / "info.json"
        if not info_path.is_file():
            return ValidationReport(ok=False, errors=(f"missing {info_path}",))
        info = json.loads(info_path.read_text(encoding="utf-8"))
        for key in (
            "codebase_version",
            "fps",
            "features",
            "total_episodes",
            "total_frames",
            "data_path",
        ):
            if key not in info:
                errors.append(f"info.json missing required key {key!r}")

        ep_meta_path = _episodes_meta_file(path)
        if not ep_meta_path.is_file():
            errors.append(f"missing episode metadata {ep_meta_path}")
            return ValidationReport(ok=False, errors=tuple(errors))
        records = pq.read_table(ep_meta_path).to_pylist()  # type: ignore[no-untyped-call]
        checked.append(str(ep_meta_path.relative_to(path)))
        if "total_episodes" in info and info["total_episodes"] != len(records):
            errors.append(
                f"total_episodes={info['total_episodes']} but episode metadata has "
                f"{len(records)} rows"
            )

        data_path = _data_file(path)
        if not data_path.is_file():
            errors.append(f"missing data shard {data_path}")
            return ValidationReport(ok=False, errors=tuple(errors), checked_files=tuple(checked))
        table = pq.read_table(data_path)  # type: ignore[no-untyped-call]
        checked.append(str(data_path.relative_to(path)))
        cols = set(table.column_names)
        all_features: dict[str, dict[str, Any]] = info.get("features", {})
        # Low-dim features are parquet columns; video features live in mp4 shards, not the table.
        video_keys = [k for k, s in all_features.items() if _is_video_feature(k, s)]
        for key in all_features:
            if key in video_keys:
                continue
            if key not in cols:
                errors.append(f"data shard missing feature column {key!r}")
        for key in _BOOKKEEPING_FEATURES:
            if key not in cols:
                errors.append(f"data shard missing bookkeeping column {key!r}")
        declared_frames = info.get("total_frames")
        if declared_frames is not None and table.num_rows != declared_frames:
            errors.append(
                f"data shard has {table.num_rows} frames but info.json declares "
                f"total_frames={declared_frames}"
            )
        # Stage-1 video pass-through: every shard a kept episode references must exist on disk.
        errors.extend(self._validate_video_shards(path, info, records, video_keys))
        return ValidationReport(ok=not errors, errors=tuple(errors), checked_files=tuple(checked))

    @staticmethod
    def _validate_video_shards(
        path: Path,
        info: dict[str, Any],
        records: list[dict[str, Any]],
        video_keys: list[str],
    ) -> list[str]:
        """Check that every per-episode video shard declared in the metadata exists on disk."""
        errors: list[str] = []
        tmpl = info.get("video_path")
        if not video_keys or tmpl is None:
            return errors
        for record in records:
            for key in video_keys:
                chunk = record.get(f"videos/{key}/chunk_index")
                file = record.get(f"videos/{key}/file_index")
                if chunk is None or file is None:
                    continue  # episode declares no shard for this camera (opaque/low-dim source)
                shard = path / str(tmpl).format(
                    video_key=key, chunk_index=int(chunk), file_index=int(file)
                )
                if not shard.is_file():
                    errors.append(f"declared video shard missing from output: {shard}")
        return errors

    # -- internals -------------------------------------------------------------------

    def _build(
        self, trajectories: Iterable[Trajectory], expected_fingerprints: list[str]
    ) -> tuple[
        EmbodimentSpec,
        list[dict[str, Any]],
        pa.Table,
        list[str],
        dict[str, dict[str, Any]],
        list[dict[str, VideoReference]],
    ]:
        """Build the concatenated data table + episode metadata + task list in one pass.

        Re-indexes kept episodes ``0..k-1`` and accumulates a global frame counter. Records, per
        episode, the fingerprint of exactly the columns the reader will reconstruct, so the
        round-trip check compares like with like. Also collects, per episode, the IMAGE-role
        :class:`VideoReference`\\ s (for Stage-1 shard copying) and the union of video feature
        specs (re-emitted into ``info.json``), both gathered without touching any pixels.
        """
        embodiment: EmbodimentSpec | None = None
        episode_records: list[dict[str, Any]] = []
        sub_tables: list[pa.Table] = []
        task_to_index: dict[str, int] = {}
        video_specs: dict[str, dict[str, Any]] = {}
        episode_refs: list[dict[str, VideoReference]] = []
        global_index = 0

        for out_index, traj in enumerate(trajectories):
            if embodiment is None:
                embodiment = traj.embodiment
            tasks = [str(t) for t in traj.meta.extra.get("tasks", [])]
            task_index = self._task_index(tasks, task_to_index)
            table, columns = self._episode_table(traj, out_index, global_index, task_index)
            sub_tables.append(table)
            # Content-only, like the reader: the rewritten bookkeeping columns are positions,
            # not content, so a v3->v3 round trip preserves each episode's source fingerprint.
            expected_fingerprints.append(_content_fingerprint(columns))
            episode_records.append(
                {
                    "episode_index": out_index,
                    "data/chunk_index": 0,
                    "data/file_index": 0,
                    "dataset_from_index": global_index,
                    "dataset_to_index": global_index + traj.num_steps,
                    "length": traj.num_steps,
                    "tasks": tasks,
                }
            )
            episode_refs.append(traj.video_references())
            self._merge_video_specs(traj, video_specs)
            global_index += traj.num_steps

        if embodiment is None:
            raise ValueError("cannot write an empty dataset (no trajectories provided)")
        data_table = pa.concat_tables(sub_tables) if sub_tables else pa.table({})
        ordered_tasks = sorted(task_to_index, key=lambda t: task_to_index[t])
        return embodiment, episode_records, data_table, ordered_tasks, video_specs, episode_refs

    @staticmethod
    def _merge_video_specs(traj: Trajectory, video_specs: dict[str, dict[str, Any]]) -> None:
        """Accumulate the source video feature spec dicts (dtype/shape/...) for ``info.json``.

        Sourced from ``meta.extra["video_feature_specs"]`` (attached by the v3 reader). Falls back
        to a minimal ``{"dtype": "video", ...}`` spec for any video key lacking an explicit spec so
        the output still declares the feature.
        """
        specs = traj.meta.extra.get("video_feature_specs", {})
        for key in traj.video_references():
            if key in video_specs:
                continue
            spec = specs.get(key) if isinstance(specs, dict) else None
            video_specs[key] = dict(spec) if isinstance(spec, dict) else {"dtype": "video"}

    @staticmethod
    def _task_index(tasks: list[str], task_to_index: dict[str, int]) -> int:
        """Map an episode's first task string to a stable task index (0 when it has no task)."""
        if not tasks:
            return 0
        task = tasks[0]
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
        return task_to_index[task]

    def _episode_table(
        self, traj: Trajectory, episode_index: int, start_index: int, task_index: int
    ) -> tuple[pa.Table, dict[str, Array]]:
        """Build one episode's per-frame rows and the column dict the reader reconstructs.

        The returned column dict mirrors what :class:`LeRobotReaderV3` materializes (low-dim
        feature columns + the five bookkeeping columns, each shaped ``(T, 1)`` for the
        ``shape == [1]`` bookkeeping specs), so its fingerprint equals the reloaded one.
        """
        num_rows = traj.num_steps
        arrays: dict[str, pa.Array] = {}
        columns: dict[str, Array] = {}
        for spec in traj.embodiment.features:
            if not _is_persisted(spec) or not traj.has(spec.key):
                continue
            if spec.key in _BOOKKEEPING_FEATURES:
                continue  # bookkeeping columns are (re)generated below, not copied from source
            data = traj.feature(spec.key)
            arrays[spec.key] = _array_to_arrow_column(data, spec)
            columns[spec.key] = data

        ts = traj.timestamps()
        timestamp: Array
        if ts is not None:
            timestamp = np.asarray(ts, dtype=np.float32).reshape(num_rows)
        else:
            fps = traj.embodiment.control_hz or 1.0
            timestamp = (np.arange(num_rows, dtype=np.float32) / float(fps)).astype(np.float32)
        frame_index = np.arange(num_rows, dtype=np.int64)
        episode_col = np.full(num_rows, episode_index, dtype=np.int64)
        index_col = np.arange(start_index, start_index + num_rows, dtype=np.int64)
        task_col = np.full(num_rows, task_index, dtype=np.int64)

        arrays["timestamp"] = pa.array(timestamp, type=pa.float32())
        arrays["frame_index"] = pa.array(frame_index, type=pa.int64())
        arrays["episode_index"] = pa.array(episode_col, type=pa.int64())
        arrays["index"] = pa.array(index_col, type=pa.int64())
        arrays["task_index"] = pa.array(task_col, type=pa.int64())

        # The reader declares bookkeeping shape [1], so it reshapes each column to (T, 1). Mirror
        # that here so the expected fingerprint matches the reloaded one exactly.
        columns["timestamp"] = timestamp.reshape(num_rows, 1)
        columns["frame_index"] = frame_index.reshape(num_rows, 1)
        columns["episode_index"] = episode_col.reshape(num_rows, 1)
        columns["index"] = index_col.reshape(num_rows, 1)
        columns["task_index"] = task_col.reshape(num_rows, 1)

        return pa.table(arrays), columns

    def _write_data(self, data_table: pa.Table) -> None:
        pq.write_table(data_table, _data_file(self.dest))  # type: ignore[no-untyped-call]

    def _copy_videos(
        self,
        episode_records: list[dict[str, Any]],
        episode_refs: list[dict[str, VideoReference]],
    ) -> dict[str, str]:
        """Copy each referenced source mp4 shard into the output and checksum it (invariant 2).

        For every (video_key, chunk_index, file_index) any kept episode references, the *whole*
        source shard file is copied to the same ``videos/<key>/chunk-NNN/file-NNN.mp4`` path under
        the output (preserving indices/timestamps so frame indexing stays consistent — see the
        module docstring on shard granularity). Each copy is verified to be byte-identical to its
        source via sha256. Per-key shard indices + timestamp slice are written into
        ``episode_records`` so the output reloads with consistent references.

        Returns ``{output-relative-path: sha256}`` for every copied shard (used by the round-trip
        assertion). A reference with no resolvable ``shard_path`` (opaque, e.g. a low-dim-only
        source) is skipped — it has no file to copy and writes no shard columns.

        Never mutates the source: the source shard is only read; the destination is a fresh copy.
        """
        copied: dict[str, str] = {}  # output-relative path -> sha256 (one entry per unique shard)
        for record, refs in zip(episode_records, episode_refs, strict=True):
            for key, ref in refs.items():
                if (
                    ref.shard_path is None
                    or ref.shard_chunk_index is None
                    or ref.shard_file_index is None
                ):
                    continue
                rel = _VIDEO_TMPL.format(
                    video_key=key,
                    chunk_index=ref.shard_chunk_index,
                    file_index=ref.shard_file_index,
                )
                dest_path = self.dest / rel
                if rel not in copied:
                    self._copy_one_shard(ref.shard_path, dest_path)
                    copied[rel] = _sha256_file(dest_path)
                # Record the shard reference on this episode (preserve source indices/timestamps).
                record[f"videos/{key}/chunk_index"] = ref.shard_chunk_index
                record[f"videos/{key}/file_index"] = ref.shard_file_index
                record[f"videos/{key}/from_timestamp"] = ref.from_timestamp
                record[f"videos/{key}/to_timestamp"] = ref.to_timestamp
        return copied

    @staticmethod
    def _copy_one_shard(source: Path, dest: Path) -> None:
        """Copy one mp4 shard byte-for-byte and verify the copy matches the source (invariant 2)."""
        if not source.is_file():
            raise ValidationError(f"video shard missing at source: {source}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)  # copy, never move — the source is read-only (invariant 1)
        if _sha256_file(dest) != _sha256_file(source):
            raise ValidationError(f"copied video shard {dest} does not match source {source}")

    def _write_meta(
        self,
        embodiment: EmbodimentSpec,
        episode_records: list[dict[str, Any]],
        tasks: list[str],
        video_specs: dict[str, dict[str, Any]],
    ) -> None:
        features: dict[str, dict[str, Any]] = {
            spec.key: {
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "names": list(spec.names) if spec.names else None,
            }
            for spec in _persisted_specs(embodiment)
            if spec.key not in _BOOKKEEPING_FEATURES
        }
        # The five bookkeeping columns are part of the v3 feature dict (the reader reads them back).
        for key, spec_dict in _BOOKKEEPING_FEATURES.items():
            features[key] = dict(spec_dict)
        # Stage-1 pass-through: re-emit the (copied) video feature specs so the output declares its
        # video features and the v3 reader records them in meta.extra["video_features"] again.
        for key, spec_dict in video_specs.items():
            features[key] = dict(spec_dict)

        total_frames = sum(int(r["length"]) for r in episode_records)
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": embodiment.embodiment_id,
            "fps": embodiment.control_hz,
            "total_episodes": len(episode_records),
            "total_frames": total_frames,
            "data_path": _DATA_TMPL,
            "video_path": _VIDEO_TMPL,
            "features": features,
        }
        (self.dest / "meta" / "info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True), encoding="utf-8"
        )

        pq.write_table(  # type: ignore[no-untyped-call]
            pa.Table.from_pylist(episode_records), _episodes_meta_file(self.dest)
        )

        task_rows = [{"task_index": i, "task": task} for i, task in enumerate(tasks)]
        if not task_rows:
            tasks_table = pa.table(
                {
                    "task_index": pa.array([], type=pa.int64()),
                    "task": pa.array([], type=pa.string()),
                }
            )
        else:
            tasks_table = pa.Table.from_pylist(task_rows)
        pq.write_table(tasks_table, self.dest / "meta" / "tasks.parquet")  # type: ignore[no-untyped-call]

    def _assert_roundtrip(
        self, expected_fingerprints: list[str], copied_video_checksums: dict[str, str]
    ) -> None:
        # Reload the just-written dataset and assert per-episode content fingerprints match exactly
        # what we intended to write (invariant 2). The fingerprint comparison is over the low-dim
        # columns the writer persists; video references are excluded from the content hash by
        # construction (pixels are never loaded), so the low-dim guarantee is unchanged.
        reader = LeRobotReaderV3(self.dest)
        reread = [t.meta.fingerprint for t in reader]
        if reread != expected_fingerprints:
            raise ValidationError(
                "round-trip reload mismatch: the written v3 dataset does not reload to the "
                "low-dim content that was written"
            )
        # Stage-1 video pass-through round-trip: every copied shard must be present on disk and
        # byte-identical to what was copied, and every reloaded VideoReference must resolve to a
        # real shard file under the output (so the kept frames are genuinely preserved).
        for rel, expected in copied_video_checksums.items():
            shard = self.dest / rel
            if not shard.is_file():
                raise ValidationError(f"round-trip: copied video shard missing from output: {rel}")
            if _sha256_file(shard) != expected:
                raise ValidationError(f"round-trip: copied video shard changed after write: {rel}")
        # Every reloaded reference that resolves to a shard must point to a real output file. A
        # reference with shard_path=None is a legitimately opaque/low-dim feature (the source
        # exposed no shard for it) and is not a pass-through failure.
        for traj in reader:
            for ref in traj.video_references().values():
                if ref.shard_path is not None and not ref.shard_path.is_file():
                    raise ValidationError(
                        f"round-trip: reloaded video shard does not exist: {ref.shard_path}"
                    )
        # If the source actually had shards (we copied at least one), the output must carry them.
        if copied_video_checksums and not any(
            ref.shard_path is not None
            for traj in reader
            for ref in traj.video_references().values()
        ):
            raise ValidationError(
                "round-trip: video shards were copied but none resolve on reload (lost references)"
            )

    def _quarantine(self) -> None:
        if self.dest.exists():
            quarantine = self.dest.with_name(self.dest.name + ".invalid")
            if quarantine.exists():
                shutil.rmtree(quarantine)
            self.dest.rename(quarantine)


__all__ = ["LeRobotWriterV3"]
