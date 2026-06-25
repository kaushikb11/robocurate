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

Scope (v1, **low-dim curation**): scalar and 1-D features (action, state/proprio, reward,
timestamp, ...) plus the five v3 bookkeeping columns the reader/spec expect
(``timestamp``/``frame_index``/``episode_index``/``index``/``task_index``). **Video / image-role
features are NOT persisted** — their pixels live in mp4 shards we do not write — so the curated
output is a low-dim dataset. The round-trip guarantee is therefore over the low-dim columns the
writer actually writes (the same columns :class:`~robocurate.adapters.lerobot_v3.LeRobotReaderV3`
reads back).

Like the v2.1 writer: the destination must not exist and must not overlap the source (invariant 1);
every write finishes with schema + checksum + round-trip validation and any failure quarantines the
partial output and re-raises (invariant 2); kept episodes are re-indexed ``0..k-1`` because the
output is a fresh dataset.
"""

from __future__ import annotations

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
    _is_image_key,
)
from robocurate.adapters.lerobot_v3 import CODEBASE_VERSION, LeRobotReaderV3
from robocurate.manifest import Manifest
from robocurate.trajectory import (
    Array,
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    Trajectory,
    fingerprint_arrays,
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

    Video / image-role features are not persisted — see the module docstring — so this writer emits
    a low-dim curated dataset and the round-trip is asserted over the low-dim columns only.
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
            embodiment, episode_records, data_table, tasks = self._build(
                trajectories, expected_fingerprints
            )
            self._write_data(data_table)
            self._write_meta(embodiment, episode_records, tasks)
            manifest_path = self.dest / "manifest.json"

            report = self.validate(self.dest)
            report.raise_if_invalid()
            self._assert_roundtrip(expected_fingerprints)

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
        feature_keys = list(info.get("features", {}))
        for key in feature_keys:
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
        return ValidationReport(ok=not errors, errors=tuple(errors), checked_files=tuple(checked))

    # -- internals -------------------------------------------------------------------

    def _build(
        self, trajectories: Iterable[Trajectory], expected_fingerprints: list[str]
    ) -> tuple[EmbodimentSpec, list[dict[str, Any]], pa.Table, list[str]]:
        """Build the concatenated data table + episode metadata + task list in one pass.

        Re-indexes kept episodes ``0..k-1`` and accumulates a global frame counter. Records, per
        episode, the fingerprint of exactly the columns the reader will reconstruct, so the
        round-trip check compares like with like.
        """
        embodiment: EmbodimentSpec | None = None
        episode_records: list[dict[str, Any]] = []
        sub_tables: list[pa.Table] = []
        task_to_index: dict[str, int] = {}
        global_index = 0

        for out_index, traj in enumerate(trajectories):
            if embodiment is None:
                embodiment = traj.embodiment
            tasks = [str(t) for t in traj.meta.extra.get("tasks", [])]
            task_index = self._task_index(tasks, task_to_index)
            table, columns = self._episode_table(traj, out_index, global_index, task_index)
            sub_tables.append(table)
            expected_fingerprints.append(fingerprint_arrays(columns))
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
            global_index += traj.num_steps

        if embodiment is None:
            raise ValueError("cannot write an empty dataset (no trajectories provided)")
        data_table = pa.concat_tables(sub_tables) if sub_tables else pa.table({})
        ordered_tasks = sorted(task_to_index, key=lambda t: task_to_index[t])
        return embodiment, episode_records, data_table, ordered_tasks

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

    def _write_meta(
        self,
        embodiment: EmbodimentSpec,
        episode_records: list[dict[str, Any]],
        tasks: list[str],
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

    def _assert_roundtrip(self, expected_fingerprints: list[str]) -> None:
        # Reload the just-written dataset and assert per-episode content fingerprints match exactly
        # what we intended to write (invariant 2). The comparison is over the low-dim columns the
        # writer persists; video features (if any on the source) are excluded by construction.
        reread = [t.meta.fingerprint for t in LeRobotReaderV3(self.dest)]
        if reread != expected_fingerprints:
            raise ValidationError(
                "round-trip reload mismatch: the written v3 dataset does not reload to the "
                "low-dim content that was written"
            )

    def _quarantine(self) -> None:
        if self.dest.exists():
            quarantine = self.dest.with_name(self.dest.name + ".invalid")
            if quarantine.exists():
                shutil.rmtree(quarantine)
            self.dest.rename(quarantine)


__all__ = ["LeRobotWriterV3"]
