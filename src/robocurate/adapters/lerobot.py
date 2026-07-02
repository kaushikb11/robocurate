"""LeRobotDataset adapter (read + write) — minimal, faithful v2.1 implementation.

This adapter reads and writes a minimal but faithful subset of the LeRobotDataset v2.1
on-disk layout::

    <root>/meta/info.json        # codebase_version, fps, robot_type, features, totals
    <root>/meta/episodes.jsonl   # one record per episode: index, length, tasks
    <root>/meta/tasks.jsonl      # task_index -> task string
    <root>/data/chunk-000/episode_000000.parquet   # per-episode tabular frames

Scope of this skeleton:

* **Implemented:** scalar and vector (1-D) features (actions, state/proprio, reward,
  timestamp, ...), the standard bookkeeping columns, schema validation, per-file
  checksums, and a post-write round-trip reload check.
* **Declared but not yet implemented:** image/video features (``observation.images.*``)
  and the v3 layout. These raise a clear error rather than silently mis-handling data.
  Full parity (video export, complete stats, and a fast path through the upstream
  ``lerobot`` library) lands in a later rung.

The reader streams one episode parquet at a time, so a dataset larger than RAM is never
fully resident. Per-feature lazy decoding within an episode is a later optimization.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robocurate.adapters.base import (
    DatasetWriter,
    LeRobotVersion,
    SourceWriteError,
    ValidationError,
    ValidationReport,
    WriteReceipt,
)
from robocurate.manifest import Manifest
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
    fingerprint_arrays,
)

CODEBASE_VERSION = "v2.1"
_CHUNK = "chunk-000"

# Standard LeRobot bookkeeping columns written alongside the declared features.
_BOOKKEEPING = ("episode_index", "frame_index", "index", "task_index")
_IMAGE_PREFIXES = ("observation.images", "observation.image")


def _content_fingerprint(columns: Mapping[str, Array]) -> str:
    """Fingerprint the *content* columns only, never the positional bookkeeping.

    ``episode_index`` / ``frame_index`` / ``index`` / ``task_index`` are positions, not
    content: the writer re-indexes kept episodes ``0..k-1``, so including them would give a
    round-tripped episode a new fingerprint whenever its position shifts — breaking the
    contract that the fingerprint links a curated episode back to its source (``diff``,
    dedup, and manifest matching all rely on it). Real v3 datasets declare the bookkeeping
    columns as features, so the exclusion must happen here, not in the feature table.
    """
    return fingerprint_arrays({k: v for k, v in columns.items() if k not in _BOOKKEEPING})


def _infer_role(key: str) -> FeatureRole:
    """Infer a :class:`FeatureRole` from a LeRobot feature key by naming convention."""
    if key.startswith(_IMAGE_PREFIXES):
        return FeatureRole.IMAGE
    if key == "action":
        return FeatureRole.ACTION
    if key == "observation.state":
        return FeatureRole.PROPRIO
    if key in ("next.reward", "reward"):
        return FeatureRole.REWARD
    if key == "timestamp":
        return FeatureRole.TIME
    if key in ("next.success", "success", "next.done"):
        return FeatureRole.SUCCESS
    if key.startswith("observation."):
        return FeatureRole.STATE
    return FeatureRole.EXTRA


def _is_image_key(key: str) -> bool:
    return key.startswith(_IMAGE_PREFIXES)


def _episode_file(root: Path, index: int) -> Path:
    """Return the parquet path for ``index`` under a dataset ``root``."""
    return root / "data" / _CHUNK / f"episode_{index:06d}.parquet"


def _require_v2_1(version: LeRobotVersion) -> None:
    if version is not LeRobotVersion.V2_1:
        raise NotImplementedError(
            f"LeRobotReader handles {LeRobotVersion.V2_1.value}; for {version.value} use "
            "LeRobotReaderV3 (robocurate.adapters.lerobot_v3), or Dataset.from_lerobot, which "
            "auto-detects the on-disk version."
        )


class LeRobotReader:
    """Read-only :class:`~robocurate.adapters.base.DatasetReader` for a LeRobot v2.1 dataset.

    Has no write/save/mutate method by construction — the source can never be written
    through this object (Invariant 1).
    """

    def __init__(self, root: str | Path, *, version: LeRobotVersion = LeRobotVersion.V2_1):
        _require_v2_1(version)
        self.root = Path(root)
        self.version = version
        self._info = self._load_info()
        self._episodes = self._load_episodes()
        self._embodiment = self._build_embodiment()
        self.meta = self._build_meta()

    # -- construction helpers --------------------------------------------------------

    def _load_info(self) -> dict[str, Any]:
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"not a LeRobot v2.1 dataset: missing {info_path}")
        data: dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
        return data

    def _load_episodes(self) -> list[dict[str, Any]]:
        ep_path = self.root / "meta" / "episodes.jsonl"
        records: list[dict[str, Any]] = []
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        records.sort(key=lambda r: r["episode_index"])
        return records

    def _build_embodiment(self) -> EmbodimentSpec:
        features: list[FeatureSpec] = []
        for key, spec in self._info["features"].items():
            if _is_image_key(key):
                raise NotImplementedError(
                    f"image/video feature {key!r} is not yet supported by this adapter; "
                    "video decoding lands in a later rung."
                )
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
        # Content hash rolls up per-episode fingerprints so identical content => equal hash.
        per_ep = sorted(self._read_record(r).meta.fingerprint for r in self._episodes)
        roll = hashlib.sha256()
        for fp in per_ep:
            roll.update(fp.encode("utf-8"))
        return DatasetFingerprint(
            dataset_id=str(self.root),
            source_format=self.version.value,
            content_hash=roll.hexdigest(),
            num_episodes=len(self._episodes),
        )

    # -- internals -------------------------------------------------------------------

    def _read_record(self, record: dict[str, Any]) -> Trajectory:
        index = record["episode_index"]
        table = pq.read_table(_episode_file(self.root, index))  # type: ignore[no-untyped-call]
        columns: dict[str, Array] = {}
        for spec in self._embodiment.features:
            columns[spec.key] = _arrow_column_to_array(table.column(spec.key), spec)
        meta = TrajectoryMeta(
            source_dataset_id=str(self.root),
            episode_index=index,
            embodiment=self._embodiment,
            fingerprint=_content_fingerprint(columns),
            num_steps=table.num_rows,
            source_format=self.version.value,
            success=self._read_success(record, columns),
            extra={"tasks": record.get("tasks", [])},
        )
        return Trajectory(meta, InMemoryFeatureStore(columns))

    def _read_success(
        self, record: dict[str, Any], columns: dict[str, Array]
    ) -> SuccessLabel | None:
        """Reconstruct the episode :class:`SuccessLabel` from the metadata or a success column.

        Precedence: an explicit episode-level ``success`` field in ``episodes.jsonl`` (which a
        curated dataset we wrote will carry), otherwise a per-step ``SUCCESS``-role feature
        column (value taken from the final step). ``None`` if the source has no success notion.
        """
        per_step = self._success_column(columns)
        if "success" in record:
            return SuccessLabel(value=record["success"], source="dataset", per_step=per_step)
        if per_step is not None and per_step.size:
            return SuccessLabel(
                value=bool(per_step.reshape(per_step.shape[0], -1)[-1].max() > 0.5),
                source="dataset",
                per_step=per_step,
            )
        return None

    def _success_column(self, columns: dict[str, Array]) -> Array | None:
        for spec in self._embodiment.features:
            if spec.role is FeatureRole.SUCCESS and spec.key in columns:
                return columns[spec.key]
        return None


class LeRobotWriter(DatasetWriter):
    """Writes a *new* LeRobot v2.1 dataset; refuses to write to the source (invariant 1).

    The destination must not already exist and must not overlap the source directory. Every
    write ends with schema + checksum + round-trip validation; any failure quarantines the
    partial output and raises (invariant 2).
    """

    def __init__(
        self,
        dest: str | Path,
        *,
        source_root: str | Path | None = None,
        version: LeRobotVersion = LeRobotVersion.V2_1,
    ):
        _require_v2_1(version)
        self.dest = Path(dest)
        self.version = version
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
        (self.dest / "meta").mkdir(parents=True, exist_ok=False)
        (self.dest / "data" / _CHUNK).mkdir(parents=True, exist_ok=False)

        written_fingerprints: list[str] = []
        try:
            embodiment, episode_records, total_frames = self._write_episodes(
                trajectories, written_fingerprints
            )
            self._write_meta(embodiment, episode_records, total_frames)
            manifest_path = self.dest / "manifest.json"

            report = self.validate(self.dest)
            report.raise_if_invalid()
            self._assert_roundtrip(written_fingerprints)

            checksums = _checksum_tree(self.dest)
            fingerprint = LeRobotReader(self.dest).fingerprint()
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
        for key in ("codebase_version", "fps", "features", "total_episodes", "total_frames"):
            if key not in info:
                errors.append(f"info.json missing required key {key!r}")
        episodes_path = path / "meta" / "episodes.jsonl"
        if not episodes_path.is_file():
            errors.append(f"missing {episodes_path}")
            return ValidationReport(ok=False, errors=tuple(errors))

        records = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if "total_episodes" in info and info["total_episodes"] != len(records):
            errors.append(
                f"total_episodes={info['total_episodes']} but episodes.jsonl has "
                f"{len(records)} records"
            )
        feature_keys = list(info.get("features", {}))
        for record in records:
            index = record["episode_index"]
            ep_path = path / "data" / _CHUNK / f"episode_{index:06d}.parquet"
            if not ep_path.is_file():
                errors.append(f"missing parquet for episode {index}: {ep_path}")
                continue
            table = pq.read_table(ep_path)  # type: ignore[no-untyped-call]
            checked.append(str(ep_path.relative_to(path)))
            cols = set(table.column_names)
            for key in feature_keys:
                if key not in cols:
                    errors.append(f"episode {index}: missing feature column {key!r}")
            for key in _BOOKKEEPING:
                if key not in cols:
                    errors.append(f"episode {index}: missing bookkeeping column {key!r}")
            if table.num_rows != record.get("length"):
                errors.append(
                    f"episode {index}: parquet has {table.num_rows} rows but episodes.jsonl "
                    f"declares length {record.get('length')}"
                )
        return ValidationReport(ok=not errors, errors=tuple(errors), checked_files=tuple(checked))

    # -- internals -------------------------------------------------------------------

    def _write_episodes(
        self, trajectories: Iterable[Trajectory], written_fingerprints: list[str]
    ) -> tuple[EmbodimentSpec | None, list[dict[str, Any]], int]:
        embodiment: EmbodimentSpec | None = None
        episode_records: list[dict[str, Any]] = []
        running_index = 0
        for out_index, traj in enumerate(trajectories):
            if embodiment is None:
                embodiment = traj.embodiment
            self._reject_unsupported(traj.embodiment)
            table, running_index = _trajectory_to_table(traj, out_index, running_index)
            pq.write_table(table, _episode_file(self.dest, out_index))  # type: ignore[no-untyped-call]
            written_fingerprints.append(traj.meta.fingerprint)
            record: dict[str, Any] = {
                "episode_index": out_index,
                "length": traj.num_steps,
                "tasks": list(traj.meta.extra.get("tasks", [])),
            }
            # Preserve the episode-level success label so it survives the round trip (the
            # key is present iff the source carried a label; its value may be null=unknown).
            if traj.meta.success is not None:
                record["success"] = traj.meta.success.value
            episode_records.append(record)
        return embodiment, episode_records, running_index

    def _reject_unsupported(self, embodiment: EmbodimentSpec) -> None:
        for spec in embodiment.features:
            if _is_image_key(spec.key):
                raise NotImplementedError(
                    f"writing image/video feature {spec.key!r} is not yet supported; video "
                    "export lands in a later rung."
                )

    def _write_meta(
        self,
        embodiment: EmbodimentSpec | None,
        episode_records: list[dict[str, Any]],
        total_frames: int,
    ) -> None:
        if embodiment is None:
            raise ValueError("cannot write an empty dataset (no trajectories provided)")
        features = {
            spec.key: {
                "dtype": spec.dtype,
                "shape": list(spec.shape),
                "names": list(spec.names) if spec.names else None,
            }
            for spec in embodiment.features
        }
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": embodiment.embodiment_id,
            "fps": embodiment.control_hz,
            "features": features,
            "total_episodes": len(episode_records),
            "total_frames": total_frames,
        }
        (self.dest / "meta" / "info.json").write_text(
            json.dumps(info, indent=2, sort_keys=True), encoding="utf-8"
        )
        with (self.dest / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as fh:
            for record in episode_records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        # Minimal single-task table; richer task handling is a later addition.
        with (self.dest / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"task_index": 0, "task": ""}) + "\n")

    def _assert_roundtrip(self, written_fingerprints: list[str]) -> None:
        # Reload the just-written dataset and assert its canonical content fingerprints
        # match exactly what we intended to write (invariant 2).
        reread = [t.meta.fingerprint for t in LeRobotReader(self.dest)]
        if reread != written_fingerprints:
            raise ValidationError(
                "round-trip reload mismatch: the written dataset does not reload to the "
                "content that was written"
            )

    def _quarantine(self) -> None:
        if self.dest.exists():
            quarantine = self.dest.with_name(self.dest.name + ".invalid")
            if quarantine.exists():
                shutil.rmtree(quarantine)
            self.dest.rename(quarantine)


# -- arrow <-> numpy conversion ------------------------------------------------------


def _trajectory_to_table(
    traj: Trajectory, episode_index: int, start_index: int
) -> tuple[pa.Table, int]:
    """Convert one trajectory to a LeRobot parquet table; return (table, next_index)."""
    arrays: dict[str, pa.Array] = {}
    for spec in traj.embodiment.features:
        if not traj.has(spec.key):
            continue
        arrays[spec.key] = _array_to_arrow_column(traj.feature(spec.key), spec)
    num_rows = traj.num_steps
    arrays["episode_index"] = pa.array([episode_index] * num_rows, type=pa.int64())
    arrays["frame_index"] = pa.array(list(range(num_rows)), type=pa.int64())
    arrays["index"] = pa.array(list(range(start_index, start_index + num_rows)), type=pa.int64())
    arrays["task_index"] = pa.array([0] * num_rows, type=pa.int64())
    table = pa.table(arrays)
    return table, start_index + num_rows


def _array_to_arrow_column(arr: Array, spec: FeatureSpec) -> pa.Array:
    pa_type = pa.from_numpy_dtype(np.dtype(spec.dtype))
    if arr.ndim == 1:  # scalar-per-step feature, shape ()
        return pa.array(arr.astype(spec.dtype), type=pa_type)
    # Vector/tensor feature: flatten each timestep row and store as a list column.
    flat = arr.reshape(arr.shape[0], -1).astype(spec.dtype)
    return pa.array(list(flat), type=pa.list_(pa_type))


def _arrow_column_to_array(column: pa.ChunkedArray, spec: FeatureSpec) -> Array:
    if len(spec.shape) == 0:  # scalar-per-step
        return np.asarray(column.to_numpy(zero_copy_only=False), dtype=spec.dtype)
    rows = column.to_pylist()
    arr = np.asarray(rows, dtype=spec.dtype)
    return arr.reshape((len(rows), *spec.shape))


def _checksum_tree(root: Path) -> dict[str, str]:
    """Return ``relative path -> sha256`` for every file under ``root``."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


__all__ = [
    "CODEBASE_VERSION",
    "LeRobotReader",
    "LeRobotWriter",
]
