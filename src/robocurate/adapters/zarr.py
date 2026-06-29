"""Configurable read-only adapter for generic one-group-per-episode Zarr stores.

Zarr is a common cloud-native chunked-array store (think "HDF5 for object storage"): a
hierarchy of *groups* and *arrays* with a near-identical access API to h5py. Plenty of
robot datasets are published as Zarr in exactly the shape the generic HDF5 reader already
handles — "one group per episode, with an ``actions`` array and an observation array/group"
— just under different names (``data/demo_*`` vs ``traj_*``, ``obs`` as a group of low-dim
keys vs a single flat ``state`` array, rewards/success present or not). :class:`ZarrReader`
reads any such store given a small schema describing where the pieces live.

The layout description is storage-agnostic, so this reader **reuses**
:class:`~robocurate.adapters.hdf5.HDF5Schema` directly rather than duplicating it; an alias
:data:`ZarrSchema` is re-exported for discoverability. It implements the read-only
:class:`~robocurate.adapters.base.DatasetReader` protocol, so a Zarr dataset flows through
the same signals, curator, and experiment harness as everything else. The store is opened
``mode="r"`` — the source is never mutated (invariant 1).

Dependency note: only ``zarr`` is needed (the ``zarr`` extra); it is imported lazily so this
module loads without it.

Scope (v1): flat low-dim observations (the de-risk path), concatenated in a deterministic
key order. Image/visual obs are a follow-up; an image-hint observation key raises a clear
error rather than mis-handling pixel data — exactly as the HDF5 / robomimic readers do.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from robocurate.adapters.hdf5 import (
    DEFAULT_CONTROL_HZ,
    HDF5Schema,
    _episode_sort_key,
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
    fingerprint_arrays,
)

# The schema is pure storage-agnostic configuration (string paths + roles), so the HDF5 one
# is reused verbatim rather than duplicated. ``ZarrSchema`` is an alias kept for symmetry and
# discoverability at the Zarr import site.
ZarrSchema = HDF5Schema

# Image observation keys are not supported in v1; their presence is an explicit error, not a
# silently mis-shaped state vector. Mirrors the HDF5 / robomimic readers' hint list.
_IMAGE_HINTS = ("image", "rgb", "depth", "camera")

# Default semantic roles for the canonical feature keys this reader emits. Overridable per
# schema via ``HDF5Schema.feature_roles`` for non-standard keys.
_DEFAULT_ROLES: dict[str, FeatureRole] = {
    "timestamp": FeatureRole.TIME,
    "observation.state": FeatureRole.PROPRIO,
    "action": FeatureRole.ACTION,
    "reward": FeatureRole.REWARD,
}


def _require_zarr() -> Any:
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading generic Zarr datasets requires zarr, an optional dependency. Install "
            "it with `uv pip install 'robocurate[zarr]'`."
        ) from exc
    return zarr


def _is_group(node: Any) -> bool:
    """Whether a Zarr node is a group (vs an array), without importing zarr at call sites.

    Zarr groups expose ``group_keys`` / ``array_keys``; arrays do not. Checking for the
    method is version-robust across the zarr 2.x / 3.x class reshuffles.
    """
    return hasattr(node, "group_keys")


class ZarrReader:
    """Read-only :class:`DatasetReader` over an arbitrary one-group-per-episode Zarr store.

    Args:
        path: Path to the Zarr store (a directory, ``.zarr`` directory, or any path
            ``zarr.open_group`` accepts).
        schema: The :class:`HDF5Schema` (re-exported as :data:`ZarrSchema`) describing the
            layout (defaults to the common root-level ``actions`` + ``obs`` shape).
        embodiment_id: Embodiment id for the produced trajectories.
        control_hz: Control rate used to synthesize per-step timestamps when the schema has
            no ``timestamp_path``.
        dataset_id: Identifier recorded in fingerprints/metadata (defaults to the store stem).
        source_format: The ``source_format`` recorded on every trajectory + fingerprint.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        # HDF5Schema is a frozen (immutable) dataclass, so a shared default instance is safe;
        # B008's mutable-default concern does not apply.
        schema: HDF5Schema = HDF5Schema(),  # noqa: B008
        embodiment_id: str = "zarr",
        control_hz: float = DEFAULT_CONTROL_HZ,
        dataset_id: str | None = None,
        source_format: str = "zarr",
    ) -> None:
        self.path = Path(path)
        self.schema = schema
        self.embodiment_id = embodiment_id
        self.control_hz = control_hz
        self.dataset_id = dataset_id or f"zarr/{self.path.stem}"
        self.source_format = source_format
        self._trajectories = self._load()
        self.meta = self._build_meta()

    # -- loading ---------------------------------------------------------------------

    def _load(self) -> list[Trajectory]:
        zarr = _require_zarr()
        schema = self.schema
        pattern = re.compile(schema.episode_pattern) if schema.episode_pattern is not None else None
        trajectories: list[Trajectory] = []
        handle = zarr.open_group(str(self.path), mode="r")
        if schema.episode_root:
            if schema.episode_root not in handle:
                raise ValueError(
                    f"{self.path}: episode_root group {schema.episode_root!r} is missing"
                )
            root = handle[schema.episode_root]
        else:
            root = handle
        names = [
            name
            for name in root.group_keys()
            if pattern is None or pattern.fullmatch(name) is not None
        ]
        for index, name in enumerate(sorted(names, key=_episode_sort_key)):
            trajectories.append(self._read_episode(index, name, root[name]))
        return trajectories

    def _read_episode(self, index: int, name: str, group: Any) -> Trajectory:
        schema = self.schema
        if schema.action_path not in group:
            raise ValueError(
                f"{self.path}: episode {name!r} has no action array at "
                f"{schema.action_path!r} (set ZarrSchema.action_path)"
            )
        actions = np.asarray(group[schema.action_path], dtype=np.float32)
        num_steps = actions.shape[0]

        columns: dict[str, Array] = {
            "timestamp": self._timestamps(group, num_steps),
            "action": actions,
        }
        columns.update(self._observations(group, num_steps))
        if schema.reward_path is not None and schema.reward_path in group:
            columns["reward"] = np.asarray(group[schema.reward_path], dtype=np.float32).reshape(-1)[
                :num_steps
            ]

        success = self._read_success(group)
        embodiment = self._build_embodiment(columns)
        meta = TrajectoryMeta(
            source_dataset_id=self.dataset_id,
            episode_index=index,
            embodiment=embodiment,
            fingerprint=fingerprint_arrays(columns),
            num_steps=num_steps,
            source_format=self.source_format,
            success=success,
            extra={"source_group": name},
        )
        return Trajectory(meta, InMemoryFeatureStore(columns))

    def _timestamps(self, group: Any, num_steps: int) -> Array:
        path = self.schema.timestamp_path
        if path is not None and path in group:
            return np.asarray(group[path], dtype=np.float32).reshape(-1)[:num_steps]
        return np.arange(num_steps, dtype=np.float32) / self.control_hz

    def _observations(self, group: Any, num_steps: int) -> dict[str, Array]:
        schema = self.schema
        if schema.obs_path not in group:
            raise ValueError(
                f"{self.path}: episode has no observation at {schema.obs_path!r} "
                "(set ZarrSchema.obs_path)"
            )
        node = group[schema.obs_path]
        is_group = _is_group(node) if schema.obs_type is None else schema.obs_type == "group"
        if not is_group:
            # A single flat observation array; the leading axis may hold T+1 entries (e.g.
            # ManiSkill stores s_0..s_T for T actions), so clip to the action length.
            return {"observation.state": np.asarray(node, dtype=np.float32)[:num_steps]}

        keys = schema.obs_keys if schema.obs_keys is not None else tuple(sorted(node.array_keys()))
        parts: dict[str, Array] = {}
        for key in keys:
            if any(hint in key.lower() for hint in _IMAGE_HINTS):
                raise NotImplementedError(
                    f"image observation {key!r} is not supported yet; set ZarrSchema.obs_keys "
                    "to select the low-dim keys, or use a low-dim-only dataset."
                )
            arr = np.asarray(node[key], dtype=np.float32)[:num_steps]
            if arr.ndim == 1:
                arr = arr.reshape(num_steps, 1)
            parts[key] = arr
        if schema.flatten_obs:
            return {"observation.state": np.concatenate(list(parts.values()), axis=1)}
        return {f"observation.{key}": arr for key, arr in parts.items()}

    def _read_success(self, group: Any) -> SuccessLabel | None:
        path = self.schema.success_path
        if path is None or path not in group:
            return None
        flags = np.asarray(group[path]).reshape(-1)
        value = bool(flags[-1]) if flags.size else None
        return SuccessLabel(value=value, source="zarr")

    def _build_embodiment(self, columns: dict[str, Array]) -> EmbodimentSpec:
        roles = dict(_DEFAULT_ROLES)
        if self.schema.feature_roles is not None:
            roles.update(self.schema.feature_roles)
        features = tuple(
            FeatureSpec(
                key=key,
                role=roles.get(key, FeatureRole.EXTRA),
                shape=tuple(arr.shape[1:]),
                dtype=str(arr.dtype),
            )
            for key, arr in columns.items()
        )
        return EmbodimentSpec(
            embodiment_id=self.embodiment_id, features=features, control_hz=self.control_hz
        )

    # -- DatasetReader protocol ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._trajectories)

    def read_episode(self, index: int) -> Trajectory:
        try:
            return self._trajectories[index]
        except IndexError as exc:
            raise IndexError(f"episode {index} out of range (0..{len(self) - 1})") from exc

    def fingerprint(self) -> DatasetFingerprint:
        roll = hashlib.sha256()
        for fp in sorted(t.meta.fingerprint for t in self._trajectories):
            roll.update(fp.encode("utf-8"))
        return DatasetFingerprint(
            dataset_id=self.dataset_id,
            source_format=self.source_format,
            content_hash=roll.hexdigest(),
            num_episodes=len(self._trajectories),
        )

    def _build_meta(self) -> DatasetMeta:
        feature_keys: tuple[str, ...] = (
            tuple(s.key for s in self._trajectories[0].embodiment.features)
            if self._trajectories
            else ()
        )
        return DatasetMeta(
            fingerprint=self.fingerprint(),
            embodiment_ids=(self.embodiment_id,),
            feature_keys=feature_keys,
        )


__all__ = ["ZarrReader", "ZarrSchema"]
