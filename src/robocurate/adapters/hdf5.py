"""Configurable read-only adapter for generic one-group-per-episode HDF5 datasets.

The robomimic and ManiSkill readers each hard-code one HDF5 layout. Plenty of robot
datasets in the wild are *also* "one HDF5 group per episode, with an ``actions`` array and
an observation array/group" — just under different names (``data/demo_*`` vs ``traj_*``,
``obs`` as a group of low-dim keys vs a single flat ``state`` array, rewards/success present
or not). Rather than write a new reader for each, :class:`GenericHDF5Reader` reads any such
file given a small :class:`HDF5Schema` describing where the pieces live.

It implements the read-only :class:`~robocurate.adapters.base.DatasetReader` protocol, so a
generic HDF5 dataset flows through the same signals, curator, and experiment harness as
everything else. The file is opened ``"r"`` — the source is never mutated (invariant 1).

Dependency note: only ``h5py`` is needed (the ``hdf5`` extra, an alias of the light
``maniskill-demos`` extra); it is imported lazily so this module loads without it.

Scope (v1): flat low-dim observations (the de-risk path), concatenated in a deterministic
key order. Image/visual obs are a follow-up; an image-hint observation key raises a clear
error rather than mis-handling pixel data — exactly as the robomimic / ManiSkill readers do.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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

DEFAULT_CONTROL_HZ = 20.0
# Image observation keys are not supported in v1; their presence is an explicit error, not a
# silently mis-shaped state vector. Mirrors the robomimic reader's hint list.
_IMAGE_HINTS = ("image", "rgb", "depth", "camera")

# Default semantic roles for the canonical feature keys this reader emits. Overridable per
# schema via ``HDF5Schema.feature_roles`` for non-standard keys.
_DEFAULT_ROLES: dict[str, FeatureRole] = {
    "timestamp": FeatureRole.TIME,
    "observation.state": FeatureRole.PROPRIO,
    "action": FeatureRole.ACTION,
    "reward": FeatureRole.REWARD,
}


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading generic HDF5 datasets requires h5py, an optional dependency. Install "
            "it with `uv pip install 'robocurate[hdf5]'`."
        ) from exc
    return h5py


def _episode_sort_key(name: str) -> tuple[int, int | str]:
    """Deterministic natural-sort key for an episode group name.

    Episode groups are sorted by the *trailing* integer parsed from the name when present
    (so ``demo_2`` sorts before ``demo_10``), else lexicographically. The two cases are kept
    in separate buckets (``0`` before ``1``) so a mixed set still has a single stable order.
    """
    match = re.search(r"(\d+)$", name)
    if match is not None:
        return (0, int(match.group(1)))
    return (1, name)


@dataclass(frozen=True)
class HDF5Schema:
    """Describes where the pieces of a one-group-per-episode HDF5 dataset live.

    Every path is relative to the per-episode group (e.g. ``"actions"``, ``"obs/state"``).
    Sensible defaults match the common "root-level groups with ``actions`` + ``obs``" shape;
    the :meth:`robomimic_like` / :meth:`maniskill_like` classmethods pin the two layouts the
    dedicated readers handle and double as worked examples.

    Attributes:
        episode_root: Group holding the per-episode groups. ``""`` (default) means the file
            root (like ManiSkill); ``"data"`` means a nested parent group (like robomimic).
        episode_pattern: Regex matched against child group names to select episodes. ``None``
            (default) takes every child that is an h5py ``Group``.
        action_path: Path to the per-step action array, relative to the episode group.
        obs_path: Path to the observation node (a group of named arrays or a flat array).
        reward_path: Path to a per-step reward array, or ``None`` if absent.
        success_path: Path to a per-step success array, or ``None`` if absent.
        timestamp_path: Path to a per-step timestamp array, or ``None`` to synthesize one
            from ``control_hz``.
        obs_type: ``"group"`` / ``"array"`` to force the obs node's kind, or ``None``
            (default) to auto-detect via ``isinstance(node, h5py.Group)``.
        obs_keys: When obs is a group, the keys to use (in this exact order). ``None``
            (default) uses every key sorted lexically — deterministic and lossless.
        flatten_obs: When obs is a group, concatenate the selected keys into a single
            ``observation.state`` (default). When ``False``, emit each key as its own
            ``observation.<key>`` feature instead.
        feature_roles: Optional ``key -> FeatureRole`` overrides for non-standard keys,
            layered on top of the built-in default role map.
    """

    episode_root: str = ""
    episode_pattern: str | None = None
    action_path: str = "actions"
    obs_path: str = "obs"
    reward_path: str | None = None
    success_path: str | None = None
    timestamp_path: str | None = None
    obs_type: str | None = None
    obs_keys: tuple[str, ...] | None = None
    flatten_obs: bool = True
    feature_roles: Mapping[str, FeatureRole] | None = None

    @classmethod
    def robomimic_like(cls) -> HDF5Schema:
        """Schema matching the robomimic layout (``data/demo_*``, ``obs`` group, rewards)."""
        return cls(
            episode_root="data",
            episode_pattern=r"demo_\d+",
            obs_type="group",
            reward_path="rewards",
        )

    @classmethod
    def maniskill_like(cls) -> HDF5Schema:
        """Schema matching the ManiSkill layout (root ``traj_*``, flat ``obs``, reward+success)."""
        return cls(
            episode_root="",
            episode_pattern=r"traj_\d+",
            obs_type="array",
            reward_path="rewards",
            success_path="success",
        )


class GenericHDF5Reader:
    """Read-only :class:`DatasetReader` over an arbitrary one-group-per-episode HDF5 file.

    Args:
        path: Path to the ``.h5`` / ``.hdf5`` file.
        schema: The :class:`HDF5Schema` describing the layout (defaults to the common
            root-level ``actions`` + ``obs`` shape).
        embodiment_id: Embodiment id for the produced trajectories.
        control_hz: Control rate used to synthesize per-step timestamps when the schema has
            no ``timestamp_path``.
        dataset_id: Identifier recorded in fingerprints/metadata (defaults to the file stem).
        source_format: The ``source_format`` recorded on every trajectory + fingerprint.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        # HDF5Schema is a frozen (immutable) dataclass, so a shared default instance is safe;
        # B008's mutable-default concern does not apply.
        schema: HDF5Schema = HDF5Schema(),  # noqa: B008
        embodiment_id: str = "generic_hdf5",
        control_hz: float = DEFAULT_CONTROL_HZ,
        dataset_id: str | None = None,
        source_format: str = "generic_hdf5",
    ) -> None:
        self.path = Path(path)
        self.schema = schema
        self.embodiment_id = embodiment_id
        self.control_hz = control_hz
        self.dataset_id = dataset_id or f"generic_hdf5/{self.path.stem}"
        self.source_format = source_format
        self._trajectories = self._load()
        self.meta = self._build_meta()

    # -- loading ---------------------------------------------------------------------

    def _load(self) -> list[Trajectory]:
        h5py = _require_h5py()
        schema = self.schema
        pattern = re.compile(schema.episode_pattern) if schema.episode_pattern is not None else None
        trajectories: list[Trajectory] = []
        with h5py.File(self.path, "r") as handle:
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
                for name in root
                if isinstance(root[name], h5py.Group)
                and (pattern is None or pattern.fullmatch(name) is not None)
            ]
            for index, name in enumerate(sorted(names, key=_episode_sort_key)):
                trajectories.append(self._read_episode(index, name, root[name]))
        return trajectories

    def _read_episode(self, index: int, name: str, group: Any) -> Trajectory:
        schema = self.schema
        if schema.action_path not in group:
            raise ValueError(
                f"{self.path}: episode {name!r} has no action array at "
                f"{schema.action_path!r} (set HDF5Schema.action_path)"
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
        h5py = _require_h5py()
        schema = self.schema
        if schema.obs_path not in group:
            raise ValueError(
                f"{self.path}: episode has no observation at {schema.obs_path!r} "
                "(set HDF5Schema.obs_path)"
            )
        node = group[schema.obs_path]
        is_group = (
            isinstance(node, h5py.Group) if schema.obs_type is None else schema.obs_type == "group"
        )
        if not is_group:
            # A single flat observation array; the leading axis may hold T+1 entries (e.g.
            # ManiSkill stores s_0..s_T for T actions), so clip to the action length.
            return {"observation.state": np.asarray(node, dtype=np.float32)[:num_steps]}

        keys = schema.obs_keys if schema.obs_keys is not None else tuple(sorted(node.keys()))
        parts: dict[str, Array] = {}
        for key in keys:
            if any(hint in key.lower() for hint in _IMAGE_HINTS):
                raise NotImplementedError(
                    f"image observation {key!r} is not supported yet; set HDF5Schema.obs_keys "
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
        return SuccessLabel(value=value, source="generic_hdf5")

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


__all__ = ["GenericHDF5Reader", "HDF5Schema"]
