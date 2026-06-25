"""Reader for ManiSkill3 demonstration datasets (HDF5) -> canonical trajectories.

ManiSkill stores demonstrations as an HDF5 file of ``traj_0``, ``traj_1``, ... groups, each
with an ``actions`` ``(T, action_dim)`` dataset and an ``obs`` ``(T+1, obs_dim)`` dataset (in
``state`` observation mode), plus optional ``success`` / ``rewards``. This reader converts
each trajectory into the canonical :class:`~robocurate.trajectory.Trajectory` and implements
the read-only :class:`~robocurate.adapters.base.DatasetReader` protocol, so ManiSkill demos
flow through the same signals, curator, and experiment harness as everything else.

Dependency note: only ``h5py`` (a light dependency, in the ``maniskill`` extra) is needed to
parse the file — *not* the heavy ``mani_skill`` package, which is for the environment. h5py is
imported lazily, so this module loads without the extra.

Scope (v1): flat ``state`` observations (the de-risk path). Dict/visual observations and the
companion JSON metadata / control-mode replay are follow-ups; a dict ``obs`` group raises a
clear error rather than mis-handling data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
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


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading ManiSkill demonstrations requires h5py, an optional dependency. Install "
            "it with `uv pip install 'robocurate[maniskill-demos]'` (or `robocurate[maniskill]`)."
        ) from exc
    return h5py


class ManiSkillDemoReader:
    """Read-only :class:`DatasetReader` over a ManiSkill3 demonstration HDF5 file.

    Args:
        path: Path to the ``.h5`` demonstration file.
        embodiment_id: Embodiment id for the produced trajectories.
        control_hz: Control rate used to synthesize per-step timestamps.
        dataset_id: Identifier recorded in fingerprints/metadata (defaults to the file stem).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        embodiment_id: str = "maniskill",
        control_hz: float = DEFAULT_CONTROL_HZ,
        dataset_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.embodiment_id = embodiment_id
        self.control_hz = control_hz
        self.dataset_id = dataset_id or f"maniskill/{self.path.stem}"
        self._trajectories = self._load()
        self.meta = self._build_meta()

    def _load(self) -> list[Trajectory]:
        h5py = _require_h5py()
        trajectories: list[Trajectory] = []
        with h5py.File(self.path, "r") as handle:
            keys = sorted(
                (k for k in handle if k.startswith("traj_")),
                key=lambda k: int(k.split("_")[1]),
            )
            for index, key in enumerate(keys):
                trajectories.append(self._read_traj(index, handle[key]))
        return trajectories

    def _read_traj(self, index: int, group: Any) -> Trajectory:
        h5py = _require_h5py()
        actions = np.asarray(group["actions"], dtype=np.float32)
        num_steps = actions.shape[0]

        obs = group["obs"]
        if isinstance(obs, h5py.Group):
            raise NotImplementedError(
                "dict/visual ManiSkill observations are not supported yet; use a flat "
                "'state' obs_mode demonstration file."
            )
        # There are T+1 observations (s_0..s_T) for T actions; pair s_t with action_t.
        state = np.asarray(obs, dtype=np.float32)[:num_steps]

        columns: dict[str, Array] = {
            "timestamp": (np.arange(num_steps, dtype=np.float32) / self.control_hz),
            "observation.state": state,
            "action": actions,
        }
        if "rewards" in group:
            columns["reward"] = np.asarray(group["rewards"], dtype=np.float32)[:num_steps]

        success = self._read_success(group)
        embodiment = self._build_embodiment(columns)
        meta = TrajectoryMeta(
            source_dataset_id=self.dataset_id,
            episode_index=index,
            embodiment=embodiment,
            fingerprint=fingerprint_arrays(columns),
            num_steps=num_steps,
            source_format="maniskill_demo",
            success=success,
        )
        return Trajectory(meta, InMemoryFeatureStore(columns))

    def _read_success(self, group: Any) -> SuccessLabel | None:
        if "success" not in group:
            return None
        flags = np.asarray(group["success"]).reshape(-1)
        value = bool(flags[-1]) if flags.size else None
        return SuccessLabel(value=value, source="maniskill_demo")

    def _build_embodiment(self, columns: dict[str, Array]) -> EmbodimentSpec:
        roles = {
            "timestamp": FeatureRole.TIME,
            "observation.state": FeatureRole.PROPRIO,
            "action": FeatureRole.ACTION,
            "reward": FeatureRole.REWARD,
        }
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
            source_format="maniskill_demo",
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


__all__ = ["ManiSkillDemoReader"]
