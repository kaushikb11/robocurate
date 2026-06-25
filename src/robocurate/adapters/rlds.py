"""RLDS read adapter — Open X-Embodiment / DROID and other TFDS robot datasets.

RLDS is the de-facto format for large real/teleop robot datasets (Open X-Embodiment, DROID).
It is a TFDS layout where a dataset is a sequence of *episodes*, and each episode carries a
nested ``steps`` sequence of per-timestep records: ``observation`` (often a dict of
sub-observations), ``action``, ``reward``, and flags like ``is_first`` / ``is_last`` /
``is_terminal``.

This reader converts RLDS episodes into the canonical
:class:`~robocurate.trajectory.Trajectory` and implements the read-only
:class:`~robocurate.adapters.base.DatasetReader` protocol, so RLDS data flows through the
same signals, curator, and experiment harness as LeRobot data.

Dependency note: the conversion uses ``numpy.asarray`` (which works on TensorFlow eager
tensors), so **the adapter code itself imports no TensorFlow**. The optional ``rlds`` extra
(``tensorflow-datasets``) is needed only to *load* a real dataset via
:meth:`RLDSReader.from_tfds`; constructing an ``RLDSReader`` from an already-obtained episode
iterable (e.g. a ``tf.data.Dataset``) needs nothing extra.

Scope (v1): episodes are materialized eagerly into canonical trajectories on construction
(streaming over a huge Open X shard is a later optimization); image observations are carried
as ordinary array features.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from robocurate.metadata import DatasetFingerprint, DatasetMeta
from robocurate.trajectory import (
    Array,
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    InMemoryFeatureStore,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_FPS = 10.0
_IMAGE_HINTS = ("image", "rgb", "depth", "wrist", "camera")


def _infer_role(key: str) -> FeatureRole:
    """Infer a :class:`FeatureRole` from a flattened RLDS feature key by naming convention."""
    lowered = key.lower()
    if any(hint in lowered for hint in _IMAGE_HINTS):
        return FeatureRole.IMAGE
    if key == "action":
        return FeatureRole.ACTION
    if key == "reward":
        return FeatureRole.REWARD
    if key == "timestamp":
        return FeatureRole.TIME
    if "state" in lowered or "proprio" in lowered or "qpos" in lowered:
        return FeatureRole.PROPRIO
    if key.startswith("observation"):
        return FeatureRole.STATE
    return FeatureRole.EXTRA


class RLDSReader:
    """Read-only :class:`~robocurate.adapters.base.DatasetReader` over RLDS episodes.

    Args:
        episodes: An iterable of RLDS episodes — each a mapping with a ``steps`` entry that is
            itself an iterable of per-step mappings. A ``tf.data.Dataset`` from ``tfds.load``
            satisfies this directly, as does a plain list of dicts (for testing).
        dataset_id: Identifier recorded in fingerprints and metadata.
        embodiment_id: Embodiment id for the produced trajectories.
        fps: Control rate used to synthesize per-step timestamps (RLDS rarely stores them).
        steps_key / action_key / observation_key / reward_key: RLDS field names.
    """

    def __init__(
        self,
        episodes: Iterable[Mapping[str, Any]],
        *,
        dataset_id: str = "rlds",
        embodiment_id: str = "rlds",
        fps: float = DEFAULT_FPS,
        steps_key: str = "steps",
        action_key: str = "action",
        observation_key: str = "observation",
        reward_key: str = "reward",
    ) -> None:
        self.dataset_id = dataset_id
        self.embodiment_id = embodiment_id
        self.fps = fps
        self._steps_key = steps_key
        self._action_key = action_key
        self._observation_key = observation_key
        self._reward_key = reward_key
        self._trajectories = self._materialize(episodes)
        self.meta = self._build_meta()

    @classmethod
    def from_tfds(
        cls,
        name: str,
        *,
        split: str = "train",
        data_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> RLDSReader:
        """Load an RLDS dataset by name via ``tensorflow_datasets`` (needs the ``rlds`` extra)."""
        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "RLDSReader.from_tfds requires tensorflow-datasets, an optional dependency. "
                "Install it with `uv pip install 'robocurate[rlds]'`."
            ) from exc
        episodes = tfds.load(name, split=split, data_dir=str(data_dir) if data_dir else None)
        return cls(episodes, dataset_id=name, **kwargs)

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
            source_format="rlds",
            content_hash=roll.hexdigest(),
            num_episodes=len(self._trajectories),
        )

    # -- conversion ------------------------------------------------------------------

    def _materialize(self, episodes: Iterable[Mapping[str, Any]]) -> list[Trajectory]:
        return [self._episode_to_trajectory(i, ep) for i, ep in enumerate(episodes)]

    def _episode_to_trajectory(self, index: int, episode: Mapping[str, Any]) -> Trajectory:
        per_step: dict[str, list[Array]] = {}
        num_steps = 0
        for step in episode[self._steps_key]:
            num_steps += 1
            for key, value in self._flatten_step(step).items():
                per_step.setdefault(key, []).append(np.asarray(value))

        columns: dict[str, Array] = {k: np.stack(v, axis=0) for k, v in per_step.items()}
        columns["timestamp"] = (np.arange(num_steps, dtype=np.float64) / self.fps).astype(
            np.float32
        )
        embodiment = self._build_embodiment(columns)
        meta = TrajectoryMeta(
            source_dataset_id=self.dataset_id,
            episode_index=index,
            embodiment=embodiment,
            fingerprint=fingerprint_arrays(columns),
            num_steps=num_steps,
            source_format="rlds",
            extra={},
        )
        return Trajectory(meta, InMemoryFeatureStore(columns))

    def _flatten_step(self, step: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        observation = step.get(self._observation_key)
        if isinstance(observation, Mapping):
            for sub_key, sub_value in observation.items():
                out[f"observation.{sub_key}"] = sub_value
        elif observation is not None:
            out["observation"] = observation
        if self._action_key in step:
            out["action"] = step[self._action_key]
        if self._reward_key in step:
            out["reward"] = step[self._reward_key]
        return out

    def _build_embodiment(self, columns: Mapping[str, Array]) -> EmbodimentSpec:
        features = tuple(
            FeatureSpec(
                key=key,
                role=_infer_role(key),
                shape=tuple(arr.shape[1:]),
                dtype=str(arr.dtype),
            )
            for key, arr in columns.items()
        )
        return EmbodimentSpec(
            embodiment_id=self.embodiment_id, features=features, control_hz=self.fps
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


__all__ = ["RLDSReader"]
