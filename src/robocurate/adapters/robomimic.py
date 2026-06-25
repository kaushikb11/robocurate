"""Reader for robomimic demonstration datasets (HDF5) -> canonical trajectories.

`robomimic <https://robomimic.github.io>`_ ships real teleoperated manipulation
demonstrations as HDF5. The layout (see the robomimic dataset docs) is a top-level ``data``
group of ``demo_0``, ``demo_1``, ... groups, each with an ``actions`` ``(N, A)`` dataset, an
``obs`` subgroup of named low-dim observation arrays (each ``(N, d)``), and optional
``rewards`` / ``dones`` ``(N,)``. A sibling ``mask`` group holds *filter keys*: each
``mask/<key>`` is a list of demo names (byte strings) belonging to a subset.

The reason this dataset is interesting for curation is the **Multi-Human (MH)** variant: its
demos are collected by operators of differing skill, and the proficiency tier of each demo
(``better`` / ``okay`` / ``worse``) is recorded as exactly those filter keys. That gives a
*ground-truth quality label* to check a curator against — do the cheap signals preferentially
flag the "worse"-operator trajectories? This reader carries that tier through on
``Trajectory.meta.extra["operator_tier"]`` so the experiment harness can score against it.

This reader converts each demo into the canonical :class:`~robocurate.trajectory.Trajectory`
and implements the read-only :class:`~robocurate.adapters.base.DatasetReader` protocol, so
robomimic data flows through the same signals, curator, and harness as everything else. The
file is opened ``"r"`` — the source is never mutated (invariant 1).

Dependency note: only ``h5py`` is needed (the ``robomimic`` extra, an alias of the light
``maniskill-demos`` extra); it is imported lazily so this module loads without it.

Scope (v1): flat low-dim observations (the de-risk path), concatenated in a fixed sorted key
order for determinism. Image/visual obs are a follow-up; an image observation key raises a
clear error rather than mis-handling pixel data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
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
# The robomimic MH proficiency tiers, stored as filter keys under ``mask/``. A demo belongs to
# exactly one of these; orthogonal splits (``train`` / ``valid`` / operator-specific keys) are
# ignored here — only the quality tier is carried through as a ground-truth label.
PROFICIENCY_TIERS = ("better", "okay", "worse")
# Image observation keys are not supported in v1; their presence is an explicit error, not a
# silently mis-shaped state vector.
_IMAGE_HINTS = ("image", "rgb", "depth", "camera")


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "reading robomimic demonstrations requires h5py, an optional dependency. Install "
            "it with `uv pip install 'robocurate[robomimic]'`."
        ) from exc
    return h5py


class RoboMimicReader:
    """Read-only :class:`DatasetReader` over a robomimic demonstration HDF5 file.

    Args:
        path: Path to the robomimic ``.hdf5`` file (e.g. a ``low_dim`` MH dataset).
        obs_keys: Low-dim observation keys to concatenate into the flat state, in this exact
            order. ``None`` (default) uses every key under ``obs`` sorted lexically, which is
            deterministic and lossless; pass an explicit list to pin a canonical subset/order.
        expose_keys: Obs keys to also emit as their own ``observation.<key>`` features (role
            ``EXTRA``), in addition to the flat state. Defaults to the Cartesian end-effector
            position so the path-efficiency signal can measure the true path; absent keys are
            skipped. Pass ``()`` to expose nothing.
        embodiment_id: Embodiment id for the produced trajectories.
        control_hz: Control rate used to synthesize per-step timestamps.
        dataset_id: Identifier recorded in fingerprints/metadata (defaults to the file stem).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        obs_keys: Sequence[str] | None = None,
        expose_keys: Sequence[str] = ("robot0_eef_pos",),
        embodiment_id: str = "robomimic",
        control_hz: float = DEFAULT_CONTROL_HZ,
        dataset_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.obs_keys = tuple(obs_keys) if obs_keys is not None else None
        self.expose_keys = tuple(expose_keys)
        self.embodiment_id = embodiment_id
        self.control_hz = control_hz
        self.dataset_id = dataset_id or f"robomimic/{self.path.stem}"
        self._trajectories = self._load()
        self.meta = self._build_meta()

    # -- loading ---------------------------------------------------------------------

    def _load(self) -> list[Trajectory]:
        h5py = _require_h5py()
        trajectories: list[Trajectory] = []
        with h5py.File(self.path, "r") as handle:
            if "data" not in handle:
                raise ValueError(
                    f"{self.path} is not a robomimic dataset (no top-level 'data' group)"
                )
            data = handle["data"]
            tier_of = self._read_tiers(handle)
            demo_names = sorted(
                (k for k in data if k.startswith("demo_")),
                key=lambda k: int(k.split("_")[1]),
            )
            for index, name in enumerate(demo_names):
                tier = tier_of.get(name)
                trajectories.append(self._read_demo(index, name, data[name], tier))
        return trajectories

    def _read_tiers(self, handle: Any) -> dict[str, str]:
        """Map ``demo_<i>`` -> proficiency tier from the ``mask/`` filter keys (if present)."""
        tier_of: dict[str, str] = {}
        if "mask" not in handle:
            return tier_of
        mask = handle["mask"]
        for tier in PROFICIENCY_TIERS:
            if tier not in mask:
                continue
            for raw in np.asarray(mask[tier]).reshape(-1):
                name = raw.decode() if isinstance(raw, bytes | np.bytes_) else str(raw)
                tier_of[name] = tier
        return tier_of

    def _read_demo(self, index: int, name: str, group: Any, tier: str | None) -> Trajectory:
        actions = np.asarray(group["actions"], dtype=np.float32)
        num_steps = actions.shape[0]
        obs = group["obs"]
        state = self._flat_state(obs, num_steps)

        columns: dict[str, Array] = {
            "timestamp": (np.arange(num_steps, dtype=np.float32) / self.control_hz),
            "observation.state": state,
            "action": actions,
        }
        if "rewards" in group:
            columns["reward"] = np.asarray(group["rewards"], dtype=np.float32).reshape(-1)[
                :num_steps
            ]
        # Expose selected low-dim obs keys (e.g. the Cartesian end-effector position) as their
        # own features so a signal can address them directly — useful for the path-efficiency
        # directness signal, which wants the true Cartesian path rather than the flat state.
        # They carry FeatureRole.EXTRA so role-globbing signals (e.g. redundancy) ignore them
        # and are not perturbed by the duplication; they are already inside observation.state.
        for key in self.expose_keys:
            if key in obs:
                columns[f"observation.{key}"] = np.asarray(obs[key], dtype=np.float32)[:num_steps]

        success = self._read_success(group)
        extra: dict[str, Any] = {"source_demo": name}
        if tier is not None:
            extra["operator_tier"] = tier

        embodiment = self._build_embodiment(columns)
        meta = TrajectoryMeta(
            source_dataset_id=self.dataset_id,
            episode_index=index,
            embodiment=embodiment,
            fingerprint=fingerprint_arrays(columns),
            num_steps=num_steps,
            source_format="robomimic",
            success=success,
            extra=extra,
        )
        return Trajectory(meta, InMemoryFeatureStore(columns))

    def _flat_state(self, obs: Any, num_steps: int) -> Array:
        h5py = _require_h5py()
        if not isinstance(obs, h5py.Group):
            raise ValueError("robomimic 'obs' is expected to be a group of named arrays")
        keys = self.obs_keys if self.obs_keys is not None else tuple(sorted(obs.keys()))
        parts: list[Array] = []
        for key in keys:
            if any(hint in key.lower() for hint in _IMAGE_HINTS):
                raise NotImplementedError(
                    f"image observation {key!r} is not supported yet; pass obs_keys to select "
                    "the low-dim keys, or use a low_dim robomimic dataset."
                )
            arr = np.asarray(obs[key], dtype=np.float32)[:num_steps]
            if arr.ndim == 1:
                arr = arr.reshape(num_steps, 1)
            parts.append(arr)
        return np.concatenate(parts, axis=1)

    def _read_success(self, group: Any) -> SuccessLabel | None:
        """robomimic demos are successful by construction; reflect that as the label.

        When sparse ``rewards`` are present the terminal reward (1.0 on task success) confirms
        it; otherwise we record ``True`` from the demonstrator with no fabricated per-step
        signal. Curation here is about *relative quality among successes*, not success itself.
        """
        if "rewards" in group:
            rewards = np.asarray(group["rewards"]).reshape(-1)
            value = bool(rewards[-1] > 0) if rewards.size else None
            return SuccessLabel(value=value, source="robomimic_reward")
        return SuccessLabel(value=True, source="robomimic_demo")

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
            source_format="robomimic",
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


__all__ = ["PROFICIENCY_TIERS", "RoboMimicReader"]
