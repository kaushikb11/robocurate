"""Canonical internal trajectory representation.

This is the in-memory form the core works in; dataset adapters convert source formats
(LeRobotDataset, RLDS, raw sim output) into it. It is designed to handle image and state
observations, proprioception, actions, rewards, success labels, variable control rates,
multiple embodiments, and variable-length episodes across both real/teleop and sim data
— *without* collapsing to a lossy lowest-common-denominator.

Design in one sentence: **separate the schema (what features exist, their roles, units,
and per-dim names) from the data (the arrays), and materialize the data lazily** through a
:class:`FeatureStore` backend so huge datasets stream episode-by-episode and cheap signals
never decode video.

Key ideas
---------
* A :class:`Trajectory` is exactly one variable-length episode. ``T`` (the number of
  timesteps) differs per trajectory; nothing pads to a global max.
* Features (images, state, proprio, actions, rewards, ...) are reached through one uniform
  ``feature(key)`` accessor over a typed feature table rather than fixed named fields, so a
  new modality or embodiment needs *no* core change (Invariant 4).
* Each feature carries a :class:`FeatureSpec` with an explicit :class:`FeatureRole`, units,
  and per-dim names, so meaning is never inferred or assumed.
* Every trajectory carries its :class:`EmbodimentSpec`, so a :class:`TrajectorySet` may be
  heterogeneous (mixed embodiments / observation+action spaces) without a global
  action-dim assumption.
* ``timestamps()`` is the authoritative source of control rate; ``control_hz`` is only a
  hint. Irregular/teleop rates are first-class.

The exchange type a signal receives from ``feature()`` is a NumPy array. The core has no
torch dependency; GPU signals convert at their own boundary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

# Canonical exchange array type. NumPy is the lingua franca of the core (see module
# docstring); ``np.generic`` is the common base of all NumPy scalar dtypes, so this covers
# uint8 images and float32 actions alike.
Array = npt.NDArray[np.generic]


@dataclass(frozen=True)
class VideoReference:
    """A *reference* to one camera's video frames for a single episode — never the pixels.

    Stage-1 image/video pass-through preserves frames without decoding them: an IMAGE-role
    feature is materialized as this lightweight reference rather than a ``(T, H, W, C)`` pixel
    array. It records exactly where the episode's frames live so a writer can **copy** the
    backing shard file(s) into a curated output and a reader can later (Stage 2) decode them.

    A v3 video shard may bundle multiple episodes; ``from_timestamp`` / ``to_timestamp`` mark
    this episode's slice within ``shard_path``. Stage-1 copies the whole shard file(s) any kept
    episode references and keeps these references consistent, so no pixels are touched.

    Attributes:
        video_key: The source feature key, e.g. ``"observation.images.wrist"``.
        num_frames: ``T`` — the number of frames this episode contributes (one per timestep).
        frame_indices: The episode-local ``(T,)`` frame indices ``0..T-1`` (an int array; the
            only array carried, and it is *bookkeeping*, not pixels). ``None`` if unknown.
        shard_path: Absolute path to the backing mp4 shard file holding these frames, or
            ``None`` when the source did not expose one (the reference is then opaque).
        shard_chunk_index: The shard's ``chunk_index`` (the ``chunk-NNN`` directory), or ``None``.
        shard_file_index: The shard's ``file_index`` (the ``file-NNN`` stem), or ``None``.
        from_timestamp: Start time (seconds) of this episode's frames within ``shard_path``.
        to_timestamp: End time (seconds) of this episode's frames within ``shard_path``.
    """

    video_key: str
    num_frames: int
    frame_indices: Array | None = None
    shard_path: Path | None = None
    shard_chunk_index: int | None = None
    shard_file_index: int | None = None
    from_timestamp: float | None = None
    to_timestamp: float | None = None


def fingerprint_arrays(columns: Mapping[str, Array | VideoReference]) -> str:
    """Return a stable content hash over a set of feature arrays.

    The hash is order-independent in the input mapping (keys are sorted) and covers both
    the key names and the raw array bytes, so two trajectories with identical content
    produce the same fingerprint. Used for reproducibility, manifest linkage, and dedup, and
    shared by adapters and fixtures so a round-tripped trajectory keeps its fingerprint.

    Image/video pixels are never loaded in Stage-1 pass-through, so a column whose value is a
    :class:`VideoReference` (a per-camera shard *reference*, not pixels) is **excluded** from
    the content hash: there is no pixel array to call ``.tobytes()`` on, and the curated output
    preserves the very same frames byte-for-byte (the writer copies the shard files), so the
    low-dim content hash is the correct, stable identity. Hashing only the low-dim arrays keeps
    the fingerprint reproducible across the read -> curate -> reload round-trip.
    """
    hasher = hashlib.sha256()
    for key in sorted(columns):
        value = columns[key]
        if isinstance(value, VideoReference):
            continue  # pixels are never materialized; references are not part of the content hash
        hasher.update(key.encode("utf-8"))
        hasher.update(np.ascontiguousarray(value).tobytes())
    return hasher.hexdigest()


class FeatureRole(Enum):
    """The semantic role of a feature, independent of its source key name.

    Roles let a signal request "all image features" or "the action array" generically,
    rather than hard-coding source-specific key strings. ``EXTRA`` is the escape hatch for
    anything that does not fit a known role (e.g. privileged sim state) — it is still
    carried losslessly, never dropped.
    """

    IMAGE = "image"
    STATE = "state"
    PROPRIO = "proprio"
    ACTION = "action"
    REWARD = "reward"
    SUCCESS = "success"
    TIME = "time"
    EXTRA = "extra"


@dataclass(frozen=True)
class FeatureSpec:
    """Describes one feature column of a trajectory (its schema, not its data).

    Attributes:
        key: The source key, e.g. ``"observation.images.wrist"`` or ``"action"``. Unique
            within an :class:`EmbodimentSpec`.
        role: The semantic :class:`FeatureRole`.
        shape: Per-timestep shape, e.g. ``(3, 224, 224)`` for an image or ``(7,)`` for a
            7-DoF action. The full materialized array is ``(T, *shape)``.
        dtype: Canonical NumPy dtype string, e.g. ``"float32"``, ``"uint8"``.
        units: Explicit physical units, e.g. ``"rad"``, ``"rad/s"``, ``"m"``,
            ``"normalized[-1,1]"``, ``"pixel"``. ``None`` only when genuinely unitless.
            Never assumed by signals — read from here.
        names: Optional per-dim names, e.g. ``("joint0", ..., "gripper")``, preserving the
            meaning of each action/state dimension across embodiments.
    """

    key: str
    role: FeatureRole
    shape: tuple[int, ...]
    dtype: str
    units: str | None = None
    names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EmbodimentSpec:
    """The observation/action space of an embodiment, travelling with every trajectory.

    Carrying the full spec per trajectory is mildly redundant for a homogeneous dataset but
    is what makes mixed-embodiment :class:`TrajectorySet`\\ s correct and the manifest
    self-describing. It is metadata only, so the cost is negligible.

    Attributes:
        embodiment_id: Stable identifier, e.g. ``"franka_panda"``, ``"so100"``.
        features: The feature schema (one :class:`FeatureSpec` per column).
        control_hz: Nominal control rate hint, or ``None`` if irregular. The authoritative
            timing source is :meth:`Trajectory.timestamps`.
    """

    embodiment_id: str
    features: tuple[FeatureSpec, ...]
    control_hz: float | None = None

    def feature(self, key: str) -> FeatureSpec | None:
        """Return the :class:`FeatureSpec` for ``key``, or ``None`` if absent."""
        for spec in self.features:
            if spec.key == key:
                return spec
        return None

    def keys_with_role(self, *roles: FeatureRole) -> tuple[str, ...]:
        """Return the keys of all features whose role is in ``roles``, in declared order."""
        wanted = set(roles)
        return tuple(spec.key for spec in self.features if spec.role in wanted)


@dataclass(frozen=True)
class SuccessLabel:
    """A tri-state success label for an episode.

    "Success" is genuinely three-valued in this domain: a demonstrator may assert success,
    a sim reward may compute it, a VLM may relabel it, or it may simply be unknown. We do
    not coerce unknown to ``False``.

    Attributes:
        value: ``True`` / ``False`` / ``None`` (unknown).
        source: Where the label came from, e.g. ``"demonstrator"``, ``"sim_reward"``,
            ``"vlm"``, ``"unlabeled"``.
        per_step: Optional ``(T,)`` per-step success/progress signal, or ``None``.
    """

    value: bool | None
    source: str
    per_step: Array | None = None


@dataclass(frozen=True)
class TrajectoryMeta:
    """Metadata that travels with a trajectory for the manifest and reproducibility.

    Attributes:
        source_dataset_id: The source dataset identifier, e.g.
            ``"lerobot/aloha_sim_insertion"``.
        episode_index: The episode's index within the source dataset.
        embodiment: The :class:`EmbodimentSpec` this trajectory conforms to.
        fingerprint: A content hash over the raw feature bytes, used for reproducibility,
            manifest linkage, and dedup. Stable for identical content.
        num_steps: ``T``, the number of timesteps in this episode.
        source_format: The exact source format + version, e.g. ``"lerobot_v2.1"``,
            ``"rlds"``, ``"maniskill3"``. Pins parsing and surfaces version churn early.
        success: The episode-level :class:`SuccessLabel`, or ``None`` if the source carries
            no notion of success at all (distinct from a ``value=None`` "unknown" label).
        extra: Free-form source metadata (task string, language instruction, sim seed,
            ...). Carried losslessly.
    """

    source_dataset_id: str
    episode_index: int
    embodiment: EmbodimentSpec
    fingerprint: str
    num_steps: int
    source_format: str
    success: SuccessLabel | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class FeatureStore(Protocol):
    """Lazy backend that materializes feature arrays for one trajectory on demand.

    Adapters supply a concrete store (memmap over parquet, decode-on-demand for video
    frames, etc.). The core only relies on this minimal protocol, which is what keeps the
    representation streamable for datasets too large for RAM.
    """

    def has(self, key: str) -> bool:
        """Return whether ``key`` is available without materializing it."""
        ...

    def read(self, key: str) -> Array:
        """Materialize and return the ``(T, *shape)`` array for ``key``.

        For a low-dim feature this is the ``(T, *shape)`` array. For an IMAGE-role video feature
        under Stage-1 pass-through the store instead holds a :class:`VideoReference` (frame
        references, never decoded pixels); the typed return is still :data:`Array` so the
        low-dim signal path stays statically typed, and :meth:`Trajectory.video_references`
        recovers the references for image keys. Callers must not request an image key through the
        array path and treat the result as pixels.

        Raises:
            KeyError: If ``key`` is not present in this store.
        """
        ...


class InMemoryFeatureStore:
    """A :class:`FeatureStore` backed by an in-memory mapping of feature values.

    Values are low-dim arrays for ordinary features and :class:`VideoReference` objects for
    IMAGE-role video features under Stage-1 pass-through (pixels are never held in memory).
    Used for synthetic test fixtures and small datasets. Real adapters supply lazy,
    memory-mapped or decode-on-demand stores instead.
    """

    def __init__(self, columns: Mapping[str, Array | VideoReference]) -> None:
        # Copy into a plain dict so the store owns its mapping and callers cannot mutate it
        # underneath us; arrays themselves are not copied (cheap, and they are read-only by
        # convention on the source path).
        self._columns: dict[str, Array | VideoReference] = dict(columns)

    def has(self, key: str) -> bool:
        return key in self._columns

    def read(self, key: str) -> Array:
        # The stored value is an Array for low-dim features and a VideoReference for IMAGE keys
        # (Stage-1 pass-through). The typed return is Array to keep the low-dim signal path
        # statically typed; image keys are recovered via Trajectory.video_references().
        try:
            return cast("Array", self._columns[key])
        except KeyError as exc:
            raise KeyError(f"feature {key!r} not present in store") from exc


class Trajectory:
    """One variable-length episode in the canonical representation.

    A trajectory is columnar (one array per feature key), lazy (arrays materialize through
    a :class:`FeatureStore` only when requested), and embodiment-aware (it carries its
    :class:`EmbodimentSpec` via :attr:`meta`). It is the unit a :class:`Signal` scores.

    Convention: the leading axis of every materialized feature is time (``T``), and ``T``
    equals :attr:`TrajectoryMeta.num_steps` for every feature.
    """

    def __init__(self, meta: TrajectoryMeta, store: FeatureStore) -> None:
        self.meta = meta
        self._store = store

    # -- introspection ---------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        """``T``: the number of timesteps in this episode."""
        return self.meta.num_steps

    @property
    def embodiment(self) -> EmbodimentSpec:
        """The :class:`EmbodimentSpec` this trajectory conforms to."""
        return self.meta.embodiment

    def has(self, key: str) -> bool:
        """Return whether feature ``key`` is available on this trajectory."""
        return self._store.has(key)

    # -- generic feature access ------------------------------------------------------

    def feature(self, key: str) -> Array:
        """Materialize and return the ``(T, *shape)`` array for feature ``key``.

        This is the one uniform accessor signals use for state, proprio, actions, rewards —
        anything low-dim. For an IMAGE-role video feature under Stage-1 pass-through the store
        instead holds a :class:`VideoReference` (frame references, *not* decoded pixels). The
        typed return stays :data:`Array` so the low-dim signal path remains statically typed and
        no existing signal changes; image-role features are recovered as references through
        :meth:`video_references`, not through this array accessor. Units and per-dim names come
        from ``self.embodiment.feature(key)``.

        Raises:
            KeyError: If ``key`` is not present on this trajectory.
        """
        return self._store.read(key)

    def select_roles(self, *roles: FeatureRole) -> dict[str, Array]:
        """Materialize all available low-dim features whose role is in ``roles``.

        Returns a mapping from key to array. Keys declared in the embodiment but absent from the
        store are skipped (a trajectory need not carry every declared feature). For IMAGE-role
        features use :meth:`video_references` instead — those carry :class:`VideoReference`
        objects, not arrays, and are not returned through this array-typed accessor.
        """
        out: dict[str, Array] = {}
        for key in self.embodiment.keys_with_role(*roles):
            if self._store.has(key):
                out[key] = self._store.read(key)
        return out

    def video_references(self) -> dict[str, VideoReference]:
        """Return the :class:`VideoReference` for every IMAGE-role feature on this trajectory.

        Stage-1 only: the values are per-camera shard references, never pixels. Used by the v3
        writer to copy kept episodes' video shard files. Empty when the trajectory carries no
        video.

        References are resolved from two complementary sources, in order:

        * ``meta.extra["video_references"]`` — a ``{video_key: VideoReference}`` mapping the
          adapter attaches. The v3 reader uses this because v3 keeps video out of the parquet
          feature table (so video keys are intentionally *not* in :attr:`embodiment`).
        * any IMAGE-role feature whose store value is a :class:`VideoReference` — the in-store
          path for adapters that do carry image features in the embodiment.
        """
        out: dict[str, VideoReference] = {}
        extra_refs = self.meta.extra.get("video_references")
        if isinstance(extra_refs, Mapping):
            for key, ref in extra_refs.items():
                if isinstance(ref, VideoReference):
                    out[str(key)] = ref
        for key in self.embodiment.keys_with_role(FeatureRole.IMAGE):
            if key in out or not self._store.has(key):
                continue
            value: object = self._store.read(key)
            if isinstance(value, VideoReference):
                out[key] = value
        return out

    # -- typed convenience views (return None when absent; never fabricate) ----------

    def timestamps(self) -> Array | None:
        """Return the ``(T,)`` per-step timestamps in seconds, or ``None`` if unavailable.

        This is the authoritative control-rate source. Signals that depend on ``dt`` (e.g.
        jerk) must read real spacing from here and must not assume uniform sampling.
        """
        keys = self.embodiment.keys_with_role(FeatureRole.TIME)
        for key in keys:
            if self._store.has(key):
                ts = self._store.read(key)
                # Honor the documented (T,) contract: some formats store a scalar-per-step time
                # as shape (T, 1) (e.g. LeRobot v3 declares timestamp shape [1]); flatten it so
                # dt-dependent signals (jerk, ...) see 1-D time, not a (T, 1) array they'd skip.
                return ts.reshape(-1) if ts.ndim == 2 and ts.shape[1] == 1 else ts
        return None

    def actions(self) -> Array | None:
        """Return the action array, or ``None`` if this trajectory has no action feature."""
        return self._first_with_role(FeatureRole.ACTION)

    def rewards(self) -> Array | None:
        """Return the reward array, or ``None`` if this trajectory has no reward feature."""
        return self._first_with_role(FeatureRole.REWARD)

    def success(self) -> SuccessLabel | None:
        """Return the episode-level :class:`SuccessLabel`, or ``None`` if unlabelled.

        Note the distinction: ``None`` here means the source has no success concept at all,
        whereas ``SuccessLabel(value=None, ...)`` means success is known-to-be-unknown.
        """
        return self.meta.success

    def _first_with_role(self, role: FeatureRole) -> Array | None:
        for key in self.embodiment.keys_with_role(role):
            if self._store.has(key):
                return self._store.read(key)
        return None
