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
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# Canonical exchange array type. NumPy is the lingua franca of the core (see module
# docstring); ``np.generic`` is the common base of all NumPy scalar dtypes, so this covers
# uint8 images and float32 actions alike.
Array = npt.NDArray[np.generic]


def fingerprint_arrays(columns: Mapping[str, Array]) -> str:
    """Return a stable content hash over a set of feature arrays.

    The hash is order-independent in the input mapping (keys are sorted) and covers both
    the key names and the raw array bytes, so two trajectories with identical content
    produce the same fingerprint. Used for reproducibility, manifest linkage, and dedup, and
    shared by adapters and fixtures so a round-tripped trajectory keeps its fingerprint.
    """
    hasher = hashlib.sha256()
    for key in sorted(columns):
        hasher.update(key.encode("utf-8"))
        hasher.update(np.ascontiguousarray(columns[key]).tobytes())
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

        Raises:
            KeyError: If ``key`` is not present in this store.
        """
        ...


class InMemoryFeatureStore:
    """A :class:`FeatureStore` backed by an in-memory mapping of arrays.

    Used for synthetic test fixtures and small datasets. Real adapters supply lazy,
    memory-mapped or decode-on-demand stores instead.
    """

    def __init__(self, columns: Mapping[str, Array]) -> None:
        # Copy into a plain dict so the store owns its mapping and callers cannot mutate it
        # underneath us; arrays themselves are not copied (cheap, and they are read-only by
        # convention on the source path).
        self._columns: dict[str, Array] = dict(columns)

    def has(self, key: str) -> bool:
        return key in self._columns

    def read(self, key: str) -> Array:
        try:
            return self._columns[key]
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

        This is the one uniform accessor signals use for images, state, proprio, actions,
        rewards — anything. Units and per-dim names come from
        ``self.embodiment.feature(key)``.

        Raises:
            KeyError: If ``key`` is not present on this trajectory.
        """
        return self._store.read(key)

    def select_roles(self, *roles: FeatureRole) -> dict[str, Array]:
        """Materialize all available features whose role is in ``roles``.

        Returns a mapping from key to array. Keys declared in the embodiment but absent
        from the store are skipped (a trajectory need not carry every declared feature).
        """
        out: dict[str, Array] = {}
        for key in self.embodiment.keys_with_role(*roles):
            if self._store.has(key):
                out[key] = self._store.read(key)
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
