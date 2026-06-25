"""The ``Signal`` protocol — the one contract every quality signal implements.

A signal takes trajectories (plus a shared :class:`SignalContext`) and emits a
:class:`TrajectoryScore` per trajectory, optionally including per-transition scores. The
contract is designed so that:

* a **cheap CPU heuristic** (e.g. action jerk) and an **expensive GPU/influence signal**
  (e.g. CUPID-style attribution) both fit the same two methods;
* a signal that must **train or precompute** something (Demo-SCORE classifier, CUPID
  embeddings) uses the optional :meth:`Signal.fit` hook, while a stateless heuristic makes
  it a no-op;
* the **engine owns batching and scheduling**, so signals stay simple and cheap/expensive
  signals both run efficiently;
* a signal **declares its cost tier and requirements** so the engine can gate it and skip
  with a clear message rather than crash when a requirement (a GPU, sim state, an action
  feature) is unmet;
* the community **adds signals via entry points** (see :mod:`robocurate.signals.registry`)
  without ever touching the core (Invariant 4).

No real curation signal is implemented in this skeleton. The only implementation is a fake
in-memory signal used by the contract tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from robocurate.trajectory import Array, Trajectory

if TYPE_CHECKING:
    import logging

    from robocurate.metadata import DatasetMeta, ResourceProbe


class CostTier(IntEnum):
    """How expensive a signal is to run. Ordered so ``<`` means "cheaper".

    The engine uses the tier to schedule work and to decide what to warn about. Mirrors the
    blueprint taxonomy.
    """

    TIER0_CPU = 0  # cheap, CPU, laptop-friendly (jerk, action-noise, dedup heuristics)
    TIER1_GPU = 1  # mid, single GPU (learned quality classifier, embedding outliers)
    TIER2_GPU_HEAVY = 2  # expensive, single GPU + time (CUPID-style influence; gated)


# Capability tokens a signal requires to run. These are free-form strings rather than a
# rigid enum so new requirements ("sim_state", "scene_spec", a named feature key, "gpu",
# "encoder") can be added without a core change. The engine matches them against the
# available features and the ResourceProbe.
FeatureRequirement = frozenset[str]

# Well-known requirement tokens (signals may also require any feature key by name).
REQUIRES_GPU = "gpu"
REQUIRES_SIM_STATE = "sim_state"
REQUIRES_ENCODER = "encoder"


@dataclass(frozen=True)
class SignalSpec:
    """Static description a signal advertises about itself.

    Attributes:
        name: Unique signal name, e.g. ``"jerk"``, ``"cupid"``. Used in scores, the
            scorecard, the manifest, and cache keys.
        version: Signal version string. Bumping it invalidates cached artifacts and is
            recorded in the manifest for reproducibility.
        cost_tier: The :class:`CostTier`.
        requires: Capability tokens that must be satisfied for the signal to run (e.g.
            ``frozenset({"action"})`` or ``frozenset({REQUIRES_SIM_STATE})``). Unmet
            requirements lead to a recorded skip, never a crash.
        produces_per_transition: Whether :meth:`Signal.score` emits ``(T,)`` per-transition
            scores in addition to the trajectory-level value.
        deterministic: Whether the signal is deterministic given ``(batch, ctx, seed)``.
            Must be ``True`` to participate in the seeded selection path
            (Invariant 3); non-deterministic signals are allowed only in report-only scoring.
        description: One-line human description for the scorecard and ``--help`` output.
    """

    name: str
    version: str
    cost_tier: CostTier
    requires: FeatureRequirement = frozenset()
    produces_per_transition: bool = False
    deterministic: bool = True
    description: str = ""


@dataclass(frozen=True)
class TrajectoryScore:
    """One signal's score for one trajectory.

    Scores are emitted on each signal's own raw scale here; the curator normalizes across
    signals before combining. ``higher_is_better`` tells the curator the orientation so it
    never has to guess.

    Attributes:
        signal: The producing signal's name.
        trajectory_fingerprint: The scored trajectory's content fingerprint (stable link to
            the episode in the manifest and scorecard).
        value: The trajectory-level scalar score. ``NaN`` is permitted only together with
            ``skipped=True``.
        higher_is_better: Orientation of ``value`` (e.g. a redundancy score may be
            lower-is-better while a quality score is higher-is-better).
        per_transition: Optional ``(T,)`` per-step scores, or ``None``. Present only if the
            signal's spec sets ``produces_per_transition``.
        skipped: Whether the signal could not score this trajectory (unmet requirement,
            missing feature, ...). A skipped score never silently becomes a removal.
        skip_reason: Human-readable reason when ``skipped`` is ``True`` (e.g.
            ``"no action feature"``, ``"requires sim_state"``). ``None`` otherwise.
        diagnostics: Optional free-form per-trajectory diagnostics for the scorecard (e.g.
            the percentile a value landed in, intermediate quantities).
    """

    signal: str
    trajectory_fingerprint: str
    value: float
    higher_is_better: bool = True
    per_transition: Array | None = None
    skipped: bool = False
    skip_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def skip(
        cls, signal: str, fingerprint: str, reason: str, *, higher_is_better: bool = True
    ) -> TrajectoryScore:
        """Construct a recorded skip for one trajectory (never a silent drop)."""
        return cls(
            signal=signal,
            trajectory_fingerprint=fingerprint,
            value=float("nan"),
            higher_is_better=higher_is_better,
            skipped=True,
            skip_reason=reason,
        )


@runtime_checkable
class CacheHandle(Protocol):
    """A keyed artifact cache a signal uses for precomputed/trained state.

    Keys are namespaced by ``(signal name, signal version)`` by the engine, so a signal just
    uses logical keys (e.g. ``"classifier"``, a trajectory fingerprint). This is how
    :meth:`Signal.fit` hands precomputed state to :meth:`Signal.score` and how repeated runs
    avoid recomputation.
    """

    def has(self, key: str) -> bool: ...

    def get(self, key: str) -> Any: ...

    def put(self, key: str, value: Any) -> None: ...


class InMemoryCache:
    """A trivial in-process :class:`CacheHandle` used by the default engine and tests.

    Keys are used verbatim; the engine is responsible for namespacing by signal name and
    version before handing a view of this to a signal. Not persisted across runs.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def has(self, key: str) -> bool:
        return key in self._store

    def get(self, key: str) -> Any:
        return self._store[key]

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value


class NamespacedCache:
    """A :class:`CacheHandle` view that prefixes every key with a fixed namespace.

    The engine wraps a single shared backing cache in one of these per signal (keyed by the
    signal's name + version), so a signal uses logical keys while different signals — and
    different signal versions — never collide. This is how :meth:`Signal.fit` state reaches
    :meth:`Signal.score` for the *same* signal without leaking across signals.
    """

    def __init__(self, backing: CacheHandle, namespace: str) -> None:
        self._backing = backing
        self._namespace = namespace

    def _key(self, key: str) -> str:
        return f"{self._namespace}::{key}"

    def has(self, key: str) -> bool:
        return self._backing.has(self._key(key))

    def get(self, key: str) -> Any:
        return self._backing.get(self._key(key))

    def put(self, key: str, value: Any) -> None:
        self._backing.put(self._key(key), value)


@dataclass
class SignalContext:
    """Shared context handed to every signal call.

    Bundling these into one object (rather than a long parameter list) means new shared
    facilities can be added without changing the :class:`Signal` method signatures.

    Attributes:
        seed: The master seed for this run. A signal that needs randomness derives a child
            stream from this and must remain deterministic given it (invariant 3).
        device: Target device string, e.g. ``"cpu"`` or ``"cuda:0"``.
        cache: A :class:`CacheHandle` namespaced to the signal.
        resources: The :class:`~robocurate.metadata.ResourceProbe` for requirement gating.
        dataset_meta: Dataset-level metadata (see
            :class:`~robocurate.metadata.DatasetMeta`).
        logger: A standard logger for progress / skip messages.
    """

    seed: int
    device: str
    cache: CacheHandle
    resources: ResourceProbe
    dataset_meta: DatasetMeta
    logger: logging.Logger


@runtime_checkable
class Signal(Protocol):
    """The contract every quality signal implements.

    Two methods:

    * :meth:`fit` — optional one-shot training/precompute over the whole dataset. Stateless
      heuristics implement it as a no-op. This is the seam that lets a trained signal
      (Demo-SCORE) or an influence signal (CUPID) fit the same contract as a stateless one.
    * :meth:`score` — scores a batch of trajectories. Must be deterministic given
      ``(batch, ctx, seed)`` when ``spec.deterministic`` is ``True``. Returns exactly one
      :class:`TrajectoryScore` per input trajectory, in the same order.

    The engine — not the signal — owns batching, parallelism, requirement gating, and
    caching, so cheap CPU and expensive GPU signals are both scheduled efficiently while
    each signal stays small.
    """

    spec: SignalSpec

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        """Optionally train or precompute over the full dataset before scoring.

        Implementations that need no fitting should simply ``return`` (no-op). The engine
        calls this at most once per run, before any :meth:`score` call. Precomputed state is
        stashed in ``ctx.cache`` for :meth:`score` to read.
        """
        ...

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        """Score a batch of trajectories.

        Returns one :class:`TrajectoryScore` per input, in input order. A trajectory the
        signal cannot score (missing feature, unmet requirement) yields a
        :meth:`TrajectoryScore.skip`, never an exception and never a silent removal.
        """
        ...
