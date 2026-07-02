"""The Curator / selector — turns signal scores into a selection decision.

The curator orchestrates a run: it builds a :class:`~robocurate.signals.base.SignalContext`,
calls each signal's optional ``fit`` once, scores all trajectories in batches into a
:class:`ScoreMatrix`, combines the per-signal scores into one keep-score per trajectory, and
selects a subset under a target budget. Alongside every selection it produces an **equal-N
random baseline** of the same size (Invariant 5), so the dataset-size-confound
comparison is always one field away.

Determinism (invariant 3) is structural: a single master ``seed`` spawns independent named
RNG streams via :class:`numpy.random.SeedSequence`, and ties break by trajectory fingerprint,
so identical (dataset, config, seed) yields byte-identical decisions. No unseeded RNG is used
anywhere in the selection path.

This module implements the selection *engine* and combiners. It does **not** implement any
quality signal.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import islice
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from robocurate.manifest import (
    MANIFEST_SCHEMA_VERSION,
    BaselineRecord,
    EpisodeDecision,
    Manifest,
    code_version,
)
from robocurate.metadata import DatasetMeta, ResourceProbe
from robocurate.signals.base import (
    REQUIRES_GPU,
    CacheHandle,
    InMemoryCache,
    NamespacedCache,
    Signal,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)

if TYPE_CHECKING:
    import logging
    from pathlib import Path

    from robocurate.adapters.base import DatasetReader, DatasetWriter, WriteReceipt
    from robocurate.scorecard import Scorecard
    from robocurate.trajectory import Array, Trajectory

DEFAULT_BATCH_SIZE = 64

# Per-trajectory float vectors (keep-scores, normalized signal values) are 1-D float64.
FloatArray = npt.NDArray[np.float64]


# --------------------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------------------


class BudgetKind(Enum):
    FRACTION = "fraction"
    COUNT = "count"


@dataclass(frozen=True)
class Budget:
    """How many episodes to keep: a fraction of the dataset or an absolute count."""

    kind: BudgetKind
    value: float

    @classmethod
    def fraction(cls, frac: float) -> Budget:
        """Keep ``frac`` of the episodes (0 < frac <= 1)."""
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"fraction budget must be in (0, 1], got {frac}")
        return cls(BudgetKind.FRACTION, float(frac))

    @classmethod
    def count(cls, n: int) -> Budget:
        """Keep exactly ``n`` episodes."""
        if n < 0:
            raise ValueError(f"count budget must be >= 0, got {n}")
        return cls(BudgetKind.COUNT, float(n))

    def resolve(self, total: int) -> int:
        """Resolve to a concrete keep-count given a dataset of ``total`` episodes."""
        if self.kind is BudgetKind.FRACTION:
            return max(0, min(total, round(self.value * total)))
        return max(0, min(total, int(self.value)))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "value": self.value}


# --------------------------------------------------------------------------------------
# Selection mode and validity gate
# --------------------------------------------------------------------------------------


class SelectionMode(Enum):
    """How the curator turns keep-scores into a selection.

    ``TOP_K`` keeps the highest-scoring trajectories under the budget. ``GREEDY_DEDUP`` keeps
    one representative per near-duplicate cluster (the highest-scoring member), which top-K
    cannot guarantee. ``COVERAGE`` greedily maximizes a submodular facility-location objective
    over the embedding distribution, keeping a representative, diverse subset that covers the
    whole distribution (so rare-but-valid modes survive instead of being crowded out by a
    high-scoring majority cluster).
    """

    TOP_K = "top_k"
    GREEDY_DEDUP = "greedy_dedup"
    COVERAGE = "coverage"


@dataclass(frozen=True)
class GateConfig:
    """A hard pre-filter that removes trajectories by thresholding one signal's value.

    Gated-out trajectories are removed unconditionally — *before* the budget applies and
    excluded from the equal-N random baseline pool (Invariant 5: both arms are then
    a fair comparison on the valid data). This is how a physically-invalid sim trajectory
    (``sim_physics_validity`` value > 0) is never kept regardless of budget, without any
    change to the frozen Signal contract.

    Attributes:
        signal: Name of the signal whose value drives the gate. Must be one of the
            curator's signals.
        reject_above: Reject when ``value > reject_above`` (e.g. ``0.0`` for sim-validity).
        reject_below: Reject when ``value < reject_below``.
    """

    signal: str
    reject_above: float | None = None
    reject_below: float | None = None

    def rejects(self, value: float) -> bool:
        """Whether ``value`` trips the gate. A skipped (NaN) score never gates."""
        if math.isnan(value):
            return False
        if self.reject_above is not None and value > self.reject_above:
            return True
        return self.reject_below is not None and value < self.reject_below

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "reject_above": self.reject_above,
            "reject_below": self.reject_below,
        }


# An embedding maps a trajectory to a fixed-length vector for dedup distance comparison.
EmbeddingFn = Callable[["Trajectory"], "Array | None"]

DEFAULT_DEDUP_EPSILON = 0.5


# --------------------------------------------------------------------------------------
# Score matrix
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryRef:
    """Lightweight reference to a scored trajectory (no heavy arrays retained)."""

    episode_index: int
    fingerprint: str
    num_steps: int


@dataclass
class ScoreMatrix:
    """All signals' per-trajectory scores for one run, in a fixed trajectory order.

    Holds the raw :class:`TrajectoryScore` objects (with orientation + skip info) so the
    curator can normalize and combine, and the scorecard can report distributions and
    per-trajectory reasons.
    """

    refs: tuple[TrajectoryRef, ...]
    signal_specs: tuple[SignalSpec, ...]
    scores: Mapping[tuple[str, str], TrajectoryScore]  # (signal_name, fingerprint) -> score

    @property
    def num_trajectories(self) -> int:
        return len(self.refs)

    def signal_values(self, signal_name: str) -> FloatArray:
        """Return the raw per-trajectory values for ``signal_name`` (NaN where skipped)."""
        out = np.full(len(self.refs), np.nan, dtype=np.float64)
        for i, ref in enumerate(self.refs):
            score = self.scores.get((signal_name, ref.fingerprint))
            if score is not None and not score.skipped:
                out[i] = score.value
        return out

    def to_numpy(self) -> FloatArray:
        """Return a ``(n_trajectories, n_signals)`` array of raw values (NaN where skipped)."""
        cols = [self.signal_values(spec.name) for spec in self.signal_specs]
        if not cols:
            return np.empty((len(self.refs), 0), dtype=np.float64)
        return np.column_stack(cols)

    def normalized_signal_scores(self, signal_name: str) -> FloatArray:
        """Per-trajectory keep-oriented normalized scores in [0, 1] for ``signal_name``.

        The exact normalization the :class:`WeightedSum` combiner applies: min-max scaled over
        this matrix, oriented so 1.0 == most keepable (respecting the signal's reported
        ``higher_is_better``), with skipped (NaN) entries imputed to the neutral 0.5. Exposed
        so report surfaces (e.g. ``robocurate rank``) attribute a combined keep-score to the
        signals responsible without reimplementing combiner logic.
        """
        return _normalize_keep_oriented(
            self.signal_values(signal_name), _orientation(self, signal_name)
        )


def _normalize_keep_oriented(values: FloatArray, higher_is_better: bool) -> FloatArray:
    """Min-max normalize to [0, 1] oriented so 1.0 == most keepable; NaN-safe.

    Skipped entries (NaN) are imputed to the neutral midpoint 0.5 so a signal that could not
    score a trajectory neither rewards nor penalizes it.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, 0.5)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:  # constant signal carries no selection information
        norm = np.full_like(values, 0.5)
    else:
        norm = (values - lo) / (hi - lo)
        if not higher_is_better:
            norm = 1.0 - norm
    norm[~np.isfinite(values)] = 0.5
    return norm


# --------------------------------------------------------------------------------------
# Combiners
# --------------------------------------------------------------------------------------


@runtime_checkable
class Combiner(Protocol):
    """Combines a :class:`ScoreMatrix` into one keep-score per trajectory (higher = keep).

    Combiners are pure and deterministic. The curator applies the budget on top of the
    keep-score, so a combiner only has to express *relative desirability*.
    """

    @property
    def name(self) -> str:
        """A short combiner name, recorded in the config snapshot and manifest."""
        ...

    def combined_score(self, matrix: ScoreMatrix) -> FloatArray: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class WeightedSum:
    """Weighted average of per-signal keep-oriented normalized scores.

    Each signal is normalized to [0, 1] with 1.0 == most keepable (respecting the score's
    ``higher_is_better`` orientation), then combined as a weighted average. Signals absent
    from ``weights`` default to weight 1.0.
    """

    weights: Mapping[str, float] = field(default_factory=dict)
    name: str = "weighted_sum"

    def combined_score(self, matrix: ScoreMatrix) -> FloatArray:
        n = matrix.num_trajectories
        if not matrix.signal_specs:
            return np.zeros(n, dtype=np.float64)
        acc = np.zeros(n, dtype=np.float64)
        wsum = 0.0
        for spec in matrix.signal_specs:
            w = float(self.weights.get(spec.name, 1.0))
            if w == 0.0:
                continue
            raw = matrix.signal_values(spec.name)
            higher_is_better = _orientation(matrix, spec.name)
            acc += w * _normalize_keep_oriented(raw, higher_is_better)
            wsum += w
        return acc / wsum if wsum > 0 else np.full(n, 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": dict(self.weights)}


def _orientation(matrix: ScoreMatrix, signal_name: str) -> bool:
    """Return the ``higher_is_better`` orientation a signal reported (default True)."""
    for ref in matrix.refs:
        score = matrix.scores.get((signal_name, ref.fingerprint))
        if score is not None:
            return score.higher_is_better
    return True


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CurationConfig:
    """Fully-resolved configuration for a curation run (serialized into the manifest)."""

    combiner_dict: Mapping[str, Any]
    budget: Budget | None
    seed: int = 0
    emit_baseline: bool = True
    selection: str = SelectionMode.TOP_K.value
    gate_dict: Mapping[str, Any] | None = None
    batch_size: int = DEFAULT_BATCH_SIZE

    def to_dict(self) -> dict[str, Any]:
        return {
            "combiner": dict(self.combiner_dict),
            "budget": self.budget.to_dict() if self.budget else None,
            "seed": self.seed,
            "emit_baseline": self.emit_baseline,
            "selection": self.selection,
            "gate": dict(self.gate_dict) if self.gate_dict else None,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CurationConfig:
        """Reconstruct a :class:`CurationConfig` from its :meth:`to_dict` form.

        Inverse of :meth:`to_dict`. The nested combiner/gate dicts are kept as plain mappings
        (they are reconstructed into live objects by the recipe loader, not here), so this
        stays a pure data round-trip with no engine imports.
        """
        budget_dict = data.get("budget")
        budget = (
            Budget(kind=BudgetKind(budget_dict["kind"]), value=float(budget_dict["value"]))
            if budget_dict is not None
            else None
        )
        gate_dict = data.get("gate")
        return cls(
            combiner_dict=dict(data.get("combiner", {})),
            budget=budget,
            seed=int(data.get("seed", 0)),
            emit_baseline=bool(data.get("emit_baseline", True)),
            selection=str(data.get("selection", SelectionMode.TOP_K.value)),
            gate_dict=dict(gate_dict) if gate_dict is not None else None,
            batch_size=int(data.get("batch_size", DEFAULT_BATCH_SIZE)),
        )


# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------


@dataclass
class CurationResult:
    """The outcome of a curation run: the selection, its justification, and the baseline."""

    kept_episode_indices: tuple[int, ...]
    removed_episode_indices: tuple[int, ...]
    decisions: tuple[EpisodeDecision, ...]
    score_matrix: ScoreMatrix
    baseline: BaselineRecord | None
    config: CurationConfig
    signal_specs: tuple[SignalSpec, ...]
    #: The combined per-trajectory keep-score the selection actually used (higher = keep),
    #: aligned with ``score_matrix.refs``. Exposed so ranking/report surfaces reuse the run's
    #: combiner output instead of recombining.
    keep_scores: tuple[float, ...] = ()
    _reader: DatasetReader | None = None

    @property
    def num_kept(self) -> int:
        return len(self.kept_episode_indices)

    @property
    def num_removed(self) -> int:
        return len(self.removed_episode_indices)

    def build_manifest(self, *, created_utc: str | None = None) -> Manifest:
        """Construct the :class:`Manifest` describing this run (no I/O)."""
        reader = self._require_reader()
        source_fp = reader.fingerprint()
        return Manifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            source=source_fp,
            output=source_fp,  # replaced with the true output fingerprint on save()
            config_dict=self.config.to_dict(),
            seed=self.config.seed,
            code_version=code_version(),
            signals=self.signal_specs,
            decisions=self.decisions,
            baseline=self.baseline,
            created_utc=created_utc,
            parent_manifest_path=_parent_manifest_path(reader),
        )

    def scorecard(self) -> Scorecard:
        """Build the human/machine-readable :class:`~robocurate.scorecard.Scorecard`."""
        from robocurate.scorecard import build_scorecard

        return build_scorecard(self)

    def save(
        self,
        dest: str | Path,
        *,
        created_utc: str | None = None,
        write_card: bool = True,
        push_to_hub: str | None = None,
    ) -> WriteReceipt:
        """Write the curated subset (kept episodes) plus the manifest to ``dest``.

        Re-reads the kept episodes from the source reader (streaming; source untouched) and
        writes a new dataset via the writer, which validates schema + checksum + round-trip.
        When ``write_card`` is set (the default), a ``README.md`` Hugging Face dataset card
        summarizing the curation is written into ``dest`` alongside the manifest. When
        ``push_to_hub`` is a repo id, the *written and validated* output directory is uploaded
        to the Hugging Face Hub **after** the local write succeeds (reading only from ``dest``,
        never the source). The source dataset is never touched (Invariant 1).
        """
        reader = self._require_reader()
        source_root = getattr(reader, "root", None)
        writer = _writer_for_source(reader, dest, source_root)
        manifest = self.build_manifest(created_utc=created_utc)
        kept = (reader.read_episode(i) for i in self.kept_episode_indices)
        receipt = writer.write(kept, manifest)
        if write_card:
            from pathlib import Path as _Path

            card = self.scorecard().to_hf_dataset_card()
            (_Path(receipt.path) / "README.md").write_text(card, encoding="utf-8")
        if push_to_hub is not None:
            # Push only the validated local output (Invariant 1: never reads the source).
            from robocurate.hub import maybe_push_to_hub

            maybe_push_to_hub(receipt.path, push_to_hub)
        return receipt

    def _require_reader(self) -> DatasetReader:
        if self._reader is None:
            raise RuntimeError(
                "this CurationResult has no attached reader; it cannot be saved or "
                "fingerprinted (was it constructed outside Curator.run?)"
            )
        return self._reader


def _writer_for_source(
    reader: DatasetReader, dest: str | Path, source_root: str | Path | None
) -> DatasetWriter:
    """Pick the output writer matching the source's on-disk LeRobot version.

    A v3.0 source writes back v3.0 (including the Stage-1 video-shard pass-through) so
    curation never silently downgrades the format; everything else (v2.1, and non-LeRobot
    readers such as RLDS/HDF5/Zarr) emits the v2.1 layout as before. The ``Dataset`` facade
    is unwrapped so the dispatch sees the concrete reader's declared ``version``.
    """
    from robocurate.adapters.base import LeRobotVersion
    from robocurate.adapters.lerobot import LeRobotWriter

    concrete = getattr(reader, "reader", reader)
    if getattr(concrete, "version", None) is LeRobotVersion.V3:
        from robocurate.adapters.lerobot_v3_writer import LeRobotWriterV3

        return LeRobotWriterV3(dest, source_root=source_root)
    return LeRobotWriter(dest, source_root=source_root)


# --------------------------------------------------------------------------------------
# Curator engine
# --------------------------------------------------------------------------------------


class Curator:
    """Runs signals over a dataset and selects a curated subset under a budget."""

    def __init__(
        self,
        signals: Sequence[Signal],
        *,
        combiner: Combiner | None = None,
        budget: Budget | None = None,
        seed: int = 0,
        emit_baseline: bool = True,
        selection: SelectionMode = SelectionMode.TOP_K,
        gate: GateConfig | None = None,
        dedup_epsilon: float = DEFAULT_DEDUP_EPSILON,
        dedup_embedding: EmbeddingFn | None = None,
        coverage_quality_weight: float = 0.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        resources: ResourceProbe | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.signals = list(signals)
        if combiner is None:
            combiner = WeightedSum()
        self.combiner = combiner
        self.budget = budget
        self.seed = seed
        self.emit_baseline = emit_baseline
        self.selection = selection
        self.gate = gate
        self.dedup_epsilon = dedup_epsilon
        self.dedup_embedding = dedup_embedding
        self.coverage_quality_weight = coverage_quality_weight
        self.batch_size = batch_size
        self.resources = resources if resources is not None else ResourceProbe()
        self._logger = logger
        if gate is not None and gate.signal not in {s.spec.name for s in self.signals}:
            raise ValueError(
                f"gate references signal {gate.signal!r}, which is not among the curator's "
                f"signals: {sorted(s.spec.name for s in self.signals)}"
            )

    @classmethod
    def from_config(
        cls, signals: Sequence[Signal], config: CurationConfig, **kwargs: Any
    ) -> Curator:
        """Construct a curator from a :class:`CurationConfig` (combiner passed separately)."""
        return cls(
            signals,
            budget=config.budget,
            seed=config.seed,
            emit_baseline=config.emit_baseline,
            batch_size=config.batch_size,
            **kwargs,
        )

    # -- run -------------------------------------------------------------------------

    def run(self, reader: DatasetReader) -> CurationResult:
        """Score every episode, combine, select under the budget, and emit the baseline."""
        dataset_meta = reader.meta
        # One backing cache for the whole run; each signal gets a namespaced view so a
        # signal's fit() state reaches its score() without colliding with other signals.
        backing: CacheHandle = InMemoryCache()
        active = self._gate_signals()
        contexts = {sig.spec.name: self._context_for(sig, dataset_meta, backing) for sig in active}
        self._fit(active, reader, contexts)
        matrix = self._score(active, reader, contexts)
        return self._select(matrix, active, reader)

    # -- internals -------------------------------------------------------------------

    def _logger_or_default(self) -> logging.Logger:
        import logging

        return self._logger or logging.getLogger("robocurate.curator")

    def _context_for(
        self, signal: Signal, dataset_meta: DatasetMeta, backing: CacheHandle
    ) -> SignalContext:
        namespace = f"{signal.spec.name}@{signal.spec.version}"
        device = "cuda:0" if self.resources.has_gpu else "cpu"
        return SignalContext(
            seed=self.seed,
            device=device,
            cache=NamespacedCache(backing, namespace),
            resources=self.resources,
            dataset_meta=dataset_meta,
            logger=self._logger_or_default(),
        )

    def _gate_signals(self) -> list[Signal]:
        """Drop signals whose resource requirements are unmet, with a clear log message."""
        logger = self._logger_or_default()
        active: list[Signal] = []
        for sig in self.signals:
            if REQUIRES_GPU in sig.spec.requires and not self.resources.has_gpu:
                logger.warning(
                    "skipping signal %s: requires a GPU but none is available",
                    sig.spec.name,
                )
                continue
            if not sig.spec.deterministic:
                raise ValueError(
                    f"signal {sig.spec.name!r} is non-deterministic and cannot run in the "
                    "selection path (invariant 3); use report-only scoring instead"
                )
            active.append(sig)
        return active

    def _fit(
        self,
        signals: list[Signal],
        reader: DatasetReader,
        contexts: dict[str, SignalContext],
    ) -> None:
        # Each signal gets a fresh iterator over the reader; the reader streams, so this does
        # not hold the whole dataset in RAM.
        for sig in signals:
            sig.fit(iter(reader), contexts[sig.spec.name])

    def _score(
        self,
        signals: list[Signal],
        reader: DatasetReader,
        contexts: dict[str, SignalContext],
    ) -> ScoreMatrix:
        refs: list[TrajectoryRef] = []
        scores: dict[tuple[str, str], TrajectoryScore] = {}
        for batch in _batched(iter(reader), self.batch_size):
            for traj in batch:
                refs.append(
                    TrajectoryRef(
                        episode_index=traj.meta.episode_index,
                        fingerprint=traj.meta.fingerprint,
                        num_steps=traj.num_steps,
                    )
                )
            for sig in signals:
                for score in sig.score(batch, contexts[sig.spec.name]):
                    scores[(sig.spec.name, score.trajectory_fingerprint)] = score
        self._warn_all_skipped(signals, scores, len(refs))
        return ScoreMatrix(
            refs=tuple(refs),
            signal_specs=tuple(s.spec for s in signals),
            scores=scores,
        )

    def _warn_all_skipped(
        self,
        signals: list[Signal],
        scores: dict[tuple[str, str], TrajectoryScore],
        num_episodes: int,
    ) -> None:
        """Warn when a signal scored nothing at all — it contributes only neutral imputes.

        Skips are recorded per-episode (never silent), but a signal that skips *every*
        episode — e.g. an image signal on a dataset with no decodable video, or a sim-only
        signal on real data — deserves one loud, actionable message rather than N quiet ones.
        """
        logger = self._logger_or_default()
        for sig in signals:
            own = [s for (name, _), s in scores.items() if name == sig.spec.name]
            if own and all(s.skipped for s in own):
                first_reason = next((s.skip_reason for s in own if s.skip_reason), "unknown")
                logger.warning(
                    "signal %s skipped all %d episodes (first reason: %s); it contributes "
                    "nothing to this selection",
                    sig.spec.name,
                    num_episodes,
                    first_reason,
                )

    def _select(
        self, matrix: ScoreMatrix, signals: list[Signal], reader: DatasetReader
    ) -> CurationResult:
        n = matrix.num_trajectories
        keep_score = self.combiner.combined_score(matrix)

        # 1. Hard validity gate: rejected trajectories are removed before the budget and
        #    excluded from the valid pool (and so from the equal-N baseline pool).
        gated = self._gated_positions(matrix)
        valid = [i for i in range(n) if i not in gated]

        # 2. Budget is of the VALID pool.
        budget = self.budget if self.budget is not None else Budget.fraction(1.0)
        k = budget.resolve(len(valid))

        # 3. Select within the valid pool by the chosen mode.
        skipped: dict[int, str]
        if self.selection is SelectionMode.GREEDY_DEDUP:
            selected, skipped = self._greedy_dedup(matrix, reader, valid, keep_score, k)
        elif self.selection is SelectionMode.COVERAGE:
            selected, skipped = self._coverage_selection(matrix, reader, valid, keep_score, k)
        else:
            order = sorted(valid, key=lambda i: (-keep_score[i], matrix.refs[i].fingerprint))
            selected = set(order[:k])
            skipped = {}

        kept_idx: list[int] = []
        removed_idx: list[int] = []
        decisions: list[EpisodeDecision] = []
        for i, ref in enumerate(matrix.refs):
            kept = i in selected
            (kept_idx if kept else removed_idx).append(ref.episode_index)
            decisions.append(
                EpisodeDecision(
                    episode_index=ref.episode_index,
                    fingerprint=ref.fingerprint,
                    kept=kept,
                    reason=self._decision_reason(i, kept, gated, skipped, keep_score, k),
                    signal_values=self._signal_values_for(matrix, ref.fingerprint),
                )
            )

        baseline = self._equal_n_baseline(matrix, k, valid) if self.emit_baseline else None
        return CurationResult(
            kept_episode_indices=tuple(kept_idx),
            removed_episode_indices=tuple(removed_idx),
            decisions=tuple(decisions),
            score_matrix=matrix,
            baseline=baseline,
            config=self._config_snapshot(),
            signal_specs=tuple(s.spec for s in signals),
            keep_scores=tuple(float(v) for v in keep_score),
            _reader=reader,
        )

    def _equal_n_baseline(self, matrix: ScoreMatrix, k: int, valid: list[int]) -> BaselineRecord:
        """Draw a same-size (N=k) random subset of the *valid* pool with a seeded RNG.

        Drawing from the gated (valid) pool keeps the curated-vs-random comparison fair: both
        arms exclude invalid data, so the contrast isolates the selection method (invariant 5).
        """
        baseline_seed = int(
            np.random.SeedSequence([self.seed, _BASELINE_STREAM]).generate_state(1)[0]
        )
        rng = np.random.default_rng(baseline_seed)
        pool = np.asarray(valid, dtype=np.int64)
        chosen = rng.choice(pool, size=k, replace=False) if 0 < k <= pool.size else pool[:k]
        kept = sorted(matrix.refs[int(i)].episode_index for i in chosen.tolist())
        return BaselineRecord(
            method="equal_n_random",
            seed=baseline_seed,
            n=k,
            kept_episode_indices=tuple(kept),
        )

    def _gated_positions(self, matrix: ScoreMatrix) -> set[int]:
        """Positions rejected by the hard validity gate (empty when no gate is configured)."""
        if self.gate is None:
            return set()
        gated: set[int] = set()
        for i, ref in enumerate(matrix.refs):
            score = matrix.scores.get((self.gate.signal, ref.fingerprint))
            if score is not None and not score.skipped and self.gate.rejects(score.value):
                gated.add(i)
        return gated

    def _greedy_dedup(
        self,
        matrix: ScoreMatrix,
        reader: DatasetReader,
        valid: list[int],
        keep_score: FloatArray,
        k: int,
    ) -> tuple[set[int], dict[int, str]]:
        """Keep one representative per near-duplicate cluster: highest keep-score first.

        Returns ``(selected positions, {skipped position -> kept neighbour fingerprint})``.
        """
        embeddings = self._dedup_embeddings(matrix, reader, valid)
        order = sorted(valid, key=lambda i: (-keep_score[i], matrix.refs[i].fingerprint))
        selected: set[int] = set()
        skipped: dict[int, str] = {}
        kept_vecs: list[tuple[int, FloatArray]] = []
        for i in order:
            if len(selected) >= k:
                break  # budget reached; the rest are budget-removed, not dedup-removed
            vec = embeddings.get(i)
            near = self._nearest_within(vec, kept_vecs) if vec is not None else None
            if near is not None:
                skipped[i] = matrix.refs[near].fingerprint
            else:
                selected.add(i)
                if vec is not None:
                    kept_vecs.append((i, vec))
        return selected, skipped

    def _nearest_within(self, vec: FloatArray, kept: list[tuple[int, FloatArray]]) -> int | None:
        for pos, kv in kept:
            if float(np.linalg.norm(vec - kv)) <= self.dedup_epsilon:
                return pos
        return None

    def _coverage_selection(
        self,
        matrix: ScoreMatrix,
        reader: DatasetReader,
        valid: list[int],
        keep_score: FloatArray,
        k: int,
    ) -> tuple[set[int], dict[int, str]]:
        """Greedy submodular facility-location: keep a representative, diverse subset.

        Picks trajectories that best *cover* the embedding distribution. Coverage of an embedded
        position ``p`` by the selected set ``S`` is ``max(sim(p, s) for s in S)`` where
        ``sim(a, b) = -||z_a - z_b||`` (negative Euclidean on the z-standardized embedding).
        The facility-location value ``sum_p max_{s in S} sim(p, s)`` is monotone submodular, so a
        lazy-greedy "pick the candidate with the largest marginal gain" is the standard
        near-optimal selector. We add ``coverage_quality_weight * keep_score[c]`` so quality can
        tilt the otherwise pure-diversity objective; ties break deterministically by
        ``(-keep_score, fingerprint)``.

        Returns ``(selected positions, {skipped position -> nearest-kept fingerprint})``.

        Determinism: no RNG; all ordering is by keep_score+fingerprint. Float64 throughout.
        Perf: O(k * |embedded| * |embedded|) — a marginal-gain pass over every embedded point
        for every embedded candidate, repeated ``k`` times. Fine for typical curation sizes;
        approximate nearest-neighbour acceleration for very large N is intentionally out of scope.
        """
        embeddings = self._dedup_embeddings(matrix, reader, valid)
        order = sorted(valid, key=lambda i: (-keep_score[i], matrix.refs[i].fingerprint))
        selected: set[int] = set()

        # Edge case: nothing is embeddable -> fall back to pure TOP_K ordering (no diversity
        # information exists, so the best we can do is keep the highest-scoring trajectories).
        if not embeddings:
            selected = set(order[:k])
            skipped = self._coverage_skipped(matrix, embeddings, valid, selected, order)
            return selected, skipped

        embedded = [i for i in order if i in embeddings]
        # Pairwise similarity matrix over embedded points (negative Euclidean distance).
        zmat = np.vstack([embeddings[i] for i in embedded]).astype(np.float64)
        diff = zmat[:, None, :] - zmat[None, :, :]
        sim = -np.linalg.norm(diff, axis=2)  # (n_embedded, n_embedded), float64

        pos_of = {pos: idx for idx, pos in enumerate(embedded)}
        coverage = np.full(len(embedded), -np.inf, dtype=np.float64)

        candidates = list(embedded)  # already in deterministic (-keep_score, fingerprint) order
        while candidates and len(selected) < k:
            best_pos = -1
            best_gain = -np.inf
            for c in candidates:
                ci = pos_of[c]
                # Marginal coverage gain: sum over embedded p of max(0, sim(c, p) - coverage[p]).
                gain = float(np.maximum(0.0, sim[ci] - coverage).sum())
                gain += self.coverage_quality_weight * float(keep_score[c])
                # `candidates` is pre-sorted by (-keep_score, fingerprint), so the first
                # strict maximum wins the deterministic tie-break automatically.
                if gain > best_gain:
                    best_gain = gain
                    best_pos = c
            selected.add(best_pos)
            candidates.remove(best_pos)
            coverage = np.maximum(coverage, sim[pos_of[best_pos]])

        # Fill any leftover budget (fewer embedded points than k) deterministically by top
        # keep_score among not-yet-selected valid positions, so len(selected) == k exactly and
        # the equal-N baseline contract (same valid pool, same k) is preserved.
        if len(selected) < k:
            remaining = [i for i in order if i not in selected]
            for i in remaining[: k - len(selected)]:
                selected.add(i)

        skipped = self._coverage_skipped(matrix, embeddings, valid, selected, order)
        return selected, skipped

    def _coverage_skipped(
        self,
        matrix: ScoreMatrix,
        embeddings: dict[int, FloatArray],
        valid: list[int],
        selected: set[int],
        order: list[int],
    ) -> dict[int, str]:
        """Map each removed valid position to the fingerprint that "represents" it.

        For an embedded removed position, that is the nearest *selected* embedded position; for a
        non-embedded removal (or when nothing was embedded), it is the top-kept fingerprint. Cheap
        and deterministic so the scorecard can explain coverage removals.
        """
        top_kept = next((i for i in order if i in selected), None)
        top_fp = matrix.refs[top_kept].fingerprint if top_kept is not None else ""
        selected_embedded = [(i, embeddings[i]) for i in order if i in selected and i in embeddings]
        skipped: dict[int, str] = {}
        for i in valid:
            if i in selected:
                continue
            vec = embeddings.get(i)
            if vec is not None and selected_embedded:
                nearest = min(
                    selected_embedded, key=lambda pair: float(np.linalg.norm(vec - pair[1]))
                )
                skipped[i] = matrix.refs[nearest[0]].fingerprint
            else:
                skipped[i] = top_fp
        return skipped

    def _dedup_embeddings(
        self, matrix: ScoreMatrix, reader: DatasetReader, valid: list[int]
    ) -> dict[int, FloatArray]:
        """Z-standardized embeddings for the valid pool (positions with no embedding omitted)."""
        embed = self.dedup_embedding
        if embed is None:
            from robocurate.signals.redundancy import statistical_embedding

            embed = statistical_embedding
        raw: dict[int, FloatArray] = {}
        for i in valid:
            traj = reader.read_episode(matrix.refs[i].episode_index)
            vector = embed(traj)
            if vector is not None:
                raw[i] = np.asarray(vector, dtype=np.float64).reshape(-1)
        if not raw:
            return {}
        stacked = np.vstack(list(raw.values()))
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        std_safe = np.where(std > 0.0, std, 1.0)
        return {i: (v - mean) / std_safe for i, v in raw.items()}

    def _decision_reason(
        self,
        position: int,
        kept: bool,
        gated: set[int],
        skipped: dict[int, str],
        keep_score: FloatArray,
        k: int,
    ) -> str:
        if position in gated:
            return f"removed by gate: {self.gate.signal} value tripped the validity threshold"  # type: ignore[union-attr]
        if kept:
            return f"kept: keep-score {keep_score[position]:.4f} (budget {k})"
        if position in skipped:
            if self.selection is SelectionMode.COVERAGE:
                return "removed: represented by a kept trajectory (coverage selection)"
            return f"removed: near-duplicate of kept trajectory {skipped[position][:12]}"
        return f"removed: keep-score {keep_score[position]:.4f} below budget {k}"

    def _signal_values_for(self, matrix: ScoreMatrix, fingerprint: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for spec in matrix.signal_specs:
            score = matrix.scores.get((spec.name, fingerprint))
            out[spec.name] = float("nan") if score is None or score.skipped else score.value
        return out

    def _config_snapshot(self) -> CurationConfig:
        return CurationConfig(
            combiner_dict=self.combiner.to_dict(),
            budget=self.budget,
            seed=self.seed,
            emit_baseline=self.emit_baseline,
            selection=self.selection.value,
            gate_dict=self.gate.to_dict() if self.gate else None,
            batch_size=self.batch_size,
        )


def _parent_manifest_path(reader: DatasetReader) -> str | None:
    """Return the source's ``manifest.json`` path if the source is itself a curated dataset.

    Curating an already-curated dataset should record its lineage (Invariant 6: honest,
    auditable provenance). We detect this purely by reading: a directory-backed reader exposes
    a ``root`` and, if that root contains a ``manifest.json``, the source was emitted by a
    prior curation run. The source is only read, never written.
    """
    from pathlib import Path

    root = getattr(reader, "root", None)
    if root is None:
        return None
    candidate = Path(root) / "manifest.json"
    return str(candidate) if candidate.is_file() else None


# A fixed stream id mixed with the master seed so the baseline RNG is independent of, but
# reproducible from, the master seed.
_BASELINE_STREAM = 0xBA5E


def _batched(it: Iterable[Trajectory], size: int) -> Iterator[list[Trajectory]]:
    """Yield lists of up to ``size`` items from ``it`` (a streaming-friendly batcher)."""
    iterator = iter(it)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


__all__ = [
    "Budget",
    "BudgetKind",
    "Combiner",
    "CurationConfig",
    "CurationResult",
    "Curator",
    "GateConfig",
    "ScoreMatrix",
    "SelectionMode",
    "TrajectoryRef",
    "WeightedSum",
]
