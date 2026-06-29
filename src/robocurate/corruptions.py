"""Synthetic trajectory corruptions for known-answer validation of quality signals.

Operator-proficiency labels (e.g. robomimic Multi-Human tiers) are a *soft, contested* proxy
for "bad data" — an adversarial review found that recovering them need not imply good curation.
Injected corruptions are the opposite: **hard ground truth**. We take clean trajectories, damage
a known subset in a known way, run a signal, and measure detection (does it rank the corrupted
ones worst?). This is the data-valuation field's standard known-answer check, and — unlike a
label-AUC — it exposes *blind spots*: a truncated demo is structurally bad yet geometrically
clean, so a directness/smoothness signal will (correctly, honestly) miss it.

Each corruption is deterministic given its ``seed`` (invariant 3) and returns a NEW
:class:`~robocurate.trajectory.Trajectory` tagged with ``meta.extra["corruption"]``; the input is
never mutated. Corruptions that change length (truncate, stall) slice/extend every feature column
together so the trajectory stays internally consistent.

The kinds are chosen to span detectable-and-blind:
  * ``jitter`` — high-frequency per-step noise on the target feature (a *smoothness* defect:
    jerk / action-noise / SPARC should catch it; directness less so).
  * ``detour`` — a smooth out-and-back lateral excursion (a *directness* defect: path-efficiency
    should catch it; smoothness signals should not, since it adds no high-frequency content).
  * ``truncate`` — drop the last fraction of the episode (a *structural* defect that geometry
    signals are blind to — the kept portion is still smooth and direct).
  * ``stall`` — insert a held (repeated-frame) segment (a *duration/efficiency* defect; mostly a
    blind spot for geometry, caught by length).
  * ``frame_skip`` — subsample every k-th step (a *temporal* defect: the surviving frames are
    farther apart, so per-step deltas jump — jerk / action-noise should catch it; directness
    barely moves because the geometry is preserved).
  * ``action_quantize`` — round the target feature onto a coarse grid (a *fidelity* defect: the
    staircase adds high-frequency content jerk / action-noise see; directness is blind).
  * ``wrong_target_offset`` — add a constant offset to the target feature for the whole episode
    (a *mis-targeted-demo* defect: shape and smoothness are untouched, so the geometric signals
    are blind — only a dataset-relative outlier view, e.g. action_noise / redundancy, can catch
    a shifted distribution).
  * ``dropped_dof`` — zero out one DoF of the target feature for the whole episode (a *sensor/
    actuator-dropout* defect: it collapses the feature's spread, which the dataset-relative
    outlier and uniqueness views catch; pure shape-smoothness can be blind).

The ``detection_matrix`` helper turns this into an honest blind-spot table: for each
(corruption kind x signal) it injects the defect into a copy of a clean dataset and reports the
orientation-aware detection-AUC (P[corrupted ranked worse than clean]). AUC~1 = detects,
AUC~0.5 = blind, AUC~0 = inverts (e.g. path_efficiency on ``truncate``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from robocurate.trajectory import (
    Array,
    InMemoryFeatureStore,
    Trajectory,
    fingerprint_arrays,
)

if TYPE_CHECKING:
    from robocurate.signals.base import Signal

F64 = npt.NDArray[np.float64]

CORRUPTIONS = (
    "jitter",
    "detour",
    "truncate",
    "stall",
    "frame_skip",
    "action_quantize",
    "wrong_target_offset",
    "dropped_dof",
)


def corrupt(
    traj: Trajectory,
    kind: str,
    *,
    feature: str,
    severity: float = 1.0,
    seed: int = 0,
) -> Trajectory:
    """Return a corrupted copy of ``traj`` (input untouched), tagged ``meta.extra["corruption"]``.

    Args:
        traj: The clean trajectory to damage.
        kind: One of :data:`CORRUPTIONS`.
        feature: Feature key the per-feature corruptions (``jitter`` / ``detour`` /
            ``action_quantize`` / ``wrong_target_offset`` / ``dropped_dof``) act on; the
            length-changing ones (``truncate`` / ``stall`` / ``frame_skip``) act on all columns
            together. E.g. ``"action"`` or ``"observation.robot0_eef_pos"``.
        severity: Corruption strength (kind-specific; ``jitter`` noise scale, ``detour`` bump
            scale, ``truncate``/``stall`` the affected fraction of timesteps, ``frame_skip`` the
            subsample stride fraction, ``action_quantize`` the grid coarseness, ``wrong_target_
            offset`` the offset scale, ``dropped_dof`` ignores severity).
        seed: Deterministic RNG seed.
    """
    if kind not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {kind!r}; known: {CORRUPTIONS}")
    rng = np.random.default_rng(seed)
    columns = {spec.key: np.asarray(traj.feature(spec.key)) for spec in traj.embodiment.features}

    # Capture the original control dt so a length-changing corruption can rebuild a strictly-
    # increasing timestamp (a stall holds position while time keeps advancing — a real pause, not
    # duplicate timestamps, which would just make derivative-based signals skip).
    ts = columns.get("timestamp")
    orig_dt = (
        float(np.median(np.diff(np.asarray(ts, dtype=np.float64).reshape(-1))))
        if ts is not None and np.asarray(ts).shape[0] > 1
        else None
    )

    if kind == "jitter":
        columns[feature] = _jitter(columns[feature], severity, rng)
    elif kind == "detour":
        columns[feature] = _detour(columns[feature], severity, rng)
    elif kind == "action_quantize":
        columns[feature] = _action_quantize(columns[feature], severity)
    elif kind == "wrong_target_offset":
        columns[feature] = _wrong_target_offset(columns[feature], severity, rng)
    elif kind == "dropped_dof":
        columns[feature] = _dropped_dof(columns[feature], rng)
    elif kind == "truncate":
        columns = _truncate(columns, severity)
    elif kind == "frame_skip":
        columns = _frame_skip(columns, severity)
    else:  # stall
        columns = _stall(columns, severity, rng)

    num_steps = int(next(iter(columns.values())).shape[0])
    if "timestamp" in columns and orig_dt is not None:
        columns["timestamp"] = (np.arange(num_steps) * orig_dt).astype(
            np.asarray(columns["timestamp"]).dtype
        )
    meta = replace(
        traj.meta,
        num_steps=num_steps,
        fingerprint=fingerprint_arrays(columns),
        extra={**dict(traj.meta.extra), "corruption": kind},
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _as2d(x: Array) -> F64:
    arr: F64 = np.asarray(x, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1)


def _jitter(x: Array, severity: float, rng: np.random.Generator) -> Array:
    """Add per-step (high-frequency) Gaussian noise scaled to each column's spread."""
    flat = _as2d(x)
    scale = flat.std(axis=0)
    scale[scale == 0.0] = 1.0
    noised = flat + severity * 0.5 * scale * rng.standard_normal(flat.shape)
    out: Array = noised.reshape(np.asarray(x).shape).astype(np.asarray(x).dtype)
    return out


def _detour(x: Array, severity: float, rng: np.random.Generator) -> Array:
    """Add a smooth out-and-back lateral excursion: net displacement unchanged, path longer."""
    flat = _as2d(x).copy()
    num_steps, dim = flat.shape
    direction = rng.standard_normal(dim)
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 0 else np.ones(dim) / np.sqrt(dim)
    span = float(np.linalg.norm(flat[-1] - flat[0])) or float(np.linalg.norm(flat).mean()) or 1.0
    bump = np.sin(np.linspace(0.0, np.pi, num_steps))  # 0 at both ends -> net unchanged
    flat = flat + severity * 0.5 * span * np.outer(bump, direction)
    out: Array = flat.reshape(np.asarray(x).shape).astype(np.asarray(x).dtype)
    return out


def _truncate(columns: dict[str, Array], severity: float) -> dict[str, Array]:
    """Drop the last ``severity`` fraction of timesteps from every column (incomplete demo)."""
    num_steps = int(next(iter(columns.values())).shape[0])
    keep = max(2, round(num_steps * (1.0 - min(max(severity, 0.0), 0.9))))
    return {k: np.asarray(v)[:keep] for k, v in columns.items()}


def _stall(
    columns: dict[str, Array], severity: float, rng: np.random.Generator
) -> dict[str, Array]:
    """Insert a held (repeated-frame) segment mid-trajectory in every column (a hesitation)."""
    num_steps = int(next(iter(columns.values())).shape[0])
    hold = max(1, round(num_steps * min(max(severity, 0.0), 0.9)))
    at = int(rng.integers(1, num_steps)) if num_steps > 1 else 0
    out: dict[str, Array] = {}
    for k, v in columns.items():
        arr = np.asarray(v)
        repeated = np.repeat(arr[at : at + 1], hold, axis=0)
        out[k] = np.concatenate([arr[:at], repeated, arr[at:]], axis=0)
    return out


def _action_quantize(x: Array, severity: float) -> Array:
    """Round each column onto a coarse grid (severity controls grid coarseness).

    The step size is ``severity * 0.5 * column_spread``, so larger severity = coarser grid =
    bigger per-step staircase jumps. A constant column is left untouched (no spread to grid).
    Deterministic (no RNG): rounding is a pure function of the data.
    """
    flat = _as2d(x).copy()
    spread = flat.std(axis=0)
    step = severity * 0.5 * spread
    active = step > 0.0
    flat[:, active] = np.round(flat[:, active] / step[active]) * step[active]
    out: Array = flat.reshape(np.asarray(x).shape).astype(np.asarray(x).dtype)
    return out


def _wrong_target_offset(x: Array, severity: float, rng: np.random.Generator) -> Array:
    """Add a constant per-DoF offset for the whole episode (a mis-targeted / mis-calibrated demo).

    Shape and smoothness are untouched (every step shifts by the same vector), so geometric
    shape signals are blind; only a dataset-relative outlier view sees the shifted distribution.
    The offset direction is seeded; its magnitude scales with each column's spread.
    """
    flat = _as2d(x).copy()
    spread = flat.std(axis=0)
    spread = np.where(spread > 0.0, spread, 1.0)
    direction = rng.standard_normal(flat.shape[1])
    offset = severity * spread * direction
    flat = flat + offset
    out: Array = flat.reshape(np.asarray(x).shape).astype(np.asarray(x).dtype)
    return out


def _dropped_dof(x: Array, rng: np.random.Generator) -> Array:
    """Zero out one (seeded) DoF of the feature for the whole episode (a stuck/dropped channel).

    Collapses that column's spread to 0, which the dataset-relative outlier / uniqueness views
    catch (its summary statistics move away from the dataset). If the feature is 1-D there is
    only one column to drop. Severity is intentionally ignored — a dropped DoF is binary.
    """
    flat = _as2d(x).copy()
    dim = flat.shape[1]
    which = int(rng.integers(0, dim)) if dim > 1 else 0
    flat[:, which] = 0.0
    out: Array = flat.reshape(np.asarray(x).shape).astype(np.asarray(x).dtype)
    return out


def _frame_skip(columns: dict[str, Array], severity: float) -> dict[str, Array]:
    """Keep every ``k``-th step (drop the rest) in every column — a dropped-frames / subsample.

    ``k`` grows with severity: ``severity`` in ``(0, 1]`` maps to a stride of ``2`` (drop every
    other frame) up to a few; the surviving frames are farther apart so per-step deltas jump,
    while the overall geometry (the sequence of visited points) is preserved.
    """
    stride = 1 + max(1, round(min(max(severity, 0.0), 1.0) * 3))  # severity 1.0 -> stride 4
    num_steps = int(next(iter(columns.values())).shape[0])
    if num_steps <= 2:
        return {k: np.asarray(v) for k, v in columns.items()}
    idx: npt.NDArray[np.intp] = np.arange(0, num_steps, stride)
    if idx.size < 2:  # always keep at least the endpoints so signals can still run
        idx = np.array([0, num_steps - 1], dtype=np.intp)
    return {k: np.asarray(v)[idx] for k, v in columns.items()}


# --------------------------------------------------------------------------------------
# Honest detection-AUC blind-spot matrix
# --------------------------------------------------------------------------------------


def rank_auc(higher: F64, lower: F64) -> float:
    """P(a random ``higher`` value > a random ``lower`` value), ties counted as 0.5.

    Returns NaN if either side is empty. This is the Mann-Whitney / AUC statistic.
    """
    if higher.size == 0 or lower.size == 0:
        return float("nan")
    comp = higher[:, None] - lower[None, :]
    return float((comp > 0).sum() + 0.5 * (comp == 0).sum()) / (higher.size * lower.size)


def detection_auc(values: F64, is_corrupt: npt.NDArray[np.bool_], higher_is_better: bool) -> float:
    """Orientation-aware AUC that a signal ranks corrupted demos worse than clean ones.

    ``1.0`` = the signal always ranks the corrupted episode as lower-quality (perfect
    detection), ``0.5`` = blind (no separation), ``0.0`` = it inverts (ranks the corrupted
    one as *better* — the honest failure mode, e.g. path_efficiency on ``truncate``). Non-finite
    (skipped) values are dropped from each side.
    """
    corrupt_vals = values[is_corrupt]
    clean_vals = values[~is_corrupt]
    corrupt_vals = corrupt_vals[np.isfinite(corrupt_vals)]
    clean_vals = clean_vals[np.isfinite(clean_vals)]
    # corrupted = lower quality = lower score when higher_is_better, else higher score.
    if higher_is_better:
        return rank_auc(clean_vals, corrupt_vals)
    return rank_auc(corrupt_vals, clean_vals)


def _signal_name(sig: Any) -> str:
    return str(sig.spec.name)


def _make_context(trajs: Sequence[Trajectory], seed: int) -> Any:
    """A minimal CPU :class:`SignalContext` over ``trajs`` (no curator needed for the matrix)."""
    import logging

    from robocurate.metadata import DatasetFingerprint, DatasetMeta, ResourceProbe
    from robocurate.signals.base import InMemoryCache, SignalContext

    embodiment_ids = tuple({t.meta.embodiment.embodiment_id for t in trajs})
    feature_keys = tuple(dict.fromkeys(s.key for t in trajs for s in t.meta.embodiment.features))
    dataset_meta = DatasetMeta(
        fingerprint=DatasetFingerprint(
            dataset_id="blindspot/in_memory",
            source_format="synthetic_v0",
            content_hash="0" * 64,
            num_episodes=len(trajs),
        ),
        embodiment_ids=embodiment_ids,
        feature_keys=feature_keys,
    )
    return SignalContext(
        seed=seed,
        device="cpu",
        cache=InMemoryCache(),
        resources=ResourceProbe(),
        dataset_meta=dataset_meta,
        logger=logging.getLogger("robocurate.blindspot"),
    )


def _run_signal(sig: Any, trajs: Sequence[Trajectory], seed: int) -> tuple[F64, bool]:
    """Fit (if the signal has a ``fit``) then score ``trajs``; return (values, higher_is_better).

    Signals are scored together in one batch so dataset-relative signals (action_noise's
    outlier z, redundancy's k-NN, structural_validity's median length) see the full mixed
    clean+corrupt population — exactly as the curator runs them.
    """
    ctx = _make_context(trajs, seed)
    fit = getattr(sig, "fit", None)
    if callable(fit):
        fit(list(trajs), ctx)
    scores = sig.score(list(trajs), ctx)
    values = np.full(len(trajs), np.nan, dtype=np.float64)
    higher_is_better = True
    for i, s in enumerate(scores):
        higher_is_better = bool(s.higher_is_better)
        if not s.skipped:
            values[i] = float(s.value)
    return values, higher_is_better


def detection_matrix(
    clean: Sequence[Trajectory],
    signals: Sequence[Signal],
    kinds: Sequence[str],
    *,
    feature: str,
    severity: float = 1.0,
    seed: int = 0,
) -> DetectionMatrix:
    """Build an honest (kind x signal) detection-AUC blind-spot matrix.

    For each corruption ``kind``, every clean trajectory is paired with a corrupted copy of
    itself (seeded per-episode for determinism), the mixed clean+corrupt population is scored by
    each signal, and the orientation-aware :func:`detection_auc` is recorded. The result honestly
    surfaces blind spots (AUC~0.5) and inversions (AUC~0), not just successes.

    Args:
        clean: The clean reference trajectories (read-only; never mutated).
        signals: The signals to evaluate (cheap low-dim ones; each must expose ``spec.name`` and
            ``score``, optionally ``fit``).
        kinds: Corruption kinds to inject (subset of :data:`CORRUPTIONS`).
        feature: Feature key the per-feature corruptions act on.
        severity: Corruption strength passed through to :func:`corrupt`.
        seed: Base seed; episode ``i`` of each kind is corrupted with ``seed + i``.
    """
    unknown = [k for k in kinds if k not in CORRUPTIONS]
    if unknown:
        raise ValueError(f"unknown corruption(s) {unknown}; known: {CORRUPTIONS}")
    names = [_signal_name(s) for s in signals]
    n = len(clean)
    auc: dict[tuple[str, str], float] = {}
    for kind in kinds:
        corrupted = [
            corrupt(t, kind, feature=feature, severity=severity, seed=seed + i)
            for i, t in enumerate(clean)
        ]
        mixed = list(clean) + corrupted
        is_corrupt = np.array([j >= n for j in range(len(mixed))])
        for sig in signals:
            values, higher = _run_signal(sig, mixed, seed)
            auc[(kind, _signal_name(sig))] = detection_auc(values, is_corrupt, higher)
    return DetectionMatrix(kinds=tuple(kinds), signals=tuple(names), auc=auc, severity=severity)


class DetectionMatrix:
    """A (kind x signal) detection-AUC table with an honest markdown renderer.

    ``auc[(kind, signal)]`` is the orientation-aware detection-AUC (see :func:`detection_auc`):
    ~1 detects, ~0.5 blind, ~0 inverts. NaN where the signal skipped every episode.
    """

    # Thresholds for the honest verbal read in to_markdown / classify.
    DETECT = 0.7
    INVERT = 0.3
    BLIND_LO = 0.4
    BLIND_HI = 0.6

    def __init__(
        self,
        *,
        kinds: tuple[str, ...],
        signals: tuple[str, ...],
        auc: dict[tuple[str, str], float],
        severity: float,
    ) -> None:
        self.kinds = kinds
        self.signals = signals
        self.auc = auc
        self.severity = severity

    def value(self, kind: str, signal: str) -> float:
        return self.auc.get((kind, signal), float("nan"))

    def classify(self, kind: str, signal: str) -> str:
        """One-word verdict for a cell: detects / inverts / blind / weak / skip."""
        v = self.value(kind, signal)
        if not np.isfinite(v):
            return "skip"
        if v >= self.DETECT:
            return "detects"
        if v <= self.INVERT:
            return "inverts"
        if self.BLIND_LO <= v <= self.BLIND_HI:
            return "blind"
        return "weak"

    def to_markdown(self) -> str:
        """Render the matrix as a markdown table plus an honest blind-spot / inversion legend."""
        header = "| corruption | " + " | ".join(self.signals) + " |"
        sep = "| --- | " + " | ".join("---" for _ in self.signals) + " |"
        rows = [header, sep]
        for kind in self.kinds:
            cells = []
            for sig in self.signals:
                v = self.value(kind, sig)
                cells.append("skip" if not np.isfinite(v) else f"{v:.2f}")
            rows.append(f"| {kind} | " + " | ".join(cells) + " |")

        blind: list[str] = []
        invert: list[str] = []
        detected: list[str] = []
        for kind in self.kinds:
            verdicts = {sig: self.classify(kind, sig) for sig in self.signals}
            det = [s for s, v in verdicts.items() if v == "detects"]
            inv = [s for s, v in verdicts.items() if v == "inverts"]
            bl = [s for s, v in verdicts.items() if v == "blind"]
            if det:
                detected.append(f"- **{kind}** detected by: {', '.join(det)}")
            else:
                detected.append(f"- **{kind}** detected by: (none — a SUITE blind spot)")
            if inv:
                invert.append(
                    f"- **{kind}** INVERTS on: {', '.join(inv)} (ranks corrupt as better)"
                )
            if bl:
                blind.append(f"- **{kind}** blind for: {', '.join(bl)}")

        legend = [
            "",
            f"Legend: AUC ~1.0 = detects, ~0.5 = blind, ~0.0 = inverts. Severity {self.severity}.",
            "",
            "Detected by:",
            *detected,
        ]
        if invert:
            legend += [
                "",
                "Inversions (the honest failure mode — ranks corrupt as *higher* quality):",
                *invert,
            ]
        if blind:
            legend += ["", "Blind spots (AUC ~0.5):", *blind]
        return "\n".join(rows + legend)


__all__ = [
    "CORRUPTIONS",
    "DetectionMatrix",
    "corrupt",
    "detection_auc",
    "detection_matrix",
    "rank_auc",
]
