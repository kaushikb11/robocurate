"""Action-noise / outlier heuristic (Tier 0, CPU).

Combines two complementary views of "this trajectory looks wrong", per the confirmed design:

1. **Jitter (per-trajectory, stateless).** The high-frequency residual of the source signal
   that a median filter removes: ``residual = source - median_filter(source, w)``. Scored as
   the per-DoF-normalized RMS of that residual, mean over time. A median filter (vs a moving
   average) is robust to isolated spikes — a true spike registers as noise instead of
   smearing into and hiding inside the baseline. This is distinct from the jerk signal:
   jerk penalizes *any* fast change (including intentional motion) via time-derivatives,
   whereas this isolates only the high-frequency component a smoother cannot explain.

2. **Outlier (dataset-relative, uses ``fit``).** Over the whole dataset, summarize each
   trajectory by ``[mean |action|, mean per-DoF std, jitter, num_steps]`` and flag those far
   from the dataset distribution via a robust modified z-score (median/MAD, with a mean/std
   fallback when MAD is 0 — the classic majority-identical degeneracy). This is the first
   signal to use the ``fit`` dataset-pass seam.

The emitted ``value`` is a single badness scalar, ``max(jitter_z, outlier_z)`` — a trajectory
is bad if it is notably jittery **or** a statistical outlier (``higher_is_better=False``).
Per-step residual magnitude is emitted as the per-transition score; ``is_outlier`` and the
component z-scores are surfaced in diagnostics. Tier 0, CPU, deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.trajectory import Array, Trajectory

# Internally everything is float64; a concrete dtype keeps NumPy's typed helpers happy.
F64 = npt.NDArray[np.float64]

DEFAULT_WINDOW = 3
OUTLIER_Z_THRESHOLD = 3.5  # Iglewicz-Hoaglin modified-z cutoff
_MIN_STEPS = 3
_STATS_KEY = "stats"


class ActionNoise:
    """Action jitter + dataset-relative outlier signal.

    Args:
        source: Feature key to analyze. ``None`` (default) uses the action feature.
        window: Odd median-filter window for the jitter residual (default 3); clamped down
            to the largest odd value that fits very short trajectories.
        name: Override the signal name.
    """

    def __init__(
        self, *, source: str | None = None, window: int = DEFAULT_WINDOW, name: str = "action_noise"
    ) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.source = source
        self.window = window
        requirement = source if source is not None else "action"
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({requirement}),
            produces_per_transition=True,
            deterministic=True,
            description=(
                f"Action jitter (median-residual RMS of {requirement}) combined with a "
                "dataset-relative robust-z outlier flag (lower is cleaner)."
            ),
        )

    # -- fit: dataset-relative statistics --------------------------------------------

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        summaries: list[F64] = []
        jitters: list[float] = []
        for traj in trajectories:
            computed = self._compute(traj)
            if computed is None:
                continue
            jitter, summary, _per_step = computed
            jitters.append(jitter)
            summaries.append(summary)

        if not summaries:
            # Nothing computable; score() will fall back to jitter-only with no outlier info.
            ctx.cache.put(_STATS_KEY, None)
            return

        summary_arr: F64 = np.vstack(summaries).astype(np.float64)
        jitter_arr: F64 = np.asarray(jitters, dtype=np.float64)
        ctx.cache.put(
            _STATS_KEY,
            {
                "summary_median": np.median(summary_arr, axis=0),
                "summary_mad": _mad(summary_arr, axis=0),
                "summary_mean": summary_arr.mean(axis=0),
                "summary_std": summary_arr.std(axis=0),
                "jitter_median": float(np.median(jitter_arr)),
                "jitter_mad": float(_mad(jitter_arr, axis=0)),
                "jitter_mean": float(jitter_arr.mean()),
                "jitter_std": float(jitter_arr.std()),
            },
        )

    # -- score -----------------------------------------------------------------------

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        stats = ctx.cache.get(_STATS_KEY) if ctx.cache.has(_STATS_KEY) else None
        return [self._score_one(traj, stats) for traj in batch]

    def _score_one(self, traj: Trajectory, stats: dict[str, Any] | None) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        computed = self._compute(traj)
        if computed is None:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason=self._skip_reason(traj),
                higher_is_better=False,
            )
        jitter, summary, per_step = computed

        if stats is None:
            # No dataset statistics (e.g. degenerate dataset); fall back to raw jitter.
            jitter_z = jitter
            outlier_z = 0.0
            is_outlier = False
        else:
            jitter_z = _z(
                jitter,
                stats["jitter_median"],
                stats["jitter_mad"],
                stats["jitter_mean"],
                stats["jitter_std"],
            )
            feature_z = np.array(
                [
                    _z(
                        summary[i],
                        stats["summary_median"][i],
                        stats["summary_mad"][i],
                        stats["summary_mean"][i],
                        stats["summary_std"][i],
                    )
                    for i in range(summary.size)
                ]
            )
            outlier_z = float(np.max(np.abs(feature_z))) if feature_z.size else 0.0
            is_outlier = outlier_z > OUTLIER_Z_THRESHOLD

        value = max(float(jitter_z), float(outlier_z))
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=value,
            higher_is_better=False,
            per_transition=per_step.astype(np.float32),
            diagnostics={
                "source": self._source_label(),
                "jitter": float(jitter),
                "jitter_z": float(jitter_z),
                "outlier_z": float(outlier_z),
                "is_outlier": bool(is_outlier),
            },
        )

    # -- shared computation ----------------------------------------------------------

    def _compute(self, traj: Trajectory) -> tuple[float, F64, F64] | None:
        """Return ``(jitter, summary_features, per_step_residual)`` or ``None`` to skip."""
        source = self._resolve_source(traj)
        if source is None or source.shape[0] < _MIN_STEPS:
            return None
        x: F64 = np.asarray(source, dtype=np.float64).reshape(source.shape[0], -1)

        # Per-DoF z-normalization so jitter is unit-/embodiment-agnostic; constant DoFs zeroed.
        std = x.std(axis=0)
        nonconst = std > 0.0
        x_norm = np.zeros_like(x)
        x_norm[:, nonconst] = (x[:, nonconst] - x[:, nonconst].mean(axis=0)) / std[nonconst]

        residual: F64 = x_norm - _median_filter(x_norm, self._effective_window(x.shape[0]))
        per_step: F64 = np.linalg.norm(residual, axis=1)
        jitter = float(per_step.mean())

        summary: F64 = np.array(
            [
                float(np.abs(x).mean()),  # overall action magnitude
                float(std.mean()),  # spread across DoFs
                jitter,  # jitter feeds the outlier view too
                float(x.shape[0]),  # episode length
            ],
            dtype=np.float64,
        )
        return jitter, summary, per_step

    def _effective_window(self, num_steps: int) -> int:
        w = min(self.window, num_steps)
        if w % 2 == 0:  # keep the window odd so the filter is centered
            w -= 1
        return max(1, w)

    def _resolve_source(self, traj: Trajectory) -> Array | None:
        if self.source is None:
            return traj.actions()
        return traj.feature(self.source) if traj.has(self.source) else None

    def _skip_reason(self, traj: Trajectory) -> str:
        if self._resolve_source(traj) is None:
            return f"no {self._source_label()} feature to analyze"
        return f"trajectory too short (< {_MIN_STEPS} steps) for noise estimation"

    def _source_label(self) -> str:
        return self.source if self.source is not None else "action"


def _median_filter(x: F64, window: int) -> F64:
    """Centered median filter over the time axis of an ``(T, D)`` array (edge-padded)."""
    if window <= 1:
        return x
    half = window // 2
    padded = np.pad(x, ((half, half), (0, 0)), mode="edge")
    windows: F64 = np.asarray(sliding_window_view(padded, window, axis=0), dtype=np.float64)
    out: F64 = np.median(windows, axis=-1).astype(np.float64)
    return out


def _mad(values: F64, axis: int) -> F64:
    """Median absolute deviation (raw, not scaled), along ``axis``."""
    median = np.median(values, axis=axis, keepdims=True)
    deviation: F64 = np.abs(values - median).astype(np.float64)
    out: F64 = np.median(deviation, axis=axis).astype(np.float64)
    return out


def _z(value: float, median: float, mad: float, mean: float, std: float) -> float:
    """Robust modified z-score, falling back to a standard z-score when MAD is 0."""
    if mad > 0.0:
        return 0.6745 * (value - median) / mad
    if std > 0.0:
        return (value - mean) / std
    return 0.0


__all__ = ["OUTLIER_Z_THRESHOLD", "ActionNoise"]
