"""Structural-validity heuristic (Tier 0, CPU) — catches the defects geometry signals MISS.

The cheap geometric signals (jerk, path-efficiency, spectral-smoothness) score the *shape* of a
motion. A known-answer corruption study on this project's own signals showed they are blind to —
and on truncation actively INVERT on — STRUCTURAL defects: a truncated (incomplete) episode is
shorter and straighter, so a directness/smoothness signal ranks it as *higher* quality. This
signal exists to close that blind spot. It is source-format-agnostic (real or sim) and flags
three structural failure modes that have nothing to do with kinematic smoothness:

1. **Truncation** — an episode far shorter than the dataset's typical length (an incomplete /
   cut-off demonstration). Needs dataset context, so the median length is learned in ``fit``;
   without it (signal not fit), this check is skipped rather than guessed.
2. **Stall** — a run of held/repeated frames (the motion paused / the logger hung): a fraction of
   steps with ~zero per-step change in the source feature beyond a small tolerance.
3. **Non-finite** — any NaN/inf in a non-image feature (catastrophic; the data is corrupt).

``value`` is the summed violation severity (``higher_is_better=False``); a structurally sound
trajectory scores 0, so the worst structural offenders rank first for removal. ``is_valid`` is
surfaced in diagnostics as the seam for a future hard validity gate. This is a *complement* to —
not a replacement for — the geometric and learned signals; a real curator combines them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.trajectory import Array, FeatureRole, Trajectory

DEFAULT_TRUNCATION_FRACTION = 0.5  # shorter than half the median length -> likely truncated
DEFAULT_STALL_TOLERANCE = 0.1  # up to 10% held frames is normal (settling); beyond is a stall
DEFAULT_STALL_EPS = 1e-8  # per-step change below this counts as a held/repeated frame
NONFINITE_PENALTY = 1.0e6  # catastrophic; dominates the bounded truncation/stall terms
_MEDIAN_KEY = "structural_validity.median_steps"


class StructuralValidity:
    """Structural-defect signal: truncation + stall + non-finite (complements geometric signals).

    Args:
        source: Feature key checked for stalls (per-step held frames). ``None`` (default) uses
            the trajectory's state/proprio features (a stall = the *world/robot* is frozen, i.e.
            the observation stops changing — not the action, which is ~constant during smooth
            constant-velocity motion), falling back to the action feature.
        truncation_fraction: An episode shorter than this fraction of the dataset median length
            is flagged as truncated (needs ``fit`` to know the median).
        stall_tolerance: Held-frame fraction tolerated before it counts as a stall.
        stall_eps: Per-step L2 change below which a step is a held/repeated frame.
        name: Override the signal name.
    """

    def __init__(
        self,
        *,
        source: str | None = None,
        truncation_fraction: float = DEFAULT_TRUNCATION_FRACTION,
        stall_tolerance: float = DEFAULT_STALL_TOLERANCE,
        stall_eps: float = DEFAULT_STALL_EPS,
        name: str = "structural_validity",
    ) -> None:
        self.source = source
        self.truncation_fraction = truncation_fraction
        self.stall_tolerance = stall_tolerance
        self.stall_eps = stall_eps
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset(),  # works on any trajectory; degrades gracefully
            produces_per_transition=False,
            deterministic=True,
            description=(
                "Structural validity: truncation, stall (held frames), and non-finite "
                "violations the geometric signals miss (lower is more valid)."
            ),
        )

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        """Learn the dataset's median episode length (for the truncation check)."""
        lengths = [int(traj.meta.num_steps) for traj in trajectories]
        ctx.cache.put(_MEDIAN_KEY, float(np.median(lengths)) if lengths else None)

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        median = ctx.cache.get(_MEDIAN_KEY) if ctx.cache.has(_MEDIAN_KEY) else None
        return [self._score_one(traj, median) for traj in batch]

    # -- internals -------------------------------------------------------------------

    def _score_one(self, traj: Trajectory, median: float | None) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        trunc_sev = self._truncation_severity(traj, median)
        stall_sev, held_frac = self._stall_severity(traj)
        has_nonfinite = self._has_nonfinite(traj)

        value = trunc_sev + stall_sev + (NONFINITE_PENALTY if has_nonfinite else 0.0)
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=float(value),
            higher_is_better=False,
            diagnostics={
                "is_valid": value == 0.0,
                "num_steps": int(traj.meta.num_steps),
                "median_steps": median,
                "truncation_severity": trunc_sev,
                "held_frame_fraction": held_frac,
                "stall_severity": stall_sev,
                "has_nonfinite": bool(has_nonfinite),
            },
        )

    def _truncation_severity(self, traj: Trajectory, median: float | None) -> float:
        if median is None or median <= 0:
            return 0.0
        floor = self.truncation_fraction * median
        if traj.meta.num_steps >= floor:
            return 0.0
        return float((floor - traj.meta.num_steps) / median)

    def _stall_severity(self, traj: Trajectory) -> tuple[float, float]:
        source = self._resolve_source(traj)
        if source is None or source.shape[0] < 2:
            return 0.0, 0.0
        x = np.asarray(source, dtype=np.float64).reshape(source.shape[0], -1)
        step_norm = np.linalg.norm(np.diff(x, axis=0), axis=1)
        held_frac = float(np.mean(step_norm < self.stall_eps))
        severity = max(0.0, held_frac - self.stall_tolerance)
        return severity, held_frac

    def _has_nonfinite(self, traj: Trajectory) -> bool:
        for spec in traj.embodiment.features:
            if spec.role is FeatureRole.IMAGE or not traj.has(spec.key):
                continue
            if not np.all(np.isfinite(np.asarray(traj.feature(spec.key), dtype=np.float64))):
                return True
        return False

    def _resolve_source(self, traj: Trajectory) -> Array | None:
        if self.source is not None:
            return traj.feature(self.source) if traj.has(self.source) else None
        # A stall is the world/robot frozen -> the STATE stops changing. Use the concatenated
        # state/proprio features (constant-velocity motion has ~constant *action* but moving
        # *state*, so the action would false-positive). Fall back to the action only if no state.
        state = traj.select_roles(FeatureRole.STATE, FeatureRole.PROPRIO)
        if state:
            parts = [
                np.asarray(v, dtype=np.float64).reshape(np.asarray(v).shape[0], -1)
                for _, v in sorted(state.items())
            ]
            return np.concatenate(parts, axis=1)
        return traj.actions()


__all__ = ["NONFINITE_PENALTY", "StructuralValidity"]
