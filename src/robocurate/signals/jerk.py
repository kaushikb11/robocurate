"""Jerk / smoothness heuristic (Tier 0, CPU) — the reference vertical slice.

This is the first real quality signal. It flags trajectories whose chosen source signal
(by default the action sequence) is *rough* — high magnitude in a time-derivative — which is
a cheap, robust proxy for low-quality / jittery demonstrations.

Algorithm (confirmed before implementation; see docs/ARCHITECTURE notes):

1. **Source.** Differentiate the action sequence by default (matches the blueprint's "action
   jerk/smoothness"); a ``source`` feature key can override it (e.g. ``"observation.state"``
   for realized-motion jerk). If the source feature is absent, the trajectory is a recorded
   skip — never a crash.
2. **Real time.** Derivatives use the actual per-step ``dt`` from ``timestamps()`` via
   ``np.gradient`` over the time coordinate, so irregular/teleop control rates are handled
   correctly. Missing or non-increasing timestamps → recorded skip.
3. **Definition.** Take the ``deriv_order``-th time-derivative. Default ``deriv_order=2``
   (acceleration/curvature of the command) — a robust roughness proxy that does *not* assume
   the source is position-like. Set ``deriv_order=3`` for textbook jerk when the source is a
   position.
4. **Units.** Per-DoF z-normalization (subtract mean, divide by std over time) *before*
   differentiating, so the score is unit- and embodiment-agnostic and one wide-range DoF
   does not dominate. Constant DoFs (std 0) contribute nothing.
5. **Aggregation.** Per step, the jerk magnitude is the L2 norm across DoFs of the derivative
   (emitted as the per-transition score). The trajectory-level value is the mean over time.
   Lower is smoother, so ``higher_is_better=False``.

Cost tier 0 (CPU, laptop-friendly). Deterministic: a pure function of the input arrays.
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
from robocurate.trajectory import Array, Trajectory

DEFAULT_DERIV_ORDER = 2


class Jerk:
    """Action-smoothness signal: mean magnitude of a time-derivative of the source signal.

    Args:
        source: Feature key to differentiate. ``None`` (default) uses the trajectory's
            action feature. Any feature key (e.g. ``"observation.state"``) may be given.
        deriv_order: Order of the time-derivative (default 2). Use 3 for true jerk on a
            position-like source.
        name: Override the signal name (rarely needed).
    """

    def __init__(
        self,
        *,
        source: str | None = None,
        deriv_order: int = DEFAULT_DERIV_ORDER,
        name: str = "jerk",
    ) -> None:
        if deriv_order < 1:
            raise ValueError(f"deriv_order must be >= 1, got {deriv_order}")
        self.source = source
        self.deriv_order = deriv_order
        requirement = source if source is not None else "action"
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({requirement}),
            produces_per_transition=True,
            deterministic=True,
            description=(
                f"Mean L2 magnitude of the order-{deriv_order} time-derivative of "
                f"{requirement} (lower is smoother)."
            ),
        )

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        """Stateless heuristic: nothing to fit."""
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [self._score_one(traj) for traj in batch]

    # -- internals -------------------------------------------------------------------

    def _score_one(self, traj: Trajectory) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        source = self._resolve_source(traj)
        if source is None:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason=f"no {self._source_label()} feature to differentiate",
                higher_is_better=False,
            )

        timestamps = traj.timestamps()
        if timestamps is None:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="no timestamps; cannot compute time-derivative",
                higher_is_better=False,
            )

        t = np.asarray(timestamps, dtype=np.float64)
        if t.ndim != 1 or t.size < self.deriv_order + 1:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason=(
                    f"trajectory too short ({t.size} steps) for an order-"
                    f"{self.deriv_order} derivative"
                ),
                higher_is_better=False,
            )
        if not np.all(np.diff(t) > 0.0):
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="timestamps are not strictly increasing",
                higher_is_better=False,
            )

        per_step = self._jerk_magnitude(np.asarray(source, dtype=np.float64), t)
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=float(per_step.mean()),
            higher_is_better=False,
            per_transition=per_step.astype(np.float32),
            diagnostics={
                "source": self._source_label(),
                "deriv_order": self.deriv_order,
                "max_jerk": float(per_step.max()),
            },
        )

    def _jerk_magnitude(self, source: Array, t: Array) -> Array:
        """Return the ``(T,)`` per-step L2 magnitude of the order-k derivative of ``source``."""
        # Flatten any per-step feature shape to (T, D); a scalar-per-step source becomes (T, 1).
        x = source.reshape(source.shape[0], -1).astype(np.float64)

        # Per-DoF z-normalization so the result is unit-/embodiment-agnostic. Constant DoFs
        # (std 0) are zeroed out so they neither contribute jerk nor produce NaNs.
        std = x.std(axis=0)
        nonconst = std > 0.0
        x_norm = np.zeros_like(x)
        x_norm[:, nonconst] = (x[:, nonconst] - x[:, nonconst].mean(axis=0)) / std[nonconst]

        # Successive time-derivatives using the real time coordinate (handles non-uniform dt).
        deriv = x_norm
        for _ in range(self.deriv_order):
            deriv = np.gradient(deriv, t, axis=0, edge_order=1)

        magnitude: Array = np.linalg.norm(deriv, axis=1)
        return magnitude

    def _resolve_source(self, traj: Trajectory) -> Array | None:
        if self.source is None:
            return traj.actions()
        return traj.feature(self.source) if traj.has(self.source) else None

    def _source_label(self) -> str:
        return self.source if self.source is not None else "action"


__all__ = ["Jerk"]
