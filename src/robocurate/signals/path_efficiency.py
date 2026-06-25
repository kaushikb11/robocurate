"""Path-efficiency / directness heuristic (Tier 0, CPU).

A cheap, deterministic, GPU-free proxy for demonstration quality: how *direct* a trajectory's
motion is. Less-skilled teleoperation tends to wander, hesitate, and self-correct, covering
more path to achieve the same net displacement; the **directness ratio** captures that without
any learned model or oracle path:

    directness = || Σ_t Δx_t ||  /  Σ_t || Δx_t ||      ∈ (0, 1]
                 (net displacement)   (total path length)

A straight, committed motion → ~1; a meandering/correcting motion covering the same net
distance → ≪ 1. It is the "straightness index" of movement-ecology and reaching studies
(Batschelet 1981; Benhamou, J Theor Biol 2004) applied to robot trajectories: dimensionless,
scale- and duration-invariant, parameter-free, and robust on short trajectories.

Algorithm (confirmed before implementation; grounded in the research notes for this signal):

1. **Source.** A vector feature over which to measure the path. By default the action
   sequence; point ``source`` at a Cartesian end-effector *position* feature (with
   ``motion="positions"``) for the truest path — that is the most reliable input.
2. **Metric subspace (important).** Only translational dimensions are a meaningful Euclidean
   path. ``dims=(0, 3)`` by default takes the leading three components — correct for
   end-effector / OSC-pose control, where the action (or eef position) leads with ``(x, y, z)``.
   Including rotation/gripper dimensions pollutes the norm and can *invert* the signal, so the
   default is a translational slice, never all dimensions. Set ``dims=None`` to use every
   column when you know they are all spatial.
3. **Increments.** ``motion="increments"`` treats each row as a per-step displacement (delta
   actions); ``motion="positions"`` takes ``np.diff`` of an absolute position feature first.
4. **Definition.** ``directness = ||net|| / path`` over the increments. Higher is more
   efficient, so ``higher_is_better=True``.
5. **Honesty.** Degenerate inputs (too short, no motion, missing/empty source) are recorded
   skips, never a fabricated value. Directness penalizes legitimately curved or multi-phase
   paths (a known limitation of the straightness index) — validate it against ground truth on
   your data rather than assuming it transfers.

Cost tier 0 (CPU, laptop-friendly). Deterministic: a pure function of the input arrays; needs
no timestamps (it is purely geometric).
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

DEFAULT_DIMS = (0, 3)
_MIN_STEPS = 2
_EPS = 1e-9


class PathEfficiency:
    """Directness signal: net displacement over total path length of the source motion.

    Args:
        source: Feature key to measure the path over. ``None`` (default) uses the action
            feature. Point at a Cartesian position feature (e.g. ``"observation.eef_pos"``)
            with ``motion="positions"`` for the true end-effector path.
        dims: ``(start, stop)`` column slice selecting the translational subspace, or ``None``
            for all columns. Default ``(0, 3)`` — the leading ``(x, y, z)`` of OSC-pose
            actions / end-effector positions. Never defaults to all dims (rotation/gripper
            dimensions would distort, even invert, the directness).
        motion: ``"increments"`` (rows are per-step displacements, e.g. delta actions) or
            ``"positions"`` (rows are absolute positions; take ``np.diff`` first).
        name: Override the signal name (rarely needed).
    """

    def __init__(
        self,
        *,
        source: str | None = None,
        dims: tuple[int, int] | None = DEFAULT_DIMS,
        motion: str = "increments",
        name: str = "path_efficiency",
    ) -> None:
        if motion not in ("increments", "positions"):
            raise ValueError(f"motion must be 'increments' or 'positions', got {motion!r}")
        if dims is not None and (len(dims) != 2 or dims[0] < 0 or dims[1] <= dims[0]):
            raise ValueError(
                f"dims must be a (start, stop) slice with stop > start >= 0, got {dims}"
            )
        self.source = source
        self.dims = dims
        self.motion = motion
        requirement = source if source is not None else "action"
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({requirement}),
            produces_per_transition=False,
            deterministic=True,
            description=(
                f"Path directness (net displacement / total path length) of {requirement} "
                "(higher is more efficient)."
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
            return self._skip(fingerprint, f"no {self._source_label()} feature to measure")

        x: Array = np.asarray(source, dtype=np.float64).reshape(source.shape[0], -1)
        if self.dims is not None:
            x = x[:, self.dims[0] : self.dims[1]]
        if x.shape[1] == 0:
            return self._skip(fingerprint, f"source has no columns in dims {self.dims}")
        if x.shape[0] < _MIN_STEPS:
            return self._skip(fingerprint, f"trajectory too short ({x.shape[0]} steps)")

        increments: Array = np.diff(x, axis=0) if self.motion == "positions" else x
        if increments.shape[0] == 0:
            return self._skip(fingerprint, "no increments to measure")
        path_length = float(np.linalg.norm(increments, axis=1).sum())
        if path_length < _EPS:
            return self._skip(fingerprint, "degenerate: no motion (zero path length)")

        net_displacement = float(np.linalg.norm(increments.sum(axis=0)))
        directness = net_displacement / path_length
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=directness,
            higher_is_better=True,
            diagnostics={
                "source": self._source_label(),
                "motion": self.motion,
                "net_displacement": net_displacement,
                "path_length": path_length,
            },
        )

    def _skip(self, fingerprint: str, reason: str) -> TrajectoryScore:
        return TrajectoryScore.skip(
            self.spec.name, fingerprint, reason=reason, higher_is_better=True
        )

    def _resolve_source(self, traj: Trajectory) -> Array | None:
        if self.source is None:
            return traj.actions()
        return traj.feature(self.source) if traj.has(self.source) else None

    def _source_label(self) -> str:
        return self.source if self.source is not None else "action"


__all__ = ["PathEfficiency"]
