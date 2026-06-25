"""Sim physics-validity heuristic (Tier 0, CPU, sim-only).

Flags physically impossible simulation trajectories — the kind of garbage a sim can emit
that no amount of "smoothness" or "uniqueness" scoring would catch. It is the first
**sim-only** signal: it reads sim-state features and is recorded as a skip on real/teleop
data, which carries none (the sim-vs-real applicability split is handled per-trajectory, so a
mixed dataset works).

Sim-state convention (the seam a sim adapter will populate): features are keyed under a
``sim.`` prefix (role ``EXTRA``). Recognized by this signal:

* ``sim.penetration_depth`` — per-step interpenetration depth in metres (``(T,)`` or
  ``(T, n_contacts)``; values ``>= 0``). The configurable ``penetration_key``.
* any ``sim.*`` feature whose name contains ``pos`` — object position(s) used for the
  teleportation/discontinuity check.

Three validity checks (confirmed), combined into one continuous severity:

1. **Penetration.** ``max_penetration - threshold`` (metres over the allowed threshold).
2. **Non-finite.** Any NaN/inf in a non-image feature ⇒ the sim blew up (catastrophic).
3. **Discontinuity.** A per-step object-position jump beyond ``max_step_displacement``
   metres ⇒ teleportation glitch.

``value`` is the summed violation magnitude (``higher_is_better=False``), with non-finite
weighted as catastrophic so it dominates; ``is_valid`` is surfaced in diagnostics. A valid
trajectory scores 0. This fits the existing per-trajectory ``Signal`` contract, so invalid
trajectories rank worst and are removed first under budget. A *hard* always-remove validity
gate (regardless of budget) is a deliberate future curator mode, not built here; the
``is_valid`` flag is the seam it will use.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from robocurate.signals.base import (
    REQUIRES_SIM_STATE,
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.trajectory import Array, FeatureRole, Trajectory

DEFAULT_PENETRATION_KEY = "sim.penetration_depth"
DEFAULT_PENETRATION_THRESHOLD = 0.005  # 5 mm
DEFAULT_MAX_STEP_DISPLACEMENT = 1.0  # metres per step; a larger jump is teleportation
DEFAULT_SIM_PREFIX = "sim."
# Non-finite sim state is catastrophic (the sim diverged); weight it so it dominates the
# continuous severity regardless of the metric-scale penetration/jump terms.
NONFINITE_PENALTY = 1.0e6


class SimPhysicsValidity:
    """Sim-only physical-validity signal (penetration + non-finite + discontinuity).

    Args:
        penetration_key: Feature key holding per-step penetration depth (metres).
        penetration_threshold: Allowed penetration before it counts as a violation (metres).
        max_step_displacement: Max plausible per-step object-position jump (metres).
        position_keys: Explicit sim position feature keys for the discontinuity check, or
            ``None`` to auto-detect ``sim.*`` features whose name contains ``"pos"``.
        sim_prefix: Feature-key prefix that marks sim-state (used to detect real-vs-sim).
        name: Override the signal name.
    """

    def __init__(
        self,
        *,
        penetration_key: str = DEFAULT_PENETRATION_KEY,
        penetration_threshold: float = DEFAULT_PENETRATION_THRESHOLD,
        max_step_displacement: float = DEFAULT_MAX_STEP_DISPLACEMENT,
        position_keys: tuple[str, ...] | None = None,
        sim_prefix: str = DEFAULT_SIM_PREFIX,
        name: str = "sim_physics_validity",
    ) -> None:
        self.penetration_key = penetration_key
        self.penetration_threshold = penetration_threshold
        self.max_step_displacement = max_step_displacement
        self.position_keys = position_keys
        self.sim_prefix = sim_prefix
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({REQUIRES_SIM_STATE}),
            produces_per_transition=True,
            deterministic=True,
            description=(
                "Sim-only physical validity: penetration, non-finite, and teleportation "
                "violations (lower is more valid)."
            ),
        )

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        """Stateless: nothing to fit."""
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [self._score_one(traj) for traj in batch]

    # -- internals -------------------------------------------------------------------

    def _score_one(self, traj: Trajectory) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        if not self._has_sim_state(traj):
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason=(
                    f"requires sim state (no {self.sim_prefix!r} features); not applicable "
                    "to real data"
                ),
                higher_is_better=False,
            )

        per_step_pen = self._per_step_penetration(traj)
        if per_step_pen is not None and per_step_pen.size > 0:
            max_pen = float(per_step_pen.max())
        else:
            max_pen = 0.0
        pen_excess = max(0.0, max_pen - self.penetration_threshold)

        has_nonfinite = self._has_nonfinite(traj)
        max_jump, jump_excess = self._max_position_jump(traj)

        value = pen_excess + jump_excess + (NONFINITE_PENALTY if has_nonfinite else 0.0)
        is_valid = value == 0.0

        per_transition = per_step_pen.astype(np.float32) if per_step_pen is not None else None
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=float(value),
            higher_is_better=False,
            per_transition=per_transition,
            diagnostics={
                "is_valid": bool(is_valid),
                "max_penetration": max_pen,
                "penetration_threshold": self.penetration_threshold,
                "penetration_excess": pen_excess,
                "max_position_jump": max_jump,
                "jump_excess": jump_excess,
                "has_nonfinite": bool(has_nonfinite),
            },
        )

    def _has_sim_state(self, traj: Trajectory) -> bool:
        return any(
            spec.key.startswith(self.sim_prefix) and traj.has(spec.key)
            for spec in traj.embodiment.features
        )

    def _per_step_penetration(self, traj: Trajectory) -> Array | None:
        if not traj.has(self.penetration_key):
            return None
        arr: Array = np.asarray(traj.feature(self.penetration_key), dtype=np.float64)
        if arr.ndim <= 1:
            return arr
        reduced: Array = arr.reshape(arr.shape[0], -1).max(axis=1)
        return reduced

    def _has_nonfinite(self, traj: Trajectory) -> bool:
        for spec in traj.embodiment.features:
            if spec.role is FeatureRole.IMAGE or not traj.has(spec.key):
                continue
            if not np.all(np.isfinite(np.asarray(traj.feature(spec.key), dtype=np.float64))):
                return True
        return False

    def _max_position_jump(self, traj: Trajectory) -> tuple[float, float]:
        max_jump = 0.0
        for key in self._position_keys(traj):
            pos = np.asarray(traj.feature(key), dtype=np.float64)
            pos = pos.reshape(pos.shape[0], -1)
            if pos.shape[0] < 2:
                continue
            jump = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).max())
            max_jump = max(max_jump, jump)
        jump_excess = max(0.0, max_jump - self.max_step_displacement)
        return max_jump, jump_excess

    def _position_keys(self, traj: Trajectory) -> list[str]:
        if self.position_keys is not None:
            return [k for k in self.position_keys if traj.has(k)]
        return [
            spec.key
            for spec in traj.embodiment.features
            if spec.key.startswith(self.sim_prefix)
            and "pos" in spec.key.lower()
            and traj.has(spec.key)
        ]


__all__ = ["NONFINITE_PENALTY", "SimPhysicsValidity"]
