"""A minimal worked example: writing your own RoboCurate quality signal.

RoboCurate's central bet is that **every quality signal is a plugin behind one contract**
(Invariant 4). A signal never touches the core engine — it just implements the
:class:`~robocurate.signals.base.Signal` protocol (``spec`` + ``fit`` + ``score``) and the
engine handles batching, scheduling, requirement gating, and caching.

This file is the smallest end-to-end example of that: a custom ``ActionRange`` signal that
scores each episode by the range of its action magnitudes (max minus min). It runs on CPU
with no optional extras. Read it top-to-bottom alongside ``docs/EXTENDING.md``.

Run it directly to score a couple of toy trajectories and verify the contract::

    uv run python examples/custom_signal.py

The ``__main__`` block at the bottom runs ``check_signal_contract`` on the signal and prints
the (empty) list of violations — exactly the check you'd put in your own test suite.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

# A signal depends only on the public signal/trajectory surface — never on engine internals.
from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.trajectory import Trajectory


class ActionRange:
    """Per-episode range of action magnitude: ``max(|a_t|) - min(|a_t|)`` over the episode.

    A larger range means the episode mixes near-still moments with large commanded motions —
    a cheap, embodiment-agnostic proxy that distinguishes "deliberate, varied" episodes from
    flat or saturated ones. This is a *toy* signal whose only job is to show the contract; it
    is not a recommended curation heuristic.

    Like the built-in :class:`~robocurate.signals.jerk.Jerk` it is a stateless Tier-0 CPU
    heuristic, so :meth:`fit` is a no-op. It requires an ``action`` feature and records a
    *skip* (never raises) for a trajectory that lacks one.

    Args:
        name: Override the signal name (rarely needed).
    """

    def __init__(self, *, name: str = "action_range") -> None:
        # The spec is the static contract the signal advertises about itself. The engine reads
        # it to schedule the signal, gate it on requirements, and label it in the scorecard.
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,  # cheap, CPU, laptop-friendly
            requires=frozenset({"action"}),  # skip (don't crash) without an action feature
            produces_per_transition=True,  # we also emit a (T,) per-step magnitude
            deterministic=True,  # a pure function of the input arrays (Invariant 3)
            description="Range (max - min) of per-step action magnitude (higher is more varied).",
        )

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        """Stateless heuristic: nothing to precompute or train.

        A learned signal would do its one-shot training / embedding precompute here and stash
        the result in ``ctx.cache`` for :meth:`score` to read.
        """
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        """Return exactly one :class:`TrajectoryScore` per input trajectory, in input order."""
        return [self._score_one(traj) for traj in batch]

    def _score_one(self, traj: Trajectory) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        actions = traj.actions()
        if actions is None:
            # Unmet requirement -> a recorded skip, never an exception, never a silent drop.
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="no action feature to measure range over",
                higher_is_better=True,
            )

        # Per-step L2 magnitude of the action across DoFs -> a (T,) array.
        flat = np.asarray(actions, dtype=np.float64).reshape(actions.shape[0], -1)
        per_step = np.linalg.norm(flat, axis=1)
        value = float(per_step.max() - per_step.min())

        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=value,
            higher_is_better=True,
            per_transition=per_step.astype(np.float32),
            diagnostics={
                "min_magnitude": float(per_step.min()),
                "max_magnitude": float(per_step.max()),
            },
        )


# Registering the signal so ``robocurate.signals.get("action_range")`` finds it.
#
# In a real package you advertise the signal through the ``robocurate.signals`` entry-point
# group in your ``pyproject.toml`` — the *same* mechanism the built-in signals use, so a
# third-party signal is discovered without any edit to the RoboCurate core::
#
#     [project.entry-points."robocurate.signals"]
#     action_range = "my_package.signals:ActionRange"
#
# For a quick experiment (or a test) you can instead register it programmatically:
#
#     from robocurate import signals
#     signals.register("action_range", ActionRange)
#     sig = signals.get("action_range")


def main() -> int:
    """Score a couple of toy trajectories and verify the signal honors the contract."""
    from robocurate.signals import check_signal_contract

    signal = ActionRange()

    # The contract-checker builds its own tiny synthetic batch + context when given none, so
    # this is the one-liner a contributor runs (or asserts in a test) to confirm the signal is
    # well-behaved before shipping it.
    violations = check_signal_contract(signal)
    print(f"signal: {signal.spec.name} v{signal.spec.version} ({signal.spec.cost_tier.name})")
    print(f"contract violations: {violations}")  # expected: [] (empty == passes)
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
