"""A contract-checker for the :class:`~robocurate.signals.base.Signal` protocol.

"Signals are plugins behind one contract" (Invariant 4) is only worth something if a
contributor can *verify* their signal honors that contract before shipping it. This module
provides :func:`check_signal_contract` — a battery of structural and behavioral checks that
runs against **any** object claiming to be a :class:`Signal` and returns a list of
human-readable violation strings. An empty list means the signal passes.

The checks are deliberately black-box: they only use the public :class:`Signal` surface
(``spec``, :meth:`Signal.fit`, :meth:`Signal.score`) plus a small synthetic batch and a
:class:`SignalContext`, so they reach into no engine internals and work for a cheap CPU
heuristic and an expensive learned signal alike. They cover exactly the promises the
protocol makes:

* ``spec`` is a well-formed :class:`SignalSpec` (name, cost tier, requirements, ...);
* :meth:`Signal.fit` runs without error on a small batch;
* :meth:`Signal.score` returns **exactly one** :class:`TrajectoryScore` per input, **in
  order**, matched by ``trajectory_fingerprint``;
* every score is either a recorded skip (with a reason) or carries a finite ``value``;
* ``produces_per_transition`` implies a ``(T,)`` per-transition array of the right length;
* a ``deterministic`` signal yields identical values across two runs on the same inputs
  (Invariant 3);
* ``higher_is_better`` is a bool.

Contributors use it two ways: :func:`check_signal_contract` to inspect violations, and the
thin :func:`assert_signal_contract` wrapper to fail a test with the joined messages.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robocurate.signals.base import Signal
    from robocurate.trajectory import Array, Trajectory


def check_signal_contract(
    signal: Signal,
    trajectories: Sequence[Trajectory] | None = None,
    *,
    ctx: SignalContext | None = None,
    seed: int = 0,
) -> list[str]:
    """Run the :class:`Signal` contract battery against ``signal``.

    Returns a list of human-readable violation strings; an **empty list means the signal
    honors the contract**. Nothing is raised for a contract *violation* — they are
    collected and returned so a caller can report all of them at once (use
    :func:`assert_signal_contract` to turn them into an ``AssertionError``).

    Args:
        signal: Any object claiming to implement the :class:`Signal` protocol.
        trajectories: A small batch to score. When ``None``, a tiny default synthetic batch
            is built (a 2-DoF toy embodiment with action / state / reward / timestamps), so
            the checker is runnable with no dataset and no arguments.
        ctx: The :class:`SignalContext` to score under. When ``None``, a minimal CPU context
            is built with the given ``seed``.
        seed: Seed for the default context (ignored when ``ctx`` is provided).

    Notes:
        The checker only uses the public ``Signal`` surface, so it never reaches into engine
        internals and is valid for cheap CPU and expensive learned signals alike. A signal
        that *raises* (rather than recording a skip) is itself a contract violation and is
        reported as one — the checker never lets that exception escape.
    """
    violations: list[str] = []

    # 1. The spec must be a well-formed SignalSpec before anything else can be trusted.
    spec_violations = _check_spec(signal)
    violations.extend(spec_violations)
    if spec_violations:
        # Without a usable spec the behavioral checks below would just produce noise.
        return violations

    if trajectories is None:
        trajectories = _default_batch()
    if ctx is None:
        ctx = _default_context(seed=seed)

    # 2. fit() must run without error on a small batch (a no-op for stateless heuristics).
    # A raising fit()/score() is itself a contract violation; catch broadly and report it
    # rather than letting it escape.
    try:
        signal.fit(list(trajectories), ctx)
    except Exception as exc:
        violations.append(f"fit() raised {type(exc).__name__}: {exc}")
        return violations

    # 3. score() must run without error and return one TrajectoryScore per input, in order.
    try:
        scores = signal.score(list(trajectories), ctx)
    except Exception as exc:
        violations.append(f"score() raised {type(exc).__name__}: {exc}")
        return violations

    violations.extend(_check_scores(signal.spec, list(trajectories), scores))

    # 4. Determinism (Invariant 3): a deterministic signal must yield identical values on a
    #    second run over the same (batch, ctx, seed). Skip the check if the first run already
    #    produced a malformed result we couldn't line up.
    if signal.spec.deterministic and len(scores) == len(trajectories):
        violations.extend(_check_determinism(signal, list(trajectories), ctx, scores))

    return violations


def assert_signal_contract(
    signal: Signal,
    trajectories: Sequence[Trajectory] | None = None,
    *,
    ctx: SignalContext | None = None,
    seed: int = 0,
) -> None:
    """Assert ``signal`` honors the :class:`Signal` contract.

    A thin wrapper over :func:`check_signal_contract` for use in contributor tests: raises
    :class:`AssertionError` with the joined violations when the signal does not pass, and
    returns ``None`` when it does.
    """
    violations = check_signal_contract(signal, trajectories, ctx=ctx, seed=seed)
    if violations:
        joined = "\n  - ".join(violations)
        raise AssertionError(
            f"signal failed the Signal contract ({len(violations)} violation(s)):\n  - {joined}"
        )


# -- individual check groups ---------------------------------------------------------


def _check_spec(signal: Signal) -> list[str]:
    """Validate ``signal.spec`` is a well-formed :class:`SignalSpec`."""
    out: list[str] = []
    spec = getattr(signal, "spec", None)
    if not isinstance(spec, SignalSpec):
        return [f"spec is not a SignalSpec (got {type(spec).__name__})"]

    if not isinstance(spec.name, str) or not spec.name.strip():
        out.append("spec.name must be a non-empty string")
    if not isinstance(spec.version, str) or not spec.version.strip():
        out.append("spec.version must be a non-empty string")
    if not isinstance(spec.cost_tier, CostTier):
        out.append(f"spec.cost_tier must be a CostTier (got {type(spec.cost_tier).__name__})")
    if not isinstance(spec.requires, frozenset):
        out.append(f"spec.requires must be a frozenset (got {type(spec.requires).__name__})")
    if not isinstance(spec.produces_per_transition, bool):
        out.append("spec.produces_per_transition must be a bool")
    if not isinstance(spec.deterministic, bool):
        out.append("spec.deterministic must be a bool")
    if not isinstance(spec.description, str) or not spec.description.strip():
        out.append("spec.description must be a non-empty string")
    return out


def _check_scores(
    spec: SignalSpec,
    trajectories: Sequence[Trajectory],
    scores: object,
) -> list[str]:
    """Validate the shape and content of one ``score()`` result."""
    out: list[str] = []
    if not isinstance(scores, list):
        return [f"score() must return a list (got {type(scores).__name__})"]
    if len(scores) != len(trajectories):
        return [
            f"score() returned {len(scores)} score(s) for {len(trajectories)} "
            "trajectory(ies); it must return exactly one per input"
        ]

    for i, (traj, score) in enumerate(zip(trajectories, scores, strict=True)):
        if not isinstance(score, TrajectoryScore):
            out.append(f"score[{i}] is not a TrajectoryScore (got {type(score).__name__})")
            continue
        out.extend(_check_one_score(spec, i, traj, score))
    return out


def _check_one_score(
    spec: SignalSpec,
    i: int,
    traj: Trajectory,
    score: TrajectoryScore,
) -> list[str]:
    """Validate a single :class:`TrajectoryScore` against its input trajectory."""
    out: list[str] = []
    fp = traj.meta.fingerprint

    # Order / identity: the i-th score must be for the i-th trajectory.
    if score.trajectory_fingerprint != fp:
        out.append(
            f"score[{i}].trajectory_fingerprint {score.trajectory_fingerprint!r} does not "
            f"match input trajectory {fp!r}; scores must be returned in input order"
        )

    if score.signal != spec.name:
        out.append(f"score[{i}].signal {score.signal!r} does not match spec.name {spec.name!r}")

    if not isinstance(score.higher_is_better, bool):
        out.append(f"score[{i}].higher_is_better must be a bool")

    # A score is either a recorded skip (with a reason) or has a finite value.
    if score.skipped:
        if not score.skip_reason:
            out.append(f"score[{i}] is skipped but has no skip_reason")
        # A skip carries no usable value or per-transition array; nothing more to check.
        return out

    if not isinstance(score.value, float) or not math.isfinite(score.value):
        out.append(
            f"score[{i}].value must be a finite float when not skipped (got {score.value!r}); "
            "a non-finite value must set skipped=True with a skip_reason"
        )

    out.extend(_check_per_transition(spec, i, traj, score))
    return out


def _check_per_transition(
    spec: SignalSpec,
    i: int,
    traj: Trajectory,
    score: TrajectoryScore,
) -> list[str]:
    """Validate ``per_transition`` against ``produces_per_transition`` and trajectory length."""
    out: list[str] = []
    if spec.produces_per_transition:
        if score.per_transition is None:
            out.append(
                f"score[{i}].per_transition is None but spec.produces_per_transition is True"
            )
            return out
        arr = np.asarray(score.per_transition)
        if arr.ndim != 1:
            out.append(f"score[{i}].per_transition must be 1-D (T,); got shape {arr.shape}")
        elif arr.shape[0] != traj.num_steps:
            out.append(
                f"score[{i}].per_transition has length {arr.shape[0]} but the trajectory has "
                f"{traj.num_steps} steps; per-transition must be (T,)"
            )
    elif score.per_transition is not None:
        out.append(f"score[{i}].per_transition is set but spec.produces_per_transition is False")
    return out


def _check_determinism(
    signal: Signal,
    trajectories: Sequence[Trajectory],
    ctx: SignalContext,
    first: Sequence[TrajectoryScore],
) -> list[str]:
    """Re-run a deterministic signal and assert identical per-trajectory values."""
    out: list[str] = []
    try:
        second = signal.score(list(trajectories), ctx)
    except Exception as exc:
        return [f"score() raised on a repeated run despite deterministic=True: {exc}"]

    if len(second) != len(first):
        return [
            "spec.deterministic is True but two runs returned different score counts "
            f"({len(first)} vs {len(second)})"
        ]

    for i, (a, b) in enumerate(zip(first, second, strict=True)):
        if a.skipped != b.skipped:
            out.append(
                f"spec.deterministic is True but score[{i}] skipped flag differs between runs"
            )
            continue
        if a.skipped:
            continue
        if not _values_equal(a.value, b.value):
            out.append(
                f"spec.deterministic is True but score[{i}].value differs between two runs "
                f"on identical inputs ({a.value!r} vs {b.value!r}); randomness must be seeded "
                "(Invariant 3)"
            )
    return out


def _values_equal(a: float, b: float) -> bool:
    """Exact equality, treating two NaNs as equal (a skipped value should not reach here)."""
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


# -- default synthetic batch + context (self-contained; no test deps) ----------------


def _default_batch() -> list[Trajectory]:
    """Build a tiny deterministic batch of toy trajectories for the contract checks.

    A 2-DoF embodiment with timestamps, an action, a proprio state, and a reward — enough to
    exercise the action/state/time/reward feature paths. Distinct ``scale`` per episode gives
    each a distinct fingerprint so the order/identity check is meaningful. This mirrors
    ``tests/synthetic.py`` but is duplicated here so the checker is usable by contributors
    without importing the test package.
    """
    from robocurate.trajectory import (
        EmbodimentSpec,
        FeatureRole,
        FeatureSpec,
        InMemoryFeatureStore,
        SuccessLabel,
        Trajectory,
        TrajectoryMeta,
        fingerprint_arrays,
    )

    embodiment = EmbodimentSpec(
        embodiment_id="contract_toy2dof",
        control_hz=10.0,
        features=(
            FeatureSpec("timestamp", FeatureRole.TIME, shape=(), dtype="float32", units="s"),
            FeatureSpec(
                "action",
                FeatureRole.ACTION,
                shape=(2,),
                dtype="float32",
                units="normalized[-1,1]",
                names=("x", "y"),
            ),
            FeatureSpec(
                "observation.state",
                FeatureRole.PROPRIO,
                shape=(2,),
                dtype="float32",
                units="m",
                names=("px", "py"),
            ),
            FeatureSpec("reward", FeatureRole.REWARD, shape=(), dtype="float32", units=None),
        ),
    )

    num_steps = 8
    out: list[Trajectory] = []
    for episode_index in range(3):
        t = np.arange(num_steps, dtype=np.float32) / 10.0
        phase = float(episode_index)
        scale = 1.0 + episode_index
        action = (scale * np.stack([np.sin(t + phase), np.cos(t + phase)], axis=-1)).astype(
            np.float32
        )
        state = np.cumsum(action, axis=0).astype(np.float32)
        reward = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
        columns: dict[str, Array] = {
            "timestamp": t,
            "action": action,
            "observation.state": state,
            "reward": reward,
        }
        meta = TrajectoryMeta(
            source_dataset_id="contract/toy",
            episode_index=episode_index,
            embodiment=embodiment,
            fingerprint=fingerprint_arrays(columns),
            num_steps=num_steps,
            source_format="contract_v0",
            success=SuccessLabel(value=True, source="contract"),
        )
        out.append(Trajectory(meta, InMemoryFeatureStore(columns)))
    return out


def _default_context(*, seed: int) -> SignalContext:
    """Build a minimal CPU :class:`SignalContext` for the contract checks."""
    import logging

    from robocurate.metadata import DatasetFingerprint, DatasetMeta, ResourceProbe
    from robocurate.signals.base import InMemoryCache

    fingerprint = DatasetFingerprint(
        dataset_id="contract/toy",
        source_format="contract_v0",
        content_hash="0" * 64,
        num_episodes=0,
    )
    dataset_meta = DatasetMeta(
        fingerprint=fingerprint,
        embodiment_ids=("contract_toy2dof",),
        feature_keys=("timestamp", "action", "observation.state", "reward"),
    )
    return SignalContext(
        seed=seed,
        device="cpu",
        cache=InMemoryCache(),
        resources=ResourceProbe(),
        dataset_meta=dataset_meta,
        logger=logging.getLogger("robocurate.signals.contract"),
    )


__all__ = ["assert_signal_contract", "check_signal_contract"]
