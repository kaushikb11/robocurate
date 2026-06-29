"""Tests for the Signal contract-checker (:mod:`robocurate.signals.contract`).

These exercise the checker itself: it should pass real built-in signals, pass the worked
example, and produce specific, non-empty violations for deliberately broken signals.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from robocurate import signals
from robocurate.signals import (
    assert_signal_contract,
    check_signal_contract,
)
from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)

if TYPE_CHECKING:
    from robocurate.trajectory import Trajectory


# -- the checker passes real, well-behaved signals -----------------------------------


@pytest.mark.parametrize("name", ["jerk", "action_noise"])
def test_builtin_signal_passes_contract(name: str) -> None:
    sig = signals.get(name)
    assert check_signal_contract(sig) == []


def test_builtin_signal_passes_via_assert() -> None:
    # The thin assert wrapper should not raise for a conforming signal.
    assert_signal_contract(signals.get("jerk"))


def test_check_is_deterministic_across_calls() -> None:
    # Same signal, same seed -> same (empty) result, every time.
    sig = signals.get("action_noise")
    first = check_signal_contract(sig, seed=7)
    second = check_signal_contract(sig, seed=7)
    assert first == second == []


# -- the worked example passes the contract ------------------------------------------


def _load_action_range() -> type:
    """Import ``ActionRange`` from ``examples/custom_signal.py`` by file path.

    ``examples/`` is not an installed package, so we load the module directly rather than via
    a regular import. This keeps the example a runnable script *and* test-covered.
    """
    example_path = Path(__file__).resolve().parents[1] / "examples" / "custom_signal.py"
    spec = importlib.util.spec_from_file_location("custom_signal_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActionRange  # type: ignore[no-any-return]


def test_example_action_range_passes_contract() -> None:
    action_range_cls = _load_action_range()
    assert check_signal_contract(action_range_cls()) == []


# -- deliberately broken signals must produce violations -----------------------------


def _toy_spec(*, deterministic: bool = True, produces_per_transition: bool = False) -> SignalSpec:
    return SignalSpec(
        name="broken",
        version="0.1.0",
        cost_tier=CostTier.TIER0_CPU,
        requires=frozenset({"action"}),
        produces_per_transition=produces_per_transition,
        deterministic=deterministic,
        description="A deliberately broken signal for contract tests.",
    )


class WrongCountSignal:
    """Returns fewer scores than inputs — violates the one-score-per-trajectory rule."""

    def __init__(self) -> None:
        self.spec = _toy_spec()

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        # Drop the last score on purpose.
        return [
            TrajectoryScore(
                signal=self.spec.name,
                trajectory_fingerprint=traj.meta.fingerprint,
                value=0.0,
            )
            for traj in batch[:-1]
        ]


class NaNWithoutSkipSignal:
    """Emits a NaN value without setting ``skipped`` — a silent non-finite value."""

    def __init__(self) -> None:
        self.spec = _toy_spec()

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [
            TrajectoryScore(
                signal=self.spec.name,
                trajectory_fingerprint=traj.meta.fingerprint,
                value=float("nan"),  # not finite, but skipped is False
            )
            for traj in batch
        ]


class NonDeterministicSignal:
    """Declares ``deterministic=True`` but returns a different value each call."""

    def __init__(self) -> None:
        self.spec = _toy_spec()
        self._rng = np.random.default_rng()  # unseeded on purpose

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [
            TrajectoryScore(
                signal=self.spec.name,
                trajectory_fingerprint=traj.meta.fingerprint,
                value=float(self._rng.random()),
            )
            for traj in batch
        ]


class WrongFingerprintSignal:
    """Returns scores in the wrong order (fingerprints don't line up with inputs)."""

    def __init__(self) -> None:
        self.spec = _toy_spec()

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [
            TrajectoryScore(
                signal=self.spec.name,
                trajectory_fingerprint="not-a-real-fingerprint",
                value=0.0,
            )
            for _ in batch
        ]


class MissingPerTransitionSignal:
    """Declares ``produces_per_transition`` but never emits a per-transition array."""

    def __init__(self) -> None:
        self.spec = _toy_spec(produces_per_transition=True)

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [
            TrajectoryScore(
                signal=self.spec.name,
                trajectory_fingerprint=traj.meta.fingerprint,
                value=0.0,
                per_transition=None,
            )
            for traj in batch
        ]


class BadSpecSignal:
    """Has an empty name and a non-frozenset ``requires`` — a malformed spec."""

    def __init__(self) -> None:
        self.spec = SignalSpec(
            name="   ",
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset(),
            description="",  # empty description, too
        )

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return []


class RaisingScoreSignal:
    """Raises from ``score`` instead of recording a skip — a contract violation."""

    def __init__(self) -> None:
        self.spec = _toy_spec()

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        raise RuntimeError("kaboom")


@pytest.mark.parametrize(
    ("factory", "needle"),
    [
        (WrongCountSignal, "exactly one per input"),
        (NaNWithoutSkipSignal, "finite float when not skipped"),
        (NonDeterministicSignal, "deterministic"),
        (WrongFingerprintSignal, "input order"),
        (MissingPerTransitionSignal, "produces_per_transition"),
        (BadSpecSignal, "spec.name"),
        (RaisingScoreSignal, "score() raised"),
    ],
)
def test_broken_signal_is_flagged(factory: type, needle: str) -> None:
    violations = check_signal_contract(factory())
    assert violations, f"{factory.__name__} should have produced at least one violation"
    assert any(needle in v for v in violations), (
        f"expected a violation containing {needle!r}, got: {violations}"
    )


def test_assert_signal_contract_raises_on_broken_signal() -> None:
    with pytest.raises(AssertionError, match="failed the Signal contract"):
        assert_signal_contract(WrongCountSignal())
