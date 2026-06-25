"""Contract tests for the Signal protocol and the plugin registry."""

from __future__ import annotations

import numpy as np
import pytest

from robocurate import signals
from robocurate.signals.base import Signal
from robocurate.trajectory import InMemoryFeatureStore as _Store
from robocurate.trajectory import Trajectory
from tests.synthetic import (
    FakeActionMagnitudeSignal,
    make_signal_context,
    make_trajectory,
)


def test_fake_signal_satisfies_protocol() -> None:
    sig = FakeActionMagnitudeSignal()
    # runtime_checkable Protocol: the fake must structurally satisfy Signal.
    assert isinstance(sig, Signal)


def test_fit_then_score_seam() -> None:
    sig = FakeActionMagnitudeSignal()
    ctx = make_signal_context()
    batch = [make_trajectory(i) for i in range(3)]

    sig.fit(batch, ctx)
    assert sig.fit_calls == 1
    assert ctx.cache.get("fitted") is True

    scores = sig.score(batch, ctx)
    assert len(scores) == len(batch)  # one score per input, in order
    for traj, score in zip(batch, scores, strict=True):
        assert score.trajectory_fingerprint == traj.meta.fingerprint
        assert score.signal == sig.spec.name
        assert not score.skipped
        assert score.higher_is_better is False
        assert score.per_transition is not None
        assert score.per_transition.shape == (traj.num_steps,)


def test_score_is_deterministic() -> None:
    sig = FakeActionMagnitudeSignal()
    ctx = make_signal_context(seed=123)
    batch = [make_trajectory(i) for i in range(4)]

    first = [s.value for s in sig.score(batch, ctx)]
    second = [s.value for s in sig.score(batch, ctx)]
    assert first == second  # byte-identical given same inputs + seed


def test_unmet_feature_is_recorded_as_skip_not_error() -> None:
    sig = FakeActionMagnitudeSignal()
    ctx = make_signal_context()
    # A trajectory with no action feature.
    store = _Store({"reward": np.zeros((3,), dtype=np.float32)})
    traj = Trajectory(make_trajectory(0).meta, store)

    [score] = sig.score([traj], ctx)
    assert score.skipped is True
    assert score.skip_reason == "no action feature"
    assert np.isnan(score.value)  # skip carries NaN, never a real number


def test_registry_register_get_available_roundtrip() -> None:
    signals.unregister("fake_action_magnitude")
    signals.register("fake_action_magnitude", FakeActionMagnitudeSignal)
    try:
        assert "fake_action_magnitude" in signals.available()
        sig = signals.get("fake_action_magnitude")
        assert sig.spec.name == "fake_action_magnitude"
    finally:
        signals.unregister("fake_action_magnitude")


def test_registry_duplicate_register_raises() -> None:
    signals.register("dup_signal", FakeActionMagnitudeSignal)
    try:
        with pytest.raises(ValueError, match="already registered"):
            signals.register("dup_signal", FakeActionMagnitudeSignal)
    finally:
        signals.unregister("dup_signal")


def test_registry_unknown_signal_lists_available() -> None:
    with pytest.raises(KeyError, match="no signal registered"):
        signals.get("does_not_exist_xyz")
