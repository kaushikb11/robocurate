"""Tests for the optional-dependency machinery (no ML extra required).

These verify the registry stays robust when a signal's optional dependency is missing: a
failed entry-point load is recorded (not raised), discovery of other signals keeps working,
and requesting the unavailable signal yields an actionable install message.
"""

from __future__ import annotations

import pytest

from robocurate import signals
from robocurate.signals import registry


def test_core_signals_discoverable() -> None:
    # The four cheap signals are always discoverable, with or without the ML extra.
    names = set(signals.available())
    assert {"jerk", "action_noise", "redundancy", "sim_physics_validity"} <= names


def test_failed_load_is_recorded_not_raised() -> None:
    # Simulate an entry point whose dependency is missing (e.g. torch not installed).
    registry._LOAD_ERRORS["fake_ml_signal"] = "ImportError: No module named 'torch'"
    try:
        # Discovery of the working signals is unaffected.
        assert "jerk" in signals.available()
        assert "fake_ml_signal" in signals.unavailable()
        # Requesting it gives an actionable message mentioning how to install it.
        with pytest.raises(KeyError, match=r"could not be imported|install"):
            signals.get("fake_ml_signal")
    finally:
        registry._LOAD_ERRORS.pop("fake_ml_signal", None)


def test_unknown_signal_distinguished_from_unavailable() -> None:
    with pytest.raises(KeyError, match="no signal registered"):
        signals.get("totally_made_up_signal_xyz")
