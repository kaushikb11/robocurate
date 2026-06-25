"""Core-only registration / without-extra UX for the CPU image-quality signals.

No marker and no decoding here: this runs in the lightweight core-only CI job. It asserts the
three image signals are *discoverable* (registered via entry points) and that, when the
``video`` extra (PyAV) is genuinely absent, instantiating one raises a clear error pointing at
``robocurate[video]`` — mirroring the demo_score without-torch UX. When ``av`` *is* installed
(the full+video env) the raise check is skipped, exactly like the demo_score test guards on
torch.
"""

from __future__ import annotations

import importlib.util

import pytest

from robocurate import signals

_IMAGE_SIGNALS = ("image_blur", "visual_stall", "visual_diversity")
_HAVE_AV = importlib.util.find_spec("av") is not None


@pytest.mark.parametrize("name", _IMAGE_SIGNALS)
def test_image_signal_is_registered(name: str) -> None:
    # Discoverable with or without the video extra: the class loads fine; only *constructing*
    # it needs PyAV (fail-fast in __init__), so the entry point never fails to import.
    assert name in signals.available()


@pytest.mark.parametrize("name", _IMAGE_SIGNALS)
@pytest.mark.skipif(_HAVE_AV, reason="video extra (av) installed; without-extra raise can't fire")
def test_image_signal_without_extra_raises_clear_error(name: str) -> None:
    with pytest.raises(ImportError, match=r"robocurate\[video\]|PyAV"):
        signals.get(name)


@pytest.mark.skipif(not _HAVE_AV, reason="requires the video extra (av) to construct the signal")
@pytest.mark.parametrize("name", _IMAGE_SIGNALS)
def test_image_signal_constructs_with_extra(name: str) -> None:
    sig = signals.get(name)
    assert sig.spec.name == name
    assert "image" in sig.spec.requires
