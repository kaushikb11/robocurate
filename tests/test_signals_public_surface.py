"""Guard against drift between the declared signal entry points and the public surface.

Every signal advertised under ``[project.entry-points."robocurate.signals"]`` in
``pyproject.toml`` must also be re-exported as an attribute on the :mod:`robocurate.signals`
module, so that ``signals.PathEfficiency()`` works for every built-in exactly as the
quickstart promises. This caught a real bug where ``PathEfficiency`` and
``SpectralSmoothness`` were registered as entry points but missing from the module's
imports and ``__all__``. The test is CPU-only and imports no optional dependency: it reads
the entry-point *target strings* (``module:ClassName``) statically and checks the class name
is exposed, rather than loading each signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10 has no stdlib tomllib; tomli is the drop-in backport.
    import tomli as tomllib

from robocurate import signals

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
_ENTRY_POINT_GROUP = "robocurate.signals"


def _declared_signal_class_names() -> dict[str, str]:
    """Map entry-point name -> exported class name, parsed from pyproject.toml.

    Each entry-point value has the form ``"module.path:ClassName"``; we take the part after
    the colon as the public attribute expected on the ``robocurate.signals`` module.
    """
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    entry_points = data["project"]["entry-points"][_ENTRY_POINT_GROUP]
    return {name: target.split(":", 1)[1] for name, target in entry_points.items()}


def test_pyproject_declares_signal_entry_points() -> None:
    declared = _declared_signal_class_names()
    # Sanity: the v1 suite is at least the eight original signals (guards an empty/garbled parse;
    # this is a floor, not an exact count, so adding a signal doesn't require editing this test).
    assert len(declared) >= 8


def test_every_entry_point_signal_is_exported() -> None:
    """Every declared signal class is importable from ``robocurate.signals`` and in ``__all__``."""
    missing_attr: list[str] = []
    missing_all: list[str] = []
    for ep_name, class_name in _declared_signal_class_names().items():
        if not hasattr(signals, class_name):
            missing_attr.append(f"{ep_name} -> {class_name}")
        elif class_name not in signals.__all__:
            missing_all.append(class_name)
    assert not missing_attr, f"signals module missing entry-point classes: {missing_attr}"
    assert not missing_all, f"classes not in robocurate.signals.__all__: {missing_all}"


def test_signal_all_matches_module_namespace() -> None:
    """Every name in ``__all__`` resolves to a real attribute (no dangling exports)."""
    module = sys.modules["robocurate.signals"]
    dangling = [name for name in signals.__all__ if not hasattr(module, name)]
    assert not dangling, f"names in __all__ with no attribute: {dangling}"
