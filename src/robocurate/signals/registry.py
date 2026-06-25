"""Signal plugin registry.

Signals are discovered through Python **entry points** in the ``robocurate.signals`` group,
so the community ships a signal as an installable package and it appears here without any
edit to the core (Invariant 4). Built-in signals will register the same way.
Signals can also be registered programmatically via :func:`register` (used by tests and for
ad-hoc/custom signals).

This skeleton ships **no** real signal; the registry is the seam they plug into.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robocurate.signals.base import Signal

# A factory takes no required arguments and returns a configured Signal instance. Signals
# with constructor parameters expose sensible defaults so ``Factory()`` always works.
SignalFactory = Callable[[], "Signal"]

ENTRY_POINT_GROUP = "robocurate.signals"

_REGISTRY: dict[str, SignalFactory] = {}
# Entry points that failed to import (e.g. a missing optional dependency): name -> error
# message. Recorded rather than raised so one unloadable signal never breaks discovery of
# the others.
_LOAD_ERRORS: dict[str, str] = {}
_ENTRY_POINTS_LOADED = False


def register(name: str, factory: SignalFactory, *, overwrite: bool = False) -> None:
    """Register a signal factory under ``name``.

    Args:
        name: The signal name (should match the produced signal's ``spec.name``).
        factory: A zero-argument callable returning a configured :class:`Signal`.
        overwrite: If ``False`` (default), registering an existing name raises; set ``True``
            to replace (useful in tests).

    Raises:
        ValueError: If ``name`` is already registered and ``overwrite`` is ``False``.
    """
    if not overwrite and name in _REGISTRY:
        raise ValueError(f"signal {name!r} is already registered")
    _REGISTRY[name] = factory


def unregister(name: str) -> None:
    """Remove a signal from the registry if present (no error if absent)."""
    _REGISTRY.pop(name, None)


def _load_entry_points() -> None:
    """Discover signals advertised via the ``robocurate.signals`` entry-point group.

    Entry points are loaded once and cached. A signal package advertises::

        [project.entry-points."robocurate.signals"]
        jerk = "my_pkg.signals:Jerk"

    where the target is a :class:`Signal` factory (class or callable).
    """
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    eps = metadata.entry_points()
    # importlib.metadata's API differs across versions; select() exists on 3.10+.
    selected = eps.select(group=ENTRY_POINT_GROUP)
    for ep in selected:
        if ep.name in _REGISTRY:
            continue
        try:
            factory: SignalFactory = ep.load()
        except Exception as exc:
            # Most commonly a missing optional dependency (the heavy ML extra isn't
            # installed). Record it so it surfaces only if the signal is actually requested.
            _LOAD_ERRORS[ep.name] = f"{type(exc).__name__}: {exc}"
            continue
        _REGISTRY[ep.name] = factory
    _ENTRY_POINTS_LOADED = True


def available() -> tuple[str, ...]:
    """Return the sorted names of all *loadable* registered signals (loads entry points)."""
    _load_entry_points()
    return tuple(sorted(_REGISTRY))


def unavailable() -> dict[str, str]:
    """Return signals whose entry point failed to import, mapped to the error.

    Typically a signal whose optional ML extra is not installed. These are discoverable by
    name but cannot be instantiated until their dependency is available.
    """
    _load_entry_points()
    return dict(_LOAD_ERRORS)


def get(name: str, **params: Any) -> Signal:
    """Instantiate and return the signal registered under ``name``.

    Any keyword ``params`` are passed to the signal's factory (its class), so a configured
    signal can be built by name — e.g. ``get("cupid", mode="self_influence")``. This is what
    lets a serializable experiment config name signals with parameters.

    Raises:
        KeyError: If no signal is registered under ``name``. If the name *is* a known signal
            that failed to import (e.g. a missing optional dependency), the error explains
            that and how to install it, rather than just "unknown".
    """
    _load_entry_points()
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        if name in _LOAD_ERRORS:
            raise KeyError(
                f"signal {name!r} is registered but could not be imported "
                f"({_LOAD_ERRORS[name]}). It likely needs an optional dependency; try "
                f"installing it, e.g. `uv pip install 'robocurate[{name}]'` or "
                "`robocurate[all]`."
            ) from exc
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"no signal registered as {name!r}; available: {known}") from exc
    return factory(**params)


def clear() -> None:
    """Reset the registry to empty and forget entry-point loading (test-support hook)."""
    global _ENTRY_POINTS_LOADED
    _REGISTRY.clear()
    _LOAD_ERRORS.clear()
    _ENTRY_POINTS_LOADED = False
