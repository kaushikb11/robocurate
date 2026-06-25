"""Quality signals: the ``Signal`` protocol, supporting types, and the plugin registry.

The public surface here is the *contract* (the :class:`Signal` protocol and its supporting
types — :class:`SignalContext`, :class:`TrajectoryScore`, :class:`FeatureRequirement`,
caches, cost tiers), the entry-point *registry* (:func:`get`, :func:`available`,
:func:`register`), and the built-in signal classes.

Nine built-in signals ship today: :class:`Jerk`, :class:`ActionNoise`,
:class:`PathEfficiency`, :class:`SpectralSmoothness`, :class:`Redundancy`,
:class:`StructuralValidity`, :class:`SimPhysicsValidity`, :class:`DemoScore`, and
:class:`Cupid`. Each one registers through the ``robocurate.signals`` entry-point group
declared in ``pyproject.toml`` — exactly the mechanism a third-party signal uses — so adding a
signal never touches the core.

Usage sketch::

    from robocurate import signals
    sig = signals.get("jerk")          # instantiate a registered signal by name
    print(signals.available())          # list registered signal names
    sig = signals.PathEfficiency()      # or construct a built-in directly
"""

from __future__ import annotations

from robocurate.signals.action_noise import ActionNoise
from robocurate.signals.base import (
    REQUIRES_ENCODER,
    REQUIRES_GPU,
    REQUIRES_SIM_STATE,
    CacheHandle,
    CostTier,
    FeatureRequirement,
    InMemoryCache,
    NamespacedCache,
    Signal,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.signals.cupid import Cupid
from robocurate.signals.demo_score import DemoScore
from robocurate.signals.jerk import Jerk
from robocurate.signals.path_efficiency import PathEfficiency
from robocurate.signals.redundancy import Redundancy, statistical_embedding
from robocurate.signals.registry import (
    available,
    get,
    register,
    unavailable,
    unregister,
)
from robocurate.signals.sim_validity import SimPhysicsValidity
from robocurate.signals.spectral_smoothness import SpectralSmoothness
from robocurate.signals.structural_validity import StructuralValidity

__all__ = [
    "REQUIRES_ENCODER",
    "REQUIRES_GPU",
    "REQUIRES_SIM_STATE",
    "ActionNoise",
    "CacheHandle",
    "CostTier",
    "Cupid",
    "DemoScore",
    "FeatureRequirement",
    "InMemoryCache",
    "Jerk",
    "NamespacedCache",
    "PathEfficiency",
    "Redundancy",
    "Signal",
    "SignalContext",
    "SignalSpec",
    "SimPhysicsValidity",
    "SpectralSmoothness",
    "StructuralValidity",
    "TrajectoryScore",
    "available",
    "get",
    "register",
    "statistical_embedding",
    "unavailable",
    "unregister",
]
