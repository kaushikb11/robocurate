"""RoboCurate — data curation for robot-learning / embodied-AI datasets.

This is the public package surface. The heavy lifting lives in submodules; the names
re-exported here are the stable entry points users are expected to import — chiefly
:class:`Dataset`, :class:`Curator`, :class:`Budget`, and the :mod:`signals` namespace.

The core works in a NumPy-exchange canonical :class:`Trajectory`; adapters convert
LeRobotDataset / RLDS / sim output into it, and the engine never depends on a source
format. The cheap heuristic signals run on a no-GPU laptop (NumPy + PyArrow only);
learned signals (Demo-SCORE, CUPID) live behind optional extras.
"""

from __future__ import annotations

from robocurate import signals
from robocurate.curator import (
    Budget,
    Combiner,
    CurationConfig,
    CurationResult,
    Curator,
    GateConfig,
    ScoreMatrix,
    SelectionMode,
    WeightedSum,
)
from robocurate.dataset import Dataset
from robocurate.metadata import DatasetFingerprint, DatasetMeta, ResourceProbe
from robocurate.scorecard import Scorecard
from robocurate.trajectory import (
    Array,
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    FeatureStore,
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)

__version__ = "0.0.1"

__all__ = [
    "Array",
    "Budget",
    "Combiner",
    "CurationConfig",
    "CurationResult",
    "Curator",
    "Dataset",
    "DatasetFingerprint",
    "DatasetMeta",
    "EmbodimentSpec",
    "FeatureRole",
    "FeatureSpec",
    "FeatureStore",
    "GateConfig",
    "InMemoryFeatureStore",
    "ResourceProbe",
    "ScoreMatrix",
    "Scorecard",
    "SelectionMode",
    "SuccessLabel",
    "Trajectory",
    "TrajectoryMeta",
    "WeightedSum",
    "__version__",
    "fingerprint_arrays",
    "signals",
]
