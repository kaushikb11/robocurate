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
from robocurate.adapters.hdf5 import GenericHDF5Reader, HDF5Schema
from robocurate.adapters.zarr import ZarrReader, ZarrSchema
from robocurate.benchmark import (
    BenchmarkResult,
    BenchmarkSpec,
    Leaderboard,
    LeaderboardEntry,
    ResolvedSubmission,
    build_spec,
    resolve_submission,
    run_submission,
)
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
from robocurate.profile import ProfileReport, dataset_profile
from robocurate.scorecard import Scorecard
from robocurate.signals import assert_signal_contract, check_signal_contract
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
    "BenchmarkResult",
    "BenchmarkSpec",
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
    "GenericHDF5Reader",
    "HDF5Schema",
    "InMemoryFeatureStore",
    "Leaderboard",
    "LeaderboardEntry",
    "ProfileReport",
    "ResolvedSubmission",
    "ResourceProbe",
    "ScoreMatrix",
    "Scorecard",
    "SelectionMode",
    "SuccessLabel",
    "Trajectory",
    "TrajectoryMeta",
    "WeightedSum",
    "ZarrReader",
    "ZarrSchema",
    "__version__",
    "assert_signal_contract",
    "build_spec",
    "check_signal_contract",
    "dataset_profile",
    "fingerprint_arrays",
    "resolve_submission",
    "run_submission",
    "signals",
]
