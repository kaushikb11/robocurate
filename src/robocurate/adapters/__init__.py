"""Dataset adapters: convert source formats to/from canonical trajectories.

The two protocols (:class:`DatasetReader`, :class:`DatasetWriter`) and the read-only
guarantee live in :mod:`robocurate.adapters.base`. LeRobotDataset is the first concrete
adapter; RLDS and raw sim-output adapters slot in later behind the same protocols.
"""

from __future__ import annotations

from robocurate.adapters.base import (
    DatasetReader,
    DatasetWriter,
    LeRobotVersion,
    SourceWriteError,
    ValidationError,
    ValidationReport,
    WriteReceipt,
)
from robocurate.adapters.hdf5 import GenericHDF5Reader, HDF5Schema
from robocurate.adapters.lerobot import LeRobotReader, LeRobotWriter
from robocurate.adapters.lerobot_v3 import LeRobotReaderV3
from robocurate.adapters.lerobot_v3_writer import LeRobotWriterV3
from robocurate.adapters.maniskill_demos import ManiSkillDemoReader
from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.adapters.rlds import RLDSReader
from robocurate.adapters.robomimic import RoboMimicReader
from robocurate.adapters.zarr import ZarrReader, ZarrSchema

__all__ = [
    "DatasetReader",
    "DatasetWriter",
    "GenericHDF5Reader",
    "HDF5Schema",
    "InMemoryDatasetReader",
    "LeRobotReader",
    "LeRobotReaderV3",
    "LeRobotVersion",
    "LeRobotWriter",
    "LeRobotWriterV3",
    "ManiSkillDemoReader",
    "RLDSReader",
    "RoboMimicReader",
    "SourceWriteError",
    "ValidationError",
    "ValidationReport",
    "WriteReceipt",
    "ZarrReader",
    "ZarrSchema",
]
