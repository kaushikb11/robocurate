"""The ``Dataset`` convenience facade used by the 5-line quickstart.

``Dataset`` is a thin, friendly wrapper over the adapter layer so the common case reads
cleanly::

    from robocurate import Dataset, Curator, signals
    ds = Dataset.from_lerobot("lerobot/aloha_sim_insertion")

Power users can drop to :class:`~robocurate.adapters.lerobot.LeRobotReader` directly. The
facade intentionally exposes only read access — there is no ``Dataset.write`` — keeping the
read-only guarantee visible at the top of the API as well (Invariant 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robocurate.adapters.base import LeRobotVersion
from robocurate.adapters.lerobot import LeRobotReader

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from robocurate.adapters.base import DatasetReader
    from robocurate.metadata import DatasetFingerprint, DatasetMeta
    from robocurate.trajectory import Trajectory


class Dataset:
    """A read-only handle to a source dataset, wrapping a :class:`DatasetReader`.

    Delegates the full reader protocol, so it can be passed straight to
    :meth:`~robocurate.curator.Curator.run`. It exposes only read access — there is no
    ``write`` method — so the read-only guarantee stays visible at the top of the API.
    """

    def __init__(self, reader: DatasetReader) -> None:
        self._reader = reader
        self.meta: DatasetMeta = reader.meta

    @classmethod
    def from_lerobot(cls, path: str | Path, *, version: LeRobotVersion | None = None) -> Dataset:
        """Open a local LeRobotDataset directory for reading (v2.1 or v3.0).

        The on-disk format is auto-detected from ``meta/info.json`` unless ``version`` is given.

        Note: remote Hub resolution (``"lerobot/<name>"`` -> local cache) is a later
        addition; today ``path`` must be a local dataset directory.
        """
        from robocurate.adapters.lerobot_v3 import LeRobotReaderV3, detect_lerobot_version

        resolved = version if version is not None else detect_lerobot_version(path)
        if resolved is LeRobotVersion.V3:
            return cls(LeRobotReaderV3(path))
        return cls(LeRobotReader(path, version=resolved))

    @property
    def reader(self) -> DatasetReader:
        """The underlying read-only :class:`DatasetReader`."""
        return self._reader

    @property
    def root(self) -> Path | None:
        """The source directory, when the underlying reader is path-backed (else ``None``).

        Surfaced so the writer's source-overlap guard still applies when a ``Dataset`` is
        passed to the curator rather than a raw reader.
        """
        return getattr(self._reader, "root", None)

    # -- DatasetReader protocol (delegated) ------------------------------------------

    def __len__(self) -> int:
        return len(self._reader)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._reader)

    def read_episode(self, index: int) -> Trajectory:
        return self._reader.read_episode(index)

    def fingerprint(self) -> DatasetFingerprint:
        return self._reader.fingerprint()


__all__ = ["Dataset"]
