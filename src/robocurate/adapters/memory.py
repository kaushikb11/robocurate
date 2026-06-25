"""An in-memory :class:`~robocurate.adapters.base.DatasetReader` over a list of trajectories.

A read-only adapter that wraps trajectories already in memory — used by the synthetic
experiment-dataset generator and handy for building small datasets programmatically (tests,
demos, the Modal worker). Like every reader it has no write method, so the source can never
be mutated (invariant 1).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from robocurate.metadata import DatasetFingerprint, DatasetMeta

if TYPE_CHECKING:
    from robocurate.trajectory import Trajectory


class InMemoryDatasetReader:
    """A :class:`DatasetReader` backed by an in-memory sequence of trajectories."""

    def __init__(
        self, trajectories: Sequence[Trajectory], *, dataset_id: str = "in_memory"
    ) -> None:
        self._trajectories = list(trajectories)
        self.dataset_id = dataset_id
        self.meta = self._build_meta()

    def __len__(self) -> int:
        return len(self._trajectories)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._trajectories)

    def read_episode(self, index: int) -> Trajectory:
        try:
            return self._trajectories[index]
        except IndexError as exc:
            raise IndexError(f"episode {index} out of range (0..{len(self) - 1})") from exc

    def fingerprint(self) -> DatasetFingerprint:
        roll = hashlib.sha256()
        for fp in sorted(t.meta.fingerprint for t in self._trajectories):
            roll.update(fp.encode("utf-8"))
        source_format = (
            self._trajectories[0].meta.source_format if self._trajectories else "in_memory"
        )
        return DatasetFingerprint(
            dataset_id=self.dataset_id,
            source_format=source_format,
            content_hash=roll.hexdigest(),
            num_episodes=len(self._trajectories),
        )

    def _build_meta(self) -> DatasetMeta:
        if not self._trajectories:
            return DatasetMeta(fingerprint=self.fingerprint(), embodiment_ids=(), feature_keys=())
        first = self._trajectories[0]
        embodiment_ids = tuple({t.meta.embodiment.embodiment_id for t in self._trajectories})
        return DatasetMeta(
            fingerprint=self.fingerprint(),
            embodiment_ids=embodiment_ids,
            feature_keys=tuple(s.key for s in first.embodiment.features),
        )


__all__ = ["InMemoryDatasetReader"]
