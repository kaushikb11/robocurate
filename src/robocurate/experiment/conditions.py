"""Experiment conditions (arms) and a read-only subset view over a dataset.

Each experiment arm is a named training set defined by a subset of the source episodes:

* ``FULL`` — train on everything.
* ``CURATED`` — our selection at the target budget.
* ``EQUAL_N_RANDOM`` — a random subset of the *same size* as ``CURATED``
  (Invariant 5: the dataset-size confound, the first thing reviewers attack).
* ``RANDOM_FILTER`` — a second, independent random subset of that size (a sanity control).
* ``ABLATION`` — one arm per signal: curate using only that signal, to isolate its
  contribution.

A :class:`SubsetReader` exposes a chosen subset of a source reader as a normal, read-only
:class:`~robocurate.adapters.base.DatasetReader`, so a policy trains on it like any dataset
and the source is never mutated.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from robocurate.metadata import DatasetFingerprint, DatasetMeta

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader
    from robocurate.trajectory import Trajectory


class Condition(Enum):
    """The kind of training set an arm represents."""

    FULL = "full"
    CURATED = "curated"
    EQUAL_N_RANDOM = "equal_n_random"
    RANDOM_FILTER = "random_filter"
    ABLATION = "ablation"


@dataclass(frozen=True)
class Arm:
    """One experiment arm: a named training subset.

    Attributes:
        name: Unique arm name (e.g. ``"curated"``, ``"ablation:jerk"``).
        condition: The :class:`Condition` this arm belongs to.
        episode_indices: The source episode indices in this arm's training set.
    """

    name: str
    condition: Condition
    episode_indices: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.episode_indices)


class SubsetReader:
    """A read-only :class:`~robocurate.adapters.base.DatasetReader` over selected episodes.

    Wraps a source reader and exposes only ``episode_indices`` (in the given order). Has no
    write method, so a training run over a subset can never mutate the source (invariant 1).
    """

    def __init__(self, source: DatasetReader, episode_indices: Sequence[int]) -> None:
        self._source = source
        self._indices = tuple(episode_indices)
        self.meta: DatasetMeta = self._build_meta(source.meta)

    def _build_meta(self, source_meta: DatasetMeta) -> DatasetMeta:
        return DatasetMeta(
            fingerprint=self.fingerprint(),
            embodiment_ids=source_meta.embodiment_ids,
            feature_keys=source_meta.feature_keys,
            extra={"subset_of": source_meta.fingerprint.dataset_id},
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[Trajectory]:
        for index in self._indices:
            yield self._source.read_episode(index)

    def read_episode(self, index: int) -> Trajectory:
        """Read the ``index``-th episode *within this subset* (0-based over the selection)."""
        return self._source.read_episode(self._indices[index])

    def fingerprint(self) -> DatasetFingerprint:
        src = self._source.fingerprint()
        roll = hashlib.sha256(src.content_hash.encode("utf-8"))
        for index in self._indices:
            roll.update(str(index).encode("utf-8"))
        return DatasetFingerprint(
            dataset_id=f"{src.dataset_id}#subset",
            source_format=src.source_format,
            content_hash=roll.hexdigest(),
            num_episodes=len(self._indices),
        )


__all__ = ["Arm", "Condition", "SubsetReader"]
