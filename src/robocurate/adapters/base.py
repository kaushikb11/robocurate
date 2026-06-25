"""Dataset adapter protocols and the structural read-only guarantee.

Adapters convert source dataset formats to/from the canonical
:class:`~robocurate.trajectory.Trajectory`. The interface is split into two protocols on
purpose:

* :class:`DatasetReader` — **read + iterate** a source. It has **no write method at all**,
  so there is physically no way to write back through the object that holds the source.
  This is how Invariant 1 (source data is read-only) becomes a *type-level*
  guarantee rather than a convention.
* :class:`DatasetWriter` — **writes a new dataset only**. A writer is constructed with a
  destination that must not exist and must not overlap the source; it refuses otherwise.
  Every write finishes with a validation pass (schema + checksum + round-trip reload); a
  failure is a hard error and the partial output is quarantined, never reported as success
  (invariant 2).

RLDS and raw sim-output adapters slot in later by implementing these same protocols.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from robocurate.manifest import Manifest
    from robocurate.metadata import DatasetFingerprint, DatasetMeta
    from robocurate.trajectory import Trajectory


class LeRobotVersion(str, Enum):
    """Supported LeRobotDataset on-disk format versions.

    The format is a live maintenance risk (see BLUEPRINT caveats), so the targeted version
    is always explicit and pinned. ``V2_1`` is the implemented path; ``V3`` is declared so
    the surface is stable but currently raises a clear error rather than guessing.
    """

    V2_1 = "lerobot_v2.1"
    V3 = "lerobot_v3"


class SourceWriteError(RuntimeError):
    """Raised when a write would target the source dataset or an existing path.

    Existence of this error is part of the read-only guarantee: the writer refuses unsafe
    destinations structurally.
    """


class ValidationError(RuntimeError):
    """Raised when a written dataset fails schema/checksum/round-trip validation.

    A validation failure is a hard error (invariant 2); the partial output is quarantined.
    """


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating a written dataset against the format schema.

    Attributes:
        ok: Whether the dataset is valid.
        errors: Human-readable schema/integrity errors (empty iff ``ok``).
        checked_files: Files whose checksums were verified.
    """

    ok: bool
    errors: tuple[str, ...] = ()
    checked_files: tuple[str, ...] = ()

    def raise_if_invalid(self) -> None:
        """Raise :class:`ValidationError` if this report is not ``ok``."""
        if not self.ok:
            joined = "\n  - ".join(self.errors)
            raise ValidationError(f"written dataset failed validation:\n  - {joined}")


@dataclass(frozen=True)
class WriteReceipt:
    """Returned by a successful :meth:`DatasetWriter.write`.

    Attributes:
        path: The directory the new dataset was written to.
        fingerprint: The output dataset's fingerprint.
        manifest_path: Path to the written manifest JSON.
        file_checksums: ``relative path -> sha256`` for every written file.
        validation: The validation report (always ``ok`` on a successful write).
    """

    path: Path
    fingerprint: DatasetFingerprint
    manifest_path: Path
    file_checksums: Mapping[str, str] = field(default_factory=dict)
    validation: ValidationReport | None = None


@runtime_checkable
class DatasetReader(Protocol):
    """Read-only access to a source dataset as canonical trajectories.

    Intentionally has no write/save/mutate method. Iterating yields trajectories lazily so
    datasets too large for RAM stream episode-by-episode.
    """

    meta: DatasetMeta

    def __len__(self) -> int:
        """Return the number of episodes in the source dataset."""
        ...

    def __iter__(self) -> Iterator[Trajectory]:
        """Iterate over all episodes lazily, in episode-index order."""
        ...

    def read_episode(self, index: int) -> Trajectory:
        """Materialize a single episode by index.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        ...

    def fingerprint(self) -> DatasetFingerprint:
        """Return the content fingerprint of the source (stable for identical content)."""
        ...


@runtime_checkable
class DatasetWriter(Protocol):
    """Writes a *new* dataset (plus manifest). Never writes back to a source.

    Implementations validate after writing and raise on any schema/checksum/round-trip
    failure.
    """

    def write(self, trajectories: Iterable[Trajectory], manifest: Manifest) -> WriteReceipt:
        """Write ``trajectories`` as a new dataset and emit ``manifest`` beside it.

        Raises:
            SourceWriteError: If the destination exists or overlaps the source.
            ValidationError: If the written dataset fails validation.
        """
        ...

    def validate(self, path: Path) -> ValidationReport:
        """Validate a written dataset at ``path`` (schema + checksum + round-trip)."""
        ...
