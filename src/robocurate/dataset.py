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

import re
from pathlib import Path
from typing import TYPE_CHECKING

from robocurate.adapters.base import LeRobotVersion
from robocurate.adapters.lerobot import LeRobotReader

if TYPE_CHECKING:
    from collections.abc import Iterator

    from robocurate.adapters.base import DatasetReader
    from robocurate.metadata import DatasetFingerprint, DatasetMeta
    from robocurate.trajectory import Trajectory

# A Hugging Face Hub dataset id: "namespace/name". Deliberately conservative — anything that
# exists on disk is treated as a path first, so an id-shaped local directory still wins.
_HUB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")

_HUB_INSTALL_HINT = (
    "reading a dataset by Hub id requires the 'huggingface_hub' package, which is part of the "
    "optional 'lerobot' extra. Install it with: pip install 'robocurate[lerobot]' "
    "(or pass a local dataset directory instead)."
)

# Low-dim-only download: metadata + parquet shards, never the mp4 video shards. Cheap signals,
# profile, and validate need no pixels, so this is the default; image signals opt in.
_LOW_DIM_PATTERNS = ["meta/*", "meta/**", "data/**"]


def _resolve_source(path_or_id: str | Path, *, include_videos: bool) -> tuple[Path, str | None]:
    """Resolve a local directory or a Hub dataset id to a local dataset root.

    Returns ``(root, hub_repo_id)``; ``hub_repo_id`` is ``None`` for a plain local path. A Hub
    id downloads (or reuses) the ``huggingface_hub`` cache snapshot — low-dim files only unless
    ``include_videos`` is set, so profiling a large video dataset never pulls the mp4 shards.
    The cache is a read-only *source* like any other; curation still writes a new dataset.
    """
    path = Path(path_or_id)
    if path.exists():
        return path, None
    raw = str(path_or_id)
    if _HUB_ID_RE.match(raw):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(_HUB_INSTALL_HINT) from exc

        root = snapshot_download(
            repo_id=raw,
            repo_type="dataset",
            allow_patterns=None if include_videos else _LOW_DIM_PATTERNS,
        )
        return Path(root), raw
    raise FileNotFoundError(
        f"{path_or_id!r} is neither an existing local dataset directory nor a "
        "'namespace/name' Hugging Face Hub dataset id"
    )


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
    def from_lerobot(
        cls,
        path: str | Path,
        *,
        version: LeRobotVersion | None = None,
        include_videos: bool = False,
    ) -> Dataset:
        """Open a LeRobotDataset for reading (v2.1 or v3.0) from a local directory or a Hub id.

        ``path`` is either a local dataset directory or a ``"namespace/name"`` Hugging Face Hub
        dataset id (an existing local path always wins). A Hub id is snapshot-downloaded into
        the ``huggingface_hub`` cache — low-dim files only by default; pass
        ``include_videos=True`` when a signal needs to decode frames. Requires the ``lerobot``
        extra for Hub ids. The on-disk format is auto-detected from ``meta/info.json`` unless
        ``version`` is given.
        """
        from robocurate.adapters.lerobot_v3 import LeRobotReaderV3, detect_lerobot_version

        root, hub_repo_id = _resolve_source(path, include_videos=include_videos)
        resolved = version if version is not None else detect_lerobot_version(root)
        if resolved is LeRobotVersion.V3:
            # Record the Hub id (not the machine-specific cache path) as the dataset id, so
            # fingerprints/manifests built from a Hub source are shareable and reproducible.
            return cls(LeRobotReaderV3(root, dataset_id=hub_repo_id))
        return cls(LeRobotReader(root, version=resolved))

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
