"""Shared, dependency-light metadata types.

These records are imported by several layers (signals, adapters, curator, scorecard). They
deliberately depend only on the standard library and the trajectory module so they can sit
at the bottom of the import graph and avoid cycles.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetFingerprint:
    """An identity + content fingerprint of a dataset, for reproducibility and the manifest.

    Attributes:
        dataset_id: Human/source identifier, e.g. ``"lerobot/droid"`` or a local path.
        source_format: Exact format + version, e.g. ``"lerobot_v2.1"``, ``"lerobot_v3"``,
            ``"rlds"``.
        content_hash: A stable hash over the dataset's content (episode fingerprints rolled
            up). Two datasets with equal content have equal ``content_hash``.
        num_episodes: Number of episodes in the dataset.
    """

    dataset_id: str
    source_format: str
    content_hash: str
    num_episodes: int


@dataclass(frozen=True)
class DatasetMeta:
    """Dataset-level metadata a signal or the engine may need.

    Attributes:
        fingerprint: The :class:`DatasetFingerprint`.
        embodiment_ids: The distinct embodiment ids present (a dataset may be mixed).
        feature_keys: The union of feature keys present across episodes.
        extra: Free-form dataset metadata (e.g. a handle to a prebuilt embedding index a
            redundancy signal populated in ``Signal.fit``). Carried losslessly.
    """

    fingerprint: DatasetFingerprint
    embodiment_ids: tuple[str, ...]
    feature_keys: tuple[str, ...]
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceProbe:
    """A snapshot of available compute, used to gate signals by their declared requirements.

    The engine fills this in; a signal never probes hardware itself. Signals declaring
    ``requires={"gpu"}`` are skipped with a clear message when ``has_gpu`` is ``False``.

    Attributes:
        has_gpu: Whether a usable GPU is available.
        num_gpus: Number of visible GPUs.
        vram_gb: Total VRAM on the primary GPU in GiB, or ``None`` if no GPU.
        num_cpus: Number of usable CPU cores.
    """

    has_gpu: bool = False
    num_gpus: int = 0
    vram_gb: float | None = None
    num_cpus: int = 1
