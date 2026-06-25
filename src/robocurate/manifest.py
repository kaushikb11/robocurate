"""The curation manifest and its per-episode decision records.

Every curation run emits a :class:`Manifest` alongside the new dataset. The manifest is the
auditable record of *what was removed and why*, plus everything needed to reproduce the run
(config, seed, code version, source + output fingerprints, the equal-N random baseline).

The manifest references the run :class:`~robocurate.curator.CurationConfig` only as a typed
field (under ``TYPE_CHECKING``) and serializes it via its ``to_dict``; it deliberately does
not import the engine, so it sits below the curator in the import graph.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robocurate.metadata import DatasetFingerprint
from robocurate.signals.base import SignalSpec

if TYPE_CHECKING:
    from pathlib import Path

MANIFEST_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class EpisodeDecision:
    """The kept/removed decision for one source episode, with its justification.

    This is what makes "why was this trajectory removed" inspectable rather than a black box
    (Invariant 6).

    Attributes:
        episode_index: The episode's index in the source dataset.
        fingerprint: The episode's content fingerprint.
        kept: Whether the episode is in the curated output.
        reason: Human-readable justification, e.g.
            ``"removed: jerk in 95th percentile (value=1.84 > threshold=1.20)"`` or
            ``"kept"``.
        signal_values: The per-signal scalar scores that drove the decision, by signal name.
            ``NaN`` indicates the signal skipped this episode.
    """

    episode_index: int
    fingerprint: str
    kept: bool
    reason: str
    signal_values: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "fingerprint": self.fingerprint,
            "kept": self.kept,
            "reason": self.reason,
            "signal_values": dict(self.signal_values),
        }


@dataclass(frozen=True)
class BaselineRecord:
    """The equal-size random baseline selection paired with every curation (invariant 5).

    Recording this makes the dataset-size-confound comparison "one flag away": the baseline
    keeps the *same number* of episodes as the curated selection, drawn with a seeded RNG.

    Attributes:
        method: The baseline method; ``"equal_n_random"`` for v1.
        seed: The seed of the RNG stream used for the baseline draw.
        n: The number of episodes kept (equal to the curated selection size).
        kept_episode_indices: The episode indices the baseline kept.
    """

    method: str
    seed: int
    n: int
    kept_episode_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "n": self.n,
            "kept_episode_indices": list(self.kept_episode_indices),
        }


@dataclass(frozen=True)
class Manifest:
    """The full, auditable record emitted next to a curated dataset.

    Attributes:
        schema_version: Version of the manifest schema itself.
        source: Fingerprint of the (read-only) source dataset.
        output: Fingerprint of the written curated dataset.
        config_dict: The fully-resolved run config, serialized (kept as a dict so the
            manifest does not import the engine).
        seed: The master seed for the run.
        code_version: Package version (plus git commit when available), for reproducibility.
        signals: The specs of the signals that ran.
        decisions: One :class:`EpisodeDecision` per source episode.
        baseline: The equal-N random :class:`BaselineRecord`, or ``None`` if not emitted.
        file_checksums: ``path -> sha256`` for every file written, so integrity is
            verifiable (Invariant 2).
        created_utc: ISO-8601 UTC timestamp the run finished (stamped by the caller).
    """

    schema_version: str
    source: DatasetFingerprint
    output: DatasetFingerprint
    config_dict: Mapping[str, Any]
    seed: int
    code_version: str
    signals: tuple[SignalSpec, ...]
    decisions: tuple[EpisodeDecision, ...]
    baseline: BaselineRecord | None
    file_checksums: Mapping[str, str] = field(default_factory=dict)
    created_utc: str | None = None

    @property
    def num_removed(self) -> int:
        """How many source episodes the curation removed."""
        return sum(1 for d in self.decisions if not d.kept)

    @property
    def num_kept(self) -> int:
        """How many source episodes the curation kept."""
        return sum(1 for d in self.decisions if d.kept)

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable manifest as a JSON-serializable dict."""
        return {
            "schema_version": self.schema_version,
            "source": _fingerprint_to_dict(self.source),
            "output": _fingerprint_to_dict(self.output),
            "config": dict(self.config_dict),
            "seed": self.seed,
            "code_version": self.code_version,
            "signals": [_spec_to_dict(s) for s in self.signals],
            "decisions": [d.to_dict() for d in self.decisions],
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "file_checksums": dict(self.file_checksums),
            "created_utc": self.created_utc,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the manifest to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write(self, path: Path) -> None:
        """Write the manifest JSON to ``path``."""
        path.write_text(self.to_json(), encoding="utf-8")


def _fingerprint_to_dict(fp: DatasetFingerprint) -> dict[str, Any]:
    return {
        "dataset_id": fp.dataset_id,
        "source_format": fp.source_format,
        "content_hash": fp.content_hash,
        "num_episodes": fp.num_episodes,
    }


def _spec_to_dict(spec: SignalSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "version": spec.version,
        "cost_tier": spec.cost_tier.name,
        "requires": sorted(spec.requires),
        "produces_per_transition": spec.produces_per_transition,
        "deterministic": spec.deterministic,
        "description": spec.description,
    }


def code_version() -> str:
    """Return the running package version (best-effort), for manifest provenance."""
    from robocurate import __version__

    return __version__


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "BaselineRecord",
    "EpisodeDecision",
    "Manifest",
    "code_version",
]
