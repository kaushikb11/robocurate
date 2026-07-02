"""Read-only dataset health check — schema readability, structural defects, and coverage.

This module powers the ``validate`` / ``doctor`` CLI command. It is a *diagnosis*, not a
curation: it never selects, mutates, or writes anything (Invariant 1). Given an opened
:class:`~robocurate.adapters.base.DatasetReader`, :func:`dataset_health` composes three
cheap, CPU-only checks into a single :class:`HealthReport`:

1. **Readability / schema** — the reader has already parsed ``meta/info.json`` and the
   per-episode parquet on construction, so a dataset that reaches this code is structurally
   loadable. The report records the episode count, feature keys, and embodiment ids.
   Episodes whose bytes cannot be read (corrupt parquet, missing file) are reported as
   :class:`UnreadableEpisode` findings — index + error — rather than crashing the check.
2. **Structural defects** — runs :class:`~robocurate.signals.structural_validity.StructuralValidity`
   over every episode (fit then score) and tallies how many are valid vs. truncated /
   stalled / non-finite, reading the per-episode diagnostics. This catches the incomplete,
   frozen, or corrupt episodes that the geometric signals miss.
3. **Per-feature coverage** — for each declared feature key, the fraction of episodes that
   actually carry it, plus finite-value statistics (min/max/mean over the finite entries of
   the non-image numeric features). A key present in the schema but absent or all-NaN on
   most episodes is a data-quality smell surfaced here rather than discovered mid-training.

The report is a frozen dataclass with ``to_dict()`` (machine-readable, for ``--json``) and
``to_markdown()`` (human-readable, the default CLI rendering). It depends only on the reader
protocol, NumPy, and the structural-validity signal, so it stays cheap and torch-free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from robocurate.metadata import ResourceProbe
from robocurate.signals.base import InMemoryCache, SignalContext
from robocurate.signals.structural_validity import StructuralValidity
from robocurate.trajectory import FeatureRole

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader
    from robocurate.trajectory import Trajectory


@dataclass(frozen=True)
class UnreadableEpisode:
    """One episode whose bytes could not be read (corrupt parquet, missing file, ...).

    The health check is read-only reporting, so an unreadable episode is always a
    *finding* (index + error), never a crash of the whole diagnosis.

    Attributes:
        episode_index: The episode's index in the source dataset.
        error: ``"<ExcType>: <msg>"`` for the exception the read raised.
    """

    episode_index: int
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {"episode_index": self.episode_index, "error": self.error}


@dataclass(frozen=True)
class FeatureCoverage:
    """Coverage and finite-value statistics for one declared feature key.

    Attributes:
        key: The feature key.
        role: The feature's :class:`~robocurate.trajectory.FeatureRole` name.
        present_episodes: How many episodes actually carry this feature.
        coverage: ``present_episodes / total_episodes`` (1.0 = every episode has it).
        finite_fraction: Fraction of materialized values that are finite (non-NaN/inf),
            over the episodes that carry the feature; ``None`` for image features (skipped,
            not decoded) or when the feature is absent everywhere.
        minimum: Minimum finite value across all carrying episodes, or ``None``.
        maximum: Maximum finite value across all carrying episodes, or ``None``.
        mean: Mean of the finite values across all carrying episodes, or ``None``.
    """

    key: str
    role: str
    present_episodes: int
    coverage: float
    finite_fraction: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "present_episodes": self.present_episodes,
            "coverage": self.coverage,
            "finite_fraction": self.finite_fraction,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
        }


@dataclass(frozen=True)
class StructuralSummary:
    """Dataset-wide tally of structural defects from :class:`StructuralValidity`.

    Attributes:
        num_episodes: Total episodes examined.
        num_valid: Episodes with no structural defect (structural score 0).
        num_truncated: Episodes flagged as truncated (far shorter than the median length).
        num_stalled: Episodes with a run of held/repeated frames beyond tolerance.
        num_nonfinite: Episodes containing any NaN/inf in a non-image feature.
        median_steps: The dataset median episode length learned by the signal's ``fit``.
        defect_episode_indices: Source episode indices of every structurally-invalid episode,
            sorted, so a caller can drill in without re-scoring.
    """

    num_episodes: int
    num_valid: int
    num_truncated: int
    num_stalled: int
    num_nonfinite: int
    median_steps: float | None
    defect_episode_indices: tuple[int, ...] = ()

    @property
    def num_invalid(self) -> int:
        """How many episodes carry at least one structural defect."""
        return self.num_episodes - self.num_valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_episodes": self.num_episodes,
            "num_valid": self.num_valid,
            "num_invalid": self.num_invalid,
            "num_truncated": self.num_truncated,
            "num_stalled": self.num_stalled,
            "num_nonfinite": self.num_nonfinite,
            "median_steps": self.median_steps,
            "defect_episode_indices": list(self.defect_episode_indices),
        }


@dataclass(frozen=True)
class HealthReport:
    """The full read-only health diagnosis of a source dataset.

    Attributes:
        dataset_id: The source dataset identifier (its path or hub id).
        source_format: The on-disk format + version (e.g. ``"lerobot_v2.1"``).
        num_episodes: Number of episodes the reader exposes.
        embodiment_ids: Distinct embodiment ids present.
        feature_keys: All declared feature keys.
        structural: The :class:`StructuralSummary` defect tally (over the readable episodes).
        coverage: One :class:`FeatureCoverage` per declared feature key (over the readable
            episodes).
        unreadable: One :class:`UnreadableEpisode` per episode whose bytes could not be
            read at all; these are reported as findings rather than crashing the check.
    """

    dataset_id: str
    source_format: str
    num_episodes: int
    embodiment_ids: tuple[str, ...]
    feature_keys: tuple[str, ...]
    structural: StructuralSummary
    coverage: tuple[FeatureCoverage, ...] = field(default_factory=tuple)
    unreadable: tuple[UnreadableEpisode, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the dataset is free of structural defects and unreadable episodes."""
        return self.structural.num_invalid == 0 and not self.unreadable

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable report as a JSON-serializable dict."""
        return {
            "dataset_id": self.dataset_id,
            "source_format": self.source_format,
            "num_episodes": self.num_episodes,
            "ok": self.ok,
            "embodiment_ids": list(self.embodiment_ids),
            "feature_keys": list(self.feature_keys),
            "structural": self.structural.to_dict(),
            "coverage": [c.to_dict() for c in self.coverage],
            "unreadable": [u.to_dict() for u in self.unreadable],
        }

    def to_markdown(self) -> str:
        """Render the report as human-readable Markdown (the default CLI output)."""
        s = self.structural
        verdict = "OK — no structural defects found" if self.ok else "DEFECTS FOUND"
        lines = [
            f"# Dataset health: {self.dataset_id}",
            "",
            f"- Format: {self.source_format}",
            f"- Episodes: {self.num_episodes}",
            f"- Embodiments: {', '.join(self.embodiment_ids) or '(none)'}",
            f"- Verdict: **{verdict}**",
            "",
            "## Structural validity",
            "",
            f"- Valid episodes: {s.num_valid}/{s.num_episodes}",
            f"- Truncated: {s.num_truncated}",
            f"- Stalled (held frames): {s.num_stalled}",
            f"- Non-finite (NaN/inf): {s.num_nonfinite}",
            f"- Median episode length: {_fmt(s.median_steps)}",
        ]
        if s.defect_episode_indices:
            shown = ", ".join(str(i) for i in s.defect_episode_indices)
            lines.append(f"- Defective episode indices: {shown}")
        if self.unreadable:
            lines += [
                "",
                "## Unreadable episodes",
                "",
                f"- {len(self.unreadable)} of {self.num_episodes} episodes could not be read:",
            ]
            lines += [f"  - episode {u.episode_index}: {u.error}" for u in self.unreadable]
        lines += [
            "",
            "## Feature coverage",
            "",
            "| feature | role | coverage | finite | min | max | mean |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in self.coverage:
            lines.append(
                f"| {c.key} | {c.role} | {c.present_episodes}/{self.num_episodes} "
                f"| {_fmt(c.finite_fraction)} | {_fmt(c.minimum)} | {_fmt(c.maximum)} "
                f"| {_fmt(c.mean)} |"
            )
        return "\n".join(lines)


def _fmt(value: float | None) -> str:
    """Format an optional float for markdown (``-`` when absent)."""
    if value is None:
        return "-"
    return f"{value:.4g}"


def dataset_health(reader: DatasetReader) -> HealthReport:
    """Diagnose a source dataset's health without curating, mutating, or writing anything.

    Composes a schema/readability summary (the reader has already parsed the on-disk
    metadata on construction), a structural-defect tally via
    :class:`~robocurate.signals.structural_validity.StructuralValidity`, and per-feature
    coverage + finite statistics. Read-only by construction (Invariant 1): it only iterates
    the reader and reads feature arrays.

    Args:
        reader: An opened, read-only :class:`DatasetReader` (e.g. a ``LeRobotReader`` or a
            ``Dataset``).

    Returns:
        A :class:`HealthReport` summarizing the dataset's health.
    """
    meta = reader.meta
    # Read per-episode by index so one unreadable episode (corrupt parquet, missing file)
    # becomes a finding rather than aborting the whole health check: this is read-only
    # reporting, so tolerating + reporting is always correct here.
    trajectories: list[Trajectory] = []
    unreadable: list[UnreadableEpisode] = []
    for index in range(len(reader)):
        try:
            trajectories.append(reader.read_episode(index))
        except Exception as exc:
            unreadable.append(
                UnreadableEpisode(episode_index=index, error=f"{type(exc).__name__}: {exc}")
            )
    num_episodes = len(trajectories) + len(unreadable)

    structural = _structural_summary(trajectories)
    coverage = _coverage(trajectories, len(trajectories))

    return HealthReport(
        dataset_id=meta.fingerprint.dataset_id,
        source_format=meta.fingerprint.source_format,
        num_episodes=num_episodes,
        embodiment_ids=tuple(meta.embodiment_ids),
        feature_keys=tuple(meta.feature_keys),
        structural=structural,
        coverage=coverage,
        unreadable=tuple(unreadable),
    )


def _structural_summary(trajectories: list[Trajectory]) -> StructuralSummary:
    """Fit + score :class:`StructuralValidity` over every episode and tally the diagnostics."""
    signal = StructuralValidity()
    ctx = _signal_context()
    signal.fit(trajectories, ctx)

    num_valid = num_truncated = num_stalled = num_nonfinite = 0
    median_steps: float | None = None
    defects: list[int] = []
    scores = signal.score(trajectories, ctx)
    for traj, score in zip(trajectories, scores, strict=True):
        diag = score.diagnostics
        median_steps = diag.get("median_steps", median_steps)
        if diag.get("is_valid"):
            num_valid += 1
        else:
            defects.append(traj.meta.episode_index)
        if float(diag.get("truncation_severity", 0.0)) > 0.0:
            num_truncated += 1
        if float(diag.get("stall_severity", 0.0)) > 0.0:
            num_stalled += 1
        if diag.get("has_nonfinite"):
            num_nonfinite += 1

    return StructuralSummary(
        num_episodes=len(trajectories),
        num_valid=num_valid,
        num_truncated=num_truncated,
        num_stalled=num_stalled,
        num_nonfinite=num_nonfinite,
        median_steps=median_steps,
        defect_episode_indices=tuple(sorted(defects)),
    )


def _coverage(trajectories: list[Trajectory], num_episodes: int) -> tuple[FeatureCoverage, ...]:
    """Compute per-feature presence + finite statistics across all episodes."""
    # Collect declared features (key -> role) in declared order, de-duplicated across the
    # (possibly mixed-embodiment) dataset while preserving first-seen order.
    declared: dict[str, FeatureRole] = {}
    for traj in trajectories:
        for spec in traj.embodiment.features:
            declared.setdefault(spec.key, spec.role)

    out: list[FeatureCoverage] = []
    for key, role in declared.items():
        present = 0
        finite_count = 0
        total_count = 0
        minimum: float | None = None
        maximum: float | None = None
        running_sum = 0.0
        for traj in trajectories:
            if not traj.has(key):
                continue
            present += 1
            if role is FeatureRole.IMAGE:
                continue  # never decode image features for stats
            arr = np.asarray(traj.feature(key), dtype=np.float64).reshape(-1)
            total_count += arr.size
            finite = arr[np.isfinite(arr)]
            finite_count += int(finite.size)
            if finite.size:
                fmin = float(finite.min())
                fmax = float(finite.max())
                minimum = fmin if minimum is None else min(minimum, fmin)
                maximum = fmax if maximum is None else max(maximum, fmax)
                running_sum += float(finite.sum())

        finite_fraction: float | None = None
        mean: float | None = None
        if role is not FeatureRole.IMAGE and total_count > 0:
            finite_fraction = finite_count / total_count
            if finite_count > 0:
                mean = running_sum / finite_count

        out.append(
            FeatureCoverage(
                key=key,
                role=role.value,
                present_episodes=present,
                coverage=(present / num_episodes) if num_episodes else 0.0,
                finite_fraction=finite_fraction,
                minimum=minimum,
                maximum=maximum,
                mean=mean,
            )
        )
    return tuple(out)


def _signal_context(seed: int = 0) -> SignalContext:
    """Build a minimal CPU :class:`SignalContext` for running the structural signal in-process."""
    return SignalContext(
        seed=seed,
        device="cpu",
        cache=InMemoryCache(),
        resources=ResourceProbe(),
        dataset_meta=_PLACEHOLDER_META,
        logger=logging.getLogger("robocurate.health"),
    )


# StructuralValidity does not read dataset_meta, so a lightweight placeholder is sufficient
# and avoids forcing the reader to expose more than the iteration protocol.
def _placeholder_meta() -> Any:
    from robocurate.metadata import DatasetFingerprint, DatasetMeta

    return DatasetMeta(
        fingerprint=DatasetFingerprint(
            dataset_id="<health>",
            source_format="<health>",
            content_hash="0" * 64,
            num_episodes=0,
        ),
        embodiment_ids=(),
        feature_keys=(),
    )


_PLACEHOLDER_META = _placeholder_meta()


__all__ = [
    "FeatureCoverage",
    "HealthReport",
    "StructuralSummary",
    "UnreadableEpisode",
    "dataset_health",
]
