"""Read-only dataset profiling — exploratory data analysis for a source dataset.

This module powers the ``profile`` CLI command. Like :mod:`robocurate.health` it is a
*description*, not a curation: it never selects, mutates, or writes anything (Invariant 1).
Given an opened :class:`~robocurate.adapters.base.DatasetReader`, :func:`dataset_profile`
composes a single :class:`ProfileReport` answering "what is in this dataset?":

1. **Shape** — episode count and the episode-length distribution (min / median / max plus a
   handful of histogram buckets over the lengths).
2. **Features** — one summary per declared low-dim feature key: its role, per-step dimension,
   and value min / median / max (computed over the finite materialized values; image
   features are reported by shape only, never decoded).
3. **Embodiment** — the distinct embodiment ids present (a dataset may be mixed).
4. **Success rate** — the fraction of episodes whose :class:`~robocurate.trajectory.SuccessLabel`
   is ``True``, over the episodes that carry a known label, with the unknown/unlabelled count
   reported alongside so the rate is never silently computed over a misleading denominator.
5. **Task balance** — when episodes carry task labels (in ``meta.extra``), the per-task
   episode counts, so an imbalanced task mix is visible up front.
6. **Diversity** — a cheap redundancy estimate: the mean nearest-neighbour distance over the
   :func:`~robocurate.signals.redundancy.statistical_embedding` (z-standardized), a coarse
   "how spread out is this dataset" number that needs no GPU or learned model.

The report is a frozen dataclass with ``to_dict()`` (machine-readable, for ``--json``) and
``to_markdown()`` (human-readable, the default CLI rendering). It depends only on the reader
protocol, NumPy, and the statistical embedding, so it stays cheap and torch-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from robocurate.signals.redundancy import statistical_embedding
from robocurate.trajectory import FeatureRole

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader
    from robocurate.trajectory import Trajectory

# Number of equal-width buckets used to summarize the episode-length histogram.
_NUM_LENGTH_BUCKETS = 5


@dataclass(frozen=True)
class Distribution:
    """A compact min / median / max summary of a 1-D distribution.

    Attributes:
        count: How many values the summary was computed over.
        minimum: The smallest value, or ``None`` when ``count`` is 0.
        median: The median value, or ``None`` when ``count`` is 0.
        maximum: The largest value, or ``None`` when ``count`` is 0.
    """

    count: int
    minimum: float | None
    median: float | None
    maximum: float | None

    @classmethod
    def from_values(cls, values: list[float]) -> Distribution:
        """Summarize a list of values (empty list yields an all-``None`` summary)."""
        if not values:
            return cls(count=0, minimum=None, median=None, maximum=None)
        arr = np.asarray(values, dtype=np.float64)
        return cls(
            count=int(arr.size),
            minimum=float(arr.min()),
            median=float(np.median(arr)),
            maximum=float(arr.max()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.minimum,
            "median": self.median,
            "max": self.maximum,
        }


@dataclass(frozen=True)
class LengthHistogramBucket:
    """One equal-width bucket of the episode-length histogram.

    Attributes:
        low: Inclusive lower bound of the bucket (an episode length).
        high: Inclusive upper bound of the bucket.
        count: How many episodes fall in ``[low, high]``.
    """

    low: int
    high: int
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high, "count": self.count}


@dataclass(frozen=True)
class FeatureSummary:
    """Per-feature shape + value summary for one declared feature key.

    Attributes:
        key: The feature key.
        role: The feature's :class:`~robocurate.trajectory.FeatureRole` name.
        dim: The flattened per-step dimension (e.g. ``2`` for a ``(2,)`` action), or ``None``
            for an image feature (not decoded).
        values: The value :class:`Distribution` over finite materialized entries across all
            carrying episodes; ``None`` for image features (skipped, not decoded).
    """

    key: str
    role: str
    dim: int | None
    values: Distribution | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "dim": self.dim,
            "values": self.values.to_dict() if self.values is not None else None,
        }


@dataclass(frozen=True)
class ProfileReport:
    """The full read-only EDA profile of a source dataset.

    Attributes:
        dataset_id: The source dataset identifier (its path or hub id).
        source_format: The on-disk format + version (e.g. ``"lerobot_v2.1"``).
        num_episodes: Number of episodes the reader exposes.
        embodiment_ids: Distinct embodiment ids present.
        episode_lengths: The :class:`Distribution` of episode lengths (``num_steps``).
        length_histogram: A few equal-width buckets over the episode lengths.
        features: One :class:`FeatureSummary` per declared feature key.
        num_success: Episodes with a known-``True`` success label.
        num_failure: Episodes with a known-``False`` success label.
        num_success_unknown: Episodes with no/unknown success label (excluded from the rate).
        task_counts: ``task label -> episode count`` when task labels are present, else empty.
        mean_nn_distance: Mean nearest-neighbour distance over the z-standardized statistical
            embedding (a coarse diversity estimate), or ``None`` when fewer than two episodes
            are embeddable.
        num_embedded: How many episodes contributed to ``mean_nn_distance``.
    """

    dataset_id: str
    source_format: str
    num_episodes: int
    embodiment_ids: tuple[str, ...]
    episode_lengths: Distribution
    length_histogram: tuple[LengthHistogramBucket, ...]
    features: tuple[FeatureSummary, ...]
    num_success: int
    num_failure: int
    num_success_unknown: int
    task_counts: Mapping[str, int] = field(default_factory=dict)
    mean_nn_distance: float | None = None
    num_embedded: int = 0

    @property
    def num_success_labelled(self) -> int:
        """How many episodes carry a known (``True``/``False``) success label."""
        return self.num_success + self.num_failure

    @property
    def success_rate(self) -> float | None:
        """Fraction of labelled episodes that succeeded, or ``None`` if none are labelled."""
        labelled = self.num_success_labelled
        return (self.num_success / labelled) if labelled else None

    def to_dict(self) -> dict[str, Any]:
        """Return the machine-readable profile as a JSON-serializable dict."""
        return {
            "dataset_id": self.dataset_id,
            "source_format": self.source_format,
            "num_episodes": self.num_episodes,
            "embodiment_ids": list(self.embodiment_ids),
            "episode_lengths": self.episode_lengths.to_dict(),
            "length_histogram": [b.to_dict() for b in self.length_histogram],
            "features": [f.to_dict() for f in self.features],
            "success": {
                "num_success": self.num_success,
                "num_failure": self.num_failure,
                "num_unknown": self.num_success_unknown,
                "num_labelled": self.num_success_labelled,
                "rate": self.success_rate,
            },
            "task_counts": dict(self.task_counts),
            "diversity": {
                "mean_nn_distance": self.mean_nn_distance,
                "num_embedded": self.num_embedded,
            },
        }

    def to_markdown(self) -> str:
        """Render the profile as human-readable Markdown (the default CLI output)."""
        lines = [
            f"# Dataset profile: {self.dataset_id}",
            "",
            f"- Format: {self.source_format}",
            f"- Episodes: {self.num_episodes}",
            f"- Embodiments: {', '.join(self.embodiment_ids) or '(none)'}",
            "",
            "## Episode length",
            "",
            f"- min / median / max: {_fmt(self.episode_lengths.minimum)} / "
            f"{_fmt(self.episode_lengths.median)} / {_fmt(self.episode_lengths.maximum)}",
        ]
        if self.length_histogram:
            lines.append("- histogram:")
            for bucket in self.length_histogram:
                lines.append(f"  - [{bucket.low}, {bucket.high}]: {bucket.count}")
        lines += [
            "",
            "## Features",
            "",
            "| feature | role | dim | min | median | max |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for feat in self.features:
            vals = feat.values
            lines.append(
                f"| {feat.key} | {feat.role} | {_fmt(feat.dim)} "
                f"| {_fmt(vals.minimum if vals else None)} "
                f"| {_fmt(vals.median if vals else None)} "
                f"| {_fmt(vals.maximum if vals else None)} |"
            )
        lines += ["", "## Success", ""]
        if self.success_rate is None:
            lines.append("- No success labels present.")
        else:
            lines.append(
                f"- Success rate: {self.success_rate:.1%} "
                f"({self.num_success}/{self.num_success_labelled} labelled episodes)"
            )
        if self.num_success_unknown:
            lines.append(f"- Unknown/unlabelled: {self.num_success_unknown}")
        if self.task_counts:
            lines += ["", "## Task balance", ""]
            for task, count in sorted(self.task_counts.items()):
                lines.append(f"- {task or '(empty)'}: {count}")
        lines += ["", "## Diversity", ""]
        if self.mean_nn_distance is None:
            lines.append("- Diversity estimate unavailable (fewer than 2 embeddable episodes).")
        else:
            lines.append(
                f"- Mean nearest-neighbour distance: {self.mean_nn_distance:.4g} "
                f"(over {self.num_embedded} embeddable episodes; higher = more diverse)"
            )
        return "\n".join(lines)


def _fmt(value: float | int | None) -> str:
    """Format an optional number for markdown (``-`` when absent)."""
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4g}"


def dataset_profile(reader: DatasetReader) -> ProfileReport:
    """Profile a source dataset without curating, mutating, or writing anything.

    Composes episode-shape, per-feature, embodiment, success-rate, task-balance, and a cheap
    diversity (mean nearest-neighbour distance) summary into one :class:`ProfileReport`.
    Read-only by construction (Invariant 1): it only iterates the reader and reads feature
    arrays.

    Args:
        reader: An opened, read-only :class:`DatasetReader` (e.g. a ``LeRobotReader`` or a
            ``Dataset``).

    Returns:
        A :class:`ProfileReport` summarizing what is in the dataset.
    """
    meta = reader.meta
    trajectories = list(reader)
    num_episodes = len(trajectories)

    lengths = [traj.num_steps for traj in trajectories]
    episode_lengths = Distribution.from_values([float(n) for n in lengths])
    histogram = _length_histogram(lengths)
    features = _feature_summaries(trajectories)
    num_success, num_failure, num_unknown = _success_counts(trajectories)
    task_counts = _task_counts(trajectories)
    mean_nn, num_embedded = _diversity(trajectories)

    return ProfileReport(
        dataset_id=meta.fingerprint.dataset_id,
        source_format=meta.fingerprint.source_format,
        num_episodes=num_episodes,
        embodiment_ids=tuple(meta.embodiment_ids),
        episode_lengths=episode_lengths,
        length_histogram=histogram,
        features=features,
        num_success=num_success,
        num_failure=num_failure,
        num_success_unknown=num_unknown,
        task_counts=task_counts,
        mean_nn_distance=mean_nn,
        num_embedded=num_embedded,
    )


def _length_histogram(lengths: list[int]) -> tuple[LengthHistogramBucket, ...]:
    """Bucket episode lengths into a few equal-width bins (inclusive bounds)."""
    if not lengths:
        return ()
    lo, hi = min(lengths), max(lengths)
    if lo == hi:
        return (LengthHistogramBucket(low=lo, high=hi, count=len(lengths)),)
    num_buckets = min(_NUM_LENGTH_BUCKETS, hi - lo + 1)
    edges = np.linspace(lo, hi, num_buckets + 1)
    buckets: list[LengthHistogramBucket] = []
    for b in range(num_buckets):
        b_low = int(np.floor(edges[b]))
        # Make bounds inclusive and contiguous; the last bucket absorbs the top edge.
        b_high = hi if b == num_buckets - 1 else int(np.floor(edges[b + 1])) - 1
        b_high = max(b_high, b_low)
        count = sum(1 for n in lengths if b_low <= n <= b_high)
        buckets.append(LengthHistogramBucket(low=b_low, high=b_high, count=count))
    return tuple(buckets)


def _feature_summaries(trajectories: list[Trajectory]) -> tuple[FeatureSummary, ...]:
    """One shape + value summary per declared feature key, in first-seen declared order."""
    declared: dict[str, FeatureRole] = {}
    for traj in trajectories:
        for spec in traj.embodiment.features:
            declared.setdefault(spec.key, spec.role)

    out: list[FeatureSummary] = []
    for key, role in declared.items():
        if role is FeatureRole.IMAGE:
            out.append(FeatureSummary(key=key, role=role.value, dim=None, values=None))
            continue
        finite: list[float] = []
        dim: int | None = None
        for traj in trajectories:
            if not traj.has(key):
                continue
            arr = np.asarray(traj.feature(key), dtype=np.float64)
            flat = arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr.reshape(arr.shape[0], 1)
            if dim is None:
                dim = int(flat.shape[1])
            col = flat.reshape(-1)
            finite.extend(float(v) for v in col[np.isfinite(col)])
        out.append(
            FeatureSummary(
                key=key,
                role=role.value,
                dim=dim,
                values=Distribution.from_values(finite),
            )
        )
    return tuple(out)


def _success_counts(trajectories: list[Trajectory]) -> tuple[int, int, int]:
    """Count (known-success, known-failure, unknown/unlabelled) episodes."""
    num_success = num_failure = num_unknown = 0
    for traj in trajectories:
        label = traj.success()
        if label is None or label.value is None:
            num_unknown += 1
        elif label.value:
            num_success += 1
        else:
            num_failure += 1
    return num_success, num_failure, num_unknown


def _task_counts(trajectories: list[Trajectory]) -> dict[str, int]:
    """Per-task episode counts when task labels are present in ``meta.extra``.

    LeRobot adapters expose ``meta.extra["tasks"]`` as a (possibly empty) list of task strings
    per episode; an episode counts toward every task it carries. Returns an empty mapping when
    no episode carries a non-empty task label, so the profile simply omits the section.
    """
    counts: dict[str, int] = {}
    for traj in trajectories:
        tasks = traj.meta.extra.get("tasks")
        if not isinstance(tasks, (list, tuple)):
            continue
        for task in tasks:
            label = str(task)
            counts[label] = counts.get(label, 0) + 1
    return counts


def _diversity(trajectories: list[Trajectory]) -> tuple[float | None, int]:
    """Mean nearest-neighbour distance over the z-standardized statistical embedding.

    A coarse, GPU-free "how spread out is this dataset" estimate. Embeds every episode with
    :func:`~robocurate.signals.redundancy.statistical_embedding`, z-standardizes per feature,
    and averages each point's distance to its nearest other point. Returns ``(None, n)`` when
    fewer than two episodes are embeddable (a single point has no neighbour).
    """
    raw: list[npt.NDArray[np.float64]] = []
    for traj in trajectories:
        emb = statistical_embedding(traj)
        if emb is not None:
            raw.append(np.asarray(emb, dtype=np.float64).reshape(-1))
    if len(raw) < 2:
        return None, len(raw)

    stacked = np.vstack(raw)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std_safe = np.where(std > 0.0, std, 1.0)
    z = (stacked - mean) / std_safe
    diff = z[:, None, :] - z[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)  # a point is never its own nearest neighbour
    nearest = dist.min(axis=1)
    return float(nearest.mean()), len(raw)


__all__ = [
    "Distribution",
    "FeatureSummary",
    "LengthHistogramBucket",
    "ProfileReport",
    "dataset_profile",
]
