"""Scorecard / report — human- and machine-readable output of a curation run.

A :class:`Scorecard` describes per-dataset quality, the per-signal score distributions, and
exactly *what a curation run removed and why*. It serializes to stable JSON (machine), to
Markdown (human / terminal), and to a Hugging Face dataset-card fragment.

Two honesty rules from the project invariants are baked in:

* **Why-removed is explicit** (invariant 6): every removed episode carries a
  :class:`TrajectoryFlag` naming the signal value(s) and the human reason — never a black
  box.
* **Effect sizes carry uncertainty** (invariant 6): an :class:`EffectReport` is included
  *only* when a downstream policy evaluation is attached, and it always pairs an effect size
  with its uncertainty. Without an eval, the scorecard reports quality distributions and
  removals and makes no downstream-gain claim.

No quality signal is implemented here; the scorecard only summarizes scores it is given.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from robocurate.manifest import BaselineRecord
from robocurate.metadata import DatasetFingerprint

if TYPE_CHECKING:
    from robocurate.curator import CurationResult

SCORECARD_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class SignalReport:
    """Distribution summary for one signal across all scored trajectories."""

    name: str
    description: str
    higher_is_better: bool
    num_scored: int
    num_skipped: int
    minimum: float | None
    median: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "higher_is_better": self.higher_is_better,
            "num_scored": self.num_scored,
            "num_skipped": self.num_skipped,
            "min": self.minimum,
            "median": self.median,
            "max": self.maximum,
        }


@dataclass(frozen=True)
class TrajectoryFlag:
    """A per-trajectory record of the kept/removed decision and its justification."""

    episode_index: int
    fingerprint: str
    kept: bool
    reason: str
    signal_values: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "fingerprint": self.fingerprint,
            "kept": self.kept,
            "reason": self.reason,
            "signal_values": dict(self.signal_values),
        }


@dataclass(frozen=True)
class QualitySummary:
    """Top-line counts for a curation run."""

    num_episodes: int
    num_kept: int
    num_removed: int

    @property
    def pct_removed(self) -> float:
        return 100.0 * self.num_removed / self.num_episodes if self.num_episodes else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_episodes": self.num_episodes,
            "num_kept": self.num_kept,
            "num_removed": self.num_removed,
            "pct_removed": self.pct_removed,
        }


@dataclass(frozen=True)
class EffectReport:
    """A downstream effect size *with* its uncertainty (invariant 6).

    Only present when a policy evaluation is attached to the run. ``ci_low``/``ci_high``
    bound the effect; ``per_task`` carries the breakdown so a task-dependent gain is never
    reported as a single universal number.
    """

    metric: str
    effect: float
    ci_low: float
    ci_high: float
    per_task: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "effect": self.effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "per_task": dict(self.per_task),
        }


@dataclass(frozen=True)
class Scorecard:
    """The full report for a curation run (machine- and human-readable)."""

    schema_version: str
    dataset: DatasetFingerprint
    summary: QualitySummary
    per_signal: tuple[SignalReport, ...]
    flags: tuple[TrajectoryFlag, ...]
    baseline: BaselineRecord | None
    effects: EffectReport | None = None

    # -- reconstruction --------------------------------------------------------------

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> Scorecard:
        """Reconstruct a :class:`Scorecard` from a serialized manifest dict.

        A saved manifest records the per-episode decisions (with the scalar ``signal_values``
        that drove them) and the equal-N baseline, but *not* the full per-signal score matrix.
        We therefore rebuild the per-signal distribution summaries from those decision-level
        values and render only what the manifest carries: orientation is unknown after the
        fact, so it is reported as the neutral default rather than fabricated, and no
        downstream-effect claim is made (the manifest attaches no policy evaluation).
        """
        src = manifest["source"]
        dataset = DatasetFingerprint(
            dataset_id=src["dataset_id"],
            source_format=src["source_format"],
            content_hash=src["content_hash"],
            num_episodes=src["num_episodes"],
        )

        decisions = manifest["decisions"]
        num_kept = sum(1 for d in decisions if d["kept"])
        summary = QualitySummary(
            num_episodes=len(decisions),
            num_kept=num_kept,
            num_removed=len(decisions) - num_kept,
        )

        flags = tuple(
            TrajectoryFlag(
                episode_index=d["episode_index"],
                fingerprint=d["fingerprint"],
                kept=d["kept"],
                reason=d["reason"],
                signal_values=dict(d.get("signal_values", {})),
            )
            for d in decisions
        )

        per_signal = tuple(
            _signal_report_from_decisions(spec, decisions) for spec in manifest.get("signals", [])
        )

        baseline_dict = manifest.get("baseline")
        baseline = (
            BaselineRecord(
                method=baseline_dict["method"],
                seed=baseline_dict["seed"],
                n=baseline_dict["n"],
                kept_episode_indices=tuple(baseline_dict["kept_episode_indices"]),
            )
            if baseline_dict
            else None
        )

        return cls(
            schema_version=SCORECARD_SCHEMA_VERSION,
            dataset=dataset,
            summary=summary,
            per_signal=per_signal,
            flags=flags,
            baseline=baseline,
            effects=None,
        )

    # -- machine-readable ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": {
                "dataset_id": self.dataset.dataset_id,
                "source_format": self.dataset.source_format,
                "content_hash": self.dataset.content_hash,
                "num_episodes": self.dataset.num_episodes,
            },
            "summary": self.summary.to_dict(),
            "per_signal": [s.to_dict() for s in self.per_signal],
            "flags": [f.to_dict() for f in self.flags],
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "effects": self.effects.to_dict() if self.effects else None,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # -- human-readable --------------------------------------------------------------

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Curation scorecard — `{self.dataset.dataset_id}`")
        lines.append("")
        s = self.summary
        lines.append(
            f"**{s.num_removed}/{s.num_episodes}** episodes removed "
            f"({s.pct_removed:.1f}%); **{s.num_kept}** kept."
        )
        if self.baseline is not None:
            lines.append(
                f"Paired equal-N random baseline keeps **{self.baseline.n}** episodes "
                "(same size) for a confound-free comparison."
            )
        lines.append("")
        lines.append("## Signals")
        lines.append("")
        if self.per_signal:
            lines.append("| signal | scored | skipped | min | median | max | orientation |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
            for sig in self.per_signal:
                orient = "higher=better" if sig.higher_is_better else "lower=better"
                lines.append(
                    f"| {sig.name} | {sig.num_scored} | {sig.num_skipped} | "
                    f"{_fmt(sig.minimum)} | {_fmt(sig.median)} | {_fmt(sig.maximum)} | "
                    f"{orient} |"
                )
        else:
            lines.append("_No signals were run._")
        lines.append("")
        lines.append("## Removed episodes")
        lines.append("")
        removed = [f for f in self.flags if not f.kept]
        if removed:
            for flag in removed:
                lines.append(f"- **episode {flag.episode_index}** — {flag.reason}")
        else:
            lines.append("_Nothing was removed._")
        lines.append("")
        if self.effects is not None:
            e = self.effects
            lines.append("## Downstream effect")
            lines.append("")
            lines.append(
                f"{e.metric}: **{e.effect:+.3f}** (95% CI [{e.ci_low:+.3f}, {e.ci_high:+.3f}])"
            )
        else:
            lines.append(
                "_No downstream policy evaluation attached; this scorecard makes no "
                "claim about training gains._"
            )
        lines.append("")
        return "\n".join(lines)

    def to_hf_dataset_card(self) -> str:
        """Return a Hugging Face dataset-card fragment summarizing the curation."""
        s = self.summary
        front = [
            "---",
            "tags:",
            "  - robocurate",
            "  - curated",
            "---",
            "",
            "## RoboCurate curation summary",
            "",
            f"- Source: `{self.dataset.dataset_id}`",
            f"- Episodes removed: {s.num_removed}/{s.num_episodes} ({s.pct_removed:.1f}%)",
            f"- Signals: {', '.join(sig.name for sig in self.per_signal) or 'none'}",
        ]
        if self.baseline is not None:
            front.append(f"- Equal-N random baseline size: {self.baseline.n}")
        front.append("")
        front.append("See the run manifest for the full per-episode rationale.")
        front.append("")
        return "\n".join(front)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _signal_report_from_decisions(
    spec: Mapping[str, Any], decisions: list[Mapping[str, Any]]
) -> SignalReport:
    """Summarize one signal's distribution from the manifest's decision-level values.

    The manifest stores a scalar value per (signal, episode) but not the score orientation,
    so ``higher_is_better`` reports the neutral default; min/median/max are computed over the
    finite values actually present, and skips (``NaN`` / missing) are counted, never invented.
    """
    name = spec["name"]
    values = np.array(
        [d.get("signal_values", {}).get(name, np.nan) for d in decisions], dtype=np.float64
    )
    finite = values[np.isfinite(values)]
    num_scored = int(finite.size)
    return SignalReport(
        name=name,
        description=spec.get("description", ""),
        higher_is_better=True,
        num_scored=num_scored,
        num_skipped=int(values.size - num_scored),
        minimum=float(finite.min()) if num_scored else None,
        median=float(np.median(finite)) if num_scored else None,
        maximum=float(finite.max()) if num_scored else None,
    )


def build_scorecard(result: CurationResult, *, effects: EffectReport | None = None) -> Scorecard:
    """Construct a :class:`Scorecard` from a :class:`~robocurate.curator.CurationResult`."""
    matrix = result.score_matrix
    summary = QualitySummary(
        num_episodes=len(result.decisions),
        num_kept=result.num_kept,
        num_removed=result.num_removed,
    )

    per_signal: list[SignalReport] = []
    for spec in matrix.signal_specs:
        values = matrix.signal_values(spec.name)
        finite = values[np.isfinite(values)]
        num_scored = int(finite.size)
        num_skipped = int(values.size - finite.size)
        per_signal.append(
            SignalReport(
                name=spec.name,
                description=spec.description,
                higher_is_better=_signal_orientation(result, spec.name),
                num_scored=num_scored,
                num_skipped=num_skipped,
                minimum=float(finite.min()) if num_scored else None,
                median=float(np.median(finite)) if num_scored else None,
                maximum=float(finite.max()) if num_scored else None,
            )
        )

    flags = tuple(
        TrajectoryFlag(
            episode_index=d.episode_index,
            fingerprint=d.fingerprint,
            kept=d.kept,
            reason=d.reason,
            signal_values=d.signal_values,
        )
        for d in result.decisions
    )

    return Scorecard(
        schema_version=SCORECARD_SCHEMA_VERSION,
        dataset=result.build_manifest().source,
        summary=summary,
        per_signal=tuple(per_signal),
        flags=flags,
        baseline=result.baseline,
        effects=effects,
    )


def _signal_orientation(result: CurationResult, signal_name: str) -> bool:
    for ref in result.score_matrix.refs:
        score = result.score_matrix.scores.get((signal_name, ref.fingerprint))
        if score is not None:
            return score.higher_is_better
    return True


__all__ = [
    "SCORECARD_SCHEMA_VERSION",
    "EffectReport",
    "QualitySummary",
    "Scorecard",
    "SignalReport",
    "TrajectoryFlag",
    "build_scorecard",
]
