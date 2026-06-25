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

import base64
import html
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

    def to_html(self) -> str:
        """Render a single self-contained HTML document for this curation run.

        The output is a standalone file (inline CSS, no external assets): a header with the
        dataset id, the quality summary, the per-signal distribution table (orientation,
        min/median/max, skips), the equal-N baseline (invariant 5), the removed-episodes table
        with per-episode reasons (invariant 6), and — only when a policy evaluation is
        attached — the downstream effect *with* its CI bounds (invariant 6).

        Matplotlib (the ``viz`` extra) is imported lazily: when present, the kept-vs-removed
        summary plot is base64-embedded as an ``<img>`` so the file stays self-contained; when
        absent, the report degrades gracefully to tables only. All free text is HTML-escaped.
        """
        parts: list[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append('<html lang="en">')
        parts.append("<head>")
        parts.append('<meta charset="utf-8">')
        parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
        parts.append(f"<title>RoboCurate scorecard — {_esc(self.dataset.dataset_id)}</title>")
        parts.append(f"<style>{_REPORT_CSS}</style>")
        parts.append("</head>")
        parts.append("<body>")
        parts.append('<main class="report">')

        parts.append(self._html_header())
        parts.append(self._html_summary())
        parts.append(self._html_summary_plot())
        parts.append(self._html_signals())
        parts.append(self._html_baseline())
        parts.append(self._html_removed())
        parts.append(self._html_effects())

        parts.append("</main>")
        parts.append("</body>")
        parts.append("</html>")
        return "\n".join(parts)

    # -- HTML section builders -------------------------------------------------------

    def _html_header(self) -> str:
        d = self.dataset
        return (
            "<header>"
            "<h1>RoboCurate curation scorecard</h1>"
            f'<p class="dataset-id"><code>{_esc(d.dataset_id)}</code></p>'
            f'<p class="meta">format: <code>{_esc(d.source_format)}</code> · '
            f"episodes: {d.num_episodes} · "
            f"content hash: <code>{_esc(d.content_hash)}</code></p>"
            "</header>"
        )

    def _html_summary(self) -> str:
        s = self.summary
        return (
            "<section>"
            "<h2>Quality summary</h2>"
            '<ul class="summary">'
            f"<li><strong>{s.num_removed}/{s.num_episodes}</strong> episodes removed "
            f"({s.pct_removed:.1f}%)</li>"
            f"<li><strong>{s.num_kept}</strong> kept</li>"
            "</ul>"
            "</section>"
        )

    def _html_summary_plot(self) -> str:
        """Embed the kept-vs-removed plot as a base64 PNG when ``viz`` is available."""
        try:
            from robocurate.viz import render_curation_summary_png
        except ImportError:  # pragma: no cover - exercised only without the extra
            return ""
        try:
            png = render_curation_summary_png(self)
        except ImportError:
            # Module imported but Matplotlib itself is missing: degrade to tables only.
            return ""
        encoded = base64.b64encode(png).decode("ascii")
        return (
            '<section class="figure">'
            f'<img alt="Kept vs removed episodes" '
            f'src="data:image/png;base64,{encoded}">'
            "</section>"
        )

    def _html_signals(self) -> str:
        rows: list[str] = []
        if self.per_signal:
            for sig in self.per_signal:
                orient = "higher = better" if sig.higher_is_better else "lower = better"
                rows.append(
                    "<tr>"
                    f"<td>{_esc(sig.name)}</td>"
                    f"<td>{_esc(sig.description)}</td>"
                    f'<td class="num">{sig.num_scored}</td>'
                    f'<td class="num">{sig.num_skipped}</td>'
                    f'<td class="num">{_esc(_fmt(sig.minimum))}</td>'
                    f'<td class="num">{_esc(_fmt(sig.median))}</td>'
                    f'<td class="num">{_esc(_fmt(sig.maximum))}</td>'
                    f"<td>{_esc(orient)}</td>"
                    "</tr>"
                )
            body = (
                "<table>"
                "<thead><tr>"
                "<th>signal</th><th>description</th>"
                '<th class="num">scored</th><th class="num">skipped</th>'
                '<th class="num">min</th><th class="num">median</th><th class="num">max</th>'
                "<th>orientation</th>"
                "</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>"
                "</table>"
            )
        else:
            body = "<p><em>No signals were run.</em></p>"
        return f"<section><h2>Signals</h2>{body}</section>"

    def _html_baseline(self) -> str:
        if self.baseline is None:
            return ""
        b = self.baseline
        return (
            "<section>"
            "<h2>Equal-N random baseline</h2>"
            f"<p>A paired random baseline (<code>{_esc(b.method)}</code>, seed "
            f"<code>{b.seed}</code>) keeps the <strong>same {b.n}</strong> episodes as the "
            "curated selection, so the dataset-size confound can be compared away.</p>"
            "</section>"
        )

    def _html_removed(self) -> str:
        removed = [f for f in self.flags if not f.kept]
        if not removed:
            return (
                "<section><h2>Removed episodes</h2><p><em>Nothing was removed.</em></p></section>"
            )
        rows: list[str] = []
        for flag in removed:
            values = ", ".join(
                f"{_esc(name)}={_esc(_fmt(value))}"
                for name, value in sorted(flag.signal_values.items())
            )
            rows.append(
                "<tr>"
                f'<td class="num">{flag.episode_index}</td>'
                f"<td>{_esc(flag.reason)}</td>"
                f"<td>{values or '—'}</td>"
                "</tr>"
            )
        return (
            "<section>"
            "<h2>Removed episodes</h2>"
            "<table>"
            '<thead><tr><th class="num">episode</th><th>reason</th>'
            "<th>signal values</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
            "</section>"
        )

    def _html_effects(self) -> str:
        if self.effects is None:
            return (
                "<section>"
                "<h2>Downstream effect</h2>"
                "<p><em>No downstream policy evaluation attached; this scorecard makes no "
                "claim about training gains.</em></p>"
                "</section>"
            )
        e = self.effects
        per_task = ""
        if e.per_task:
            rows = "".join(
                f'<tr><td>{_esc(task)}</td><td class="num">{value:+.3f}</td></tr>'
                for task, value in sorted(e.per_task.items())
            )
            per_task = (
                "<table>"
                '<thead><tr><th>task</th><th class="num">effect</th></tr></thead>'
                f"<tbody>{rows}</tbody>"
                "</table>"
            )
        return (
            "<section>"
            "<h2>Downstream effect</h2>"
            f"<p><strong>{_esc(e.metric)}: {e.effect:+.3f}</strong> "
            f"(95% CI [{e.ci_low:+.3f}, {e.ci_high:+.3f}])</p>"
            f"{per_task}"
            "</section>"
        )

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


def _esc(text: str) -> str:
    """HTML-escape free text (including quotes) so user-supplied strings can't break markup."""
    return html.escape(text, quote=True)


_REPORT_CSS = (
    "body{margin:0;background:#f6f7f9;color:#1a1d24;"
    "font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}"
    ".report{max-width:900px;margin:0 auto;padding:2rem;background:#fff;"
    "box-shadow:0 1px 3px rgba(0,0,0,.08)}"
    "h1{font-size:1.6rem;margin:0 0 .25rem}"
    "h2{font-size:1.15rem;margin:1.75rem 0 .5rem;border-bottom:1px solid #e3e6ea;"
    "padding-bottom:.25rem}"
    ".dataset-id code{font-size:1rem}"
    ".meta{color:#5b6472;font-size:.85rem;margin:.25rem 0 0;word-break:break-all}"
    "code{background:#f0f2f4;padding:.05rem .3rem;border-radius:3px;font-size:.85em}"
    "ul.summary{list-style:none;padding:0;display:flex;gap:2rem;flex-wrap:wrap}"
    "table{border-collapse:collapse;width:100%;margin:.5rem 0;font-size:.9rem}"
    "th,td{text-align:left;padding:.35rem .6rem;border-bottom:1px solid #eceef1}"
    "th{background:#fafbfc;font-weight:600}"
    ".num{text-align:right;font-variant-numeric:tabular-nums}"
    ".figure img{max-width:100%;height:auto}"
    "em{color:#5b6472}"
)


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
