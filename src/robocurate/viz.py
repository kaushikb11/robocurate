"""Optional Matplotlib visualizations for curation scorecards (``viz`` extra).

Three plots that turn a :class:`~robocurate.curator.CurationResult` /
:class:`~robocurate.scorecard.Scorecard` into PNGs: per-signal score histograms, a
kept-vs-removed summary (with the equal-N baseline size, invariant 5), and an optional
per-signal box/strip plot grouped by a per-episode label (e.g. operator tier).

Matplotlib is an optional dependency: it is imported lazily inside each function (the module
imports cleanly without it), and the non-interactive Agg backend is forced before pyplot so
the plots render headless. Constructing a plot without the extra raises a clear "install
``robocurate[viz]``" error, mirroring the torch-backed signals.

Nothing here is in the selection path; these are reporting helpers only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from robocurate.curator import CurationResult
    from robocurate.scorecard import Scorecard


def _require_matplotlib() -> Any:
    """Import pyplot with the Agg backend forced, or raise an actionable install error."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless, non-interactive; must precede the pyplot import
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the 'viz' helpers require Matplotlib, which is an optional dependency. "
            "Install it with `uv pip install 'robocurate[viz]'`."
        ) from exc
    return plt


def plot_signal_distributions(result: CurationResult, out_path: str | Path) -> Path:
    """Plot a grid of per-signal histograms of the raw per-trajectory scores.

    One subplot per signal over the finite (non-skipped) values from
    ``result.score_matrix.signal_values``, titled with the signal name and orientation. Saved
    to ``out_path``; returns the output :class:`~pathlib.Path`.
    """
    from pathlib import Path

    plt = _require_matplotlib()
    matrix = result.score_matrix
    specs = list(matrix.signal_specs)

    n = max(1, len(specs))
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)
    flat = [ax for row in axes for ax in row]

    for ax, spec in zip(flat, specs, strict=False):
        values = matrix.signal_values(spec.name)
        finite = values[np.isfinite(values)]
        orient = "higher=better" if _orientation(result, spec.name) else "lower=better"
        if finite.size:
            ax.hist(finite, bins=min(20, max(1, finite.size)), color="#4c72b0", edgecolor="white")
        else:
            ax.text(0.5, 0.5, "no scored values", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{spec.name} ({orient})")
        ax.set_xlabel("score")
        ax.set_ylabel("count")

    for ax in flat[len(specs) :]:  # hide unused cells in the grid
        ax.set_visible(False)

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def plot_curation_summary(scorecard: Scorecard, out_path: str | Path) -> Path:
    """Plot kept vs removed counts, plus the equal-N baseline size if a baseline is present.

    Saved to ``out_path``; returns the output :class:`~pathlib.Path`.
    """
    from pathlib import Path

    plt = _require_matplotlib()
    summary = scorecard.summary

    labels = ["kept", "removed"]
    counts = [summary.num_kept, summary.num_removed]
    colors = ["#55a868", "#c44e52"]
    if scorecard.baseline is not None:
        labels.append("baseline (N)")
        counts.append(scorecard.baseline.n)
        colors.append("#8172b3")

    fig, ax = plt.subplots(figsize=(1.6 * len(labels) + 2.0, 4.0))
    bars = ax.bar(labels, counts, color=colors)
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("episodes")
    ax.set_title(
        f"Curation summary — {summary.num_removed}/{summary.num_episodes} removed "
        f"({summary.pct_removed:.1f}%)"
    )

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def plot_scores_by_group(
    result: CurationResult,
    groups: Mapping[int, str],
    signal_name: str,
    out_path: str | Path,
) -> Path:
    """Plot one signal's per-trajectory values as a box+strip plot grouped by a label.

    ``groups`` maps ``episode_index -> group string`` (e.g. operator tier). Episodes without a
    finite score or without a group label are dropped. Saved to ``out_path``; returns the
    output :class:`~pathlib.Path`.
    """
    from pathlib import Path

    plt = _require_matplotlib()
    matrix = result.score_matrix
    values = matrix.signal_values(signal_name)

    by_group: dict[str, list[float]] = {}
    for ref, value in zip(matrix.refs, values, strict=True):
        label = groups.get(ref.episode_index)
        if label is None or not np.isfinite(value):
            continue
        by_group.setdefault(label, []).append(float(value))

    ordered = sorted(by_group)
    data = [by_group[label] for label in ordered]

    fig, ax = plt.subplots(figsize=(1.5 * max(1, len(ordered)) + 2.0, 4.5))
    if data:
        positions = list(range(1, len(ordered) + 1))
        ax.boxplot(data, positions=positions, tick_labels=ordered, showfliers=False)
        rng = np.random.default_rng(0)  # seeded jitter only; not in the selection path
        for pos, points in zip(positions, data, strict=True):
            jitter = pos + rng.uniform(-0.12, 0.12, size=len(points))
            ax.scatter(jitter, points, s=14, color="#4c72b0", alpha=0.7, zorder=3)
    else:
        ax.text(0.5, 0.5, "no grouped values", ha="center", va="center", transform=ax.transAxes)
    orient = "higher=better" if _orientation(result, signal_name) else "lower=better"
    ax.set_title(f"{signal_name} by group ({orient})")
    ax.set_ylabel("score")

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def _orientation(result: CurationResult, signal_name: str) -> bool:
    """Return the ``higher_is_better`` orientation a signal reported (default True)."""
    for ref in result.score_matrix.refs:
        score = result.score_matrix.scores.get((signal_name, ref.fingerprint))
        if score is not None:
            return score.higher_is_better
    return True


__all__ = [
    "plot_curation_summary",
    "plot_scores_by_group",
    "plot_signal_distributions",
]
