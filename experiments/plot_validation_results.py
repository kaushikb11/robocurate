"""Plot downstream BC-validation results: curated signals vs an equal-N random baseline.

This renders the headline fairness comparison (Invariant 5): a curated subset is
only worth anything if it beats an *equal-size random* subset on a downstream policy. The Modal
validation harness (``experiments/robomimic_bc_validation_modal.py``) trains a robomimic BC
policy on each arm and rolls it out; this script turns its JSON dump into a bar chart you can
put in front of a skeptic.

The input is the JSON emitted by::

    modal run experiments/robomimic_bc_validation_modal.py::main --as-json > results.json

which is a LIST of per-run dicts, one per (arm, signal, seed)::

    {"arm": "full",    "signal": "-",               "seed": 0, "success": 0.42, "ok": true}
    {"arm": "random",  "signal": "-",               "seed": 1, "success": 0.38, "ok": true}
    {"arm": "curated", "signal": "path_efficiency", "seed": 2, "success": 0.55, "ok": true}
    {"arm": "curated", "signal": "action_noise",    "seed": 0, "success": 0.49, "ok": true}
    {"arm": "curated", "signal": "jerk",            "seed": 1, "error": "...", "ok": false}

Arms are ``full`` (all train data), ``random`` (the equal-N control, signal ``"-"``), and
``curated`` (one group per signal). ``success`` is a rollout success rate in ``[0, 1]``; a
negative value (e.g. ``-1.0``) means "no rollout ran" and is treated as missing.

Each group becomes one bar at the mean success across seeds, with the individual per-seed
points overlaid and a thin +/- standard-deviation error bar. The ``random`` baseline is drawn
distinctly (hatched) with a light reference line at its mean, and every ``curated:*`` bar is
annotated with its gap vs that random mean (the only number that actually matters).

Usage::

    python experiments/plot_validation_results.py results.json
    python experiments/plot_validation_results.py results.json --task can --out can.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np


def load_groups(path: Path) -> tuple[dict[str, list[float]], int, int]:
    """Read the results JSON and bucket usable successes into named groups.

    Returns ``(groups, n_failed, n_skipped)`` where ``groups`` maps a label
    (``"full"``, ``"random"``, or ``"curated:<signal>"``) to its list of per-seed success
    rates, ``n_failed`` counts runs with ``ok=false``, and ``n_skipped`` counts ok runs that
    were dropped for a missing/negative success (no rollout ran).
    """
    runs = json.loads(path.read_text())
    if not isinstance(runs, list):
        raise ValueError(f"expected a JSON list of run dicts, got {type(runs).__name__}")

    groups: dict[str, list[float]] = {}
    n_failed = 0
    n_skipped = 0
    for run in runs:
        if not run.get("ok", False):
            n_failed += 1
            continue
        success = run.get("success")
        if success is None or success < 0:
            n_skipped += 1
            continue
        arm = run.get("arm")
        if arm == "curated":
            label = f"curated:{run.get('signal')}"
        elif arm in ("full", "random"):
            label = arm
        else:
            n_skipped += 1
            continue
        groups.setdefault(label, []).append(float(success))
    return groups, n_failed, n_skipped


def ordered_labels(groups: dict[str, list[float]]) -> list[str]:
    """Order bars as full, random, then curated signals sorted by descending mean success."""
    curated = sorted(
        (lbl for lbl in groups if lbl.startswith("curated:")),
        key=lambda lbl: float(np.mean(groups[lbl])),
        reverse=True,
    )
    return [lbl for lbl in ("full", "random") if lbl in groups] + curated


def render(
    groups: dict[str, list[float]],
    labels: list[str],
    task: str,
    out: Path,
) -> dict[str, float]:
    """Draw the bar chart to ``out`` and return each group's mean success."""
    means = {lbl: float(np.mean(groups[lbl])) for lbl in labels}
    stds = {lbl: float(np.std(groups[lbl])) for lbl in labels}
    random_mean = means.get("random")

    fig, ax = plt.subplots(figsize=(max(7.0, 1.4 * len(labels)), 5.0))
    xs = np.arange(len(labels))

    for x, lbl in zip(xs, labels, strict=True):
        is_random = lbl == "random"
        color = "#bdbdbd" if is_random else ("#7f7f7f" if lbl == "full" else "#3b7dd8")
        ax.bar(
            x,
            means[lbl],
            width=0.66,
            color=color,
            hatch="//" if is_random else None,
            edgecolor="white" if is_random else "none",
            zorder=2,
            label="equal-N random (control)" if is_random else None,
        )
        ax.errorbar(
            x,
            means[lbl],
            yerr=stds[lbl],
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=4,
            zorder=4,
        )
        pts = np.asarray(groups[lbl], dtype=float)
        jitter = (np.random.RandomState(0).rand(pts.size) - 0.5) * 0.18
        ax.scatter(
            np.full(pts.size, x) + jitter,
            pts,
            s=26,
            color="#1a1a1a",
            alpha=0.65,
            zorder=5,
        )

    # Reference line at the random mean, plus the curated-vs-random gap annotations.
    if random_mean is not None:
        ax.axhline(random_mean, color="#bdbdbd", linestyle="--", linewidth=1.2, zorder=1)
        for x, lbl in zip(xs, labels, strict=True):
            if lbl.startswith("curated:"):
                gap = means[lbl] - random_mean
                ax.annotate(
                    f"{gap:+.2f}",
                    xy=(x, means[lbl]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color="#1a6e1a" if gap >= 0 else "#b22222",
                )

    pretty = [lbl.replace("curated:", "") for lbl in labels]
    ax.set_xticks(xs)
    ax.set_xticklabels(pretty, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("rollout success rate")
    ax.set_title(f"robomimic {task} MH — curated vs equal-N random (BC rollout success)")
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    if random_mean is not None:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return means


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="path to the validation results JSON (a list)")
    parser.add_argument("--out", type=Path, default=Path("validation.png"), help="output PNG path")
    parser.add_argument(
        "--task", default="square", help="robomimic task name for the title (JSON omits it)"
    )
    args = parser.parse_args()

    groups, n_failed, n_skipped = load_groups(args.results)
    if n_failed:
        print(f"note: {n_failed} run(s) failed (ok=false) and were excluded.")
    if n_skipped:
        print(f"note: {n_skipped} ok run(s) had no rollout (negative success) and were excluded.")

    if not groups:
        print("No usable runs in the results — nothing to plot. Not writing a file.")
        sys.exit(1)

    labels = ordered_labels(groups)
    means = render(groups, labels, args.task, args.out)
    print(f"saved plot to {args.out}")

    random_mean = means.get("random")
    parts = []
    for lbl in labels:
        piece = f"{lbl}={means[lbl]:.2f}"
        if random_mean is not None and lbl.startswith("curated:"):
            piece += f" ({means[lbl] - random_mean:+.2f} vs random)"
        parts.append(piece)
    print("summary: " + ", ".join(parts))


if __name__ == "__main__":
    main()
