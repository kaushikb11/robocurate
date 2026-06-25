"""Real-data scorecard: validate RoboCurate's cheap signals against ground truth.

This is the "wow demo" on a *real* public dataset — no GPU, no sim. It runs RoboCurate's
GPU-free signals over the robomimic **Multi-Human (MH)** demonstrations and checks them against
the dataset's ground-truth operator-proficiency labels (``better`` / ``okay`` / ``worse``),
which robomimic stores as ``mask/<tier>`` filter keys.

Why this is an honest test (Invariants 5 & 6): the MH demos are collected by
operators of differing skill, so "which trajectories are worse" is *known* independently of any
signal. We can therefore measure, not assert, whether a cheap signal recovers proficiency:

  1. **Diagnostic** — for each signal, the rank-AUC that a random ``worse`` demo scores worse
     than a random ``better`` demo (0.5 = no separation; 1.0 = perfect).
  2. **Curation** — drop the worst third by the chosen signal and report the operator-tier
     composition of what was removed, the *enrichment* of ``worse`` demos over their base rate,
     and the same numbers for an equal-size **random baseline** (the confound control).

The headline finding on ``lift`` is deliberately reported warts-and-all: ``action_noise``
recovers proficiency well, ``jerk`` does not, and naively stacking signals *dilutes* the good
one — a result you can only trust because it is measured against ground truth and a random
control.

Usage (downloads ~50-120 MB on first run, into experiments/data/, gitignored):

    uv run --extra robomimic python experiments/robomimic_scorecard.py
    uv run --extra robomimic python experiments/robomimic_scorecard.py --task can
"""

from __future__ import annotations

import argparse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

from robocurate.adapters import RoboMimicReader
from robocurate.curator import Budget, Curator
from robocurate.signals import get as get_signal
from robocurate.signals.base import Signal

DATA_DIR = Path(__file__).parent / "data"
HF_BASE = "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5"
TIERS = ("better", "okay", "worse")
# Cheap (Tier-0, CPU-only) signals to put on trial against the operator labels.
DIAGNOSTIC_SIGNALS = (
    "jerk",
    "action_noise",
    "redundancy",
    "path_efficiency",
    "spectral_smoothness",
)
# path_efficiency / spectral_smoothness measure the true Cartesian end-effector path (exposed by
# RoboMimicReader), which is more faithful than integrating the normalized OSC delta actions.
EEF_KEY = "observation.robot0_eef_pos"
KEEP_FRACTION = 0.67  # drop the worst third


def build_signal(name: str) -> Signal:
    """Construct a diagnostic/curation signal by name, configuring robomimic-specific inputs."""
    if name == "path_efficiency":
        from robocurate.signals.path_efficiency import PathEfficiency

        return PathEfficiency(source=EEF_KEY, motion="positions")
    if name == "spectral_smoothness":
        from robocurate.signals.spectral_smoothness import SpectralSmoothness

        return SpectralSmoothness(source=EEF_KEY, motion="positions")
    return get_signal(name)


def ensure_dataset(task: str) -> Path:
    """Return a local path to the MH low_dim HDF5 for ``task``, downloading it if absent."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{task}_mh_low_dim_v15.hdf5"
    if not path.exists():
        url = f"{HF_BASE}/{task}/mh/low_dim_v15.hdf5"
        print(f"downloading {task} MH demos from {url} ...")
        urllib.request.urlretrieve(url, path)
    return path


def rank_auc(higher: np.ndarray, lower: np.ndarray) -> float:
    """P(a random ``higher`` value > a random ``lower`` value), ties counted as 0.5."""
    if higher.size == 0 or lower.size == 0:
        return float("nan")
    comparisons = higher[:, None] - lower[None, :]
    wins = float((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum())
    return wins / (higher.size * lower.size)


def _orientation(matrix: object, name: str) -> bool:
    """Whether ``name`` is higher-is-better, read from any of its scores in the matrix."""
    for ref in matrix.refs:  # type: ignore[attr-defined]
        score = matrix.scores.get((name, ref.fingerprint))  # type: ignore[attr-defined]
        if score is not None:
            return bool(score.higher_is_better)
    return False


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two arrays (finite pairs only), without scipy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="lift", choices=["lift", "can", "square"])
    parser.add_argument(
        "--signal",
        default="action_noise",
        help="signal to curate with (default: action_noise, the one that works on lift)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = ensure_dataset(args.task)
    reader = RoboMimicReader(path)
    tier_of = {t.meta.episode_index: t.meta.extra.get("operator_tier") for t in reader}
    counts = Counter(tier_of.values())
    print(
        f"\nrobomimic/{args.task} MH: {len(reader)} demos "
        f"(better={counts['better']}, okay={counts['okay']}, worse={counts['worse']})"
    )

    # 1) Diagnostic: how well does each cheap signal recover operator proficiency?
    diag = Curator([build_signal(s) for s in DIAGNOSTIC_SIGNALS], budget=None, seed=args.seed).run(
        reader
    )
    matrix = diag.score_matrix
    order = [r.episode_index for r in matrix.refs]
    tiers = np.array([tier_of[i] for i in order])
    lengths = np.array([reader.read_episode(i).meta.num_steps for i in order], dtype=float)

    def separation_auc(values: np.ndarray, higher_is_better: bool) -> float:
        """Oriented AUC: P(the signal ranks a 'worse' demo as lower-quality than a 'better')."""
        worse = values[tiers == "worse"]
        better = values[tiers == "better"]
        worse, better = worse[np.isfinite(worse)], better[np.isfinite(better)]
        # lower-quality = lower score if higher_is_better, else higher score.
        return rank_auc(better, worse) if higher_is_better else rank_auc(worse, better)

    print("\n=== Signal diagnostics vs ground-truth operator tiers ===")
    print(f"{'signal':16s} {'better':>9s} {'okay':>9s} {'worse':>9s}   skill-separation AUC")
    for name in DIAGNOSTIC_SIGNALS:
        vals = matrix.signal_values(name)
        means = {tr: float(np.nanmean(vals[tiers == tr])) for tr in TIERS}
        hib = _orientation(matrix, name)
        auc = separation_auc(vals, hib)
        verdict = "recovers skill" if auc >= 0.65 else "~ no separation"
        print(
            f"{name:16s} {means['better']:9.3f} {means['okay']:9.3f} {means['worse']:9.3f}   "
            f"{auc:5.3f}  ({verdict})"
        )
    # A non-signal reference: trajectory length (efficiency) is the strongest cheap predictor.
    len_means = {tr: float(lengths[tiers == tr].mean()) for tr in TIERS}
    len_auc = separation_auc(lengths, higher_is_better=False)  # longer = worse
    print(
        f"{'(length)':16s} {len_means['better']:9.1f} {len_means['okay']:9.1f} "
        f"{len_means['worse']:9.1f}   {len_auc:5.3f}  (reference: efficiency)"
    )

    # 2) Curate with the chosen signal; report the scorecard.
    result = Curator(
        [build_signal(args.signal)], budget=Budget.fraction(KEEP_FRACTION), seed=args.seed
    ).run(reader)
    print("\n" + result.scorecard().to_markdown())

    # 3) Validation against ground truth + the equal-N random control.
    removed = [d.episode_index for d in result.decisions if not d.kept]
    base_rate = counts["worse"] / len(reader)
    rc = Counter(tier_of[i] for i in removed)
    worse_frac = rc["worse"] / len(removed)
    recall = rc["worse"] / counts["worse"]

    # The baseline keeps a random equal-N subset; everything else is what *random* would drop.
    kept_random = set(result.baseline.kept_episode_indices) if result.baseline else set()
    random_removed = [i for i in tier_of if i not in kept_random]
    rand_worse_frac = (
        Counter(tier_of[i] for i in random_removed)["worse"] / len(random_removed)
        if random_removed
        else float("nan")
    )

    print(
        f"=== Validation: curating by '{args.signal}' (dropped {len(removed)} of {len(reader)}) ==="
    )
    print(f"removed-set tiers: {dict(rc)}")
    print(
        f"worse-operator share of removed:  {worse_frac:.2f}  "
        f"(base rate {base_rate:.2f}  ->  {worse_frac / base_rate:.2f}x enrichment)"
    )
    print(
        f"recall of known worse demos:      {recall:.2f}  "
        f"(a random drop would catch ~{base_rate:.2f})"
    )
    print(
        f"equal-N random control removes worse at: {rand_worse_frac:.2f} share (the confound check)"
    )
    print(
        "\nHonest read: the AUC above is a DIAGNOSTIC (does the signal track operator "
        "experience), NOT a validation. Recovering operator labels does not imply better "
        "curation — CUPID showed on robomimic that perceived quality can diverge from what "
        "maximizes policy success, and directness/smoothness risk deleting the rare, high-value "
        "recovery/corrective demos. It is still useful (the ORIENTED diagnostic shows "
        "path_efficiency / action_noise track skill, jerk is flat, and redundancy's keep-"
        "direction is backwards here). But the only real proof is the downstream gate: a BC "
        "policy on the curated subset beating BOTH an equal-N AND a length-matched random "
        "subset, CI-separated, with a non-floored baseline "
        "(experiments/robomimic_bc_validation_modal.py)."
    )

    # 4) Confound probe: is the curation just "drop the short / easy episodes"? Worse-operator
    # demos are longer (length AUC ~0.95), so a signal that mostly tracks length would select
    # short episodes — and a downstream win could then be a length confound, not a skill effect.
    len_by_ep = dict(zip(order, lengths, strict=True))
    directness = matrix.signal_values("path_efficiency")
    rho = spearman(directness, lengths)
    print("\n=== Confound probe: is the gain just 'keep the short/easy episodes'? ===")
    print(
        f"Spearman corr(path_efficiency directness, episode length) = {rho:+.2f}  "
        "(near 0 -> directness is NOT merely length; strongly negative -> it mostly IS length)"
    )
    full_len = float(np.mean(lengths))
    for sig in ("path_efficiency", "action_noise"):
        res = Curator(
            [build_signal(sig)], budget=Budget.fraction(KEEP_FRACTION), seed=args.seed
        ).run(reader)
        kept = [d.episode_index for d in res.decisions if d.kept]
        rnd = list(res.baseline.kept_episode_indices) if res.baseline else []
        cur_len = float(np.mean([len_by_ep[i] for i in kept]))
        rnd_len = float(np.mean([len_by_ep[i] for i in rnd])) if rnd else float("nan")
        print(
            f"  {sig:15s} curated-kept mean length {cur_len:6.1f}  vs equal-N random {rnd_len:6.1f}"
            f"  vs full {full_len:6.1f}"
        )
    print(
        "  If curated-kept length is close to the random/full length, curation is keeping demos "
        "on quality, not just shortness."
    )


if __name__ == "__main__":
    main()
