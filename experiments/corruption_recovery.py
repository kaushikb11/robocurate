"""Known-answer corruption recovery: do the signals detect INJECTED defects (hard ground truth)?

Operator-tier AUC is a soft, contested proxy (an adversarial review found recovering it need not
mean good curation). This is the opposite test: take clean robomimic demos, damage a copy of each
in a KNOWN way, and measure whether each signal ranks the corrupted copy as worse. We report a
detection-AUC table per (corruption, signal) — 1.0 = always caught, 0.5 = blind. The honest point
is the blind spots: ``truncate`` / ``stall`` are structural / temporal defects a geometry signal
cannot see, and the table shows it (this is the data-valuation field's standard known-answer
check, and the kind of probe the audit literature says label-AUC hides).

Usage (uses the local robomimic data downloaded by robomimic_scorecard.py; lift by default):

    uv run --extra robomimic python experiments/corruption_recovery.py --task lift --n 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from robocurate.adapters import RoboMimicReader
from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.corruptions import CORRUPTIONS, corrupt
from robocurate.curator import Curator
from robocurate.signals.action_noise import ActionNoise
from robocurate.signals.jerk import Jerk
from robocurate.signals.path_efficiency import PathEfficiency
from robocurate.signals.redundancy import Redundancy
from robocurate.signals.spectral_smoothness import SpectralSmoothness
from robocurate.signals.structural_validity import StructuralValidity

DATA_DIR = Path(__file__).parent / "data"
EEF = "observation.robot0_eef_pos"  # the true Cartesian end-effector path RoboMimicReader exposes


def build_signals() -> list[tuple[str, object]]:
    """The cheap signals, configured to read the end-effector path where geometric."""
    return [
        ("jerk", Jerk(source=EEF)),
        ("action_noise", ActionNoise(source=EEF)),
        ("path_efficiency", PathEfficiency(source=EEF, dims=None, motion="positions")),
        ("spectral_smoothness", SpectralSmoothness(source=EEF, motion="positions")),
        ("redundancy", Redundancy()),
        # the structural verifier closes the geometry blind spot (truncate/stall)
        ("structural_validity", StructuralValidity()),
    ]


def rank_auc(higher: np.ndarray, lower: np.ndarray) -> float:
    """P(a random ``higher`` value > a random ``lower`` value), ties = 0.5."""
    if higher.size == 0 or lower.size == 0:
        return float("nan")
    comp = higher[:, None] - lower[None, :]
    return float((comp > 0).sum() + 0.5 * (comp == 0).sum()) / (higher.size * lower.size)


def detection_auc(values: np.ndarray, is_corrupt: np.ndarray, higher_is_better: bool) -> float:
    """Oriented AUC that the signal ranks corrupted demos as lower-quality than clean ones."""
    corrupt_vals = values[is_corrupt]
    clean_vals = values[~is_corrupt]
    corrupt_vals = corrupt_vals[np.isfinite(corrupt_vals)]
    clean_vals = clean_vals[np.isfinite(clean_vals)]
    # corrupted = lower quality = lower score if higher_is_better, else higher score.
    if higher_is_better:
        return rank_auc(clean_vals, corrupt_vals)
    return rank_auc(corrupt_vals, clean_vals)


def _orientation(matrix: object, name: str) -> bool:
    for ref in matrix.refs:  # type: ignore[attr-defined]
        score = matrix.scores.get((name, ref.fingerprint))  # type: ignore[attr-defined]
        if score is not None:
            return bool(score.higher_is_better)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="lift", choices=["lift", "can", "square"])
    parser.add_argument("--n", type=int, default=40, help="number of clean demos to corrupt")
    parser.add_argument("--severity", type=float, default=1.0)
    args = parser.parse_args()

    path = DATA_DIR / f"{args.task}_mh_low_dim_v15.hdf5"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run experiments/robomimic_scorecard.py --task "
            f"{args.task} first to download it."
        )
    reader = RoboMimicReader(path)
    n = min(args.n, len(reader))
    clean = [reader.read_episode(i) for i in range(n)]
    sigs = build_signals()

    print(
        f"\nrobomimic/{args.task} — corruption detection AUC ({n} clean demos, "
        f"severity {args.severity}); 1.0 = always caught, ~0.5 = blind\n"
    )
    header = f"{'corruption':12s} " + " ".join(f"{name:>13s}" for name, _ in sigs)
    print(header)
    print("-" * len(header))

    for kind in CORRUPTIONS:
        # Pair each clean demo with a corrupted copy; reindex so labels map unambiguously.
        corrupted = [
            corrupt(t, kind, feature=EEF, severity=args.severity, seed=i)
            for i, t in enumerate(clean)
        ]
        mixed = clean + corrupted
        is_corrupt_by_fp = {t.meta.fingerprint: (j >= n) for j, t in enumerate(mixed)}
        curator = Curator([s for _, s in sigs], budget=None, seed=0)
        result = curator.run(InMemoryDatasetReader(mixed))
        matrix = result.score_matrix
        labels = np.array([is_corrupt_by_fp[ref.fingerprint] for ref in matrix.refs])

        cells = []
        for name, _ in sigs:
            vals = matrix.signal_values(name)
            auc = detection_auc(vals, labels, _orientation(matrix, name))
            cells.append(f"{auc:13.2f}" if np.isfinite(auc) else f"{'skip':>13s}")
        print(f"{kind:12s} " + " ".join(cells))

    print(
        "\nHonest read: no SINGLE signal catches everything — on `truncate`, "
        "path_efficiency/spectral_smoothness INVERT (AUC ~0): an incomplete demo is shorter and "
        "straighter, so they rank it as *higher* quality, a structural-defect failure a label-AUC "
        "hides entirely. The fix is a complementary SUITE: the geometric signals catch kinematic "
        "defects (jitter/detour) and `structural_validity` catches the structural ones "
        "(truncate/stall, AUC ~1.0) the geometry misses. Takeaway: directness/smoothness are "
        "useful features but must never be standalone keep/drop filters — combine them with a "
        "structural verifier (and, ultimately, outcome-aware signals)."
    )


if __name__ == "__main__":
    main()
