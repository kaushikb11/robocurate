"""Sim-free downstream check on real data: curated-vs-control held-out BC loss (no Modal, no GPU).

This is the laptop-runnable, $0 cross-check of the Modal rollout gate
(``experiments/robomimic_bc_validation_modal.py``). It carves out a fixed held-out split of a
robomimic dataset, trains a small behavior-cloning policy on each arm (full / equal-N random /
length-matched random / curated), and compares the **action-prediction MSE on the held-out
split**. Lower loss = the subset taught a more predictive policy.

It is a *proxy*, not the faithful metric (closed-loop success is): held-out BC loss measures
imitation accuracy, which correlates with but is not task success. When this and the rollout
gate agree on which signal helps, that is a cheap, double-confirmed result; disagreement is
itself worth reporting (invariant 6).

Usage (uses the local robomimic data downloaded by robomimic_scorecard.py; needs torch + h5py):

    uv run --extra policy --extra robomimic python experiments/heldout_validation.py --task lift
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robocurate.adapters import RoboMimicReader
from robocurate.experiment.heldout import compare_curation_heldout
from robocurate.signals.action_noise import ActionNoise
from robocurate.signals.path_efficiency import PathEfficiency

DATA_DIR = Path(__file__).parent / "data"
EEF = "observation.robot0_eef_pos"  # the true Cartesian end-effector path RoboMimicReader exposes


def build_signal(name: str) -> object:
    if name == "path_efficiency":
        return PathEfficiency(source=EEF, dims=None, motion="positions")
    if name == "action_noise":
        return ActionNoise(source=EEF)
    raise SystemExit(f"unknown signal {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="lift", choices=["lift", "can", "square"])
    parser.add_argument(
        "--signal", default="path_efficiency", choices=["path_efficiency", "action_noise"]
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    path = DATA_DIR / f"{args.task}_mh_low_dim_v15.hdf5"
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `python experiments/robomimic_scorecard.py --task "
            f"{args.task}` first to download it."
        )
    reader = RoboMimicReader(path)
    report = compare_curation_heldout(
        reader,
        build_signal(args.signal),  # type: ignore[arg-type]
        seeds=tuple(range(args.seeds)),
        val_frac=args.val_frac,
        epochs=args.epochs,
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"\nrobomimic/{args.task} — held-out BC loss (sim-free proxy), curating by "
        f"{report['signal']} ({report['n_curated']} kept of {report['n_train_pool']} train pool, "
        f"{report['n_val']} held-out):"
    )
    for arm in ("full", "random", "random_steps", "curated"):
        m = report["mean_loss_by_arm"][arm]
        print(f"  {arm:14s} loss {m['mean']:.4f}  (95% CI [{m['ci_low']:.4f}, {m['ci_high']:.4f}])")
    controls = (
        ("curated_vs_random", "equal-N random"),
        ("curated_vs_random_steps", "len-matched random"),
    )
    for key, control in controls:
        e = report[key]
        better = "curated LOWER loss" if e["effect"] < 0 else "curated higher loss"
        sep = "CI-separated" if e["separated"] else "CI overlaps 0"
        print(f"  curated vs {control:18s} Δloss = {e['effect']:+.4f} ({better}; {sep})")
    print(
        "\nLower loss is better; a real improvement is a NEGATIVE, CI-separated Δ vs BOTH "
        "controls. This is a proxy — cross-check against the Modal rollout gate before claiming "
        "a signal helps."
    )


if __name__ == "__main__":
    main()
