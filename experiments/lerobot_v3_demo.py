"""Curate a real LeRobotDataset **v3.0** from the Hub and print a scorecard (laptop, no GPU).

The "LeRobot-native" promise, end to end on real data: download a real v3.0 Hub dataset (low-dim
parts only — no video), read it with the v3 adapter, curate it with the cheap CPU signals, and
print the scorecard (what was removed, why, + the equal-N random baseline). The source is never
mutated (invariant 1); a curated copy + manifest is what you'd save.

Defaults to ``lerobot/svla_so101_pickplace`` (50 SO-100 teleop episodes, ~12k frames, 2 cameras
recorded-but-not-loaded). Needs the ``lerobot`` extra (huggingface_hub) for the download; the v3
reader + curation themselves need only pyarrow (core).

    uv run --extra lerobot python experiments/lerobot_v3_demo.py
    uv run --extra lerobot python experiments/lerobot_v3_demo.py --repo-id lerobot/<name>
"""

from __future__ import annotations

import argparse

from robocurate.adapters.lerobot_v3 import LeRobotReaderV3
from robocurate.curator import Budget, Curator
from robocurate.signals.action_noise import ActionNoise
from robocurate.signals.jerk import Jerk
from robocurate.signals.redundancy import Redundancy
from robocurate.signals.structural_validity import StructuralValidity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="lerobot/svla_so101_pickplace")
    parser.add_argument("--budget", type=float, default=0.8, help="fraction of episodes to keep")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "this demo needs huggingface_hub to download the dataset; install the extra with "
            "`uv sync --extra lerobot` (the v3 reader + curation themselves need only pyarrow)."
        ) from exc

    print(f"downloading low-dim parts of {args.repo_id} (no video) ...")
    root = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["meta/*", "meta/**", "data/**"],  # skip the mp4 video shards
    )

    reader = LeRobotReaderV3(root)
    print(
        f"\nread {len(reader)} episodes from a real LeRobotDataset v3.0 "
        f"(robot={reader._embodiment.embodiment_id}, fps={reader._embodiment.control_hz})"
    )
    print(f"video features recorded but not loaded: {reader.meta.extra['video_features']}")

    # Cheap CPU signals only (no extras): smoothness, action-noise, dedup, and the structural
    # verifier. Curate to the requested budget; the source stays read-only.
    signals = [Jerk(), ActionNoise(), Redundancy(), StructuralValidity()]
    result = Curator(signals, budget=Budget.fraction(args.budget), seed=0).run(reader)

    print("\n" + result.scorecard().to_markdown())
    print(
        "Honest note: these cheap signals rank trajectories by kinematic smoothness / "
        "redundancy / structural validity — useful as a first pass, but on robomimic our "
        "downstream checks suggest they don't beat a random subset for policy quality. Treat the "
        "scorecard as a diagnostic; the real proof is a downstream policy eval."
    )


if __name__ == "__main__":
    main()
