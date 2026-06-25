"""Generate a tiny synthetic demo LeRobotDataset for the onboarding tutorial.

This is the first command in the "run this first" path (see docs/GETTING_STARTED.md). It
writes a deterministic mix of clearly-good (smooth, direct) and clearly-bad (jittery,
wandering) episodes so that running a curation signal visibly removes the bad ones. No
GPU, no network, no optional extras.

Usage::

    python examples/make_demo_dataset.py [path] [--episodes N] [--seed S]

Example::

    uv run python examples/make_demo_dataset.py ./demo_dataset
"""

from __future__ import annotations

import argparse

from robocurate.examples import write_demo_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="./demo_dataset",
        help="destination directory (must not already exist). Default: ./demo_dataset",
    )
    parser.add_argument(
        "--episodes", type=int, default=8, help="number of episodes to generate (default 8)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="master seed for deterministic output (default 0)."
    )
    args = parser.parse_args()

    out = write_demo_dataset(args.path, num_episodes=args.episodes, seed=args.seed)
    print(f"wrote demo dataset: {out}")
    print(f"next: robocurate curate {out} --out ./demo_curated --signals jerk --budget 0.8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
