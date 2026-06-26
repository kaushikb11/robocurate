"""Runnable open-benchmark v0 proof on the synthetic identity dataset (CPU; `policy` extra).

What this shows end-to-end, with zero data plumbing:

1. Build the synthetic identity dataset (helpful episodes have ``action ≈ observation`` — the
   rewarded task; a harmful minority have ``action ≈ -observation``).
2. ``build_spec`` freezes a pool + a fixed held-out eval split + a fixed BC training config.
3. Score TWO submissions as raw index-sets:
   * a **good** one — keep only the helpful episodes;
   * a **random** one — an equal-size arbitrary slice (here, includes the harmful episodes).
4. Print a leaderboard: the good submission wins (clearly lower held-out loss, a negative,
   ``separated`` effect vs the equal-N random control).

Honest framing: held-out BC loss is a CPU *proxy* with a coverage bias toward the random
control (printed in the result + leaderboard caveat). This is a scaffolding proof, not the
field's benchmark. Run with: ``uv run --extra policy python examples/benchmark_identity.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from robocurate.benchmark import append_entry, build_spec, run_submission
from robocurate.benchmark.leaderboard import load_leaderboard
from robocurate.experiment.synthetic import make_identity_experiment_dataset

NUM_HELPFUL = 12
NUM_HARMFUL = 4


def _write_index_submission(path: Path, indices: list[int]) -> None:
    path.write_text(json.dumps({"kept_episode_indices": indices}), encoding="utf-8")


def main() -> None:
    reader = make_identity_experiment_dataset(
        num_helpful=NUM_HELPFUL, num_harmful=NUM_HARMFUL, seed=0
    )
    # A small, fast training config so the whole proof runs in seconds on a CPU.
    spec = build_spec(
        reader, eval_frac=0.25, seed=0, training={"hidden_dim": 32, "epochs": 120, "lr": 0.01}
    )

    helpful = set(range(NUM_HELPFUL))  # episodes 0..11 are the helpful (rewarded) majority
    train_pool = list(spec.train_pool_indices)
    good_indices = sorted(i for i in train_pool if i in helpful)
    # A "bad" submission of the same size that deliberately includes the harmful episodes.
    harmful = [i for i in train_pool if i not in helpful]
    bad_indices = sorted((harmful + [i for i in train_pool if i in helpful])[: len(good_indices)])

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        good_path = tmpdir / "good.json"
        bad_path = tmpdir / "bad.json"
        board_path = tmpdir / "leaderboard.json"
        _write_index_submission(good_path, good_indices)
        _write_index_submission(bad_path, bad_indices)

        for path, name in ((good_path, "helpful_only"), (bad_path, "includes_harmful")):
            result = run_submission(spec, path, reader, seeds=(0, 1, 2))
            append_entry(board_path, result, name=name, created_utc="1970-01-01T00:00:00Z")

        board = load_leaderboard(board_path)
        print(board.to_markdown())


if __name__ == "__main__":
    main()
