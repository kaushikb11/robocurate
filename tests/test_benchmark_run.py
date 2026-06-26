"""End-to-end ``run_submission`` on the synthetic identity dataset (needs the policy extra).

Known answer (the benchmark mechanics, no signal): a submission that keeps only the **helpful**
episodes (``action ≈ observation`` — the rewarded task) trains a BC policy that predicts the
held-out clean actions far better — clearly LOWER held-out loss — than the equal-N random
control (which, drawn uniformly from the train pool, includes the harmful ``action ≈ -obs``
episodes). So the submission's effect vs equal-N is negative and ``separated``.

Plus determinism: the same spec + submission + seeds yield a byte-identical
``BenchmarkResult.to_dict()`` (Invariant 3). Kept fast: small dataset, few seeds, modest epochs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.benchmark.runner import run_submission
from robocurate.benchmark.spec import build_spec
from robocurate.experiment.synthetic import make_identity_experiment_dataset

pytestmark = pytest.mark.ml

NUM_HELPFUL = 12
NUM_HARMFUL = 4
TRAINING = {"hidden_dim": 32, "epochs": 120, "lr": 0.01}


def _write_indices(path: Path, indices: list[int]) -> None:
    path.write_text(json.dumps({"kept_episode_indices": indices}), encoding="utf-8")


def _pool() -> InMemoryDatasetReader:
    return make_identity_experiment_dataset(
        num_helpful=NUM_HELPFUL, num_harmful=NUM_HARMFUL, seed=0
    )


def test_good_submission_beats_equal_n_random(tmp_path: Path) -> None:
    reader = _pool()
    spec = build_spec(reader, eval_frac=0.25, seed=0, training=TRAINING)
    helpful = set(range(NUM_HELPFUL))  # episodes 0..11 are the helpful majority
    good = sorted(i for i in spec.train_pool_indices if i in helpful)

    sub = tmp_path / "good.json"
    _write_indices(sub, good)
    result = run_submission(spec, sub, reader, seeds=(0, 1, 2))

    submitted = result.mean_loss_by_arm["submitted"]["mean"]
    equal_n = result.mean_loss_by_arm["equal_n_random"]["mean"]
    eff = result.submitted_vs_equal_n

    # Known answer: helpful-only trains a much more predictive policy than the equal-N random
    # control (which includes the contradictory harmful episodes). Margin is large and clean.
    assert submitted < equal_n
    assert equal_n - submitted > 0.2  # a clear, not-marginal separation
    assert eff["effect"] < 0.0  # lower loss == a win == NEGATIVE effect
    assert eff["separated"] is True  # the bootstrap CI excludes zero
    # The coverage-bias caveat is always present (Invariant 6).
    assert "proxy" in result.note


def test_run_is_deterministic(tmp_path: Path) -> None:
    reader = _pool()
    spec = build_spec(reader, eval_frac=0.25, seed=0, training=TRAINING)
    helpful = set(range(NUM_HELPFUL))
    good = sorted(i for i in spec.train_pool_indices if i in helpful)
    sub = tmp_path / "good.json"
    _write_indices(sub, good)

    a = run_submission(spec, sub, reader, seeds=(0, 1))
    b = run_submission(spec, sub, reader, seeds=(0, 1))
    assert a.to_dict() == b.to_dict()
