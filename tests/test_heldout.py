"""Tests for the sim-free held-out BC-loss evaluator (needs the policy extra; torch).

Known answer (the evaluator mechanics, no signal): a policy trained on *clean* identity data
(action = observation) predicts held-out clean actions far better — lower MSE — than one trained
on *contradictory* data (action = -observation). Plus a smoke test that the full
curated-vs-controls comparison returns a well-formed report on a held-out split.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.experiment.heldout import compare_curation_heldout, held_out_bc_loss
from robocurate.experiment.policies import BCPolicy
from robocurate.experiment.synthetic import make_identity_experiment_dataset
from robocurate.signals.jerk import Jerk

pytestmark = pytest.mark.ml


def _clean_reader(n: int, *, seed: int) -> InMemoryDatasetReader:
    ds = make_identity_experiment_dataset(num_helpful=n, num_harmful=0, seed=seed)
    return ds


def _contradictory_reader(n: int, *, seed: int) -> InMemoryDatasetReader:
    ds = make_identity_experiment_dataset(num_helpful=0, num_harmful=n, seed=seed)
    return ds


def test_held_out_loss_lower_for_clean_than_contradictory() -> None:
    # Held-out set is clean (the rewarded identity task). A policy trained on clean data should
    # predict it well; one trained on action=-observation data should predict it badly.
    val = _clean_reader(6, seed=999)
    clean_train = _clean_reader(10, seed=0)
    bad_train = _contradictory_reader(10, seed=0)
    policy = BCPolicy(epochs=200)

    clean_loss = held_out_bc_loss(policy, clean_train, val, seed=0)
    bad_loss = held_out_bc_loss(policy, bad_train, val, seed=0)
    assert np.isfinite(clean_loss) and np.isfinite(bad_loss)
    assert clean_loss < bad_loss  # the known answer
    assert clean_loss < 0.1  # clean identity is easy to imitate


def test_held_out_loss_is_deterministic() -> None:
    val = _clean_reader(4, seed=7)
    train = _clean_reader(8, seed=1)
    policy = BCPolicy(epochs=100)
    a = held_out_bc_loss(policy, train, val, seed=3)
    b = held_out_bc_loss(policy, train, val, seed=3)
    assert a == b


def test_compare_curation_heldout_returns_wellformed_report() -> None:
    reader = make_identity_experiment_dataset(num_helpful=16, num_harmful=6, seed=0)
    report = compare_curation_heldout(
        reader, Jerk(), budget=0.67, seeds=(0, 1), val_frac=0.25, epochs=80
    )
    assert report["signal"] == "jerk"
    assert report["n_val"] >= 1 and report["n_curated"] >= 1
    for arm in ("full", "random", "random_steps", "curated"):
        assert len(report["loss_by_arm"][arm]) == 2  # one per seed
        assert all(np.isfinite(v) for v in report["loss_by_arm"][arm])
    # effect dicts carry the paired-difference fields + a separation verdict
    for key in ("curated_vs_random", "curated_vs_random_steps"):
        eff = report[key]
        assert {"effect", "ci_low", "ci_high", "n", "separated"} <= set(eff)
