"""Known-answer DOWNSTREAM test: curating out demonstrably-bad demos lowers held-out BC loss.

This is the end-to-end version of the corruption known-answer check (see
``tests/test_corruptions.py``), carried all the way through to the metric that actually matters:
**does the curated subset train a better policy than the equal-N random control (invariant 5)?**

Why this is a valid known-answer test (not a vibe check):

* We start from clean identity demos (action = observation — the rewarded task the
  :class:`~robocurate.experiment.policy.FakeEnvironment` is built around) and corrupt a *known*
  subset by injecting heavy ``jitter`` onto the **action** feature. Jittered action targets are
  ground-truth-bad for behavior cloning: BC regresses the policy onto the recorded actions, so
  noisy targets directly poison it. We know exactly which episodes are bad and why.
* The held-out validation split is a *separate, fully clean* identity dataset, so it measures
  "did the policy learn the true identity task", uncoupled from the training subset's
  distribution. This sidesteps the coverage bias documented in ``experiment/heldout.py`` (where a
  uniform-random held-out split favors a uniform-random training subset): a clean held-out set
  rewards *quality*, which is what we are testing.
* ``action_noise`` is precisely the signal designed to catch per-step action jitter, so curating
  with it should keep the clean demos and drop the corrupted ones. We assert detection (recall)
  *and* the downstream consequence (lower held-out loss than equal-N random).

The separation is enormous and stable: removing the jittered demos yields ~3 orders of magnitude
lower held-out loss than an equal-size random subset (which keeps several poisoned demos), every
seed. We assert a deliberately conservative margin (random loss > 5x curated loss per seed) so
the test is robust, not a knife-edge on the exact numbers.

Needs the ``policy`` extra (torch, CPU is fine).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import numpy as np

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.corruptions import corrupt
from robocurate.curator import Budget, Curator
from robocurate.experiment.conditions import SubsetReader
from robocurate.experiment.heldout import held_out_bc_loss
from robocurate.experiment.policies import BCPolicy
from robocurate.experiment.synthetic import make_identity_experiment_dataset
from robocurate.signals.action_noise import ActionNoise

pytestmark = pytest.mark.ml

# Counts/seeds chosen so the result is fast (seconds, CPU) and stable: 16 clean + 8 jittered
# demos, a 2/3 budget keeps 16 of 24 (enough room to drop all 8 bad), and a clean held-out set.
_N_CLEAN = 16
_N_BAD = 8
_JITTER_SEVERITY = 3.0  # heavy per-step action noise: unambiguous ground-truth-bad for BC
_BUDGET = 2.0 / 3.0
_SEEDS = (0, 1, 2, 3)


def _build_known_dataset(*, seed: int = 0) -> tuple[InMemoryDatasetReader, set[int]]:
    """Clean identity demos with a KNOWN jittered-action subset (episodes ``_N_CLEAN..``).

    Returns the reader and the set of known-bad episode indices. ``corrupt`` returns a new
    trajectory and never mutates its input, so the clean demos are untouched.
    """
    base = make_identity_experiment_dataset(num_helpful=_N_CLEAN + _N_BAD, num_harmful=0, seed=seed)
    bad_indices = set(range(_N_CLEAN, _N_CLEAN + _N_BAD))
    trajectories = []
    for i, traj in enumerate(base):
        if i in bad_indices:
            traj = corrupt(
                traj, "jitter", feature="action", severity=_JITTER_SEVERITY, seed=seed * 100 + i
            )
        trajectories.append(traj)
    reader = InMemoryDatasetReader(trajectories, dataset_id="synthetic/known-answer")
    return reader, bad_indices


def _clean_val_reader() -> InMemoryDatasetReader:
    """A separate, fully clean held-out identity set (different seed from training)."""
    return make_identity_experiment_dataset(num_helpful=6, num_harmful=0, seed=777)


def test_action_noise_detects_the_known_jittered_demos() -> None:
    # Detection half of the known answer: action_noise should remove exactly the jittered demos.
    reader, bad_indices = _build_known_dataset(seed=0)
    result = Curator([ActionNoise()], budget=Budget.fraction(_BUDGET), seed=0).run(reader)
    removed = set(result.removed_episode_indices)
    # Every known-bad demo is removed (the budget leaves room for exactly the clean ones).
    assert bad_indices <= removed
    # And the removals are *only* the bad ones (no clean demo collateral at this budget).
    assert removed == bad_indices


def test_curated_subset_beats_equal_n_random_on_held_out_bc_loss() -> None:
    # Downstream half of the known answer: curating out the jittered demos trains a policy that
    # imitates the clean held-out task far better than an equal-size random subset (invariant 5).
    reader, bad_indices = _build_known_dataset(seed=0)
    val_reader = _clean_val_reader()
    train_pool = list(range(len(reader)))

    curated_losses: list[float] = []
    random_losses: list[float] = []
    for seed in _SEEDS:
        # Curated arm: keep what action_noise judges clean.
        result = Curator([ActionNoise()], budget=Budget.fraction(_BUDGET), seed=seed).run(reader)
        curated = sorted(result.kept_episode_indices)
        assert bad_indices.isdisjoint(curated)  # curated arm holds no poisoned demos

        # Equal-N random control: same size, seeded draw from the full pool (invariant 5).
        rng = np.random.default_rng(seed * 31 + 7)
        random_subset = sorted(rng.choice(train_pool, size=len(curated), replace=False).tolist())

        policy = BCPolicy(epochs=300)
        curated_loss = held_out_bc_loss(
            policy, SubsetReader(reader, curated), val_reader, seed=seed
        )
        random_loss = held_out_bc_loss(
            policy, SubsetReader(reader, random_subset), val_reader, seed=seed
        )
        assert np.isfinite(curated_loss) and np.isfinite(random_loss)
        curated_losses.append(curated_loss)
        random_losses.append(random_loss)

        # Per-seed clear separation: equal-N random (keeping poisoned demos) is more than 5x
        # worse than curated. The true gap is ~100-500x; 5x is a conservative, stable margin.
        assert random_loss > 5.0 * curated_loss

    # Aggregate over seeds: curated mean loss is far below the random control's mean.
    assert float(np.mean(curated_losses)) < float(np.mean(random_losses))
