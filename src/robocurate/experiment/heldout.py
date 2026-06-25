"""Sim-free downstream check: curated-vs-control behavior-cloning loss on a held-out split.

The faithful downstream metric is closed-loop rollout success (see :mod:`.policy`), which needs
a GPU/simulator. This module provides a **CPU-only proxy** that needs neither: carve out a fixed
held-out validation split, train a small behavior-cloning policy on each arm's subset, and
compare the **action-prediction MSE on the held-out split** (a DataMIL-style validation-loss
proxy). Lower held-out loss = the training subset taught a more predictive policy.

It is an *independent cross-check* of the rollout gate, not a replacement: held-out BC loss
measures how well a policy imitates held-out actions, which correlates with — but is not — task
success. When the rollout gate and this proxy agree on which signal helps, that is a cheap,
double-confirmed result; when they disagree, that is itself informative (and honestly reported).

**Known bias (read before interpreting):** the held-out split is a *uniform-random* sample of
the dataset, so an arm trained on a uniform-random subset is distribution-matched to it, while a
*curated* (deliberately non-uniform) subset is not. This biases the proxy **toward the random
control** via coverage, independent of demo quality: removing "different-but-valid" demos lowers
held-out coverage even when those demos are lower-quality. So a curated arm showing higher
held-out loss is *suggestive* that the signal isn't capturing policy-relevant quality, but it is
not proof of harm — the unbiased arbiter is closed-loop task success (the rollout gate), which
does not reward matching the held-out *demo* distribution. Treat a negative here as a yellow
flag that the rollout gate should confirm, not a verdict.

Design (respects the invariants): the held-out split is carved out ONCE and shared by every arm
(no arm ever trains on it); curation and the random controls are drawn only from the remaining
train pool; everything is seeded (invariant 3); the curated subset is compared against an
equal-N **and** a length-matched random control (invariant 5); effects carry bootstrap CIs
(invariant 6). The source dataset is never mutated — arms are read-only :class:`SubsetReader`
views.

Needs the ``policy`` extra (torch, CPU is fine).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from robocurate.curator import Budget, Curator
from robocurate.experiment.conditions import SubsetReader
from robocurate.experiment.policies import BCPolicy, _bc_pair
from robocurate.experiment.stats import EffectEstimate, bootstrap_mean, paired_effect

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader
    from robocurate.signals.base import Signal


def held_out_bc_loss(
    policy: BCPolicy, train_reader: DatasetReader, val_reader: DatasetReader, *, seed: int
) -> float:
    """Train ``policy`` on ``train_reader``; return mean per-transition MSE on ``val_reader``.

    The validation actions are never seen in training. A single batched forward per validation
    trajectory (via :meth:`BCTrainedPolicy.predict`) keeps it fast and CPU-friendly.
    """
    trained = policy.train(train_reader, seed=seed)
    total_sq = 0.0
    total_n = 0
    for traj in val_reader:
        pair = _bc_pair(traj)
        if pair is None:
            continue
        state, action = pair
        pred = trained.predict(state)
        total_sq += float(np.square(pred - action).sum())
        total_n += int(action.shape[0] * action.shape[1])
    return total_sq / total_n if total_n else float("nan")


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministically split ``range(n)`` into (train_pool, val) by ``val_frac``."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_val = max(1, round(n * val_frac))
    val = sorted(perm[:n_val])
    train_pool = sorted(perm[n_val:])
    return train_pool, val


def compare_curation_heldout(
    reader: DatasetReader,
    signal: Signal,
    *,
    budget: float = 0.67,
    seeds: Sequence[int] = (0, 1, 2),
    val_frac: float = 0.2,
    split_seed: int = 0,
    epochs: int = 300,
    hidden_dim: int = 64,
) -> dict[str, Any]:
    """Curated-vs-control held-out BC loss for one ``signal``, on a fixed held-out split.

    Arms (all trained on disjoint subsets of the train pool, all evaluated on the same held-out
    split): ``full`` (whole train pool), ``random`` (equal-N), ``random_steps`` (length-matched),
    and ``curated``. Returns per-arm mean loss + bootstrap CI and the paired curated-vs-control
    effects. NOTE: lower loss is better, so a curated *improvement* is a **negative** effect.
    """
    train_pool, val = _split_indices(len(reader), val_frac, split_seed)
    val_reader = SubsetReader(reader, val)

    # Curate within the train pool only. SubsetReader yields source trajectories with their
    # original meta.episode_index, so the curator's kept indices are already global.
    pool_reader = SubsetReader(reader, train_pool)
    result = Curator([signal], budget=Budget.fraction(budget), seed=split_seed).run(pool_reader)
    pool_set = set(train_pool)
    curated = sorted(
        d.episode_index for d in result.decisions if d.kept and d.episode_index in pool_set
    )
    len_of = {i: reader.read_episode(i).meta.num_steps for i in train_pool}

    def arm_indices(arm: str, seed: int) -> list[int]:
        if arm == "full":
            return train_pool
        if arm == "curated":
            return curated
        rng = np.random.default_rng((seed + 1) * 100003 + split_seed)
        if arm == "random":
            return sorted(rng.choice(train_pool, size=len(curated), replace=False).tolist())
        # length-matched: add random demos until total transitions reach the curated total
        target = sum(len_of[i] for i in curated)
        keep: list[int] = []
        total = 0
        for i in rng.permutation(train_pool).tolist():
            if total >= target:
                break
            keep.append(i)
            total += len_of[i]
        return sorted(keep)

    arms = ("full", "random", "random_steps", "curated")
    losses: dict[str, list[float]] = {arm: [] for arm in arms}
    for seed in seeds:
        policy = BCPolicy(hidden_dim=hidden_dim, epochs=epochs)
        for arm in arms:
            train_reader = SubsetReader(reader, arm_indices(arm, seed))
            losses[arm].append(held_out_bc_loss(policy, train_reader, val_reader, seed=seed))

    def effect(control: str) -> EffectEstimate:
        # treatment - baseline; curated better => lower loss => negative effect.
        return paired_effect(losses["curated"], losses[control], seed=split_seed)

    return {
        "signal": signal.spec.name,
        "budget": budget,
        "n_train_pool": len(train_pool),
        "n_val": len(val),
        "n_curated": len(curated),
        "seeds": list(seeds),
        "loss_by_arm": {arm: [round(v, 6) for v in losses[arm]] for arm in arms},
        "mean_loss_by_arm": {
            arm: bootstrap_mean(losses[arm], seed=split_seed).to_dict() for arm in arms
        },
        "curated_vs_random": effect("random").to_dict(),
        "curated_vs_random_steps": effect("random_steps").to_dict(),
        "note": "lower loss is better; a curated improvement is a NEGATIVE effect (CI fully < 0).",
    }


__all__ = ["compare_curation_heldout", "held_out_bc_loss"]
