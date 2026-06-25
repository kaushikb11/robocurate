"""The experiment runner: builds the controlled arms and trains/evaluates each.

The crux of the design is that the **equal-N random baseline is reused from the curator
itself** — a :class:`~robocurate.curator.Curator` run already emits the curated selection
*and* a same-size random baseline (Invariant 5), so the headline fair comparison is
wired in by construction rather than re-derived here.

Flow: run the curator once → derive the arms (full / curated / equal-N random / random-filter
/ one ablation per signal) → for each arm and seed, train the policy on the
:class:`~robocurate.experiment.conditions.SubsetReader` and evaluate it in the environment →
aggregate into an :class:`~robocurate.experiment.report.ExperimentReport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robocurate.curator import CurationResult, Curator, WeightedSum
from robocurate.experiment.conditions import Arm, Condition, SubsetReader
from robocurate.experiment.report import ExperimentReport, build_report

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader
    from robocurate.experiment.policy import Environment, EvalResult, Policy

# Independent RNG stream id (mixed with the curator seed) for the random-filter control, so it
# is reproducible but distinct from the curator's equal-N baseline draw.
_RANDOM_FILTER_STREAM = 0xF117E2


@dataclass(frozen=True)
class ExperimentSpec:
    """Everything needed to run the headline experiment.

    Attributes:
        source: The (read-only) source dataset.
        curator: A configured :class:`~robocurate.curator.Curator` (its signals, combiner,
            budget, and seed drive curation and the equal-N baseline).
        policy: The :class:`~robocurate.experiment.policy.Policy` to train per arm.
        environment: The :class:`~robocurate.experiment.policy.Environment` to evaluate in.
        seeds: Training/eval seeds; every arm is run once per seed.
        eval_episodes: Rollout episodes per evaluation.
        include_random_filter: Whether to include the random-filter control arm.
        include_ablations: Whether to include one ablation arm per signal.
        stats_seed: Seed for the report's bootstrap resampling.
    """

    source: DatasetReader
    curator: Curator
    policy: Policy
    environment: Environment
    seeds: tuple[int, ...] = (0, 1, 2)
    eval_episodes: int = 100
    include_random_filter: bool = True
    include_ablations: bool = True
    stats_seed: int = 0


def run(spec: ExperimentSpec) -> ExperimentReport:
    """Run the experiment described by ``spec`` and return its report."""
    total = len(spec.source)
    result = spec.curator.run(spec.source)
    arms = _build_arms(spec, total, result)

    arm_results: list[tuple[Arm, list[EvalResult]]] = []
    for arm in arms:
        subset = SubsetReader(spec.source, arm.episode_indices)
        results: list[EvalResult] = []
        for seed in spec.seeds:
            trained = spec.policy.train(subset, seed=seed)
            results.append(
                spec.environment.evaluate(trained, episodes=spec.eval_episodes, seed=seed)
            )
        arm_results.append((arm, results))

    return build_report(
        dataset_id=spec.source.fingerprint().dataset_id,
        total_episodes=total,
        seeds=spec.seeds,
        eval_episodes=spec.eval_episodes,
        arm_results=arm_results,
        stats_seed=spec.stats_seed,
    )


def _build_arms(spec: ExperimentSpec, total: int, result: CurationResult) -> list[Arm]:
    kept = result.kept_episode_indices
    arms: list[Arm] = [
        Arm("full", Condition.FULL, tuple(range(total))),
        Arm("curated", Condition.CURATED, kept),
    ]
    if result.baseline is not None:
        arms.append(
            Arm("equal_n_random", Condition.EQUAL_N_RANDOM, result.baseline.kept_episode_indices)
        )
    if spec.include_random_filter:
        arms.append(
            Arm("random_filter", Condition.RANDOM_FILTER, _random_subset(total, len(kept), spec))
        )
    if spec.include_ablations:
        arms.extend(_ablation_arms(spec))
    return arms


def _random_subset(total: int, k: int, spec: ExperimentSpec) -> tuple[int, ...]:
    stream = np.random.SeedSequence([spec.curator.seed, _RANDOM_FILTER_STREAM])
    rng = np.random.default_rng(int(stream.generate_state(1)[0]))
    if not 0 < k <= total:
        return tuple(range(min(k, total)))
    return tuple(sorted(int(i) for i in rng.choice(total, size=k, replace=False)))


def _ablation_arms(spec: ExperimentSpec) -> list[Arm]:
    """One arm per signal: curate using only that signal at the same budget/seed."""
    arms: list[Arm] = []
    for signal in spec.curator.signals:
        solo = Curator(
            [signal],
            combiner=WeightedSum(),
            budget=spec.curator.budget,
            seed=spec.curator.seed,
            emit_baseline=False,
            resources=spec.curator.resources,
        )
        kept = solo.run(spec.source).kept_episode_indices
        arms.append(Arm(f"ablation:{signal.spec.name}", Condition.ABLATION, kept))
    return arms


__all__ = ["ExperimentSpec", "run"]
