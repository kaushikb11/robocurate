"""Policy and environment abstractions for the experiment harness.

The headline experiment trains a policy on a (curated or control) subset and evaluates it by
**rolling it out in an environment** to measure a success rate — the faithful metric. The
abstractions here are environment-rollout-first:

* :class:`Policy` — trains on a dataset, returning a :class:`TrainedPolicy` that can act.
* :class:`Environment` — rolls a trained policy out over episodes and returns an
  :class:`EvalResult` (success rate + per-task breakdown).

Real components (Diffusion Policy, SmolVLA; a ManiSkill3 / RoboMimic environment) implement
these protocols later, behind extras. For scaffolding — so the whole harness is testable
*now*, before any heavy training/sim dependency — this module ships a deterministic
:class:`FakePolicy` + :class:`FakeEnvironment` whose rollout success tracks the quality of
the training subset, so the equal-N separation the experiment exists to detect actually
appears in tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from robocurate.trajectory import Array, FeatureRole

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader

Observation = Mapping[str, Array]


@dataclass(frozen=True)
class EvalResult:
    """The outcome of evaluating a trained policy in an environment.

    Attributes:
        success_rate: Fraction of successful rollout episodes in ``[0, 1]``.
        per_task: Success rate broken out per task id (honest reporting never collapses a
            task-dependent result to one number).
        n_episodes: Number of rollout episodes the rate is computed over.
    """

    success_rate: float
    per_task: Mapping[str, float] = field(default_factory=dict)
    n_episodes: int = 0


@runtime_checkable
class TrainedPolicy(Protocol):
    """A trained policy that maps an observation to an action during rollouts."""

    def act(self, observation: Observation, *, seed: int) -> Array:
        """Return an action for ``observation``. Deterministic given ``seed``."""
        ...


@runtime_checkable
class Policy(Protocol):
    """Trains a :class:`TrainedPolicy` from a dataset.

    Implementations must be deterministic given the seed (Invariant 3): the same
    training subset + seed yields the same trained policy.
    """

    name: str

    def train(self, train_set: DatasetReader, *, seed: int) -> TrainedPolicy:
        """Train on ``train_set`` and return a :class:`TrainedPolicy`."""
        ...


@runtime_checkable
class Environment(Protocol):
    """Evaluates a trained policy by rolling it out over episodes."""

    task_ids: tuple[str, ...]

    def evaluate(self, policy: TrainedPolicy, *, episodes: int, seed: int) -> EvalResult:
        """Roll ``policy`` out for ``episodes`` episodes and return the :class:`EvalResult`.

        Deterministic given ``seed`` so experiment conditions are reproducible.
        """
        ...


# --------------------------------------------------------------------------------------
# Deterministic fakes for scaffolding / tests
# --------------------------------------------------------------------------------------


def _mean_smoothness(reader: DatasetReader) -> float:
    """A cheap, deterministic quality proxy: negative mean action 'jerk' over a dataset.

    Smoother (less jerky) action data scores higher. Used only by :class:`FakePolicy` as a
    stand-in for "this training data produces a better policy".
    """
    totals: list[float] = []
    for traj in reader:
        action = traj.actions()
        if action is None or action.shape[0] < 3:
            continue
        a = np.asarray(action, dtype=np.float64).reshape(action.shape[0], -1)
        jerk = float(np.abs(np.diff(a, n=2, axis=0)).mean())
        totals.append(jerk)
    if not totals:
        return 0.0
    return -float(np.mean(totals))


class FakeTrainedPolicy:
    """A trained :class:`FakePolicy`: acts by echoing the observation with competence-scaled noise.

    The optimal action in :class:`FakeEnvironment` is the observation itself; a more competent
    policy adds less noise and therefore succeeds more often.
    """

    def __init__(self, competence: float, *, action_key: str = "action") -> None:
        self.competence = float(np.clip(competence, 0.0, 1.0))
        self._action_key = action_key

    def act(self, observation: Observation, *, seed: int) -> Array:
        target = np.asarray(next(iter(observation.values())), dtype=np.float64)
        noise_scale = 1.0 - self.competence
        noise = np.random.default_rng(seed).normal(0.0, noise_scale, size=target.shape)
        return target + noise


class FakePolicy:
    """A deterministic, quality-sensitive stand-in policy for harness scaffolding/tests.

    Its competence — and thus downstream success — increases with the quality of the training
    subset (via ``quality_fn``, defaulting to action smoothness), so a good curation actually
    trains a "better" fake policy than an equal-size random subset. Not a real policy.
    """

    def __init__(
        self,
        *,
        quality_fn: Callable[[DatasetReader], float] = _mean_smoothness,
        sensitivity: float = 6.0,
        name: str = "fake_policy",
    ) -> None:
        self.name = name
        self._quality_fn = quality_fn
        self._sensitivity = sensitivity

    def train(self, train_set: DatasetReader, *, seed: int) -> TrainedPolicy:
        quality = self._quality_fn(train_set)
        # Map quality -> competence in (0, 1) with a smooth, monotonic squashing. A tiny
        # seed-dependent jitter models run-to-run training variance without breaking
        # determinism (same seed => same value).
        jitter = (np.random.default_rng(seed).random() - 0.5) * 0.02
        competence = 1.0 / (1.0 + np.exp(-(self._sensitivity * quality + jitter)))
        return FakeTrainedPolicy(float(competence))


class FakeEnvironment:
    """A deterministic environment whose optimal action equals the observation.

    Each episode samples a random target observation; a rollout succeeds when the policy's
    action is within ``tolerance`` of it. Success probability therefore rises with policy
    competence, giving the harness a real (seeded) signal to measure.
    """

    def __init__(
        self,
        *,
        obs_dim: int = 2,
        tolerance: float = 0.5,
        task_ids: tuple[str, ...] = ("reach",),
    ) -> None:
        self.obs_dim = obs_dim
        self.tolerance = tolerance
        self.task_ids = task_ids
        self._role = FeatureRole.STATE  # documents what the observation represents

    def evaluate(self, policy: TrainedPolicy, *, episodes: int, seed: int) -> EvalResult:
        successes = {task: 0 for task in self.task_ids}
        counts = {task: 0 for task in self.task_ids}
        for episode in range(episodes):
            task = self.task_ids[episode % len(self.task_ids)]
            ep_seed = int(np.random.SeedSequence([seed, episode]).generate_state(1)[0])
            target = np.random.default_rng(ep_seed).normal(0.0, 1.0, size=self.obs_dim)
            action = np.asarray(policy.act({"observation": target}, seed=ep_seed), dtype=np.float64)
            counts[task] += 1
            if float(np.linalg.norm(action - target)) <= self.tolerance:
                successes[task] += 1
        per_task = {t: successes[t] / counts[t] for t in self.task_ids if counts[t]}
        overall = sum(successes.values()) / max(1, sum(counts.values()))
        return EvalResult(success_rate=overall, per_task=per_task, n_episodes=episodes)


__all__ = [
    "Environment",
    "EvalResult",
    "FakeEnvironment",
    "FakePolicy",
    "FakeTrainedPolicy",
    "Observation",
    "Policy",
    "TrainedPolicy",
]
