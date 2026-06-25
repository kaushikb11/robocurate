"""Tests for the ManiSkill environment's rollout logic (real SAPIEN runs on Modal only).

The real ManiSkill factory needs a GPU and cannot run in this sandbox, so these tests drive
the generic rollout loop with a deterministic fake VecEnv — covering success latching, the
policy->action wiring, and per-task reporting.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from robocurate.experiment.maniskill import (
    ManiSkillEnvironment,
    RandomPolicy,
    RandomTrainedPolicy,
)
from robocurate.trajectory import Array

F64 = npt.NDArray[np.float64]


class _FakeVecEnv:
    """A deterministic vectorized env: env i 'succeeds' once it has taken >= (i+1) positive steps.

    Used only to exercise the rollout loop (latching, counting), not real physics.
    """

    def __init__(self, num_envs: int, obs_dim: int = 3) -> None:
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._positive_steps = np.zeros(num_envs, dtype=np.int64)

    def reset(self, *, seed: int) -> F64:
        self._positive_steps = np.zeros(self.num_envs, dtype=np.int64)
        return np.zeros((self.num_envs, self._obs_dim), dtype=np.float64)

    def step(self, actions: F64) -> tuple[F64, npt.NDArray[np.bool_]]:
        took_positive = actions.sum(axis=1) > 0
        self._positive_steps += took_positive.astype(np.int64)
        threshold = np.arange(1, self.num_envs + 1)
        success = self._positive_steps >= threshold
        obs = np.zeros((self.num_envs, self._obs_dim), dtype=np.float64)
        return obs, success


class _AlwaysPositivePolicy:
    def act(self, observation: Mapping[str, Array], *, seed: int) -> F64:
        return np.ones(2, dtype=np.float64)


def test_rollout_counts_latched_success() -> None:
    # With an always-positive policy over enough steps, every env eventually succeeds.
    env = ManiSkillEnvironment(
        task_id="FakeTask-v0", max_steps=10, env_factory=lambda n, s: _FakeVecEnv(n)
    )
    result = env.evaluate(_AlwaysPositivePolicy(), episodes=4, seed=0)
    assert result.success_rate == 1.0
    assert result.n_episodes == 4
    assert set(result.per_task) == {"FakeTask-v0"}


def test_rollout_partial_success_with_short_horizon() -> None:
    # Only 3 steps: env i needs i+1 positive steps, so envs 0,1,2 succeed, env 3 does not.
    env = ManiSkillEnvironment(
        task_id="FakeTask-v0", max_steps=3, env_factory=lambda n, s: _FakeVecEnv(n)
    )
    result = env.evaluate(_AlwaysPositivePolicy(), episodes=4, seed=0)
    assert result.success_rate == 0.75


def test_random_policy_acts_with_right_dimension() -> None:
    policy = RandomTrainedPolicy(action_dim=7)
    action = policy.act({"observation": np.zeros(3)}, seed=1)
    assert action.shape == (7,)
    # Deterministic given the seed.
    again = policy.act({"observation": np.zeros(3)}, seed=1)
    np.testing.assert_array_equal(action, again)


def test_random_policy_trains_to_trained_policy() -> None:
    trained = RandomPolicy(action_dim=4).train(train_set=None, seed=0)  # type: ignore[arg-type]
    assert isinstance(trained, RandomTrainedPolicy)
    assert trained.act({"observation": np.zeros(2)}, seed=0).shape == (4,)


def test_random_policy_drives_the_environment() -> None:
    # The smoke-test shape: a random policy rolled out in the (fake) env yields a rate in [0,1].
    env = ManiSkillEnvironment(
        task_id="FakeTask-v0", max_steps=5, env_factory=lambda n, s: _FakeVecEnv(n)
    )
    result = env.evaluate(RandomTrainedPolicy(action_dim=2), episodes=6, seed=3)
    assert 0.0 <= result.success_rate <= 1.0
