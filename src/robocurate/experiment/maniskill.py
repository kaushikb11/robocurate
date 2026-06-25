"""ManiSkill3 sim environment for real success-rate rollouts (optional, GPU/Modal).

Replaces the :class:`~robocurate.experiment.policy.FakeEnvironment` with real physics:
:class:`ManiSkillEnvironment` rolls a policy out in a GPU-parallel ManiSkill3 task and reports
the fraction of episodes that succeed. It implements the same
:class:`~robocurate.experiment.policy.Environment` protocol, so it drops into the existing
runner.

Design for testability: the **rollout loop** is generic over a tiny vectorized-env interface
(:class:`VecEnv`) and is fully unit-tested here against a fake env. The **real ManiSkill
factory** (SAPIEN, GPU, possibly Vulkan) is imported lazily and only runs on a GPU worker
(Modal) — it cannot be exercised in a CPU sandbox. This module is the first de-risking step:
a state-observation task (no rendering) rolled out by a random policy, to confirm ManiSkill
installs and runs on Modal before training is wired in.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from robocurate.experiment.policy import EvalResult

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader
    from robocurate.experiment.policy import Observation, TrainedPolicy

F64 = npt.NDArray[np.float64]

DEFAULT_TASK = "PickCube-v1"
DEFAULT_MAX_STEPS = 50


@runtime_checkable
class VecEnv(Protocol):
    """The minimal vectorized-env interface the rollout needs (a thin view over ManiSkill)."""

    num_envs: int

    def reset(self, *, seed: int) -> F64:
        """Reset all envs; return the ``(num_envs, obs_dim)`` observation."""
        ...

    def step(self, actions: F64) -> tuple[F64, npt.NDArray[np.bool_]]:
        """Step with ``(num_envs, action_dim)`` actions; return (next_obs, success-mask)."""
        ...


# --------------------------------------------------------------------------------------
# Random policy (for the smoke test — no training, just samples actions)
# --------------------------------------------------------------------------------------


class RandomTrainedPolicy:
    """Samples uniform random actions of a fixed dimension. Deterministic given the seed."""

    def __init__(self, action_dim: int, *, low: float = -1.0, high: float = 1.0) -> None:
        self.action_dim = action_dim
        self._low = low
        self._high = high

    def act(self, observation: Observation, *, seed: int) -> F64:
        rng = np.random.default_rng(seed)
        return rng.uniform(self._low, self._high, size=self.action_dim)


class RandomPolicy:
    """A :class:`~robocurate.experiment.policy.Policy` that ignores data and acts randomly."""

    def __init__(self, action_dim: int, *, name: str = "random") -> None:
        self.name = name
        self.action_dim = action_dim

    def train(self, train_set: DatasetReader, *, seed: int) -> RandomTrainedPolicy:
        return RandomTrainedPolicy(self.action_dim)


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------


VecEnvFactory = Callable[[int, int], "VecEnv"]  # (num_envs, seed) -> VecEnv


class ManiSkillEnvironment:
    """Evaluate a policy by rolling it out in a (state-observation) ManiSkill3 task.

    Args:
        task_id: ManiSkill task, e.g. ``"PickCube-v1"``.
        obs_mode: Observation mode; ``"state"`` avoids rendering (no Vulkan).
        max_steps: Rollout horizon; success is latched (an episode counts once it ever
            reaches a success state within the horizon).
        env_factory: ``(num_envs, seed) -> VecEnv`` builder. Defaults to the real ManiSkill
            factory (lazy import; GPU only). Injectable for testing.
    """

    def __init__(
        self,
        *,
        task_id: str = DEFAULT_TASK,
        obs_mode: str = "state",
        control_mode: str | None = None,
        max_steps: int | None = None,
        env_factory: VecEnvFactory | None = None,
    ) -> None:
        self.task_id = task_id
        self.obs_mode = obs_mode
        self.control_mode = control_mode  # must match the demos' control mode for BC
        self.max_steps = max_steps  # None => use the task's max_episode_steps
        self.task_ids: tuple[str, ...] = (task_id,)
        self._env_factory = env_factory

    def evaluate(self, policy: TrainedPolicy, *, episodes: int, seed: int) -> EvalResult:
        factory = self._env_factory or self._make_real_env
        vec = factory(episodes, seed)
        return self._evaluate_on(policy, vec, episodes, seed)

    def smoke_rollout(self, *, episodes: int, seed: int) -> EvalResult:
        """Build the real env once and roll out a random policy matched to its action space.

        A single env (no separate probe), used by the de-risk smoke test to confirm ManiSkill
        runs on the GPU. A random policy should not succeed — a clean number is the goal.
        """
        vec = self._make_real_env(episodes, seed)
        return self._evaluate_on(RandomTrainedPolicy(vec.action_dim), vec, episodes, seed)

    def _evaluate_on(
        self, policy: TrainedPolicy, vec: VecEnv, episodes: int, seed: int
    ) -> EvalResult:
        horizon = self.max_steps or int(getattr(vec, "max_episode_steps", DEFAULT_MAX_STEPS))
        success = _rollout(policy, vec, max_steps=horizon, seed=seed)
        rate = float(success.mean()) if success.size else 0.0
        return EvalResult(success_rate=rate, per_task={self.task_id: rate}, n_episodes=episodes)

    def _make_real_env(self, num_envs: int, seed: int) -> _ManiSkillVec:
        """Build a real ManiSkill3 vectorized env (lazy import; runs on a GPU worker only).

        Wrapped in ``ManiSkillVectorEnv(ignore_terminations=True)`` so partial resets are off:
        each sub-env runs exactly one fixed-length episode, making the OR-over-steps success
        latch correct (otherwise a success would auto-reset the env mid-rollout).
        """
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401  - registers the ManiSkill tasks with gymnasium
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        kwargs: dict[str, Any] = {"num_envs": num_envs, "obs_mode": self.obs_mode}
        if self.control_mode is not None:
            kwargs["control_mode"] = self.control_mode
        env = gym.make(self.task_id, **kwargs)
        env = ManiSkillVectorEnv(env, ignore_terminations=True, record_metrics=True)
        return _ManiSkillVec(env)


def _rollout(
    policy: TrainedPolicy, vec: VecEnv, *, max_steps: int, seed: int
) -> npt.NDArray[np.bool_]:
    """Roll ``policy`` out in ``vec`` for ``max_steps``; return the per-env latched success mask."""
    obs = np.asarray(vec.reset(seed=seed), dtype=np.float64)
    latched = np.zeros(vec.num_envs, dtype=np.bool_)
    for step in range(max_steps):
        actions = np.stack(
            [
                np.asarray(
                    # Distinct seed per (step, env) so a random policy acts differently across
                    # envs; a deterministic policy ignores it.
                    policy.act({"observation": obs[i]}, seed=seed * 1_000_003 + step * 10_007 + i),
                    dtype=np.float64,
                )
                for i in range(vec.num_envs)
            ]
        )
        obs_next, success = vec.step(actions)
        obs = np.asarray(obs_next, dtype=np.float64)
        latched |= np.asarray(success, dtype=np.bool_)
    return latched


class _ManiSkillVec:
    """Adapts a ManiSkill3 ``ManiSkillVectorEnv`` to the :class:`VecEnv` interface (GPU-only)."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self.num_envs = int(env.num_envs)
        self.action_dim = int(env.single_action_space.shape[0])  # per-env action dim
        self.max_episode_steps = _read_max_episode_steps(env)

    def reset(self, *, seed: int) -> F64:
        obs, _info = self._env.reset(seed=seed)
        return np.asarray(_to_numpy(obs), dtype=np.float64)

    def step(self, actions: F64) -> tuple[F64, npt.NDArray[np.bool_]]:
        # NumPy actions are accepted; ManiSkill's to_tensor moves them to the sim device and
        # downcasts float64 -> float32. info["success"] is a per-step (num_envs,) bool tensor.
        obs, _reward, _terminated, _truncated, info = self._env.step(actions)
        success = info["success"] if "success" in info else info.get("is_success")
        if success is None:
            raise RuntimeError(
                "ManiSkill step info has no 'success'/'is_success' key; this task reports "
                "success differently and needs a custom success extractor"
            )
        obs_np: F64 = np.asarray(_to_numpy(obs), dtype=np.float64)
        success_np: npt.NDArray[np.bool_] = np.asarray(_to_numpy(success), dtype=np.bool_).reshape(
            -1
        )
        if success_np.shape != (self.num_envs,):
            raise RuntimeError(f"success shape {success_np.shape} != expected ({self.num_envs},)")
        return obs_np, success_np


def _read_max_episode_steps(env: Any) -> int:
    """Read the task horizon from a ManiSkill env, failing loudly rather than guessing.

    The vector wrapper does not expose ``max_episode_steps`` directly, so read it from the
    underlying env spec; a silent default would roll out the wrong horizon for non-PickCube
    tasks.
    """
    for source in ("spec", "wrapper_attr", "direct"):
        try:
            if source == "spec":
                value = env.unwrapped.spec.max_episode_steps
            elif source == "wrapper_attr":
                value = env.get_wrapper_attr("max_episode_steps")
            else:
                value = env.max_episode_steps
        except Exception:  # try the next accessor
            continue
        if value:
            return int(value)
    raise RuntimeError(
        "could not determine max_episode_steps from the ManiSkill env; pass max_steps explicitly"
    )


def _to_numpy(value: Any) -> Any:
    """Convert a torch tensor (possibly on GPU) or array-like to a NumPy array."""
    if hasattr(value, "detach"):  # torch tensor
        return value.detach().cpu().numpy()
    return np.asarray(value)


__all__ = [
    "ManiSkillEnvironment",
    "RandomPolicy",
    "RandomTrainedPolicy",
    "VecEnv",
]
