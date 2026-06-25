"""Tests for the real behavior-cloning policy (requires torch). Marked ``ml``."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from robocurate.curator import Budget, Curator
from robocurate.experiment import (
    BCPolicy,
    ExperimentSpec,
    FakeEnvironment,
    run,
)
from robocurate.trajectory import (
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)
from tests.synthetic import TOY_EMBODIMENT
from tests.test_jerk_signal import _ListReader

pytestmark = pytest.mark.ml


def _identity_traj(
    idx: int, *, noise: float, seed: int, sign: float = 1.0, num_steps: int = 24
) -> Trajectory:
    """A trajectory whose action is ``sign * observation + noise``.

    The FakeEnvironment's optimal action equals the observation, so ``sign=+1`` is "good"
    identity data and ``sign=-1`` is contradictory data that corrupts a BC policy trained on
    it (the mean of ``+x`` and ``-x`` is ~0).
    """
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 1.0, size=(num_steps, 2)).astype(np.float32)
    action = (sign * state + rng.normal(0.0, noise, size=(num_steps, 2))).astype(np.float32)
    columns = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) * 0.1),
        "action": action,
        "observation.state": state,
        "reward": np.zeros(num_steps, dtype=np.float32),
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/bc",
        episode_index=idx,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
        success=SuccessLabel(value=True, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _clean_reader() -> _ListReader:
    return _ListReader([_identity_traj(i, noise=0.02, seed=i) for i in range(8)])


def _noisy_reader() -> _ListReader:
    return _ListReader([_identity_traj(i, noise=1.0, seed=i) for i in range(8)])


def test_bc_policy_learns_the_mapping() -> None:
    trained = BCPolicy(epochs=400).train(_clean_reader(), seed=0)
    # The optimal action is the observation itself; a policy trained on clean identity data
    # should act close to it.
    target = np.array([0.7, -0.4], dtype=np.float64)
    action = np.asarray(trained.act({"observation": target}, seed=0), dtype=np.float64)
    assert float(np.linalg.norm(action - target)) < 0.4


def test_real_policy_quality_sensitivity() -> None:
    env = FakeEnvironment()
    clean = env.evaluate(BCPolicy().train(_clean_reader(), seed=0), episodes=200, seed=1)
    noisy = env.evaluate(BCPolicy().train(_noisy_reader(), seed=0), episodes=200, seed=1)
    assert clean.success_rate > noisy.success_rate


def test_bc_policy_is_deterministic() -> None:
    env = FakeEnvironment()
    a = env.evaluate(BCPolicy().train(_clean_reader(), seed=0), episodes=100, seed=3)
    b = env.evaluate(BCPolicy().train(_clean_reader(), seed=0), episodes=100, seed=3)
    assert a.success_rate == b.success_rate


def test_real_policy_drives_headline_separation() -> None:
    # Majority helpful (action=+state, the identity task) + a minority of contradictory
    # trajectories (action=-state). CUPID influence keeps the helpful consensus; a REAL BC
    # policy trained on them acts near the optimum, while one trained on an equal-size random
    # (mixed) subset is dragged toward zero by the contradictory examples.
    from robocurate.signals.cupid import Cupid

    helpful = [_identity_traj(i, noise=0.02, sign=1.0, seed=i) for i in range(12)]
    harmful = [_identity_traj(12 + i, noise=0.02, sign=-1.0, seed=12 + i) for i in range(4)]
    source = _ListReader(helpful + harmful)
    spec = ExperimentSpec(
        source=source,
        curator=Curator([Cupid(mode="tracin")], budget=Budget.fraction(0.5), seed=0),
        policy=BCPolicy(epochs=250),
        environment=FakeEnvironment(),
        seeds=(0, 1, 2, 3, 4),
        eval_episodes=200,
        include_ablations=False,
    )
    report = run(spec)
    effect = report.curated_vs_equal_n
    assert effect is not None
    assert effect.effect > 0.0
    assert effect.separated  # a real trained policy, CI-separated from equal-N random
    # CUPID kept only helpful trajectories (episode indices < 12).
    curated = report.arm("curated")
    assert curated is not None and curated.size == 8


def test_train_without_state_or_action_raises() -> None:
    store = InMemoryFeatureStore({"timestamp": np.arange(4, dtype=np.float32)})
    bad = Trajectory(_identity_traj(0, noise=0.0, seed=0).meta, store)
    with pytest.raises(ValueError, match="no usable"):
        BCPolicy().train(_ListReader([bad]), seed=0)
