"""Tests for the experiment building blocks: SubsetReader and the deterministic fakes."""

from __future__ import annotations

from robocurate.experiment import (
    FakeEnvironment,
    FakePolicy,
    SubsetReader,
)
from robocurate.experiment.conditions import Condition
from tests.test_jerk_signal import _jerky_action, _ListReader, _smooth_action, _traj_with_action


def _smooth_reader() -> _ListReader:
    return _ListReader([_traj_with_action(i, _smooth_action()) for i in range(6)])


def _jerky_reader() -> _ListReader:
    return _ListReader([_traj_with_action(i, _jerky_action()) for i in range(6)])


def test_subset_reader_is_read_only_view() -> None:
    source = _smooth_reader()
    subset = SubsetReader(source, [0, 2, 4])
    assert len(subset) == 3
    assert not hasattr(SubsetReader, "write")
    # Iterates exactly the selected episodes, in order.
    episodes = [t.meta.episode_index for t in subset]
    assert episodes == [0, 2, 4]
    assert subset.read_episode(1).meta.episode_index == 2


def test_subset_fingerprint_depends_on_selection() -> None:
    source = _smooth_reader()
    a = SubsetReader(source, [0, 1, 2]).fingerprint()
    b = SubsetReader(source, [0, 1, 2]).fingerprint()
    c = SubsetReader(source, [0, 1, 3]).fingerprint()
    assert a.content_hash == b.content_hash  # same selection => same fingerprint
    assert a.content_hash != c.content_hash
    assert a.num_episodes == 3


def test_fake_environment_is_deterministic() -> None:
    env = FakeEnvironment()
    policy = FakePolicy().train(_smooth_reader(), seed=0)
    first = env.evaluate(policy, episodes=50, seed=7)
    second = env.evaluate(policy, episodes=50, seed=7)
    assert first.success_rate == second.success_rate
    assert first.n_episodes == 50


def test_fake_policy_quality_sensitivity() -> None:
    # A policy trained on smoother data should evaluate better than one trained on jerky data.
    env = FakeEnvironment()
    smooth = env.evaluate(FakePolicy().train(_smooth_reader(), seed=0), episodes=200, seed=1)
    jerky = env.evaluate(FakePolicy().train(_jerky_reader(), seed=0), episodes=200, seed=1)
    assert smooth.success_rate > jerky.success_rate


def test_per_task_breakdown_present() -> None:
    env = FakeEnvironment(task_ids=("reach", "push"))
    result = env.evaluate(FakePolicy().train(_smooth_reader(), seed=0), episodes=40, seed=2)
    assert set(result.per_task) == {"reach", "push"}


def test_condition_enum_values() -> None:
    assert {c.value for c in Condition} == {
        "full",
        "curated",
        "equal_n_random",
        "random_filter",
        "ablation",
    }
