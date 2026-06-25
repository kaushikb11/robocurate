"""End-to-end experiment harness tests and statistics tests."""

from __future__ import annotations

import json

from robocurate.curator import Budget, Curator
from robocurate.experiment import (
    ExperimentSpec,
    FakeEnvironment,
    FakePolicy,
    run,
)
from robocurate.experiment.stats import bootstrap_mean, paired_effect
from robocurate.signals.jerk import Jerk
from tests.test_jerk_signal import _jerky_action, _ListReader, _smooth_action, _traj_with_action


def _mixed_dataset() -> _ListReader:
    # Eight smooth + eight jerky trajectories; jerk-curation should keep the smooth ones,
    # which the FakePolicy rewards -> curated beats equal-N random. Episode indices are
    # sequential 0..15 (as real readers number them), matching positional lookups.
    smooth = [_traj_with_action(i, _smooth_action()) for i in range(8)]
    jerky = [_traj_with_action(8 + i, _jerky_action()) for i in range(8)]
    return _ListReader(smooth + jerky)


def _spec(**overrides: object) -> ExperimentSpec:
    base = dict(
        source=_mixed_dataset(),
        curator=Curator([Jerk()], budget=Budget.fraction(0.5), seed=0),
        policy=FakePolicy(),
        environment=FakeEnvironment(),
        seeds=(0, 1, 2, 3, 4),
        eval_episodes=200,
    )
    base.update(overrides)
    return ExperimentSpec(**base)  # type: ignore[arg-type]


def test_runner_builds_all_arms() -> None:
    report = run(_spec())
    names = {a.name for a in report.arms}
    assert {"full", "curated", "equal_n_random", "random_filter", "ablation:jerk"} <= names
    # curated and equal_n_random are the same size (invariant 5).
    curated = report.arm("curated")
    equal_n = report.arm("equal_n_random")
    assert curated is not None and equal_n is not None
    assert curated.size == equal_n.size == 8


def test_headline_curated_beats_equal_n_random() -> None:
    report = run(_spec())
    effect = report.curated_vs_equal_n
    assert effect is not None
    # The curated (smooth) subset trains a better fake policy than an equal-size random one.
    assert effect.effect > 0.0
    assert effect.separated  # CI excludes zero across the 5 seeds


def test_report_is_machine_and_human_readable() -> None:
    report = run(_spec())
    data = json.loads(report.to_json())
    assert data["headline"]["curated_vs_equal_n_random"]["separated"] is True
    assert len(data["arms"]) >= 5

    md = report.to_markdown()
    assert "Headline" in md
    assert "equal-N random" in md
    assert "separated" in md


def test_report_is_deterministic() -> None:
    a = run(_spec())
    b = run(_spec())
    assert a.to_json() == b.to_json()


def test_ablation_and_random_filter_toggle_off() -> None:
    report = run(_spec(include_ablations=False, include_random_filter=False))
    names = {a.name for a in report.arms}
    assert "ablation:jerk" not in names
    assert "random_filter" not in names
    assert {"full", "curated", "equal_n_random"} <= names


def test_bootstrap_mean_brackets_the_mean() -> None:
    est = bootstrap_mean([0.2, 0.4, 0.6, 0.8], seed=0)
    assert est.ci_low <= est.mean <= est.ci_high
    assert est.n == 4


def test_paired_effect_separation_verdict() -> None:
    # A consistent positive gap across seeds is separated; identical values are not.
    sep = paired_effect([0.9, 0.92, 0.88, 0.91], [0.5, 0.52, 0.48, 0.51], seed=0)
    assert sep.effect > 0 and sep.separated
    null = paired_effect([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], seed=0)
    assert not null.separated
