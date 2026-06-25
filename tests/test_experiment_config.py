"""Tests for the serializable ExperimentConfig, synthetic dataset, and run_config."""

from __future__ import annotations

import json

import pytest

from robocurate.experiment import (
    ExperimentConfig,
    make_identity_experiment_dataset,
    run_config,
)


def test_synthetic_identity_dataset_shape() -> None:
    reader = make_identity_experiment_dataset(num_helpful=5, num_harmful=2, num_steps=10)
    assert len(reader) == 7
    traj = reader.read_episode(0)
    assert traj.feature("action").shape == (10, 2)
    assert traj.feature("observation.state").shape == (10, 2)
    # Helpful trajectories are labelled success, harmful labelled failure.
    assert reader.read_episode(0).success().value is True  # type: ignore[union-attr]
    assert reader.read_episode(6).success().value is False  # type: ignore[union-attr]


def test_config_round_trips_through_json() -> None:
    config = ExperimentConfig(
        signals=[{"name": "jerk", "params": {}}],
        budget={"kind": "count", "value": 4},
        seeds=[0, 1],
    )
    restored = ExperimentConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored.to_dict() == config.to_dict()


def test_config_defaults_are_serializable() -> None:
    data = ExperimentConfig().to_dict()
    # Must survive a JSON round trip (it is shipped to a Modal worker as a dict).
    assert json.loads(json.dumps(data))["signals"][0]["name"] == "cupid"


def test_run_config_with_cheap_signal_local() -> None:
    # A fully core-only path: jerk curation + the fake policy/environment, no torch.
    config = ExperimentConfig(
        dataset={"kind": "identity_synthetic", "params": {"num_helpful": 6, "num_harmful": 2}},
        signals=[{"name": "jerk", "params": {}}],
        budget={"kind": "fraction", "value": 0.5},
        policy={"name": "fake", "params": {}},
        environment={"name": "fake", "params": {}},
        seeds=[0, 1, 2],
        eval_episodes=100,
    )
    report = run_config(config)
    assert report.curated_vs_equal_n is not None
    assert {a.name for a in report.arms} >= {"full", "curated", "equal_n_random"}


def test_run_config_rejects_unknown_dataset() -> None:
    with pytest.raises(KeyError, match="unknown dataset kind"):
        run_config(ExperimentConfig(dataset={"kind": "nope", "params": {}}))


def test_run_config_rejects_unknown_policy() -> None:
    with pytest.raises(KeyError, match="unknown policy"):
        run_config(ExperimentConfig(policy={"name": "nope", "params": {}}))


@pytest.mark.ml
def test_run_config_real_headline_separation() -> None:
    # The Modal-bound config: CUPID curation + real BC policy. Verifies the exact path the
    # GPU worker runs (here on CPU), producing the CI-separated headline.
    config = ExperimentConfig(
        signals=[{"name": "cupid", "params": {"mode": "tracin"}}],
        policy={"name": "bc", "params": {"epochs": 250}},
        seeds=[0, 1, 2, 3, 4],
        eval_episodes=200,
    )
    report = run_config(config)
    effect = report.curated_vs_equal_n
    assert effect is not None
    assert effect.effect > 0.0 and effect.separated
