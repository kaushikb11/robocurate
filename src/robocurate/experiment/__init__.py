"""The headline experiment harness (validity-critical; see Invariant 5).

Trains a policy on curated vs control subsets across multiple seeds and reports the effect
with uncertainty, always alongside the **equal-N random** baseline so the dataset-size
confound is controlled by construction. The reusable framework lives here; concrete headline
experiment definitions live under the top-level ``experiments/`` directory.

Scaffolding-first: a deterministic :class:`FakePolicy` + :class:`FakeEnvironment` make the
whole pipeline testable now; real policies and a real sim environment implement the same
:class:`Policy` / :class:`Environment` protocols later, behind extras.
"""

from __future__ import annotations

from robocurate.experiment.conditions import Arm, Condition, SubsetReader
from robocurate.experiment.config import ExperimentConfig, run_config
from robocurate.experiment.maniskill import (
    ManiSkillEnvironment,
    RandomPolicy,
    RandomTrainedPolicy,
)
from robocurate.experiment.policies import BCPolicy, BCTrainedPolicy
from robocurate.experiment.policy import (
    Environment,
    EvalResult,
    FakeEnvironment,
    FakePolicy,
    FakeTrainedPolicy,
    Policy,
    TrainedPolicy,
)
from robocurate.experiment.report import ArmReport, ExperimentReport, build_report
from robocurate.experiment.runner import ExperimentSpec, run
from robocurate.experiment.stats import EffectEstimate, Estimate, bootstrap_mean, paired_effect
from robocurate.experiment.synthetic import make_identity_experiment_dataset

__all__ = [
    "Arm",
    "ArmReport",
    "BCPolicy",
    "BCTrainedPolicy",
    "Condition",
    "EffectEstimate",
    "Environment",
    "Estimate",
    "EvalResult",
    "ExperimentConfig",
    "ExperimentReport",
    "ExperimentSpec",
    "FakeEnvironment",
    "FakePolicy",
    "FakeTrainedPolicy",
    "ManiSkillEnvironment",
    "Policy",
    "RandomPolicy",
    "RandomTrainedPolicy",
    "SubsetReader",
    "TrainedPolicy",
    "bootstrap_mean",
    "build_report",
    "make_identity_experiment_dataset",
    "paired_effect",
    "run",
    "run_config",
]
