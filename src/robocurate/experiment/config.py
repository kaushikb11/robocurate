"""Serializable experiment configuration and a config -> run -> report entry point.

An :class:`ExperimentConfig` is a plain, JSON-serializable description of a headline
experiment: which dataset to build, which signals (by name + params) to curate with, the
budget, the policy and environment, and the seeds. :func:`run_config` turns it into live
objects, runs the experiment, and returns the report.

This is the unit an execution backend runs. Locally, :func:`run_config` runs in-process
(using a local GPU if present — the torch components auto-detect ``cuda``). On Modal, the
same ``config.to_dict()`` is shipped to a GPU worker that calls :func:`run_config`. It also
doubles as a reproducible, checked-in experiment definition.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from robocurate.curator import Budget, Combiner, Curator, WeightedSum
from robocurate.experiment.policy import Environment, FakeEnvironment, FakePolicy, Policy
from robocurate.experiment.runner import ExperimentSpec, run
from robocurate.experiment.synthetic import make_identity_experiment_dataset
from robocurate.signals import get as get_signal


def _build_dataset(kind: str, **params: Any) -> Any:
    if kind == "identity_synthetic":
        return make_identity_experiment_dataset(**params)
    if kind == "maniskill_demos":
        from robocurate.adapters.maniskill_demos import ManiSkillDemoReader

        return ManiSkillDemoReader(**params)
    known = ["identity_synthetic", "maniskill_demos"]
    raise KeyError(f"unknown dataset kind {kind!r}; known: {known}")


def _build_policy(name: str, **params: Any) -> Policy:
    from robocurate.experiment.policies import BCPolicy

    builders: dict[str, Any] = {"bc": BCPolicy, "fake": FakePolicy}
    if name not in builders:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(builders)}")
    return builders[name](**params)  # type: ignore[no-any-return]


def _build_environment(name: str, **params: Any) -> Environment:
    if name == "fake":
        return FakeEnvironment(**params)
    if name == "maniskill":
        from robocurate.experiment.maniskill import ManiSkillEnvironment

        return ManiSkillEnvironment(**params)
    raise KeyError(f"unknown environment {name!r}; known: ['fake', 'maniskill']")


def _build_budget(spec: Mapping[str, Any] | None) -> Budget | None:
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "fraction":
        return Budget.fraction(float(spec["value"]))
    if kind == "count":
        return Budget.count(int(spec["value"]))
    raise ValueError(f"unknown budget kind {kind!r}")


def _build_combiner(spec: Mapping[str, Any] | None) -> Combiner:
    if spec is None or spec.get("name", "weighted_sum") == "weighted_sum":
        weights = dict(spec.get("weights", {})) if spec else {}
        return WeightedSum(weights=weights)
    raise ValueError(f"unknown combiner {spec.get('name')!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    """A JSON-serializable description of a headline experiment.

    Attributes:
        dataset: ``{"kind": ..., "params": {...}}`` selecting a dataset builder.
        signals: list of ``{"name": ..., "params": {...}}`` curation signals.
        budget: ``{"kind": "fraction"|"count", "value": ...}`` or ``None``.
        combiner: optional ``{"name": "weighted_sum", "weights": {...}}``.
        seed: curator seed (selection + baseline).
        seeds: per-arm training/eval seeds.
        eval_episodes: rollout episodes per evaluation.
        policy / environment: ``{"name": ..., "params": {...}}``.
        include_ablations / include_random_filter / stats_seed: runner options.
    """

    dataset: Mapping[str, Any] = field(
        default_factory=lambda: {"kind": "identity_synthetic", "params": {}}
    )
    signals: Sequence[Mapping[str, Any]] = field(
        default_factory=lambda: [{"name": "cupid", "params": {"mode": "tracin"}}]
    )
    budget: Mapping[str, Any] | None = field(
        default_factory=lambda: {"kind": "fraction", "value": 0.5}
    )
    combiner: Mapping[str, Any] | None = None
    seed: int = 0
    seeds: Sequence[int] = (0, 1, 2)
    eval_episodes: int = 200
    policy: Mapping[str, Any] = field(default_factory=lambda: {"name": "bc", "params": {}})
    environment: Mapping[str, Any] = field(default_factory=lambda: {"name": "fake", "params": {}})
    include_ablations: bool = False
    include_random_filter: bool = True
    stats_seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": dict(self.dataset),
            "signals": [dict(s) for s in self.signals],
            "budget": dict(self.budget) if self.budget else None,
            "combiner": dict(self.combiner) if self.combiner else None,
            "seed": self.seed,
            "seeds": list(self.seeds),
            "eval_episodes": self.eval_episodes,
            "policy": dict(self.policy),
            "environment": dict(self.environment),
            "include_ablations": self.include_ablations,
            "include_random_filter": self.include_random_filter,
            "stats_seed": self.stats_seed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def run_config(config: ExperimentConfig) -> Any:
    """Build live objects from ``config``, run the experiment, and return the report."""
    # Build the cheap selectors (dataset / policy / environment) first so an invalid config
    # name fails with a clear KeyError, rather than being masked by a signal's optional-
    # dependency import (e.g. requesting an unknown policy should not require torch).
    source = _build_dataset(
        config.dataset.get("kind", "identity_synthetic"), **config.dataset.get("params", {})
    )
    policy = _build_policy(config.policy["name"], **config.policy.get("params", {}))
    environment = _build_environment(
        config.environment["name"], **config.environment.get("params", {})
    )

    signals = [get_signal(s["name"], **s.get("params", {})) for s in config.signals]
    curator = Curator(
        signals,
        combiner=_build_combiner(config.combiner),
        budget=_build_budget(config.budget),
        seed=config.seed,
    )
    spec = ExperimentSpec(
        source=source,
        curator=curator,
        policy=policy,
        environment=environment,
        seeds=tuple(config.seeds),
        eval_episodes=config.eval_episodes,
        include_ablations=config.include_ablations,
        include_random_filter=config.include_random_filter,
        stats_seed=config.stats_seed,
    )
    return run(spec)


__all__ = ["ExperimentConfig", "run_config"]
