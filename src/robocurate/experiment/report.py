"""The experiment report: machine- and human-readable, effect sizes with uncertainty.

Leads with the question reviewers attack first — does the curated policy beat an **equal-N
random** subset of the same size? — reported as an effect with a confidence interval and a
clear "separated" verdict, never a single cherry-picked number (Invariant 6). Also
reports curated-vs-full, a per-arm success table with CIs, the per-task breakdown, and the
per-signal ablation contributions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from robocurate.experiment.conditions import Condition
from robocurate.experiment.stats import (
    EffectEstimate,
    Estimate,
    bootstrap_mean,
    paired_effect,
)

if TYPE_CHECKING:
    from robocurate.experiment.conditions import Arm
    from robocurate.experiment.policy import EvalResult

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class ArmReport:
    """Aggregated results for one experiment arm across seeds."""

    name: str
    condition: Condition
    size: int
    success: Estimate
    per_task: Mapping[str, Estimate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition": self.condition.value,
            "size": self.size,
            "success": self.success.to_dict(),
            "per_task": {t: e.to_dict() for t, e in self.per_task.items()},
        }


@dataclass(frozen=True)
class ExperimentReport:
    """The full experiment outcome."""

    schema_version: str
    dataset_id: str
    total_episodes: int
    seeds: tuple[int, ...]
    eval_episodes: int
    arms: tuple[ArmReport, ...]
    curated_vs_equal_n: EffectEstimate | None
    curated_vs_full: EffectEstimate | None

    def arm(self, name: str) -> ArmReport | None:
        return next((a for a in self.arms if a.name == name), None)

    # -- machine-readable ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "total_episodes": self.total_episodes,
            "seeds": list(self.seeds),
            "eval_episodes": self.eval_episodes,
            "arms": [a.to_dict() for a in self.arms],
            "headline": {
                "curated_vs_equal_n_random": (
                    self.curated_vs_equal_n.to_dict() if self.curated_vs_equal_n else None
                ),
                "curated_vs_full": (
                    self.curated_vs_full.to_dict() if self.curated_vs_full else None
                ),
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # -- human-readable --------------------------------------------------------------

    def to_markdown(self) -> str:
        lines: list[str] = [f"# Experiment report — `{self.dataset_id}`", ""]
        lines.append(
            f"{len(self.seeds)} seed(s), {self.eval_episodes} eval episodes/arm, "
            f"{self.total_episodes} source episodes."
        )
        lines.append("")
        lines.append("## Headline")
        lines.append("")
        lines.append(self._effect_line("Curated vs equal-N random", self.curated_vs_equal_n))
        lines.append(self._effect_line("Curated vs full data", self.curated_vs_full))
        lines.append("")
        lines.append("## Arms")
        lines.append("")
        lines.append("| arm | condition | size | success rate (95% CI) |")
        lines.append("| --- | --- | ---: | --- |")
        for arm in self.arms:
            s = arm.success
            lines.append(
                f"| {arm.name} | {arm.condition.value} | {arm.size} | "
                f"{s.mean:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}] |"
            )
        lines.append("")
        ablations = [a for a in self.arms if a.condition is Condition.ABLATION]
        if ablations:
            lines.append("## Per-signal ablation")
            lines.append("")
            for arm in ablations:
                lines.append(
                    f"- **{arm.name}** — success {arm.success.mean:.3f} "
                    f"[{arm.success.ci_low:.3f}, {arm.success.ci_high:.3f}]"
                )
            lines.append("")
        lines.append(self._per_task_section())
        return "\n".join(lines)

    def _effect_line(self, label: str, effect: EffectEstimate | None) -> str:
        if effect is None:
            return f"- {label}: _not available_"
        verdict = "separated (CI excludes 0)" if effect.separated else "not separated"
        return (
            f"- {label}: **{effect.effect:+.3f}** "
            f"(95% CI [{effect.ci_low:+.3f}, {effect.ci_high:+.3f}]) — {verdict}"
        )

    def _per_task_section(self) -> str:
        tasks = sorted({t for a in self.arms for t in a.per_task})
        if len(tasks) <= 1:
            return ""
        lines = ["## Per-task success (mean)", "", "| arm | " + " | ".join(tasks) + " |"]
        lines.append("| --- |" + " ---: |" * len(tasks))
        for arm in self.arms:
            cells = [f"{arm.per_task[t].mean:.3f}" if t in arm.per_task else "—" for t in tasks]
            lines.append(f"| {arm.name} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


def build_report(
    *,
    dataset_id: str,
    total_episodes: int,
    seeds: Sequence[int],
    eval_episodes: int,
    arm_results: Sequence[tuple[Arm, Sequence[EvalResult]]],
    stats_seed: int = 0,
) -> ExperimentReport:
    """Aggregate raw per-seed rollout results into an :class:`ExperimentReport`."""
    arms: list[ArmReport] = []
    by_name: dict[str, list[float]] = {}
    for arm, results in arm_results:
        successes = [r.success_rate for r in results]
        by_name[arm.name] = successes
        tasks = sorted({t for r in results for t in r.per_task})
        per_task = {
            task: bootstrap_mean(
                [r.per_task[task] for r in results if task in r.per_task], seed=stats_seed
            )
            for task in tasks
        }
        arms.append(
            ArmReport(
                name=arm.name,
                condition=arm.condition,
                size=arm.size,
                success=bootstrap_mean(successes, seed=stats_seed),
                per_task=per_task,
            )
        )

    curated = by_name.get("curated")
    equal_n = by_name.get("equal_n_random")
    full = by_name.get("full")
    vs_equal_n = paired_effect(curated, equal_n, seed=stats_seed) if curated and equal_n else None
    vs_full = paired_effect(curated, full, seed=stats_seed) if curated and full else None

    return ExperimentReport(
        schema_version=SCHEMA_VERSION,
        dataset_id=dataset_id,
        total_episodes=total_episodes,
        seeds=tuple(seeds),
        eval_episodes=eval_episodes,
        arms=tuple(arms),
        curated_vs_equal_n=vs_equal_n,
        curated_vs_full=vs_full,
    )


__all__ = ["SCHEMA_VERSION", "ArmReport", "ExperimentReport", "build_report"]
