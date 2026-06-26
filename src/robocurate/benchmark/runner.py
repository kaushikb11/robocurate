"""Scoring a submission against a benchmark spec: held-out BC loss vs an equal-N baseline.

:func:`run_submission` is the heart of the open benchmark. Given a frozen
:class:`~robocurate.benchmark.spec.BenchmarkSpec` and a submission, it:

1. resolves the submission to kept episodes and restricts them to the spec's **train pool**
   (the held-out eval episodes are never trainable — that is the whole point of the split);
2. builds three arms — ``submitted``, an **equal-N random** control of the same size drawn from
   the train pool (Invariant 5: the dataset-size confound is always controlled), and a ``full``
   reference (the whole train pool);
3. for each arm and each seed, trains a :class:`~robocurate.experiment.policies.BCPolicy` under
   the spec's fixed training config and measures held-out BC loss on the fixed eval split;
4. aggregates per-arm bootstrap means and the paired ``submitted`` vs ``equal_n_random`` effect.

**Lower loss is better**, so a *win* for the submission is a **negative** effect. Every result
carries the held-out-loss coverage-bias caveat (it tilts toward the random control), so the
metric is never read as an unbiased verdict — rollout success is the future unbiased arbiter.

The equal-N draw and the held-out split are both seeded (Invariant 3): the same spec +
submission + seeds produce a byte-identical :class:`BenchmarkResult`.

Needs the ``policy`` extra (torch; CPU is fine).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robocurate.benchmark.submission import resolve_submission
from robocurate.experiment.conditions import SubsetReader
from robocurate.experiment.heldout import held_out_bc_loss
from robocurate.experiment.policies import BCPolicy
from robocurate.experiment.stats import bootstrap_mean, paired_effect
from robocurate.manifest import code_version

if TYPE_CHECKING:
    from pathlib import Path

    from robocurate.adapters.base import DatasetReader
    from robocurate.benchmark.spec import BenchmarkSpec

# A fixed stream id mixed with the master seed so the equal-N draw is independent of, but
# reproducible from, the spec — mirroring the curator's baseline RNG construction.
_EQUAL_N_STREAM = 0xBE5E

COVERAGE_BIAS_NOTE = (
    "Held-out BC loss is a CPU proxy with a documented coverage bias toward the equal-N "
    "random control: the eval split is a uniform sample of the pool, so a uniform-random "
    "subset is distribution-matched to it while a deliberately non-uniform selection is not. "
    "A selection that loses here is a yellow flag, not proof of harm; closed-loop rollout "
    "success is the unbiased arbiter (a future metric backend)."
)


@dataclass(frozen=True)
class BenchmarkResult:
    """The scored outcome of one submission against a :class:`BenchmarkSpec`.

    Attributes:
        spec_version / metric / seeds: Echoed from the spec, for provenance.
        pool: The pool's ``DatasetFingerprint.to_dict()`` form the spec was built on.
        submission_name / submission_kind: Who/what was scored.
        num_kept: How many train-pool episodes the submission selected (after restriction).
        losses_by_arm: Per-arm list of held-out losses, one per seed.
        mean_loss_by_arm: Per-arm :class:`~robocurate.experiment.stats.Estimate` (bootstrap mean).
        submitted_vs_equal_n: Paired effect ``submitted - equal_n_random`` (negative == a win).
        code_version: The package version that produced the result.
        note: The coverage-bias caveat (always present — Invariant 6).
    """

    spec_version: str
    metric: str
    seeds: tuple[int, ...]
    pool: dict[str, Any]
    submission_name: str
    submission_kind: str
    num_kept: int
    losses_by_arm: dict[str, list[float]]
    mean_loss_by_arm: dict[str, dict[str, float | int]]
    submitted_vs_equal_n: dict[str, float | int | bool]
    code_version: str
    note: str = COVERAGE_BIAS_NOTE

    @property
    def submitted_mean(self) -> float:
        """The submission's mean held-out loss (lower is better)."""
        return float(self.mean_loss_by_arm["submitted"]["mean"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "metric": self.metric,
            "seeds": list(self.seeds),
            "pool": dict(self.pool),
            "submission_name": self.submission_name,
            "submission_kind": self.submission_kind,
            "num_kept": self.num_kept,
            "losses_by_arm": {k: list(v) for k, v in self.losses_by_arm.items()},
            "mean_loss_by_arm": {k: dict(v) for k, v in self.mean_loss_by_arm.items()},
            "submitted_vs_equal_n": dict(self.submitted_vs_equal_n),
            "code_version": self.code_version,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        return cls(
            spec_version=str(data["spec_version"]),
            metric=str(data["metric"]),
            seeds=tuple(int(s) for s in data["seeds"]),
            pool=dict(data["pool"]),
            submission_name=str(data["submission_name"]),
            submission_kind=str(data["submission_kind"]),
            num_kept=int(data["num_kept"]),
            losses_by_arm={k: [float(x) for x in v] for k, v in data["losses_by_arm"].items()},
            mean_loss_by_arm={k: dict(v) for k, v in data["mean_loss_by_arm"].items()},
            submitted_vs_equal_n=dict(data["submitted_vs_equal_n"]),
            code_version=str(data["code_version"]),
            note=str(data.get("note", COVERAGE_BIAS_NOTE)),
        )


def _equal_n_indices(train_pool: tuple[int, ...], n: int, *, seed: int) -> list[int]:
    """Draw a same-size (N=``n``) random subset of ``train_pool`` with a seeded RNG.

    Seeded from the spec's master ``seed`` mixed with a fixed stream id via ``SeedSequence``,
    so the control is independent of but reproducible from the spec (mirrors the curator's
    equal-N baseline construction — Invariants 3 and 5).
    """
    draw_seed = int(np.random.SeedSequence([seed, _EQUAL_N_STREAM]).generate_state(1)[0])
    rng = np.random.default_rng(draw_seed)
    pool = np.asarray(train_pool, dtype=np.int64)
    chosen = rng.choice(pool, size=n, replace=False) if 0 < n <= pool.size else pool[:n]
    return sorted(int(i) for i in chosen.tolist())


def run_submission(
    spec: BenchmarkSpec,
    submission_path: str | Path,
    pool_reader: DatasetReader,
    *,
    seeds: Sequence[int] | None = None,
    master_seed: int = 0,
) -> BenchmarkResult:
    """Score the submission at ``submission_path`` against ``spec`` on ``pool_reader``.

    Arms: ``submitted`` (the selection, restricted to the train pool), ``equal_n_random`` (a
    same-size seeded random draw from the train pool — Invariant 5), and ``full`` (the whole
    train pool, a reference). Each arm is trained under ``spec.training`` for every seed and
    evaluated on the fixed eval split. Returns per-arm bootstrap means and the paired
    ``submitted`` vs ``equal_n_random`` effect (negative == the submission wins).
    """
    run_seeds = tuple(int(s) for s in (seeds if seeds is not None else spec.seeds))
    train_pool = spec.train_pool_indices
    train_pool_set = set(train_pool)

    resolved = resolve_submission(submission_path, pool_reader)
    # Restrict to the train pool: eval episodes are never trainable (Invariant 1/5).
    submitted = sorted(i for i in resolved.kept_episode_indices if i in train_pool_set)
    if not submitted:
        raise ValueError(
            f"submission {resolved.name!r} selects no train-pool episodes; nothing to train on "
            "(did it select only held-out eval episodes?)."
        )

    equal_n = _equal_n_indices(train_pool, len(submitted), seed=master_seed)

    arms: dict[str, list[int]] = {
        "submitted": submitted,
        "equal_n_random": equal_n,
        "full": list(train_pool),
    }

    eval_reader = SubsetReader(pool_reader, spec.eval_split_indices)
    losses_by_arm: dict[str, list[float]] = {arm: [] for arm in arms}
    for arm, indices in arms.items():
        train_reader = SubsetReader(pool_reader, indices)
        for seed in run_seeds:
            policy = BCPolicy(**dict(spec.training))
            loss = held_out_bc_loss(policy, train_reader, eval_reader, seed=seed)
            losses_by_arm[arm].append(float(loss))

    mean_loss_by_arm = {
        arm: bootstrap_mean(losses, seed=master_seed).to_dict()
        for arm, losses in losses_by_arm.items()
    }
    effect = paired_effect(
        losses_by_arm["submitted"], losses_by_arm["equal_n_random"], seed=master_seed
    )

    return BenchmarkResult(
        spec_version=spec.spec_version,
        metric=spec.metric,
        seeds=run_seeds,
        pool=_pool_dict(spec),
        submission_name=resolved.name,
        submission_kind=resolved.kind,
        num_kept=len(submitted),
        losses_by_arm=losses_by_arm,
        mean_loss_by_arm=mean_loss_by_arm,
        submitted_vs_equal_n=effect.to_dict(),
        code_version=code_version(),
    )


def _pool_dict(spec: BenchmarkSpec) -> dict[str, Any]:
    fp = spec.pool
    return {
        "dataset_id": fp.dataset_id,
        "source_format": fp.source_format,
        "content_hash": fp.content_hash,
        "num_episodes": fp.num_episodes,
    }


__all__ = ["COVERAGE_BIAS_NOTE", "BenchmarkResult", "run_submission"]
