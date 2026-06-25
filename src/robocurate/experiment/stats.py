"""Statistics for the experiment harness: bootstrap CIs and effect sizes.

Honest reporting (Invariant 6) means every reported number carries uncertainty.
Success rates across seeds are summarised by a mean and a bootstrap confidence interval;
the headline effect (curated vs a baseline) is a paired-by-seed difference with its own
bootstrap CI, and we report whether that CI excludes zero — the "is the separation real"
question reviewers actually care about. All resampling is seeded for determinism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]

DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval."""

    mean: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n}


@dataclass(frozen=True)
class EffectEstimate:
    """A paired difference (effect) with a confidence interval and a separation verdict."""

    effect: float
    ci_low: float
    ci_high: float
    n: int
    separated: bool  # True iff the CI excludes 0 (the difference is statistically resolved)

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "effect": self.effect,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "separated": self.separated,
        }


def _bootstrap_means(values: F64, *, n_resamples: int, seed: int) -> F64:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    # (n_resamples, n) indices sampled with replacement, vectorised.
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled: F64 = values[idx].mean(axis=1)
    return resampled


def bootstrap_mean(
    values: list[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Estimate:
    """Mean of ``values`` with a percentile bootstrap CI."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return Estimate(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n=0)
    if arr.size == 1:
        v = float(arr[0])
        return Estimate(mean=v, ci_low=v, ci_high=v, n=1)
    means = _bootstrap_means(arr, n_resamples=n_resamples, seed=seed)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return Estimate(mean=float(arr.mean()), ci_low=float(low), ci_high=float(high), n=int(arr.size))


def paired_effect(
    treatment: list[float],
    baseline: list[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> EffectEstimate:
    """Paired (by seed) mean difference ``treatment - baseline`` with a bootstrap CI.

    Pairing by seed controls shared run-to-run variance. ``separated`` is ``True`` when the
    CI excludes 0.
    """
    a = np.asarray(treatment, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("treatment and baseline must be non-empty and the same length")
    diffs = a - b
    if diffs.size == 1:
        d = float(diffs[0])
        return EffectEstimate(effect=d, ci_low=d, ci_high=d, n=1, separated=False)
    means = _bootstrap_means(diffs, n_resamples=n_resamples, seed=seed)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    separated = bool(low > 0.0 or high < 0.0)
    return EffectEstimate(
        effect=float(diffs.mean()),
        ci_low=float(low),
        ci_high=float(high),
        n=int(diffs.size),
        separated=separated,
    )


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RESAMPLES",
    "EffectEstimate",
    "Estimate",
    "bootstrap_mean",
    "paired_effect",
]
