"""Redundancy / dedup heuristic (Tier 0-1, dataset-relative).

Scores each trajectory's **uniqueness**: its distance to its nearest neighbours in an
embedding space, computed over the whole dataset in ``fit``. A trajectory far from
everything else is unique (keep it); a near-duplicate sits ~0 from its twin and is flagged
as redundant bloat. The emitted ``value`` is the mean distance to the ``k`` nearest
neighbours, with ``higher_is_better=True`` (more distant = more keepable).

Confirmed design choices:

* **Embedding is pluggable** (``embedding=`` callable). The default is a cheap, fixed-length
  *statistical* embedding of DoF-count-independent scalars, so it needs no GPU/model and
  works across mixed embodiments. It is deliberately coarse — genuinely different
  trajectories with similar gross statistics can look falsely redundant; a learned encoder
  (Tier 1) is the upgrade through the same ``embedding=`` seam.
* **Metric:** mean distance to the ``k`` nearest neighbours (default ``k=1`` ≡ 1-NN). ``k>1``
  is more robust to a single near-duplicate and distinguishes a tight cluster from a lone
  pair.
* **Distance:** Euclidean on the **z-standardized** embedding (per-feature mean/std from
  ``fit``), so heterogeneous summary features (length vs magnitude) are comparable.

Scope note: this is a per-trajectory *uniqueness ranking*, not greedy
keep-one-representative-per-cluster dedup. The latter is a future selection/combiner mode
(it does not fit the per-trajectory ``Signal.score`` contract); the embedding index this
signal builds is the seam it will reuse.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.trajectory import Array, FeatureRole, Trajectory

F64 = npt.NDArray[np.float64]
# An embedding maps a trajectory to a fixed-length vector (any numeric dtype; converted to
# float64 internally) or returns None to exclude it from the index.
EmbeddingFn = Callable[[Trajectory], "Array | None"]

DEFAULT_K = 1
_INDEX_KEY = "index"


class Redundancy:
    """Dataset-relative uniqueness signal via k-NN distance in an embedding space.

    Args:
        embedding: Maps a trajectory to a fixed-length vector, or ``None`` to skip it.
            Defaults to :func:`statistical_embedding`. A custom (e.g. learned) embedding can
            be supplied here without any core change.
        k: Number of nearest neighbours to average (default 1).
        name: Override the signal name.
    """

    def __init__(
        self,
        *,
        embedding: EmbeddingFn | None = None,
        k: int = DEFAULT_K,
        name: str = "redundancy",
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.embedding: EmbeddingFn = embedding if embedding is not None else statistical_embedding
        self.k = k
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset(),  # the default embedding degrades gracefully; no hard reqs
            produces_per_transition=False,
            deterministic=True,
            description=(
                f"Uniqueness = mean distance to the {k} nearest neighbour(s) in a "
                "statistical embedding (higher is more unique)."
            ),
        )

    # -- fit: build the embedding index ----------------------------------------------

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        vectors: list[F64] = []
        fingerprints: list[str] = []
        for traj in trajectories:
            emb = self.embedding(traj)
            if emb is None:
                continue
            vectors.append(np.asarray(emb, dtype=np.float64).reshape(-1))
            fingerprints.append(traj.meta.fingerprint)

        if not vectors:
            ctx.cache.put(_INDEX_KEY, None)
            return

        raw: F64 = np.vstack(vectors).astype(np.float64)
        mean: F64 = raw.mean(axis=0)
        std: F64 = raw.std(axis=0)
        std_safe = np.where(std > 0.0, std, 1.0)
        z: F64 = (raw - mean) / std_safe
        ctx.cache.put(
            _INDEX_KEY,
            {
                "z": z,
                "mean": mean,
                "std_safe": std_safe,
                "fingerprints": fingerprints,
            },
        )

    # -- score -----------------------------------------------------------------------

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        index = ctx.cache.get(_INDEX_KEY) if ctx.cache.has(_INDEX_KEY) else None
        return [self._score_one(traj, index) for traj in batch]

    def _score_one(self, traj: Trajectory, index: dict[str, Any] | None) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        emb = self.embedding(traj)
        if emb is None:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="trajectory has no embeddable features (no action or state)",
                higher_is_better=True,
            )
        if index is None:
            return TrajectoryScore.skip(
                self.spec.name, fingerprint, reason="empty embedding index", higher_is_better=True
            )

        z_all: F64 = index["z"]
        fingerprints: list[str] = index["fingerprints"]
        z_self = (np.asarray(emb, dtype=np.float64).reshape(-1) - index["mean"]) / index["std_safe"]
        distances: F64 = np.linalg.norm(z_all - z_self, axis=1)

        # Exclude exactly one self-occurrence so a trajectory is never its own neighbour, while
        # genuine (near-)duplicates of it remain in the pool and correctly register as close.
        keep = np.ones(distances.shape[0], dtype=bool)
        self_positions = [i for i, fp in enumerate(fingerprints) if fp == fingerprint]
        if self_positions:
            keep[self_positions[0]] = False
        others = distances[keep]
        if others.size == 0:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="dataset has no other trajectories to compare against",
                higher_is_better=True,
            )

        kk = min(self.k, others.size)
        nearest = np.partition(others, kk - 1)[:kk]
        value = float(nearest.mean())
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=value,
            higher_is_better=True,
            diagnostics={
                "nn_distance": float(others.min()),
                "mean_knn_distance": value,
                "k": kk,
                "n_neighbors": int(others.size),
                "embedding_dim": int(z_all.shape[1]),
            },
        )


def statistical_embedding(traj: Trajectory) -> Array | None:
    """Default fixed-length, DoF-independent statistical embedding of a trajectory.

    Returns a length-7 vector of scalars that do not depend on the action/state dimension, so
    it is comparable across embodiments. Returns ``None`` if the trajectory carries neither an
    action nor a state/proprio feature (nothing to embed).
    """
    action = traj.actions()
    proprio = traj.select_roles(FeatureRole.PROPRIO, FeatureRole.STATE)
    if action is None and not proprio:
        return None

    feats: list[float] = [float(traj.num_steps)]
    feats.extend(_signal_stats(action))
    state = _concat_proprio(proprio)
    feats.extend(_signal_stats(state))
    return np.asarray(feats, dtype=np.float64)


def _signal_stats(arr: Array | None) -> list[float]:
    """Three DoF-independent scalars for a per-step signal: mean & std magnitude, mean |delta|."""
    if arr is None or arr.shape[0] == 0:
        return [0.0, 0.0, 0.0]
    x = np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1)
    magnitude = np.linalg.norm(x, axis=1)
    delta = float(np.linalg.norm(np.diff(x, axis=0), axis=1).mean()) if x.shape[0] > 1 else 0.0
    return [float(magnitude.mean()), float(magnitude.std()), delta]


def _concat_proprio(proprio: dict[str, Array]) -> F64 | None:
    if not proprio:
        return None
    arrays = [np.asarray(v, dtype=np.float64).reshape(v.shape[0], -1) for v in proprio.values()]
    return np.concatenate(arrays, axis=1)


__all__ = ["Redundancy", "statistical_embedding"]
