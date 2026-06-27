"""Tests for the COVERAGE selection mode (greedy submodular facility-location).

Builds a synthetic dataset with three well-separated clusters in ``statistical_embedding``
space and asserts the coverage selector spreads its budget across clusters (diversity), is
deterministic, honors the budget exactly, respects the validity gate, and leaves the equal-N
random baseline (Invariant 5) and recipe round-trip untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from robocurate.curator import (
    Budget,
    Curator,
    GateConfig,
    SelectionMode,
)
from robocurate.recipe import load_recipe, save_recipe
from robocurate.signals.base import CostTier, SignalContext, SignalSpec, TrajectoryScore
from robocurate.signals.redundancy import statistical_embedding
from robocurate.signals.sim_validity import SimPhysicsValidity
from robocurate.trajectory import Trajectory
from tests.test_jerk_signal import _ListReader
from tests.test_redundancy_signal import _ramp, _traj
from tests.test_sim_validity_signal import _sim_traj

# Cluster layout: three well-separated clusters of three near-copies each. Clusters differ in
# action magnitude AND length so their statistical embeddings are far apart; members within a
# cluster differ only by tiny noise.
_CLUSTERS = {
    "A": (24, 1.0),  # positions 0, 1, 2
    "B": (48, 8.0),  # positions 3, 4, 5
    "C": (24, 30.0),  # positions 6, 7, 8
}


def _clustered_reader() -> _ListReader:
    trajs: list[Trajectory] = []
    idx = 0
    for c, (steps, amp) in enumerate(_CLUSTERS.values()):
        for m in range(3):
            trajs.append(_traj(idx, _ramp(steps, amp=amp, noise=1e-4, seed=1000 + 10 * c + m)))
            idx += 1
    return _ListReader(trajs)


def _cluster_of_position(pos: int) -> str:
    return ("A", "B", "C")[pos // 3]


class _PresetSignal:
    """Returns preset keep-values keyed by fingerprint (higher is better)."""

    def __init__(self, values: dict[str, float]) -> None:
        self._values = values
        self.spec = SignalSpec(
            name="preset",
            version="0.0.0",
            cost_tier=CostTier.TIER0_CPU,
            produces_per_transition=False,
            deterministic=True,
        )

    def fit(self, trajectories: object, ctx: SignalContext) -> None:
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [
            TrajectoryScore(
                signal=self.spec.name,
                trajectory_fingerprint=t.meta.fingerprint,
                value=self._values.get(t.meta.fingerprint, 0.0),
                higher_is_better=True,
            )
            for t in batch
        ]


def _preset_favoring_cluster_a(reader: _ListReader) -> _PresetSignal:
    """A keep-score that ranks every cluster-A member above all of clusters B and C.

    This is the setup where TOP_K piles its whole budget into cluster A.
    """
    values: dict[str, float] = {}
    for pos in range(len(reader)):
        fp = reader.read_episode(pos).meta.fingerprint
        values[fp] = 1.0 if _cluster_of_position(pos) == "A" else 0.0
    return _PresetSignal(values)


def _kept_clusters(kept: Sequence[int]) -> set[str]:
    return {_cluster_of_position(p) for p in kept}


def _min_pairwise_embedding_distance(reader: _ListReader, kept: Sequence[int]) -> float:
    vecs = [
        np.asarray(statistical_embedding(reader.read_episode(p)), dtype=np.float64) for p in kept
    ]
    best = np.inf
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            best = min(best, float(np.linalg.norm(vecs[i] - vecs[j])))
    return best


# --- diversity known-answer ----------------------------------------------------------


def test_coverage_covers_more_clusters_than_top_k() -> None:
    reader = _clustered_reader()
    preset = _preset_favoring_cluster_a(reader)
    budget = Budget.count(3)

    top_k = Curator([preset], selection=SelectionMode.TOP_K, budget=budget, seed=0).run(reader)
    coverage = Curator([preset], selection=SelectionMode.COVERAGE, budget=budget, seed=0).run(
        reader
    )

    top_k_kept = top_k.kept_episode_indices
    coverage_kept = coverage.kept_episode_indices

    # TOP_K piles its whole budget into the single high-scoring cluster A.
    assert _kept_clusters(top_k_kept) == {"A"}
    # COVERAGE spreads across all three clusters: strictly more distinct clusters covered, and
    # a strictly larger minimum pairwise embedding distance among kept trajectories.
    assert _kept_clusters(coverage_kept) == {"A", "B", "C"}
    assert len(_kept_clusters(coverage_kept)) > len(_kept_clusters(top_k_kept))
    assert _min_pairwise_embedding_distance(
        reader, coverage_kept
    ) > _min_pairwise_embedding_distance(reader, top_k_kept)

    dropped = next(d for d in coverage.decisions if d.episode_index not in set(coverage_kept))
    assert "coverage selection" in dropped.reason


# --- determinism ---------------------------------------------------------------------


def test_coverage_is_deterministic() -> None:
    reader = _clustered_reader()
    preset = _preset_favoring_cluster_a(reader)

    def run() -> tuple[tuple[int, ...], tuple[str, ...]]:
        result = Curator(
            [preset], selection=SelectionMode.COVERAGE, budget=Budget.count(4), seed=0
        ).run(reader)
        reasons = tuple(d.reason for d in result.decisions)
        return result.kept_episode_indices, reasons

    first = run()
    second = run()
    assert first == second


# --- budget --------------------------------------------------------------------------


def test_coverage_respects_budget_exactly() -> None:
    reader = _clustered_reader()
    preset = _preset_favoring_cluster_a(reader)
    for k in (1, 3, 5, 9):
        result = Curator(
            [preset], selection=SelectionMode.COVERAGE, budget=Budget.count(k), seed=0
        ).run(reader)
        assert result.num_kept == k


# --- gate ----------------------------------------------------------------------------


def test_coverage_never_selects_gated_positions() -> None:
    # Four valid sim trajectories + two physically-invalid ones; the gate must win.
    trajs = [_sim_traj(i, penetration=0.0) for i in range(4)]
    trajs += [_sim_traj(4, penetration=0.05), _sim_traj(5, penetration=0.08)]
    reader = _ListReader(trajs)
    result = Curator(
        [SimPhysicsValidity()],
        gate=GateConfig("sim_physics_validity", reject_above=0.0),
        selection=SelectionMode.COVERAGE,
        budget=Budget.count(5),
        seed=0,
    ).run(reader)
    assert set(result.kept_episode_indices) <= {0, 1, 2, 3}
    assert {4, 5} <= set(result.removed_episode_indices)


# --- equal-N baseline unchanged (Invariant 5) ----------------------------------------


def test_coverage_leaves_equal_n_baseline_identical() -> None:
    reader = _clustered_reader()
    preset = _preset_favoring_cluster_a(reader)
    budget = Budget.count(4)

    top_k = Curator([preset], selection=SelectionMode.TOP_K, budget=budget, seed=0).run(reader)
    coverage = Curator([preset], selection=SelectionMode.COVERAGE, budget=budget, seed=0).run(
        reader
    )

    assert top_k.baseline is not None and coverage.baseline is not None
    # The equal-N random baseline draws from the SAME valid pool with the SAME k regardless of
    # selection mode, so it must be byte-identical across modes (Invariant 5).
    assert top_k.baseline.to_dict() == coverage.baseline.to_dict()


# --- recipe round-trip ---------------------------------------------------------------


def test_coverage_recipe_round_trips(tmp_path: object) -> None:
    reader = _clustered_reader()
    # The recipe reconstructs signals from the combiner's weight keys; use the registry-backed
    # redundancy signal (statistical embedding) so the loaded curator is fully rebuildable.
    from robocurate.curator import WeightedSum
    from robocurate.signals.redundancy import Redundancy

    curator = Curator(
        [Redundancy()],
        combiner=WeightedSum(weights={"redundancy": 1.0}),
        selection=SelectionMode.COVERAGE,
        budget=Budget.count(3),
        seed=0,
    )
    direct = curator.run(reader).kept_episode_indices

    path = tmp_path / "recipe.json"  # type: ignore[operator]
    save_recipe(curator, path)
    loaded = load_recipe(path)
    assert loaded.selection is SelectionMode.COVERAGE
    assert loaded.run(reader).kept_episode_indices == direct
