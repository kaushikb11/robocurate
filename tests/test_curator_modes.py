"""Tests for the curator's hard validity-gate and greedy-dedup selection mode."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from robocurate.curator import Budget, Curator, GateConfig, SelectionMode
from robocurate.signals.base import CostTier, SignalContext, SignalSpec, TrajectoryScore
from robocurate.signals.jerk import Jerk
from robocurate.signals.sim_validity import SimPhysicsValidity
from robocurate.trajectory import Trajectory
from tests.test_jerk_signal import _ListReader
from tests.test_redundancy_signal import _dataset_with_cluster, _ramp, _traj
from tests.test_sim_validity_signal import _sim_traj

# --- hard validity gate --------------------------------------------------------------


def _gated_curator(budget: Budget) -> Curator:
    return Curator(
        [SimPhysicsValidity()],
        gate=GateConfig("sim_physics_validity", reject_above=0.0),
        budget=budget,
        seed=0,
    )


def test_gate_removes_invalid_regardless_of_budget() -> None:
    # Four valid sim trajectories + two physically-invalid ones.
    trajs = [_sim_traj(i, penetration=0.0) for i in range(4)]
    trajs += [_sim_traj(4, penetration=0.05), _sim_traj(5, penetration=0.08)]
    reader = _ListReader(trajs)

    # Budget would allow keeping 5, but the 2 invalid must never be kept.
    result = _gated_curator(Budget.count(5)).run(reader)
    assert {4, 5} <= set(result.removed_episode_indices)
    assert set(result.kept_episode_indices) <= {0, 1, 2, 3}
    assert result.num_kept == 4  # budget clamped to the valid pool size

    invalid = next(d for d in result.decisions if d.episode_index == 4)
    assert "gate" in invalid.reason


def test_gate_baseline_drawn_from_valid_pool() -> None:
    trajs = [_sim_traj(i, penetration=0.0) for i in range(4)]
    trajs += [_sim_traj(4, penetration=0.05), _sim_traj(5, penetration=0.08)]
    result = _gated_curator(Budget.count(2)).run(_ListReader(trajs))
    assert result.baseline is not None
    assert result.baseline.n == 2
    # Invalid episodes (4, 5) can never appear in the equal-N baseline.
    assert set(result.baseline.kept_episode_indices) <= {0, 1, 2, 3}


def test_gate_misconfiguration_rejected() -> None:
    with pytest.raises(ValueError, match="not among"):
        Curator([Jerk()], gate=GateConfig("does_not_exist"))


def test_gate_recorded_in_config_snapshot() -> None:
    trajs = [_sim_traj(i, penetration=0.0) for i in range(3)]
    result = _gated_curator(Budget.count(2)).run(_ListReader(trajs))
    cfg = result.config.to_dict()
    assert cfg["gate"]["signal"] == "sim_physics_validity"
    assert cfg["selection"] == "top_k"


# --- greedy dedup --------------------------------------------------------------------


def test_greedy_dedup_collapses_cluster_to_one_representative() -> None:
    # Three well-separated distinct trajectories + a tight cluster of three near-copies.
    reader = _dataset_with_cluster()
    result = Curator([], selection=SelectionMode.GREEDY_DEDUP, budget=Budget.count(6), seed=0).run(
        reader
    )
    kept = set(result.kept_episode_indices)
    assert {0, 1, 2} <= kept  # the distinct ones survive
    assert len({3, 4, 5} & kept) == 1  # the cluster collapses to a single representative

    dropped = next(d for d in result.decisions if d.episode_index in ({3, 4, 5} - kept))
    assert "near-duplicate" in dropped.reason


def test_greedy_dedup_keeps_highest_keep_score_member() -> None:
    # A cluster of three near-identical trajectories; a preset signal prefers member 1.
    cluster = [_traj(i, _ramp(24, amp=4.0, noise=1e-4, seed=200 + i)) for i in range(3)]
    reader = _ListReader(cluster)
    fps = [t.meta.fingerprint for t in cluster]
    preset = _PresetSignal({fps[0]: 0.0, fps[1]: 1.0, fps[2]: 0.0})

    result = Curator(
        [preset],
        selection=SelectionMode.GREEDY_DEDUP,
        dedup_epsilon=5.0,
        budget=Budget.count(3),
        seed=0,
    ).run(reader)
    assert result.num_kept == 1
    assert result.kept_episode_indices[0] == 1  # the highest-scoring cluster member


def test_greedy_dedup_is_deterministic() -> None:
    reader = _dataset_with_cluster()

    def run() -> tuple[int, ...]:
        cur = Curator([], selection=SelectionMode.GREEDY_DEDUP, budget=Budget.count(6), seed=0)
        return cur.run(reader).kept_episode_indices

    assert run() == run()


class _PresetSignal:
    """A test signal that returns preset values keyed by fingerprint (higher is better)."""

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
