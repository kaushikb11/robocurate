"""Synthetic experiment datasets (library-level, usable on a Modal worker).

Generates a tiny, fully-deterministic dataset designed to *demonstrate* curation: most
trajectories are "helpful" (the action equals the observation — the identity task the
:class:`~robocurate.experiment.policy.FakeEnvironment` rewards) and a minority are
"contradictory" (action = -observation), which corrupts a policy trained on them. CUPID
influence keeps the helpful consensus, so curated training beats an equal-size random subset
— the separation the experiment exists to detect, with zero data plumbing.

This lives in the library (not the test suite) so the Modal job can build its dataset on the
worker without shipping data.
"""

from __future__ import annotations

import numpy as np

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.trajectory import (
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)

_EMBODIMENT = EmbodimentSpec(
    embodiment_id="identity2d",
    control_hz=10.0,
    features=(
        FeatureSpec("timestamp", FeatureRole.TIME, shape=(), dtype="float32", units="s"),
        FeatureSpec("action", FeatureRole.ACTION, shape=(2,), dtype="float32"),
        FeatureSpec("observation.state", FeatureRole.PROPRIO, shape=(2,), dtype="float32"),
    ),
)


def _identity_trajectory(
    index: int, *, sign: float, noise: float, num_steps: int, seed: int
) -> Trajectory:
    rng = np.random.default_rng(seed)
    state = rng.normal(0.0, 1.0, size=(num_steps, 2)).astype(np.float32)
    action = (sign * state + rng.normal(0.0, noise, size=(num_steps, 2))).astype(np.float32)
    columns = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) / 10.0),
        "action": action,
        "observation.state": state,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/identity",
        episode_index=index,
        embodiment=_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_identity_v0",
        success=SuccessLabel(value=sign > 0, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def make_identity_experiment_dataset(
    *,
    num_helpful: int = 12,
    num_harmful: int = 4,
    noise: float = 0.02,
    num_steps: int = 24,
    seed: int = 0,
) -> InMemoryDatasetReader:
    """Build the helpful-majority / contradictory-minority identity dataset.

    Helpful trajectories have ``action = observation`` (the rewarded task); harmful ones have
    ``action = -observation``. Influence-based curation keeps the helpful consensus.
    """
    helpful = [
        _identity_trajectory(i, sign=1.0, noise=noise, num_steps=num_steps, seed=seed + i)
        for i in range(num_helpful)
    ]
    harmful = [
        _identity_trajectory(
            num_helpful + i, sign=-1.0, noise=noise, num_steps=num_steps, seed=seed + 1000 + i
        )
        for i in range(num_harmful)
    ]
    return InMemoryDatasetReader(helpful + harmful, dataset_id="synthetic/identity")


__all__ = ["make_identity_experiment_dataset"]
