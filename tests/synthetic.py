"""Synthetic in-memory fixtures shared across contract tests.

These builders construct tiny, fully-deterministic trajectories so tests exercise the
*contracts* (the protocols and data flow) without depending on any real dataset or any real
curation signal.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from robocurate.metadata import (
    DatasetFingerprint,
    DatasetMeta,
    ResourceProbe,
)
from robocurate.signals.base import (
    CostTier,
    InMemoryCache,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.trajectory import (
    Array,
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    InMemoryFeatureStore,
    SuccessLabel,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)

# A minimal 2-DoF embodiment: timestamps, a 2-d action, a 2-d proprio state, and a reward.
TOY_EMBODIMENT = EmbodimentSpec(
    embodiment_id="toy2dof",
    control_hz=10.0,
    features=(
        FeatureSpec("timestamp", FeatureRole.TIME, shape=(), dtype="float32", units="s"),
        FeatureSpec(
            "action",
            FeatureRole.ACTION,
            shape=(2,),
            dtype="float32",
            units="normalized[-1,1]",
            names=("x", "y"),
        ),
        FeatureSpec(
            "observation.state",
            FeatureRole.PROPRIO,
            shape=(2,),
            dtype="float32",
            units="m",
            names=("px", "py"),
        ),
        FeatureSpec("reward", FeatureRole.REWARD, shape=(), dtype="float32", units=None),
    ),
)


def make_trajectory(
    episode_index: int,
    num_steps: int = 8,
    *,
    success: bool | None = True,
    scale: float = 1.0,
) -> Trajectory:
    """Build a deterministic toy :class:`Trajectory`.

    ``scale`` simply scales the action magnitude so different episodes have distinguishable
    content (and distinct fingerprints). Nothing here computes a quality signal.
    """
    t = np.arange(num_steps, dtype=np.float32) / TOY_EMBODIMENT.control_hz  # type: ignore[operator]
    phase = float(episode_index)
    action = scale * np.stack([np.sin(t + phase), np.cos(t + phase)], axis=-1).astype(np.float32)
    state = np.cumsum(action, axis=0).astype(np.float32)
    reward = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)

    columns: dict[str, Array] = {
        "timestamp": t,
        "action": action,
        "observation.state": state,
        "reward": reward,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/toy",
        episode_index=episode_index,
        embodiment=TOY_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
        success=SuccessLabel(value=success, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


class FakeActionMagnitudeSignal:
    """A fake, deterministic in-memory signal used only to exercise the Signal contract.

    It is **not** a real curation signal — it simply scores each trajectory by the mean
    absolute action magnitude (lower-is-better as a stand-in for "less aggressive"), and
    records a fit() call so the fit→score seam is testable. A trajectory without an action
    feature is recorded as a skip, never an error.
    """

    def __init__(self, *, name: str = "fake_action_magnitude") -> None:
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({"action"}),
            produces_per_transition=True,
            deterministic=True,
            description="Mean absolute action magnitude (fake test signal).",
        )
        self.fit_calls = 0

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        # Stateless heuristics may no-op; we count the call to prove the seam works and to
        # show where a real signal would train a classifier / precompute embeddings.
        self.fit_calls += 1
        ctx.cache.put("fitted", True)

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        out: list[TrajectoryScore] = []
        for traj in batch:
            actions = traj.actions()
            if actions is None:
                out.append(
                    TrajectoryScore.skip(
                        self.spec.name,
                        traj.meta.fingerprint,
                        reason="no action feature",
                        higher_is_better=False,
                    )
                )
                continue
            per_step = np.abs(actions).mean(axis=tuple(range(1, actions.ndim)))
            out.append(
                TrajectoryScore(
                    signal=self.spec.name,
                    trajectory_fingerprint=traj.meta.fingerprint,
                    value=float(per_step.mean()),
                    higher_is_better=False,
                    per_transition=per_step.astype(np.float32),
                )
            )
        return out


def write_synthetic_lerobot_dataset(
    root: str | Path,
    *,
    num_episodes: int = 4,
    num_steps: int = 8,
    success: list[bool | None] | None = None,
) -> Path:
    """Write a tiny, valid LeRobot v2.1 dataset to ``root`` directly (not via our writer).

    Building the source independently of :class:`LeRobotWriter` is deliberate: the
    round-trip test then asserts the *source* is byte-for-byte untouched after a curation
    run that writes a *new* dataset, which is the whole point of Invariant 1.

    ``success`` optionally provides a per-episode success label written into
    ``episodes.jsonl``; omit it for a dataset with no success labels.
    """
    # Imported here to avoid a module-level dependency from fixtures on the adapter internals.
    from robocurate.adapters.lerobot import _trajectory_to_table

    root = Path(root)
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)

    trajectories = [make_trajectory(i, num_steps=num_steps) for i in range(num_episodes)]
    running = 0
    total_frames = 0
    episode_records = []
    for i, traj in enumerate(trajectories):
        table, running = _trajectory_to_table(traj, i, running)
        pq.write_table(  # type: ignore[no-untyped-call]
            table, root / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
        )
        total_frames += traj.num_steps
        record: dict[str, object] = {"episode_index": i, "length": traj.num_steps, "tasks": []}
        if success is not None:
            record["success"] = success[i]
        episode_records.append(record)

    features = {
        spec.key: {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "names": list(spec.names) if spec.names else None,
        }
        for spec in TOY_EMBODIMENT.features
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": TOY_EMBODIMENT.embodiment_id,
        "fps": TOY_EMBODIMENT.control_hz,
        "features": features,
        "total_episodes": num_episodes,
        "total_frames": total_frames,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True))
    with (root / "meta" / "episodes.jsonl").open("w") as fh:
        for record in episode_records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": ""}) + "\n")
    return root


def make_signal_context(seed: int = 0) -> SignalContext:
    """Build a minimal :class:`SignalContext` for unit-testing signals in isolation."""
    fingerprint = DatasetFingerprint(
        dataset_id="synthetic/toy",
        source_format="synthetic_v0",
        content_hash="0" * 64,
        num_episodes=0,
    )
    dataset_meta = DatasetMeta(
        fingerprint=fingerprint,
        embodiment_ids=(TOY_EMBODIMENT.embodiment_id,),
        feature_keys=tuple(spec.key for spec in TOY_EMBODIMENT.features),
    )
    return SignalContext(
        seed=seed,
        device="cpu",
        cache=InMemoryCache(),
        resources=ResourceProbe(),
        dataset_meta=dataset_meta,
        logger=logging.getLogger("robocurate.test"),
    )
