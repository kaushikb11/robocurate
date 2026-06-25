"""Self-contained synthetic demo dataset for the "run this first" onboarding path.

:func:`write_demo_dataset` deterministically builds a handful of tiny trajectories — a
mix of clearly *good* episodes (smooth, direct reaches) and clearly *bad* ones (jittery
and/or wandering) — and writes them out as a valid LeRobotDataset v2.1 directory via the
public :class:`~robocurate.adapters.lerobot.LeRobotWriter`. Running a cheap curation
signal (e.g. ``jerk``) over the result visibly removes the bad episodes.

It is intentionally dependency-light (NumPy plus the core package only): no GPU, no
network, no optional extras, importable from a clean ``uv sync``. The output is a real
LeRobotDataset directory, so the rest of the tutorial — ``robocurate curate`` /
``robocurate report`` — runs against it unchanged.

Usage::

    from robocurate.examples import write_demo_dataset
    path = write_demo_dataset("./demo_dataset")
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

from robocurate import __version__
from robocurate.adapters.lerobot import LeRobotWriter
from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.manifest import MANIFEST_SCHEMA_VERSION, EpisodeDecision, Manifest
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

_NUM_STEPS = 24
_CONTROL_HZ = 10.0
_SOURCE_ID = "robocurate/demo"

# A minimal 2-DoF reaching embodiment: timestamps, a 2-d action, a 2-d proprio state.
# Kept deliberately tiny so the whole demo runs in well under a second with no extras.
_DEMO_EMBODIMENT = EmbodimentSpec(
    embodiment_id="demo2dof",
    control_hz=_CONTROL_HZ,
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
    ),
)


def _good_trajectory(episode_index: int, rng: np.random.Generator) -> Trajectory:
    """A smooth, direct reach toward a per-episode goal — low jerk, the kind we keep."""
    goal = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    # A smooth minimum-jerk-like easing from the origin to the goal, so the velocity
    # command (the action) is gentle and the path is direct.
    s = np.linspace(0.0, 1.0, _NUM_STEPS, dtype=np.float32)
    ease = (3.0 * s**2 - 2.0 * s**3).astype(np.float32)  # smoothstep
    state = (ease[:, None] * goal[None, :]).astype(np.float32)
    action = np.diff(state, axis=0, prepend=state[:1]).astype(np.float32)
    return _build(episode_index, action, state, success=True)


def _bad_trajectory(episode_index: int, rng: np.random.Generator) -> Trajectory:
    """A jittery, wandering reach — high jerk and an indirect path: the kind we drop."""
    goal = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    s = np.linspace(0.0, 1.0, _NUM_STEPS, dtype=np.float32)
    base = (s[:, None] * goal[None, :]).astype(np.float32)
    # Heavy step-to-step jitter (high-frequency noise) plus a low-frequency wander away
    # from the straight line, so both the jerk and path-efficiency signals see it.
    jitter = rng.normal(0.0, 0.45, size=base.shape).astype(np.float32)
    wander = (0.6 * np.sin(8.0 * np.pi * s)[:, None] * np.array([1.0, -1.0], np.float32)).astype(
        np.float32
    )
    state = (base + jitter + wander).astype(np.float32)
    action = np.diff(state, axis=0, prepend=state[:1]).astype(np.float32)
    return _build(episode_index, action, state, success=False)


def _build(episode_index: int, action: Array, state: Array, *, success: bool) -> Trajectory:
    """Assemble a :class:`Trajectory` from action/state arrays and a success label."""
    t = (np.arange(_NUM_STEPS, dtype=np.float32) / _CONTROL_HZ).astype(np.float32)
    columns: dict[str, Array] = {
        "timestamp": t,
        "action": action,
        "observation.state": state,
    }
    meta = TrajectoryMeta(
        source_dataset_id=_SOURCE_ID,
        episode_index=episode_index,
        embodiment=_DEMO_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=_NUM_STEPS,
        source_format="synthetic_v0",
        success=SuccessLabel(value=success, source="synthetic"),
        extra={"tasks": ["reach the goal"], "quality": "good" if success else "bad"},
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _identity_manifest(reader: InMemoryDatasetReader) -> Manifest:
    """A trivial "everything kept" manifest so the source can go through the public writer.

    The demo dataset is a *source*, not a curation output, so no episode is removed; the
    manifest just satisfies :meth:`LeRobotWriter.write` and records the provenance of the
    synthetic data. The real curation manifest is produced later by ``robocurate curate``.
    """
    fingerprint = reader.fingerprint()
    decisions = tuple(
        EpisodeDecision(
            episode_index=traj.meta.episode_index,
            fingerprint=traj.meta.fingerprint,
            kept=True,
            reason="kept (synthetic demo source; no curation applied)",
        )
        for traj in reader
    )
    return Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        source=fingerprint,
        output=fingerprint,
        config_dict={"generator": "robocurate.examples.write_demo_dataset"},
        seed=0,
        code_version=__version__,
        signals=(),
        decisions=decisions,
        baseline=None,
        created_utc=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    )


def write_demo_dataset(path: str | Path, *, num_episodes: int = 8, seed: int = 0) -> Path:
    """Write a tiny deterministic demo LeRobotDataset to ``path`` and return the path.

    The dataset is a mix of smooth/direct "good" episodes and jittery/wandering "bad"
    ones, so a cheap signal such as ``jerk`` visibly removes the bad ones during the
    onboarding tutorial. Good and bad episodes are interleaved by index.

    Args:
        path: Destination directory. It must not already exist (the writer refuses to
            overwrite — Invariant 1).
        num_episodes: How many episodes to generate (default 8). At least 2 so the mix
            contains both a good and a bad episode.
        seed: Master seed; given the same seed the output is byte-identical.

    Returns:
        The destination :class:`~pathlib.Path` that was written.
    """
    if num_episodes < 2:
        raise ValueError(f"num_episodes must be >= 2 to include both kinds, got {num_episodes}")

    rng = np.random.default_rng(seed)
    trajectories: list[Trajectory] = []
    for i in range(num_episodes):
        # Interleave good/bad by index so the dataset is an obvious, balanced mix.
        if i % 2 == 0:
            trajectories.append(_good_trajectory(i, rng))
        else:
            trajectories.append(_bad_trajectory(i, rng))

    reader = InMemoryDatasetReader(trajectories, dataset_id=_SOURCE_ID)
    writer = LeRobotWriter(path)
    receipt = writer.write(reader, _identity_manifest(reader))
    return receipt.path


__all__ = ["write_demo_dataset"]
