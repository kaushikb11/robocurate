"""Honest detection-AUC blind-spot matrix over the cheap low-dim signals x all corruptions.

This is the standalone, dependency-free sibling of ``corruption_recovery.py``: instead of needing
the downloaded robomimic HDF5s, it builds a small *synthetic* dataset in memory, injects every
corruption kind (the originals — jitter / detour / truncate / stall — plus the expanded ones —
frame_skip / action_quantize / wrong_target_offset / dropped_dof) into a copy of it, and prints
the orientation-aware detection-AUC for each (corruption x signal) cell.

The point is HONESTY, not a hero number: the table shows which signals are BLIND to which
corruption (AUC ~0.5) and which INVERT on it (AUC ~0 — they rank the corrupted demo as *higher*
quality, e.g. path_efficiency on truncate). No single cheap signal catches everything; the
takeaway is that they must be combined into a complementary SUITE (and ultimately backed by
outcome-aware signals).

Usage::

    uv run python experiments/blindspot_matrix.py
    uv run python experiments/blindspot_matrix.py --n 24 --severity 1.0
"""

from __future__ import annotations

import argparse

import numpy as np

from robocurate.corruptions import CORRUPTIONS, detection_matrix
from robocurate.signals.action_noise import ActionNoise
from robocurate.signals.jerk import Jerk
from robocurate.signals.path_efficiency import PathEfficiency
from robocurate.signals.redundancy import Redundancy
from robocurate.signals.spectral_smoothness import SpectralSmoothness
from robocurate.signals.structural_validity import StructuralValidity
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

_FS = 20.0
STATE = "observation.eef"  # the 3-D end-effector path the geometric signals read
ACTION = "action"


def _embodiment(columns: dict[str, Array]) -> EmbodimentSpec:
    roles = {
        "timestamp": FeatureRole.TIME,
        STATE: FeatureRole.STATE,
        ACTION: FeatureRole.ACTION,
    }
    features = tuple(
        FeatureSpec(
            key=key,
            role=roles.get(key, FeatureRole.EXTRA),
            shape=tuple(np.asarray(arr).shape[1:]),
            dtype=str(np.asarray(arr).dtype),
        )
        for key, arr in columns.items()
    )
    return EmbodimentSpec(embodiment_id="toy_eef", features=features, control_hz=_FS)


def _episode(index: int, num_steps: int, rng: np.random.Generator) -> Trajectory:
    """A smooth, direct 3-D reach with a small per-episode goal — clean reference data."""
    t = np.linspace(0.0, 1.0, num_steps)
    goal = 0.5 + 0.5 * rng.random(3)  # distinct per-episode target so episodes are not identical
    path = (np.outer(t, goal)).astype(np.float32)
    action = np.diff(path, axis=0, prepend=path[:1]).astype(np.float32)
    columns: dict[str, Array] = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) / _FS),
        STATE: path,
        ACTION: action,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/blindspot",
        episode_index=index,
        embodiment=_embodiment(columns),
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
        success=SuccessLabel(value=True, source="synthetic"),
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def build_dataset(n: int, *, num_steps: int = 64, seed: int = 0) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    return [_episode(i, num_steps, rng) for i in range(n)]


def build_signals() -> list[object]:
    """The cheap, CPU low-dim signals, reading the end-effector path where geometric.

    Scoped to the cheap heuristics: jerk, action_noise, path_efficiency, spectral_smoothness,
    redundancy, structural_validity. The image signals (blur / visual_stall / visual_diversity)
    are excluded — they need video decode and have their own tests; the learned ones (demo_score
    / cupid) are excluded here as they need labels / extra compute.
    """
    return [
        Jerk(source=STATE),
        ActionNoise(source=STATE),
        PathEfficiency(source=STATE, dims=None, motion="positions"),
        SpectralSmoothness(source=STATE, motion="positions"),
        Redundancy(),
        StructuralValidity(),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=24, help="number of clean episodes")
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    clean = build_dataset(args.n, seed=args.seed)
    signals = build_signals()
    matrix = detection_matrix(
        clean,
        signals,  # type: ignore[arg-type]
        CORRUPTIONS,
        feature=STATE,
        severity=args.severity,
        seed=args.seed,
    )

    print(
        f"\nSynthetic blind-spot matrix — corruption detection AUC "
        f"({args.n} clean episodes, severity {args.severity})\n"
    )
    print(matrix.to_markdown())
    print(
        "\nHonest read: no SINGLE cheap signal catches everything. The geometric signals "
        "(jerk / spectral_smoothness / path_efficiency) catch kinematic defects but are blind to "
        "(and on `truncate` INVERT on) structural ones; `structural_validity` closes truncate / "
        "stall / frame_skip; action_noise's dataset-relative outlier z catches the shifted "
        "distribution of `wrong_target_offset`. And it stays honest about what NOTHING here "
        "catches: on a goal-diverse reach dataset `dropped_dof` is a SUITE blind spot (it looks "
        "like a different valid target) — the cheap low-dim signals near chance is exactly the "
        "gap a learned / outcome-aware signal must close. The fix is a complementary suite, never "
        "a standalone geometric keep/drop filter."
    )


if __name__ == "__main__":
    main()
