"""Tests for StructuralValidity: catches the truncation/stall/non-finite defects geometry misses.

Known answer: inject the exact corruptions a directness/smoothness signal is blind to (or inverts
on) — a truncated episode, a stalled episode, and a non-finite episode — and assert the
structural verifier flags each as worse than the clean reference. Plus determinism and the
truncation-needs-fit behaviour.
"""

from __future__ import annotations

import numpy as np

from robocurate.corruptions import corrupt
from robocurate.signals.structural_validity import NONFINITE_PENALTY, StructuralValidity
from robocurate.trajectory import (
    Array,
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    InMemoryFeatureStore,
    Trajectory,
    TrajectoryMeta,
    fingerprint_arrays,
)
from tests.synthetic import make_signal_context

_FS = 20.0
_ROLES = {
    "timestamp": FeatureRole.TIME,
    "observation.eef": FeatureRole.STATE,
    "action": FeatureRole.ACTION,
}


def _embodiment(columns: dict[str, Array]) -> EmbodimentSpec:
    features = tuple(
        FeatureSpec(
            key=key,
            role=_ROLES.get(key, FeatureRole.EXTRA),
            shape=tuple(np.asarray(arr).shape[1:]),
            dtype=str(np.asarray(arr).dtype),
        )
        for key, arr in columns.items()
    )
    return EmbodimentSpec(embodiment_id="toy", features=features, control_hz=_FS)


def _clean(index: int = 0, num_steps: int = 64) -> Trajectory:
    t = np.linspace(0.0, 1.0, num_steps)
    path = np.stack([t, 0.3 * t, np.zeros(num_steps)], axis=-1).astype(np.float32)
    action = np.diff(path, axis=0, prepend=path[:1]).astype(np.float32)
    columns: dict[str, Array] = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) / _FS),
        "observation.eef": path,
        "action": action,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/structural",
        episode_index=index,
        embodiment=_embodiment(columns),
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _value_of(sig: StructuralValidity, traj: Trajectory, fit_on: list[Trajectory]) -> float:
    ctx = make_signal_context()
    sig.fit(fit_on, ctx)
    [s] = sig.score([traj], ctx)
    return float(s.value)


def test_truncation_is_flagged_with_dataset_context() -> None:
    sig = StructuralValidity()
    clean = [_clean(i) for i in range(6)]  # all length 64
    truncated = corrupt(clean[0], "truncate", feature="observation.eef", severity=0.6, seed=1)
    assert truncated.meta.num_steps < clean[0].meta.num_steps
    # fit on the clean-length population so the median (64) is known
    fit_on = [*clean, truncated]
    assert _value_of(sig, truncated, fit_on) > _value_of(sig, clean[0], fit_on)


def test_stall_is_flagged() -> None:
    sig = StructuralValidity()
    clean = [_clean(i) for i in range(4)]
    stalled = corrupt(clean[0], "stall", feature="observation.eef", severity=0.4, seed=2)
    fit_on = [*clean, stalled]
    assert _value_of(sig, stalled, fit_on) > _value_of(sig, clean[0], fit_on)


def test_nonfinite_is_catastrophic() -> None:
    sig = StructuralValidity()
    clean = _clean(0)
    cols = {
        k: np.asarray(clean.feature(k)).copy() for k in ("timestamp", "observation.eef", "action")
    }
    cols["action"][3, 0] = np.inf
    bad = Trajectory(
        TrajectoryMeta(
            source_dataset_id="synthetic/structural",
            episode_index=1,
            embodiment=_embodiment(cols),
            fingerprint=fingerprint_arrays(cols),
            num_steps=cols["action"].shape[0],
            source_format="synthetic_v0",
        ),
        InMemoryFeatureStore(cols),
    )
    assert _value_of(sig, bad, [clean, bad]) >= NONFINITE_PENALTY


def test_clean_trajectory_scores_zero() -> None:
    sig = StructuralValidity()
    clean = [_clean(i) for i in range(5)]
    assert _value_of(sig, clean[0], clean) == 0.0


def test_truncation_check_skipped_without_fit() -> None:
    # Without fit (no median), a short trajectory is NOT flagged as truncated (no guessing).
    sig = StructuralValidity()
    ctx = make_signal_context()  # no fit() called -> cache miss -> median unknown
    short = _clean(0, num_steps=8)
    [s] = sig.score([short], ctx)
    assert s.diagnostics["truncation_severity"] == 0.0


def test_score_is_deterministic() -> None:
    sig = StructuralValidity()
    clean = [_clean(i) for i in range(4)]
    stalled = corrupt(clean[0], "stall", feature="observation.eef", severity=0.3, seed=5)
    fit_on = [*clean, stalled]
    assert _value_of(sig, stalled, fit_on) == _value_of(sig, stalled, fit_on)
