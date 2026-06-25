"""Known-answer tests: injected corruptions are hard ground truth for the signals.

Unlike the soft operator-proficiency labels, here we KNOW exactly which trajectory is bad and
how. We assert the *expected detections* (a smoothness signal flags jitter; a directness signal
flags a detour) AND the *expected blind spots* (geometry signals do NOT flag a truncation —
honest, because a truncated demo is structurally bad but geometrically clean). The corruption is
deterministic and never mutates the input.
"""

from __future__ import annotations

import numpy as np

from robocurate.corruptions import CORRUPTIONS, corrupt
from robocurate.signals.jerk import Jerk
from robocurate.signals.path_efficiency import PathEfficiency
from robocurate.signals.spectral_smoothness import SpectralSmoothness
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
    """Build an embodiment whose features match the columns (as the real readers do)."""
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


def _clean(num_steps: int = 64) -> Trajectory:
    """A smooth, direct 3-D path (and matching action) — the clean reference."""
    t = np.linspace(0.0, 1.0, num_steps)
    path = np.stack([t, 0.3 * t, np.zeros(num_steps)], axis=-1).astype(np.float32)
    action = np.diff(path, axis=0, prepend=path[:1]).astype(np.float32)
    columns: dict[str, Array] = {
        "timestamp": (np.arange(num_steps, dtype=np.float32) / _FS),
        "observation.eef": path,
        "action": action,
    }
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/corrupt",
        episode_index=0,
        embodiment=_embodiment(columns),
        fingerprint=fingerprint_arrays(columns),
        num_steps=num_steps,
        source_format="synthetic_v0",
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _score(sig: object, traj: Trajectory) -> float:
    [s] = sig.score([traj], make_signal_context())  # type: ignore[attr-defined]
    return float(s.value) if not s.skipped else float("nan")


def test_corrupt_does_not_mutate_input_and_tags_kind() -> None:
    clean = _clean()
    before = np.asarray(clean.feature("observation.eef")).copy()
    bad = corrupt(clean, "jitter", feature="observation.eef", seed=0)
    assert np.array_equal(np.asarray(clean.feature("observation.eef")), before)  # input untouched
    assert bad.meta.extra["corruption"] == "jitter"
    assert "corruption" not in clean.meta.extra


def test_corruptions_are_deterministic() -> None:
    clean = _clean()
    for kind in CORRUPTIONS:
        a = corrupt(clean, kind, feature="observation.eef", seed=7)
        b = corrupt(clean, kind, feature="observation.eef", seed=7)
        assert np.array_equal(
            np.asarray(a.feature("observation.eef")), np.asarray(b.feature("observation.eef"))
        )


def test_jitter_is_caught_by_smoothness_signals() -> None:
    clean = _clean()
    jittered = corrupt(clean, "jitter", feature="observation.eef", severity=1.0, seed=1)
    sparc = SpectralSmoothness(source="observation.eef", motion="positions")
    jerk = Jerk(source="observation.eef")
    # jitter adds high-frequency content -> less smooth (more negative SPARC), higher jerk.
    assert _score(sparc, jittered) < _score(sparc, clean)
    assert _score(jerk, jittered) > _score(jerk, clean)


def test_detour_is_caught_by_directness() -> None:
    clean = _clean()
    detoured = corrupt(clean, "detour", feature="observation.eef", severity=1.0, seed=2)
    direct = PathEfficiency(source="observation.eef", dims=None, motion="positions")
    # a smooth out-and-back covers extra path for the same net displacement -> less direct.
    assert _score(direct, detoured) < _score(direct, clean)


def test_truncation_is_a_blind_spot_for_geometry_signals() -> None:
    # Honest known-answer: a truncated (structurally incomplete) demo is still smooth and direct,
    # so directness barely moves. This is the documented blind spot, asserted explicitly.
    clean = _clean()
    truncated = corrupt(clean, "truncate", feature="observation.eef", severity=0.5, seed=3)
    assert truncated.meta.num_steps < clean.meta.num_steps  # it really is shorter
    direct = PathEfficiency(source="observation.eef", dims=None, motion="positions")
    assert abs(_score(direct, truncated) - _score(direct, clean)) < 0.05  # ~undetected


def test_stall_inserts_held_frames() -> None:
    clean = _clean()
    stalled = corrupt(clean, "stall", feature="observation.eef", severity=0.3, seed=4)
    assert stalled.meta.num_steps > clean.meta.num_steps  # a hold was inserted
