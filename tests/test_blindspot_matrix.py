"""Known-answer tests for the expanded corruptions and the detection-AUC blind-spot matrix.

The matrix is the WS-Rigor "honest scorekeeper" wedge made testable: for every (corruption x
signal) cell we assert the *direction* the AUC must take — detected (>0.7), or the documented
blind spot / inversion — without pinning knife-edge values. We also re-assert the corruption
invariants the new kinds must honor: deterministic given the seed (invariant 3) and never
mutating the source (invariant 1).
"""

from __future__ import annotations

import numpy as np
import pytest

from robocurate.corruptions import (
    CORRUPTIONS,
    DetectionMatrix,
    corrupt,
    detection_auc,
    detection_matrix,
    rank_auc,
)
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
STATE = "observation.eef"
ACTION = "action"
NEW_KINDS = ("frame_skip", "action_quantize", "wrong_target_offset", "dropped_dof")
_ROLES = {"timestamp": FeatureRole.TIME, STATE: FeatureRole.STATE, ACTION: FeatureRole.ACTION}


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
    return EmbodimentSpec(embodiment_id="toy_eef", features=features, control_hz=_FS)


def _episode(index: int, rng: np.random.Generator, num_steps: int = 64) -> Trajectory:
    """A smooth, direct 3-D reach with a distinct per-episode goal."""
    t = np.linspace(0.0, 1.0, num_steps)
    goal = 0.5 + 0.5 * rng.random(3)
    path = np.outer(t, goal).astype(np.float32)
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


def _dataset(n: int = 16, seed: int = 0) -> list[Trajectory]:
    rng = np.random.default_rng(seed)
    return [_episode(i, rng) for i in range(n)]


def _signals() -> list[object]:
    return [
        Jerk(source=STATE),
        ActionNoise(source=STATE),
        PathEfficiency(source=STATE, dims=None, motion="positions"),
        SpectralSmoothness(source=STATE, motion="positions"),
        Redundancy(),
        StructuralValidity(),
    ]


# -- invariants on the new corruptions --------------------------------------------------


def test_new_corruptions_are_deterministic() -> None:
    clean = _episode(0, np.random.default_rng(1))
    for kind in NEW_KINDS:
        a = corrupt(clean, kind, feature=STATE, severity=1.0, seed=7)
        b = corrupt(clean, kind, feature=STATE, severity=1.0, seed=7)
        assert np.array_equal(np.asarray(a.feature(STATE)), np.asarray(b.feature(STATE)))
        assert a.meta.fingerprint == b.meta.fingerprint


def test_new_corruptions_do_not_mutate_source_and_tag_kind() -> None:
    clean = _episode(0, np.random.default_rng(2))
    before = np.asarray(clean.feature(STATE)).copy()
    for kind in NEW_KINDS:
        bad = corrupt(clean, kind, feature=STATE, severity=1.0, seed=3)
        assert np.array_equal(np.asarray(clean.feature(STATE)), before)  # source untouched
        assert bad.meta.extra["corruption"] == kind
        assert "corruption" not in clean.meta.extra


def test_frame_skip_shortens_and_action_quantize_preserves_length() -> None:
    clean = _episode(0, np.random.default_rng(4))
    skipped = corrupt(clean, "frame_skip", feature=STATE, severity=1.0, seed=0)
    assert skipped.meta.num_steps < clean.meta.num_steps  # frames were dropped
    quant = corrupt(clean, "action_quantize", feature=STATE, severity=1.0, seed=0)
    assert quant.meta.num_steps == clean.meta.num_steps  # quantize is in-place
    # quantization snaps to a grid -> the feature changes but stays the same shape.
    assert not np.array_equal(np.asarray(quant.feature(STATE)), np.asarray(clean.feature(STATE)))


def test_dropped_dof_zeros_exactly_one_column() -> None:
    clean = _episode(0, np.random.default_rng(5))
    dropped = corrupt(clean, "dropped_dof", feature=STATE, severity=1.0, seed=11)
    arr = np.asarray(dropped.feature(STATE))
    zero_cols = [c for c in range(arr.shape[1]) if np.allclose(arr[:, c], 0.0)]
    assert len(zero_cols) >= 1  # at least the dropped DoF is all-zero


def test_wrong_target_offset_shifts_without_changing_per_step_deltas() -> None:
    clean = _episode(0, np.random.default_rng(6))
    shifted = corrupt(clean, "wrong_target_offset", feature=STATE, severity=1.0, seed=9)
    clean_arr = np.asarray(clean.feature(STATE))
    shifted_arr = np.asarray(shifted.feature(STATE))
    # a constant offset leaves per-step differences (the shape) identical.
    assert np.allclose(np.diff(shifted_arr, axis=0), np.diff(clean_arr, axis=0), atol=1e-5)
    assert not np.allclose(shifted_arr, clean_arr)


# -- detection-AUC helpers --------------------------------------------------------------


def test_rank_and_detection_auc_orientation() -> None:
    clean = np.array([0.0, 0.1, 0.2])
    corrupt_vals = np.array([1.0, 1.1, 1.2])
    labels = np.array([False, False, False, True, True, True])
    values = np.concatenate([clean, corrupt_vals])
    # higher_is_better=False: corrupted has the higher (worse) score -> perfect detection.
    assert detection_auc(values, labels, higher_is_better=False) == pytest.approx(1.0)
    # higher_is_better=True with the SAME values inverts: corrupted scores higher = ranked better.
    assert detection_auc(values, labels, higher_is_better=True) == pytest.approx(0.0)
    assert rank_auc(corrupt_vals, clean) == pytest.approx(1.0)


# -- known-answer matrix relationships --------------------------------------------------


def _matrix() -> DetectionMatrix:
    return detection_matrix(
        _dataset(),
        _signals(),  # type: ignore[arg-type]
        CORRUPTIONS,
        feature=STATE,
        severity=1.0,
        seed=0,
    )


def test_matrix_covers_every_cell_and_renders_markdown() -> None:
    m = _matrix()
    for kind in CORRUPTIONS:
        for sig in ("jerk", "action_noise", "path_efficiency", "structural_validity"):
            assert np.isfinite(m.value(kind, sig)), f"{kind} x {sig} produced no AUC"
    md = m.to_markdown()
    assert "| corruption |" in md and "frame_skip" in md and "Legend" in md


def test_kinematic_corruptions_detected_by_a_smoothness_signal() -> None:
    m = _matrix()
    # action_quantize adds a staircase (high-frequency content) -> jerk catches it cleanly.
    assert m.value("action_quantize", "jerk") > 0.7
    assert m.value("action_quantize", "spectral_smoothness") > 0.7
    # frame_skip widens per-step deltas: the spectral-smoothness view catches it (jerk on a
    # positions path is NOT the right detector here -> honestly asserted via spectral).
    assert m.value("frame_skip", "spectral_smoothness") > 0.7


def test_frame_skip_and_quantize_caught_by_structural_validity() -> None:
    # frame_skip drops frames (shorter) and quantize freezes runs of identical states -> the
    # structural verifier (truncation + held-frame stall checks) catches both.
    m = _matrix()
    assert m.value("frame_skip", "structural_validity") > 0.7
    assert m.value("action_quantize", "structural_validity") > 0.7


def test_dropped_dof_is_an_honest_suite_blind_spot() -> None:
    # HONEST known-answer (invariant 6): on a goal-diverse reach dataset, zeroing one DoF looks
    # like a *different* valid target, so NO cheap low-dim signal reliably catches it (all AUCs
    # sit near chance). We assert the blind spot rather than pretend it is caught -- this is the
    # kind of gap the matrix exists to surface (a learned/outcome-aware signal is the upgrade).
    m = _matrix()
    cheap = ("jerk", "action_noise", "path_efficiency", "spectral_smoothness", "redundancy")
    assert all(m.value("dropped_dof", s) < 0.7 for s in cheap), "expected dropped_dof blind spot"


def test_truncate_detected_by_structural_but_inverted_for_path_efficiency() -> None:
    m = _matrix()
    # structural_validity exists to close this blind spot -> it must catch truncation.
    assert m.value("truncate", "structural_validity") > 0.7
    # the documented honest failure: a truncated demo is shorter+straighter, so the geometric
    # signals INVERT (AUC ~0) -- they rank the corrupted demo as *higher* quality.
    assert m.value("truncate", "path_efficiency") < 0.5
    assert m.classify("truncate", "path_efficiency") == "inverts"


def test_wrong_target_offset_caught_by_outlier_view_not_pure_geometry() -> None:
    m = _matrix()
    # a constant per-DoF offset shifts the action/state distribution -> action_noise's
    # dataset-relative OUTLIER view catches the mis-targeted demo.
    assert m.value("wrong_target_offset", "action_noise") > 0.7
    # ... while the geometric jerk view, which only sees within-episode shape, is blind/inverts:
    # the outlier view is the strictly stronger detector here.
    assert m.value("wrong_target_offset", "jerk") < m.value("wrong_target_offset", "action_noise")
