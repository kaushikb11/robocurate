"""Invariant 3: same input + config + seed -> byte-identical curation decisions.

Determinism is a load-bearing invariant (CLAUDE.md #3): a curated dataset must be reproducible
to the bit, so a reviewer can re-run a selection and get the *exact* same kept/removed episodes,
per-episode reasons, signal values, and manifest. This test runs a full curation twice on the
same in-memory dataset with the same seed and asserts the two runs are indistinguishable, down
to the serialized manifest dict (ignoring only the wall-clock ``created_utc`` timestamp, which is
stamped by the caller and is intentionally not part of the selection).

CPU-only and torch-free (no ``ml`` marker): it exercises only the cheap Tier-0 signals and the
selection engine, so it runs in the core-only CI lane that guards the no-GPU laptop install.
"""

from __future__ import annotations

import copy
import json

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.corruptions import corrupt
from robocurate.curator import Budget, Curator, WeightedSum
from robocurate.experiment.synthetic import make_identity_experiment_dataset
from robocurate.signals.action_noise import ActionNoise
from robocurate.signals.jerk import Jerk
from robocurate.signals.structural_validity import StructuralValidity

_SEED = 42
_CREATED_UTC = "created_utc"


def _build_reader(seed: int = 0) -> InMemoryDatasetReader:
    """Clean identity demos with a corrupted subset, so the signals have real spread to rank."""
    base = make_identity_experiment_dataset(num_helpful=16, num_harmful=0, seed=seed)
    trajectories = []
    for i, traj in enumerate(base):
        if i >= 10:
            kind = "jitter" if i % 2 == 0 else "truncate"
            traj = corrupt(traj, kind, feature="action", severity=2.0, seed=seed * 100 + i)
        trajectories.append(traj)
    return InMemoryDatasetReader(trajectories, dataset_id="synthetic/determinism")


def _run() -> object:
    # Multiple signals + an explicit combiner exercise the full normalize/combine/select path.
    reader = _build_reader()
    curator = Curator(
        [ActionNoise(), Jerk(), StructuralValidity()],
        combiner=WeightedSum(),
        budget=Budget.fraction(0.6),
        seed=_SEED,
        emit_baseline=True,
    )
    return curator.run(reader)


def test_selection_is_byte_identical_across_runs() -> None:
    a = _run()
    b = _run()

    assert a.kept_episode_indices == b.kept_episode_indices  # type: ignore[attr-defined]
    assert a.removed_episode_indices == b.removed_episode_indices  # type: ignore[attr-defined]


def test_per_episode_reasons_and_signal_values_are_identical() -> None:
    a = _run()
    b = _run()

    decisions_a = a.decisions  # type: ignore[attr-defined]
    decisions_b = b.decisions  # type: ignore[attr-defined]
    assert len(decisions_a) == len(decisions_b)
    for da, db in zip(decisions_a, decisions_b, strict=True):
        assert da.episode_index == db.episode_index
        assert da.kept == db.kept
        assert da.reason == db.reason
        # signal_values are exact floats (NaN where skipped) — assert bit-identical via repr,
        # which compares NaN==NaN as equal (unlike ==) and catches any float drift.
        assert {k: repr(v) for k, v in da.signal_values.items()} == {
            k: repr(v) for k, v in db.signal_values.items()
        }


def test_equal_n_baseline_draw_is_identical() -> None:
    a = _run()
    b = _run()

    base_a = a.baseline  # type: ignore[attr-defined]
    base_b = b.baseline  # type: ignore[attr-defined]
    assert base_a is not None and base_b is not None
    assert base_a.seed == base_b.seed
    assert base_a.kept_episode_indices == base_b.kept_episode_indices


def test_built_manifest_is_identical_ignoring_timestamp() -> None:
    a = _run()
    b = _run()

    # Two different wall-clock timestamps prove the manifests match *despite* differing stamps.
    manifest_a = a.build_manifest(created_utc="2026-01-01T00:00:00Z").to_dict()  # type: ignore[attr-defined]
    manifest_b = b.build_manifest(created_utc="2099-12-31T23:59:59Z").to_dict()  # type: ignore[attr-defined]

    # Sanity: the timestamps really do differ before we neutralize them.
    assert manifest_a[_CREATED_UTC] != manifest_b[_CREATED_UTC]

    neutral_a = copy.deepcopy(manifest_a)
    neutral_b = copy.deepcopy(manifest_b)
    neutral_a[_CREATED_UTC] = None
    neutral_b[_CREATED_UTC] = None
    # Compare the serialized JSON, not the raw dicts: skipped signals carry NaN values, and
    # ``NaN != NaN`` would make two structurally-identical dicts compare unequal. JSON
    # serialization emits a stable ``NaN`` token, so equal JSON == byte-identical manifest —
    # which is exactly the determinism claim (invariant 3) we want to assert.
    assert json.dumps(neutral_a, sort_keys=True) == json.dumps(neutral_b, sort_keys=True)
