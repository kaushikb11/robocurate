"""Tests for the RLDS read adapter.

The conversion logic is TF-free (it accepts any iterable of episodes), so the core tests use
plain Python/NumPy synthetic RLDS episodes and run in the core-only suite. One ``rlds``-marked
test verifies that real TensorFlow eager tensors (a ``tf.data.Dataset``) convert identically.
"""

from __future__ import annotations

import numpy as np
import pytest

from robocurate.adapters import RLDSReader
from robocurate.curator import Budget, Curator
from robocurate.signals.jerk import Jerk
from robocurate.trajectory import FeatureRole


def _rlds_episode(
    num_steps: int, *, seed: int, state_dim: int = 4, action_dim: int = 2
) -> dict[str, object]:
    """A synthetic RLDS episode: a dict with a ``steps`` iterable of per-step records."""
    rng = np.random.default_rng(seed)
    steps = []
    for t in range(num_steps):
        steps.append(
            {
                "observation": {
                    "state": rng.normal(size=state_dim).astype(np.float32),
                    "wrist_image": np.zeros((4, 4, 3), dtype=np.uint8),
                },
                "action": rng.normal(size=action_dim).astype(np.float32),
                "reward": np.float32(t),
                "is_first": t == 0,
                "is_last": t == num_steps - 1,
                "is_terminal": t == num_steps - 1,
            }
        )
    return {"steps": steps}


def _synthetic_rlds(num_episodes: int = 3, num_steps: int = 6) -> list[dict[str, object]]:
    return [_rlds_episode(num_steps, seed=i) for i in range(num_episodes)]


def test_rlds_converts_to_canonical_trajectories() -> None:
    reader = RLDSReader(_synthetic_rlds(3, 6), dataset_id="oxe/toy", fps=5.0)
    assert len(reader) == 3

    traj = reader.read_episode(0)
    assert traj.num_steps == 6
    assert traj.feature("action").shape == (6, 2)
    assert traj.feature("observation.state").shape == (6, 4)
    assert traj.feature("observation.wrist_image").shape == (6, 4, 4, 3)
    # Timestamps are synthesized from fps.
    ts = traj.timestamps()
    assert ts is not None
    np.testing.assert_allclose(np.diff(ts), 0.2, atol=1e-6)


def test_rlds_role_inference() -> None:
    emb = RLDSReader(_synthetic_rlds(1, 4)).read_episode(0).embodiment

    def role(key: str) -> FeatureRole:
        spec = emb.feature(key)
        assert spec is not None
        return spec.role

    assert role("action") is FeatureRole.ACTION
    assert role("observation.state") is FeatureRole.PROPRIO
    assert role("observation.wrist_image") is FeatureRole.IMAGE
    assert role("reward") is FeatureRole.REWARD
    assert role("timestamp") is FeatureRole.TIME


def test_rlds_reader_has_no_write_method() -> None:
    # Read-only by construction (invariant 1).
    assert not hasattr(RLDSReader, "write")
    assert not hasattr(RLDSReader, "save")


def test_rlds_fingerprint_is_content_addressed() -> None:
    a = RLDSReader(_synthetic_rlds(2, 5)).fingerprint()
    b = RLDSReader(_synthetic_rlds(2, 5)).fingerprint()
    c = RLDSReader(_synthetic_rlds(3, 5)).fingerprint()
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash
    assert a.source_format == "rlds"


def test_rlds_flows_through_the_curator() -> None:
    # An RLDS dataset is a normal DatasetReader: signals + curation work unchanged.
    reader = RLDSReader(_synthetic_rlds(6, 8), dataset_id="oxe/toy")
    result = Curator([Jerk()], budget=Budget.fraction(0.5), seed=0).run(reader)
    assert result.num_kept == 3
    assert result.baseline is not None


def test_index_out_of_range() -> None:
    reader = RLDSReader(_synthetic_rlds(2, 4))
    with pytest.raises(IndexError):
        reader.read_episode(5)


@pytest.mark.rlds
def test_rlds_converts_real_tf_eager_tensors() -> None:
    tf = pytest.importorskip("tensorflow")

    # A real RLDS-shaped episode where `steps` is a tf.data.Dataset of eager tensors.
    num_steps = 5
    steps = tf.data.Dataset.from_tensor_slices(
        {
            "observation": {"state": np.zeros((num_steps, 3), dtype=np.float32)},
            "action": np.ones((num_steps, 2), dtype=np.float32),
            "reward": np.arange(num_steps, dtype=np.float32),
        }
    )
    reader = RLDSReader([{"steps": steps}], dataset_id="oxe/tf")
    traj = reader.read_episode(0)
    assert traj.num_steps == num_steps
    assert traj.feature("action").shape == (num_steps, 2)
    np.testing.assert_array_equal(traj.feature("action"), np.ones((num_steps, 2)))
