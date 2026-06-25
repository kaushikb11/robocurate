"""Contract tests for the canonical trajectory representation."""

from __future__ import annotations

import numpy as np
import pytest

from robocurate.trajectory import FeatureRole, InMemoryFeatureStore, Trajectory
from tests.synthetic import TOY_EMBODIMENT, make_trajectory


def test_feature_access_is_lazy_and_uniform() -> None:
    traj = make_trajectory(episode_index=0, num_steps=8)

    action = traj.feature("action")
    assert action.shape == (8, 2)
    # The same uniform accessor reaches state and reward.
    assert traj.feature("observation.state").shape == (8, 2)
    assert traj.feature("reward").shape == (8,)


def test_typed_views_return_arrays_when_present() -> None:
    traj = make_trajectory(episode_index=1, num_steps=5)

    np.testing.assert_array_equal(traj.actions(), traj.feature("action"))
    np.testing.assert_array_equal(traj.rewards(), traj.feature("reward"))
    assert traj.timestamps() is not None
    assert traj.timestamps().shape == (5,)  # type: ignore[union-attr]


def test_typed_views_return_none_when_absent_never_fabricate() -> None:
    # A trajectory that carries only an action — no reward, no timestamps.
    store = InMemoryFeatureStore({"action": np.zeros((3, 2), dtype=np.float32)})
    meta = make_trajectory(0).meta
    traj = Trajectory(meta, store)

    assert traj.rewards() is None
    assert traj.timestamps() is None
    assert traj.has("action") is True
    assert traj.has("reward") is False


def test_select_roles_skips_absent_features() -> None:
    traj = make_trajectory(episode_index=2)
    images = traj.select_roles(FeatureRole.IMAGE)
    assert images == {}  # toy embodiment has no images; skipped, not errored

    motion = traj.select_roles(FeatureRole.ACTION, FeatureRole.PROPRIO)
    assert set(motion) == {"action", "observation.state"}


def test_missing_feature_raises_keyerror() -> None:
    traj = make_trajectory(episode_index=0)
    with pytest.raises(KeyError):
        traj.feature("observation.images.nonexistent")


def test_embodiment_spec_lookup_helpers() -> None:
    assert TOY_EMBODIMENT.feature("action") is not None
    assert TOY_EMBODIMENT.feature("missing") is None
    assert TOY_EMBODIMENT.keys_with_role(FeatureRole.TIME) == ("timestamp",)


def test_timestamps_drive_variable_rate() -> None:
    # Control rate is read from timestamps, not assumed uniform.
    traj = make_trajectory(episode_index=0, num_steps=4)
    ts = traj.timestamps()
    assert ts is not None
    dt = np.diff(ts)
    # Toy data is uniform at 10 Hz, but the point is dt comes from the array.
    np.testing.assert_allclose(dt, 0.1, atol=1e-6)


def test_success_label_is_tristate() -> None:
    known = make_trajectory(0, success=True).success()
    assert known is not None and known.value is True

    unknown = make_trajectory(0, success=None).success()
    assert unknown is not None and unknown.value is None  # unknown != False
    assert unknown.source == "synthetic"


def test_fingerprint_is_content_addressed() -> None:
    a = make_trajectory(episode_index=0, scale=1.0)
    b = make_trajectory(episode_index=0, scale=1.0)
    c = make_trajectory(episode_index=0, scale=2.0)
    assert a.meta.fingerprint == b.meta.fingerprint  # identical content
    assert a.meta.fingerprint != c.meta.fingerprint  # different content
