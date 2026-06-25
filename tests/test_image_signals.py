"""Known-answer tests for the CPU image-quality signals (require the ``video`` extra).

Builds a tiny dataset of trajectories whose camera frames are *real* h264 mp4s (encoded with
the same helper :mod:`tests.test_video_decode` uses), where the defective episodes are known by
construction:

* ``BLURRY`` — uniform / low-frequency frames (no edges): the variance-of-Laplacian sharpness
  is ~0, so :class:`ImageBlur` must rank it the blurriest.
* ``FROZEN`` — the same frame duplicated for every step: every adjacent pair is held, so
  :class:`VisualStall` must flag it (held fraction ~1).
* ``DUP_A`` / ``DUP_B`` — two near-identical textured episodes (a copy with a tiny pixel tweak):
  each is the other's visual near-duplicate, so :class:`VisualDiversity` must rank them the
  *least* unique.
* ``SHARP`` / ``MOVING`` / ``UNIQUE`` — well-behaved contrast episodes that should not be the
  worst on any of the three checks.

These are known-answer assertions (the bad episodes are known up front), plus a determinism
check (two runs -> byte-identical scores). h264 is lossy, so we assert *rankings/flags* and
generous margins, never exact pixel values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("av")  # skip the whole module if the video extra is absent

from robocurate.signals.image_blur import ImageBlur
from robocurate.signals.visual_diversity import VisualDiversity
from robocurate.signals.visual_stall import VisualStall
from robocurate.trajectory import (
    EmbodimentSpec,
    FeatureRole,
    FeatureSpec,
    InMemoryFeatureStore,
    Trajectory,
    TrajectoryMeta,
    VideoReference,
    fingerprint_arrays,
)
from robocurate.video import clear_decode_cache
from tests.synthetic import make_signal_context
from tests.test_video_decode import _FPS, _HEIGHT, _WIDTH, _write_tiny_mp4

pytestmark = pytest.mark.video

_U8 = npt.NDArray[np.uint8]
_CAM = "observation.images.cam"

# Minimal image-bearing embodiment: a 2-d action plus one camera (IMAGE role).
_EMBODIMENT = EmbodimentSpec(
    embodiment_id="toy_cam",
    control_hz=float(_FPS),
    features=(
        FeatureSpec("action", FeatureRole.ACTION, shape=(2,), dtype="float32"),
        FeatureSpec(_CAM, FeatureRole.IMAGE, shape=(_HEIGHT, _WIDTH, 3), dtype="uint8"),
    ),
)


# -- frame generators (deterministic, no RNG that leaks into scores) ------------------


def _checkerboard(shift: int, square: int = 6) -> _U8:
    """A high-contrast checkerboard (lots of edges -> high sharpness), shifted by ``shift``."""
    yy, xx = np.mgrid[0:_HEIGHT, 0:_WIDTH]
    board = (((yy + shift) // square + (xx + shift) // square) % 2).astype(np.uint8) * 255
    return np.repeat(board[..., None], 3, axis=2)


def _uniform(level: int) -> _U8:
    """A flat solid-grey frame: no edges -> variance-of-Laplacian ~0 (blurry)."""
    return np.full((_HEIGHT, _WIDTH, 3), level, dtype=np.uint8)


def _textured(seed: int) -> _U8:
    """A fixed, distinctive coarse texture (sharp, with its own colour palette)."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(_HEIGHT // 4, _WIDTH // 4, 3), dtype=np.uint8)
    return np.kron(coarse, np.ones((4, 4, 1), dtype=np.uint8)).astype(np.uint8)


def _frames_blurry(n: int) -> list[_U8]:
    # Near-uniform, only a slow brightness drift: no high-frequency content at all.
    return [_uniform(110 + i) for i in range(n)]


def _frames_frozen(n: int) -> list[_U8]:
    # A single textured frame duplicated -> every adjacent pair is held (visual stall).
    held = _textured(seed=7)
    return [held.copy() for _ in range(n)]


def _frames_sharp_moving(n: int) -> list[_U8]:
    # Sharp AND moving: the well-behaved control episode.
    return [_checkerboard(shift=i * 3) for i in range(n)]


def _frames_dup(n: int, *, tweak: int) -> list[_U8]:
    # A textured moving episode; ``tweak`` nudges a corner pixel so DUP_A/DUP_B differ slightly
    # yet remain near-duplicates of each other and far from the other episodes.
    base = _textured(seed=42)
    out: list[_U8] = []
    for i in range(n):
        frame = np.roll(base, shift=i, axis=1).copy()
        frame[0, 0, :] = np.uint8((frame[0, 0, 0].astype(int) + tweak) % 256)
        out.append(frame)
    return out


def _frames_unique(n: int) -> list[_U8]:
    # A visually distinct moving episode (different palette/structure) -> high diversity.
    return [np.roll(_textured(seed=999), shift=i * 2, axis=0).copy() for i in range(n)]


# -- dataset assembly -----------------------------------------------------------------


def _make_traj(idx: int, name: str, frames: list[_U8], shard: Path) -> Trajectory:
    _write_tiny_mp4(shard, frames)
    n = len(frames)
    ref = VideoReference(
        video_key=_CAM,
        num_frames=n,
        shard_path=shard,
        from_timestamp=0.0,
        to_timestamp=(n - 1) / _FPS,
    )
    action = (np.arange(n, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32)) + idx
    columns = {"action": action}
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/image_signals",
        episode_index=idx,
        embodiment=_EMBODIMENT,
        fingerprint=f"{name}-{fingerprint_arrays(columns)}",  # distinct per episode
        num_steps=n,
        source_format="synthetic_v0",
        extra={"video_references": {_CAM: ref}},
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _build_dataset(tmp_path: Path) -> dict[str, Trajectory]:
    n = 12
    specs: dict[str, list[_U8]] = {
        "BLURRY": _frames_blurry(n),
        "FROZEN": _frames_frozen(n),
        "SHARP": _frames_sharp_moving(n),
        "DUP_A": _frames_dup(n, tweak=0),
        "DUP_B": _frames_dup(n, tweak=2),
        "UNIQUE": _frames_unique(n),
    }
    trajs: dict[str, Trajectory] = {}
    for idx, (name, frames) in enumerate(specs.items()):
        trajs[name] = _make_traj(idx, name, frames, tmp_path / f"{name}.mp4")
    return trajs


def _score_map(sig: object, trajs: dict[str, Trajectory]) -> dict[str, float]:
    ctx = make_signal_context(seed=0)
    batch = list(trajs.values())
    sig.fit(iter(batch), ctx)  # type: ignore[attr-defined]
    scored = sig.score(batch, ctx)  # type: ignore[attr-defined]
    by_fp = {s.trajectory_fingerprint: s for s in scored}
    return {
        name: by_fp[t.meta.fingerprint].value
        for name, t in trajs.items()
        if not by_fp[t.meta.fingerprint].skipped
    }


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_decode_cache()


# -- known-answer assertions ----------------------------------------------------------


def test_image_blur_ranks_blurry_worst(tmp_path: Path) -> None:
    trajs = _build_dataset(tmp_path)
    values = _score_map(ImageBlur(), trajs)  # higher_is_better=False -> larger = blurrier
    assert len(values) == len(trajs)
    worst = max(values, key=lambda k: values[k])
    assert worst == "BLURRY", values
    # And by a comfortable margin over the sharpest control episode.
    assert values["BLURRY"] > values["SHARP"]


def test_visual_stall_flags_frozen(tmp_path: Path) -> None:
    trajs = _build_dataset(tmp_path)
    values = _score_map(VisualStall(), trajs)  # severity = max(0, held_frac - tol)
    flagged = max(values, key=lambda k: values[k])
    assert flagged == "FROZEN", values
    assert values["FROZEN"] > 0.0  # a genuine stall is flagged
    # The genuinely-moving episodes are not stalls.
    assert values["SHARP"] == 0.0
    assert values["UNIQUE"] == 0.0


def test_visual_diversity_ranks_near_duplicate_lowest(tmp_path: Path) -> None:
    trajs = _build_dataset(tmp_path)
    values = _score_map(VisualDiversity(), trajs)  # higher_is_better=True -> smaller = less unique
    assert len(values) == len(trajs)
    least_unique = min(values, key=lambda k: values[k])
    assert least_unique in {"DUP_A", "DUP_B"}, values
    # Both near-duplicates score below the genuinely-unique episode.
    assert values["DUP_A"] < values["UNIQUE"]
    assert values["DUP_B"] < values["UNIQUE"]


@pytest.mark.parametrize("factory", [ImageBlur, VisualStall, VisualDiversity])
def test_scores_are_deterministic(tmp_path: Path, factory: type) -> None:
    trajs = _build_dataset(tmp_path)
    first = _score_map(factory(), trajs)
    clear_decode_cache()  # force a real re-decode, not a cache hit
    second = _score_map(factory(), trajs)
    assert first == second  # CPU + no RNG + fixed seed => byte-identical


def test_signals_skip_gracefully_without_video() -> None:
    # A trajectory with no video references must be a recorded skip, never an error.
    columns = {"action": np.zeros((4, 2), dtype=np.float32)}
    meta = TrajectoryMeta(
        source_dataset_id="synthetic/image_signals",
        episode_index=0,
        embodiment=_EMBODIMENT,
        fingerprint=fingerprint_arrays(columns),
        num_steps=4,
        source_format="synthetic_v0",
    )
    traj = Trajectory(meta, InMemoryFeatureStore(columns))
    ctx = make_signal_context(seed=0)
    for sig in (ImageBlur(), VisualStall(), VisualDiversity()):
        sig.fit(iter([traj]), ctx)
        (score,) = sig.score([traj], ctx)
        assert score.skipped
        assert score.skip_reason
