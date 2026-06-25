"""Round-trip tests for software CPU video-frame decode (:mod:`robocurate.video`).

These encode a *real* tiny mp4 with PyAV from a list of known deterministic solid-colour
frames, then decode it back through :func:`decode_frames` and assert the frames come back in
order with the right shape. h264/mp4 is lossy, so we assert *approximate* per-frame colour
(mean-abs-diff tolerance) and correct ordering — never exact pixel equality. The module is
skipped cleanly when the ``video`` extra is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("av")  # skip the whole module if the video extra is absent

import av

from robocurate.trajectory import VideoReference
from robocurate.video import clear_decode_cache, decode_frames

pytestmark = pytest.mark.video

_FPS = 10
_HEIGHT = 48
_WIDTH = 64


def _known_colour(i: int) -> tuple[int, int, int]:
    """A deterministic distinct solid colour for frame ``i``."""
    return (i * 17 % 256, i * 29 % 256, i * 53 % 256)


def _make_frames(n: int) -> list[npt.NDArray[np.uint8]]:
    frames: list[npt.NDArray[np.uint8]] = []
    for i in range(n):
        frame = np.empty((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
        frame[:, :] = _known_colour(i)
        frames.append(frame)
    return frames


def _write_tiny_mp4(path: Path, frames: list[npt.NDArray[np.uint8]]) -> None:
    """Encode ``frames`` (list of (H, W, 3) uint8 RGB) into a real h264 mp4 at ``path``."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=_FPS)
        stream.width = _WIDTH
        stream.height = _HEIGHT
        stream.pix_fmt = "yuv420p"
        for arr in frames:
            av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():  # flush the encoder
            container.mux(packet)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    clear_decode_cache()


def _ref(shard_path: Path | None, n: int) -> VideoReference:
    to_ts = None if shard_path is None else (n - 1) / _FPS
    return VideoReference(
        video_key="observation.images.cam",
        num_frames=n,
        shard_path=shard_path,
        from_timestamp=0.0 if shard_path is not None else None,
        to_timestamp=to_ts,
    )


def test_decode_returns_known_frames_in_order(tmp_path: Path) -> None:
    n = 8
    frames = _make_frames(n)
    mp4 = tmp_path / "clip.mp4"
    _write_tiny_mp4(mp4, frames)

    decoded = decode_frames(_ref(mp4, n))
    assert decoded is not None
    assert decoded.shape == (n, _HEIGHT, _WIDTH, 3)
    assert decoded.dtype == np.uint8

    # h264 is lossy: assert each frame's mean colour is *close* to the encoded colour, and that
    # the distinct colours come back in the right order (so we know frames aren't shuffled).
    for i in range(n):
        mean = decoded[i].reshape(-1, 3).mean(axis=0)
        expected = np.asarray(_known_colour(i), dtype=np.float64)
        assert np.abs(mean - expected).mean() < 10.0, f"frame {i} colour off: {mean} vs {expected}"


def test_decode_returns_none_for_no_shard() -> None:
    assert decode_frames(_ref(None, 5)) is None


def test_decode_returns_none_for_missing_path(tmp_path: Path) -> None:
    assert decode_frames(_ref(tmp_path / "does_not_exist.mp4", 5)) is None


def test_decode_is_deterministic(tmp_path: Path) -> None:
    n = 6
    mp4 = tmp_path / "clip.mp4"
    _write_tiny_mp4(mp4, _make_frames(n))

    first = decode_frames(_ref(mp4, n))
    clear_decode_cache()  # force a real re-decode, not a cache hit
    second = decode_frames(_ref(mp4, n))
    assert first is not None and second is not None
    assert first.tobytes() == second.tobytes()  # byte-identical


def test_max_frames_subsamples(tmp_path: Path) -> None:
    n = 8
    mp4 = tmp_path / "clip.mp4"
    _write_tiny_mp4(mp4, _make_frames(n))

    decoded = decode_frames(_ref(mp4, n), max_frames=3)
    assert decoded is not None
    assert decoded.shape == (3, _HEIGHT, _WIDTH, 3)

    # max_frames >= N returns all frames unchanged.
    full = decode_frames(_ref(mp4, n), max_frames=n + 5)
    assert full is not None and full.shape[0] == n
