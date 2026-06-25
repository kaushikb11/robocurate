"""Video-frame decoding for image signals (Stage 2, optional ``video`` extra).

Stage-1 pass-through keeps a :class:`~robocurate.trajectory.VideoReference` per camera —
a pointer into an mp4 shard, never the pixels (see :mod:`robocurate.trajectory`). Stage 2
image signals need the actual frames, so this module turns one reference into a decoded
``(N, H, W, C)`` uint8 RGB array via software CPU decode (PyAV / libav). It is the *only*
place pixels are materialized, and it only ever READS shard files (invariant 1: source
data is read-only).

Decode-by-timestamp-window
--------------------------
A v3 video shard may bundle multiple episodes; a reference marks this episode's slice with
``from_timestamp`` / ``to_timestamp``. We decode every frame whose presentation timestamp
(PTS, in seconds) falls within ``[from_timestamp, to_timestamp]``, **in decode order**, and
return them as a contiguous array. This window+order approach is deliberately robust to the
ambiguity of whether the reference's timestamps are episode-relative or shard-absolute: in
the common one-episode-per-shard case the window spans the whole shard either way, and when
a shard bundles episodes the absolute PTS selects the right slice. When both timestamps are
``None`` we decode the whole shard. The result count is ``N`` (``~= ref.num_frames``); we do
not pad or truncate to ``num_frames`` — codec/keyframe rounding can make them differ by one,
and callers that need exactly ``num_frames`` should align explicitly.

Determinism (invariant 3): decoding is a pure function of the shard bytes plus the window
and ``max_frames``, so the same inputs always yield byte-identical output. That also makes a
process-local LRU decode cache safe — repeated decodes of the same shard are free.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from robocurate.trajectory import VideoReference

if TYPE_CHECKING:
    import numpy.typing as npt

    FrameArray = npt.NDArray[np.uint8]

# Bounded process-local decode cache. Decoding is deterministic (see module docstring), so
# caching cannot change results; the bound keeps memory in check since each entry holds a
# full decoded ``(N, H, W, C)`` uint8 array. Keyed by (shard path, from_ts, to_ts, max_frames).
_CACHE_MAXSIZE = 8
_DECODE_CACHE: OrderedDict[tuple[str, float | None, float | None, int | None], FrameArray] = (
    OrderedDict()
)


def _require_av() -> Any:
    """Import and return PyAV (``av``), with an actionable error if the extra is missing."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "decoding video frames requires PyAV, which is an optional dependency. "
            "Install it with `uv pip install 'robocurate[video]'` (or `robocurate[all]`)."
        ) from exc
    return av


def clear_decode_cache() -> None:
    """Empty the process-local decode cache. Intended for tests that assert decode behavior."""
    _DECODE_CACHE.clear()


def decode_frames(ref: VideoReference, *, max_frames: int | None = None) -> FrameArray | None:
    """Decode this episode's video frames into an ``(N, H, W, C)`` uint8 RGB array.

    See the module docstring for the decode-by-timestamp-window approach and the determinism
    guarantee. Only ever reads ``ref.shard_path``; it is never written.

    Args:
        ref: The :class:`~robocurate.trajectory.VideoReference` to decode. If its ``shard_path``
            is ``None`` or does not exist on disk (an opaque / low-dim-only source), this returns
            ``None`` rather than raising — there is simply nothing to decode.
        max_frames: If set and fewer than the available ``N`` frames are wanted, the frames are
            deterministically and evenly subsampled (indices ``np.linspace(0, N-1, max_frames)``
            rounded to int). ``None`` (default) returns every frame in the window.

    Returns:
        An ``(N, H, W, C)`` uint8 RGB array (``N`` is the window's frame count, ``~= num_frames``,
        or ``max_frames`` when subsampled), or ``None`` when there is no shard to decode.

    Raises:
        Exception: Propagated from PyAV only when ``shard_path`` exists but is genuinely
            corrupt/unreadable. A *missing optional extra* raises at :func:`_require_av`, not here.
    """
    if ref.shard_path is None:
        return None
    shard_path = Path(ref.shard_path)
    if not shard_path.is_file():
        return None

    key = (str(shard_path), ref.from_timestamp, ref.to_timestamp, max_frames)
    cached = _DECODE_CACHE.get(key)
    if cached is not None:
        _DECODE_CACHE.move_to_end(key)  # LRU: mark as most-recently used
        return cached

    frames = _decode_window(shard_path, ref.from_timestamp, ref.to_timestamp)
    if max_frames is not None and 0 <= max_frames < len(frames):
        frames = _subsample(frames, max_frames)

    _DECODE_CACHE[key] = frames
    _DECODE_CACHE.move_to_end(key)
    while len(_DECODE_CACHE) > _CACHE_MAXSIZE:
        _DECODE_CACHE.popitem(last=False)  # evict the least-recently used entry
    return frames


def _decode_window(
    shard_path: Path, from_timestamp: float | None, to_timestamp: float | None
) -> FrameArray:
    """Decode, in order, every frame whose PTS falls in ``[from_timestamp, to_timestamp]``.

    ``None`` bounds are treated as open (``-inf`` / ``+inf``), so both ``None`` decodes the whole
    shard. Frames are returned as a stacked ``(N, H, W, C)`` uint8 RGB array.
    """
    av = _require_av()
    lo = float("-inf") if from_timestamp is None else from_timestamp
    hi = float("inf") if to_timestamp is None else to_timestamp

    decoded: list[FrameArray] = []
    with av.open(str(shard_path)) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base  # PTS unit -> seconds conversion factor
        for frame in container.decode(stream):
            # frame.time is already in seconds when available; fall back to pts * time_base.
            if frame.time is not None:
                pts_seconds = float(frame.time)
            elif frame.pts is not None and time_base is not None:
                pts_seconds = float(frame.pts * time_base)
            else:
                # No timestamp at all: only includable if the window is fully open.
                if from_timestamp is None and to_timestamp is None:
                    decoded.append(frame.to_ndarray(format="rgb24"))
                continue
            if lo <= pts_seconds <= hi:
                decoded.append(frame.to_ndarray(format="rgb24"))

    if not decoded:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.stack(decoded).astype(np.uint8, copy=False)


def _subsample(frames: FrameArray, max_frames: int) -> FrameArray:
    """Deterministically pick ``max_frames`` evenly-spaced frames from ``frames``."""
    n = len(frames)
    indices = np.linspace(0, n - 1, max_frames).round().astype(int)
    subsampled: FrameArray = frames[indices]
    return subsampled


__all__ = ["clear_decode_cache", "decode_frames"]
