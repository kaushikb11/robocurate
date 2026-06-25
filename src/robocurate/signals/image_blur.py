"""Image-blur / sharpness signal (Tier 0, CPU; optional ``video`` extra).

Decodes a handful of evenly-spaced frames from one camera and scores how *blurry* the
episode looks via the classic **variance-of-Laplacian** focus measure: a sharp, in-focus
frame has strong high-frequency edges, so the Laplacian of its grayscale has a large
variance; a blurry, smeared, or uniform frame has a small one. We implement the 3x3
Laplacian as a tiny NumPy convolution so the signal needs no OpenCV — only the ``video``
extra (PyAV) for decoding (see :mod:`robocurate.video`).

The emitted ``value`` is a blur *severity* with ``higher_is_better=False`` (the negative
mean per-frame sharpness), so the blurriest episodes rank first for removal. Diagnostics
carry the mean / min sharpness and the fraction of frames below ``sharpness_threshold`` (a
human-readable "how many frames look out-of-focus"). This is a per-frame appearance check;
it complements — does not replace — the geometric and learned signals.

Determinism (invariant 3): decoding is a pure function of the shard bytes + window +
``max_frames`` (see :mod:`robocurate.video`) and the Laplacian variance is a pure NumPy
reduction, so the same inputs always yield byte-identical scores. No RNG is used.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from robocurate.signals.base import (
    REQUIRES_IMAGE,
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.trajectory import Trajectory
from robocurate.video import _require_av, decode_frames

F64 = npt.NDArray[np.float64]

DEFAULT_MAX_FRAMES = 16
DEFAULT_SHARPNESS_THRESHOLD = 100.0  # variance-of-Laplacian below this looks out-of-focus

# 3x3 discrete Laplacian kernel (4-neighbour). Convolving grayscale with it and taking the
# variance is the standard focus measure; we apply it with NumPy so there is no OpenCV dep.
_LAPLACIAN_KERNEL: F64 = np.asarray(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


class ImageBlur:
    """Per-episode blur severity from the variance-of-Laplacian sharpness of decoded frames.

    Args:
        camera: Which camera (image feature key) to score. ``None`` (default) picks the first
            key in sorted order, so the choice is deterministic across runs.
        max_frames: Max number of evenly-spaced frames to decode and measure per episode.
        sharpness_threshold: Variance-of-Laplacian below which a frame counts as blurry (only
            surfaced as a diagnostic fraction; it does not change ``value``).
        name: Override the signal name.
    """

    def __init__(
        self,
        *,
        camera: str | None = None,
        max_frames: int = DEFAULT_MAX_FRAMES,
        sharpness_threshold: float = DEFAULT_SHARPNESS_THRESHOLD,
        name: str = "image_blur",
    ) -> None:
        _require_av()  # fail fast with a clear "install robocurate[video]" message
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        self.camera = camera
        self.max_frames = max_frames
        self.sharpness_threshold = sharpness_threshold
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({REQUIRES_IMAGE}),  # advertised; the curator does not gate on it
            produces_per_transition=False,
            deterministic=True,
            description=(
                "Image blur severity via variance-of-Laplacian sharpness of decoded frames "
                "(lower sharpness is blurrier; higher severity is worse)."
            ),
        )

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        """No fitting needed — blur is a per-episode appearance measure."""
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [self._score_one(traj) for traj in batch]

    def _score_one(self, traj: Trajectory) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        refs = traj.video_references()
        if not refs:
            return TrajectoryScore.skip(
                self.spec.name, fingerprint, reason="no video references", higher_is_better=False
            )
        camera = self.camera if self.camera is not None else sorted(refs)[0]
        if camera not in refs:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason=f"camera {camera!r} not present (have {sorted(refs)})",
                higher_is_better=False,
            )
        frames = decode_frames(refs[camera], max_frames=self.max_frames)
        if frames is None or frames.shape[0] == 0:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="no decodable frames (absent or opaque shard)",
                higher_is_better=False,
            )

        sharpness = np.asarray([_frame_sharpness(frames[i]) for i in range(frames.shape[0])])
        mean_sharpness = float(sharpness.mean())
        min_sharpness = float(sharpness.min())
        blurry_fraction = float(np.mean(sharpness < self.sharpness_threshold))
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=-mean_sharpness,  # severity: blurrier (lower sharpness) -> larger value
            higher_is_better=False,
            diagnostics={
                "camera": camera,
                "num_frames": int(frames.shape[0]),
                "mean_sharpness": mean_sharpness,
                "min_sharpness": min_sharpness,
                "blurry_fraction": blurry_fraction,
                "sharpness_threshold": self.sharpness_threshold,
            },
        )


def _frame_sharpness(frame: npt.NDArray[np.uint8]) -> float:
    """Variance of the Laplacian of a frame's grayscale — the higher, the sharper."""
    gray = _to_grayscale(frame)
    lap = _laplacian(gray)
    return float(lap.var())


def _to_grayscale(frame: npt.NDArray[np.uint8]) -> F64:
    """Luma (Rec. 601) grayscale of an ``(H, W, C)`` RGB frame as float64."""
    rgb = np.asarray(frame, dtype=np.float64)
    if rgb.ndim == 2:  # already single-channel
        return rgb
    weights = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
    return rgb[..., :3] @ weights


def _laplacian(gray: F64) -> F64:
    """Apply the 3x3 Laplacian via an edge-replicated NumPy convolution (no SciPy/OpenCV)."""
    padded = np.pad(gray, 1, mode="edge")
    out = np.zeros_like(gray)
    for di in range(3):
        for dj in range(3):
            w = _LAPLACIAN_KERNEL[di, dj]
            if w == 0.0:
                continue
            out += w * padded[di : di + gray.shape[0], dj : dj + gray.shape[1]]
    return out


__all__ = ["ImageBlur"]
