"""Visual-stall signal (Tier 0, CPU; optional ``video`` extra).

The low-dim :class:`~robocurate.signals.structural_validity.StructuralValidity` stall check
catches a *frozen world/robot state*; this is its image-space analogue: a **frozen camera**.
It decodes the episode's frames and measures, for each adjacent pair, the mean-absolute pixel
difference; a pair whose difference is below ``stall_eps`` is a "held" frame (the camera image
did not change). When the held fraction exceeds ``stall_tolerance`` the episode has a visual
stall — a dropped/duplicated video stream, a paused recording, or a genuinely static scene —
which is usually low-value data.

``value`` is the stall severity ``max(0, held_fraction - stall_tolerance)`` with
``higher_is_better=False`` (a moving camera scores 0; the most-frozen episodes rank first for
removal). Diagnostics carry the held fraction and the pair count. Long episodes are capped
with a stride so the per-pair scan stays cheap.

Determinism (invariant 3): decode + mean-abs-diff is a pure NumPy reduction over the shard
bytes, so the same inputs always yield byte-identical scores. No RNG is used.
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

# Mirror structural_validity's stall vocabulary (held-frame fraction + tolerance) so the
# image-space stall reads the same as the low-dim one.
DEFAULT_STALL_TOLERANCE = 0.1  # up to 10% held frames is normal (settling); beyond is a stall
DEFAULT_STALL_EPS = 1.0  # mean-abs pixel diff (0-255 scale) below this counts as a held frame
DEFAULT_MAX_FRAMES = 64  # cap a long episode's pair scan; frames are evenly subsampled to this


class VisualStall:
    """Image-space stall signal: fraction of adjacent frame pairs with a near-frozen camera.

    Args:
        camera: Which camera (image feature key) to score. ``None`` (default) picks the first
            key in sorted order, so the choice is deterministic across runs.
        max_frames: Cap on decoded frames (evenly subsampled); bounds the per-pair scan.
        stall_tolerance: Held-frame fraction tolerated before it counts as a stall.
        stall_eps: Mean-absolute pixel difference (0-255 scale) below which an adjacent pair is
            a held/duplicated frame.
        name: Override the signal name.
    """

    def __init__(
        self,
        *,
        camera: str | None = None,
        max_frames: int = DEFAULT_MAX_FRAMES,
        stall_tolerance: float = DEFAULT_STALL_TOLERANCE,
        stall_eps: float = DEFAULT_STALL_EPS,
        name: str = "visual_stall",
    ) -> None:
        _require_av()  # fail fast with a clear "install robocurate[video]" message
        if max_frames < 2:
            raise ValueError(f"max_frames must be >= 2, got {max_frames}")
        self.camera = camera
        self.max_frames = max_frames
        self.stall_tolerance = stall_tolerance
        self.stall_eps = stall_eps
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({REQUIRES_IMAGE}),  # advertised; the curator does not gate on it
            produces_per_transition=False,
            deterministic=True,
            description=(
                "Visual stall: fraction of adjacent frame pairs with a near-frozen camera "
                "(a held/duplicated video stream; lower is better)."
            ),
        )

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        """No fitting needed — the stall is a per-episode adjacent-frame measure."""
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
        if frames is None or frames.shape[0] < 2:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="fewer than two decodable frames (absent/opaque shard)",
                higher_is_better=False,
            )

        # Mean-abs pixel diff per adjacent pair, on the 0-255 scale.
        diffs = _adjacent_pair_diffs(frames)
        held_fraction = float(np.mean(diffs < self.stall_eps))
        severity = max(0.0, held_fraction - self.stall_tolerance)
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=severity,
            higher_is_better=False,
            diagnostics={
                "camera": camera,
                "num_frames": int(frames.shape[0]),
                "num_pairs": int(diffs.shape[0]),
                "held_frame_fraction": held_fraction,
                "stall_tolerance": self.stall_tolerance,
                "stall_eps": self.stall_eps,
                "min_pair_diff": float(diffs.min()),
                "mean_pair_diff": float(diffs.mean()),
            },
        )


def _adjacent_pair_diffs(frames: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """Mean-absolute pixel difference for every adjacent frame pair, on the 0-255 scale."""
    f = np.asarray(frames, dtype=np.float64)
    abs_diff = np.abs(f[1:] - f[:-1])
    diffs: npt.NDArray[np.float64] = abs_diff.reshape(abs_diff.shape[0], -1).mean(axis=1)
    return diffs


__all__ = ["VisualStall"]
