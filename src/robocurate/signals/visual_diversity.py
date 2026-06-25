"""Visual-diversity signal (Tier 0, CPU; optional ``video`` extra).

The image-space sibling of :class:`~robocurate.signals.redundancy.Redundancy`: instead of a
statistical embedding of the low-dim arrays, it embeds each episode's *appearance* and ranks
it by how visually unique it is across the dataset. A near-duplicate camera trace (the same
scene re-recorded, a copy-pasted episode) sits ~0 from its twin in appearance space and is
flagged as low-diversity bloat; a visually distinct episode is far from everything else.

It reuses redundancy's exact machinery: ``fit`` builds a z-standardized embedding index in
``ctx.cache``; ``score`` embeds the trajectory and returns the **mean distance to its ``k``
nearest neighbours** (excluding one self-occurrence), ``higher_is_better=True`` (more unique).
The embedding is a cheap, fixed-length CPU image descriptor — the mean downsampled-grayscale
"thumbnail" (flattened) concatenated with per-RGB-channel intensity histograms — averaged over
a handful of decoded frames, so it needs no GPU or model. It is deliberately coarse (gross
appearance, not semantics); a learned encoder is the obvious Tier-1 upgrade.

Determinism (invariant 3): decode + downsample + histogram + z-standardized k-NN are all pure
NumPy reductions over the shard bytes, so the same input + index yields byte-identical scores.
No RNG is used.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

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

DEFAULT_K = 1
DEFAULT_MAX_FRAMES = 8
DEFAULT_THUMB = 8  # downsampled-grayscale thumbnail edge (THUMB*THUMB cells)
DEFAULT_HIST_BINS = 8  # per-RGB-channel intensity histogram bins
# Cache key + version: bumping the version invalidates a stale index (mirrors redundancy).
_INDEX_KEY = "visual_index.v1"


class VisualDiversity:
    """Dataset-relative visual uniqueness via k-NN distance in a cheap CPU image embedding.

    Args:
        camera: Which camera (image feature key) to embed. ``None`` (default) picks the first
            key in sorted order, so the choice is deterministic across runs.
        k: Number of nearest neighbours to average (default 1 ≡ 1-NN).
        max_frames: Max evenly-spaced frames decoded per episode to build the embedding.
        thumb: Edge length of the downsampled-grayscale thumbnail (``thumb*thumb`` features).
        hist_bins: Per-RGB-channel histogram bins appended to the thumbnail.
        name: Override the signal name.
    """

    def __init__(
        self,
        *,
        camera: str | None = None,
        k: int = DEFAULT_K,
        max_frames: int = DEFAULT_MAX_FRAMES,
        thumb: int = DEFAULT_THUMB,
        hist_bins: int = DEFAULT_HIST_BINS,
        name: str = "visual_diversity",
    ) -> None:
        _require_av()  # fail fast with a clear "install robocurate[video]" message
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if max_frames < 1:
            raise ValueError(f"max_frames must be >= 1, got {max_frames}")
        if thumb < 1:
            raise ValueError(f"thumb must be >= 1, got {thumb}")
        if hist_bins < 1:
            raise ValueError(f"hist_bins must be >= 1, got {hist_bins}")
        self.camera = camera
        self.k = k
        self.max_frames = max_frames
        self.thumb = thumb
        self.hist_bins = hist_bins
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({REQUIRES_IMAGE}),  # advertised; the curator does not gate on it
            produces_per_transition=False,
            deterministic=True,
            description=(
                f"Visual uniqueness = mean distance to the {k} nearest neighbour(s) in a cheap "
                "CPU image embedding (higher is more visually unique)."
            ),
        )

    # -- fit: build the embedding index ----------------------------------------------

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        vectors: list[F64] = []
        fingerprints: list[str] = []
        for traj in trajectories:
            emb = self._embed(traj)
            if emb is None:
                continue
            vectors.append(emb)
            fingerprints.append(traj.meta.fingerprint)

        if not vectors:
            ctx.cache.put(_INDEX_KEY, None)
            return

        raw: F64 = np.vstack(vectors).astype(np.float64)
        mean: F64 = raw.mean(axis=0)
        std: F64 = raw.std(axis=0)
        std_safe = np.where(std > 0.0, std, 1.0)
        z: F64 = (raw - mean) / std_safe
        ctx.cache.put(
            _INDEX_KEY,
            {"z": z, "mean": mean, "std_safe": std_safe, "fingerprints": fingerprints},
        )

    # -- score -----------------------------------------------------------------------

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        index = ctx.cache.get(_INDEX_KEY) if ctx.cache.has(_INDEX_KEY) else None
        return [self._score_one(traj, index) for traj in batch]

    def _score_one(self, traj: Trajectory, index: dict[str, Any] | None) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        emb = self._embed(traj)
        if emb is None:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="no decodable frames to embed (no video references or opaque shard)",
                higher_is_better=True,
            )
        if index is None:
            return TrajectoryScore.skip(
                self.spec.name, fingerprint, reason="empty embedding index", higher_is_better=True
            )

        z_all: F64 = index["z"]
        fingerprints: list[str] = index["fingerprints"]
        z_self = (emb - index["mean"]) / index["std_safe"]
        distances: F64 = np.linalg.norm(z_all - z_self, axis=1)

        # Exclude exactly one self-occurrence so a trajectory is never its own neighbour, while
        # genuine (near-)duplicates of it remain in the pool and correctly register as close.
        keep = np.ones(distances.shape[0], dtype=bool)
        self_positions = [i for i, fp in enumerate(fingerprints) if fp == fingerprint]
        if self_positions:
            keep[self_positions[0]] = False
        others = distances[keep]
        if others.size == 0:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="dataset has no other embeddable trajectories to compare against",
                higher_is_better=True,
            )

        kk = min(self.k, others.size)
        nearest = np.partition(others, kk - 1)[:kk]
        value = float(nearest.mean())
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=value,
            higher_is_better=True,
            diagnostics={
                "camera": self.camera if self.camera is not None else _first_camera(traj),
                "nn_distance": float(others.min()),
                "mean_knn_distance": value,
                "k": kk,
                "n_neighbors": int(others.size),
                "embedding_dim": int(z_all.shape[1]),
            },
        )

    # -- embedding -------------------------------------------------------------------

    def _embed(self, traj: Trajectory) -> F64 | None:
        """Cheap fixed-length appearance descriptor averaged over a few decoded frames."""
        refs = traj.video_references()
        if not refs:
            return None
        camera = self.camera if self.camera is not None else sorted(refs)[0]
        if camera not in refs:
            return None
        frames = decode_frames(refs[camera], max_frames=self.max_frames)
        if frames is None or frames.shape[0] == 0:
            return None
        per_frame = [self._frame_descriptor(frames[i]) for i in range(frames.shape[0])]
        emb: F64 = np.mean(np.vstack(per_frame), axis=0)
        return emb

    def _frame_descriptor(self, frame: npt.NDArray[np.uint8]) -> F64:
        thumb = _downsample_gray(frame, self.thumb).reshape(-1)
        hist = _rgb_histogram(frame, self.hist_bins)
        return np.concatenate([thumb, hist])


def _first_camera(traj: Trajectory) -> str | None:
    refs = traj.video_references()
    return sorted(refs)[0] if refs else None


def _downsample_gray(frame: npt.NDArray[np.uint8], thumb: int) -> F64:
    """Block-average an ``(H, W, C)`` RGB frame's grayscale into a ``(thumb, thumb)`` thumbnail."""
    rgb = np.asarray(frame, dtype=np.float64)
    gray = rgb[..., :3] @ np.asarray([0.299, 0.587, 0.114]) if rgb.ndim == 3 else rgb
    h, w = gray.shape
    row_idx = (np.arange(h) * thumb // max(h, 1)).clip(0, thumb - 1)
    col_idx = (np.arange(w) * thumb // max(w, 1)).clip(0, thumb - 1)
    out = np.zeros((thumb, thumb), dtype=np.float64)
    counts = np.zeros((thumb, thumb), dtype=np.float64)
    np.add.at(out, (row_idx[:, None], col_idx[None, :]), gray)
    np.add.at(counts, (row_idx[:, None], col_idx[None, :]), 1.0)
    averaged: F64 = out / np.where(counts > 0.0, counts, 1.0)
    return averaged


def _rgb_histogram(frame: npt.NDArray[np.uint8], bins: int) -> F64:
    """Per-channel normalized intensity histogram, concatenated over R, G, B."""
    rgb = np.asarray(frame)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    parts: list[F64] = []
    n = rgb.shape[0] * rgb.shape[1]
    for c in range(3):
        counts, _ = np.histogram(rgb[..., c], bins=bins, range=(0.0, 255.0))
        parts.append(counts.astype(np.float64) / max(n, 1))
    return np.concatenate(parts)


__all__ = ["VisualDiversity"]
