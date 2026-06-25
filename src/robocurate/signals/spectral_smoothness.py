"""Spectral-smoothness (SPARC) heuristic (Tier 0, CPU).

SPARC — Spectral Arc Length (Balasubramanian, Melendez-Calderon, Roby-Brami & Burdet, "On the
analysis of movement smoothness", J NeuroEngineering Rehabil 2015) — measures how *smooth* a
motion is from the arc length of the normalized Fourier magnitude spectrum of its **speed
profile**. A smooth, single-stroke motion has a compact low-frequency spectrum (short arc,
value near 0); a jerky / hesitant / self-correcting motion adds high-frequency content (longer
arc, more negative). It is dimensionless and robust to movement duration and amplitude.

This is the orthogonal complement to :class:`~robocurate.signals.path_efficiency.PathEfficiency`
(directness): directness flags a *meandering* path; SPARC flags a *jerky-but-straight* one
(high-frequency wobble in the speed profile that directness is blind to).

Algorithm (the authors' reference ``sparc(speed, fs)``, faithfully reproduced):

1. **Speed profile.** From the chosen source feature, form the per-step speed (scalar): the L2
   norm of the per-step displacement over the translational subspace (``dims``). ``positions``
   takes ``np.diff`` first; ``increments`` (e.g. delta/velocity actions) takes the row norm.
2. **Sampling rate.** ``fs`` is recovered from the real timestamps (``1 / median dt``); without
   strictly increasing timestamps the trajectory is a recorded skip, never a guess.
3. **Spectrum.** Zero-pad to ``2**(ceil(log2 N) + padlevel)``, take ``|FFT|``, normalize by its
   peak. Keep frequencies up to ``cutoff_hz``, then adaptively trim to the band where the
   normalized magnitude stays ``>= amp_threshold`` (so the arc does not accumulate length over
   the noise floor).
4. **Arc length.** ``SPARC = -sum sqrt((d f_norm)^2 + (d magnitude)^2)`` over that band, with the
   frequency axis normalized to its own span. More negative = less smooth, so
   ``higher_is_better=True`` (closer to 0 = smoother = more keepable).

Honest caveats: SPARC is a Fourier measure, so it is *coarse on very short trajectories* (a
handful of timesteps barely resolve a spectrum) — validate it against ground truth on your data
rather than assuming it transfers. The two thresholds (``cutoff_hz``, ``amp_threshold``) are
fixed and reported for determinism.

Cost tier 0 (CPU, laptop-friendly). Deterministic: a pure function of the input arrays.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.trajectory import Array, Trajectory

F64 = npt.NDArray[np.float64]

DEFAULT_DIMS = (0, 3)
DEFAULT_CUTOFF_HZ = 10.0
DEFAULT_AMP_THRESHOLD = 0.05
DEFAULT_PADLEVEL = 4
_MIN_STEPS = 4
_EPS = 1e-9


class SpectralSmoothness:
    """SPARC smoothness signal: spectral arc length of the source's speed profile.

    Args:
        source: Feature key whose motion is analyzed. ``None`` (default) uses the action
            feature; point at a Cartesian position feature with ``motion="positions"`` for the
            true end-effector speed profile.
        dims: ``(start, stop)`` column slice selecting the translational subspace (default
            ``(0, 3)``), or ``None`` for all columns.
        motion: ``"increments"`` (rows are per-step displacements) or ``"positions"`` (absolute
            positions; ``np.diff`` first).
        cutoff_hz: Max frequency considered (the adaptive band is trimmed within this).
        amp_threshold: Normalized-magnitude floor for the adaptive band.
        name: Override the signal name (rarely needed).
    """

    def __init__(
        self,
        *,
        source: str | None = None,
        dims: tuple[int, int] | None = DEFAULT_DIMS,
        motion: str = "increments",
        cutoff_hz: float = DEFAULT_CUTOFF_HZ,
        amp_threshold: float = DEFAULT_AMP_THRESHOLD,
        name: str = "spectral_smoothness",
    ) -> None:
        if motion not in ("increments", "positions"):
            raise ValueError(f"motion must be 'increments' or 'positions', got {motion!r}")
        if dims is not None and (len(dims) != 2 or dims[0] < 0 or dims[1] <= dims[0]):
            raise ValueError(
                f"dims must be a (start, stop) slice with stop > start >= 0, got {dims}"
            )
        self.source = source
        self.dims = dims
        self.motion = motion
        self.cutoff_hz = cutoff_hz
        self.amp_threshold = amp_threshold
        requirement = source if source is not None else "action"
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({requirement}),
            produces_per_transition=False,
            deterministic=True,
            description=(
                f"Spectral arc length (SPARC) smoothness of the {requirement} speed profile "
                "(closer to 0 is smoother)."
            ),
        )

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        """Stateless heuristic: nothing to fit."""
        return

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        return [self._score_one(traj) for traj in batch]

    # -- internals -------------------------------------------------------------------

    def _score_one(self, traj: Trajectory) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        source = self._resolve_source(traj)
        if source is None:
            return self._skip(fingerprint, f"no {self._source_label()} feature to measure")

        x: Array = np.asarray(source, dtype=np.float64).reshape(source.shape[0], -1)
        if self.dims is not None:
            x = x[:, self.dims[0] : self.dims[1]]
        if x.shape[1] == 0:
            return self._skip(fingerprint, f"source has no columns in dims {self.dims}")
        if x.shape[0] < _MIN_STEPS:
            return self._skip(fingerprint, f"trajectory too short ({x.shape[0]} steps) for SPARC")

        fs = self._sampling_rate(traj)
        if fs is None:
            return self._skip(fingerprint, "no strictly-increasing timestamps; cannot get fs")

        increments = np.diff(x, axis=0) if self.motion == "positions" else x
        speed: F64 = np.linalg.norm(increments, axis=1).astype(np.float64)
        if speed.shape[0] < 2 or float(speed.max()) < _EPS:
            return self._skip(fingerprint, "degenerate: no motion in the speed profile")

        sparc = self._sparc(speed, fs)
        if not np.isfinite(sparc):
            return self._skip(fingerprint, "spectrum has no content above the amplitude floor")
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=sparc,
            higher_is_better=True,
            diagnostics={
                "source": self._source_label(),
                "motion": self.motion,
                "fs": fs,
                "cutoff_hz": self.cutoff_hz,
            },
        )

    def _sparc(self, speed: F64, fs: float) -> float:
        """Spectral arc length of ``speed`` sampled at ``fs`` (authors' reference algorithm)."""
        n = speed.shape[0]
        nfft = int(2 ** (np.ceil(np.log2(n)) + DEFAULT_PADLEVEL))
        freq: F64 = np.linspace(0.0, fs, nfft, endpoint=False).astype(np.float64)
        mag: F64 = np.abs(np.fft.fft(speed, nfft)).astype(np.float64)
        peak = float(mag.max())
        if peak < _EPS:
            return float("nan")
        mag = mag / peak

        in_band = freq <= self.cutoff_hz
        freq, mag = freq[in_band], mag[in_band]
        above = np.where(mag >= self.amp_threshold)[0]
        if above.size < 2:
            return float("nan")
        lo, hi = int(above[0]), int(above[-1]) + 1
        freq, mag = freq[lo:hi], mag[lo:hi]
        span = float(freq[-1] - freq[0])
        if freq.size < 2 or span < _EPS:
            return float("nan")

        df = np.diff(freq) / span
        dm = np.diff(mag)
        return float(-np.sum(np.sqrt(df**2 + dm**2)))

    def _sampling_rate(self, traj: Trajectory) -> float | None:
        timestamps = traj.timestamps()
        if timestamps is None:
            return None
        t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if t.size < 2 or not np.all(np.diff(t) > 0.0):
            return None
        dt = float(np.median(np.diff(t)))
        return 1.0 / dt if dt > _EPS else None

    def _skip(self, fingerprint: str, reason: str) -> TrajectoryScore:
        return TrajectoryScore.skip(
            self.spec.name, fingerprint, reason=reason, higher_is_better=True
        )

    def _resolve_source(self, traj: Trajectory) -> Array | None:
        if self.source is None:
            return traj.actions()
        return traj.feature(self.source) if traj.has(self.source) else None

    def _source_label(self) -> str:
        return self.source if self.source is not None else "action"


__all__ = ["SpectralSmoothness"]
