"""Synthetic trajectory corruptions for known-answer validation of quality signals.

Operator-proficiency labels (e.g. robomimic Multi-Human tiers) are a *soft, contested* proxy
for "bad data" — an adversarial review found that recovering them need not imply good curation.
Injected corruptions are the opposite: **hard ground truth**. We take clean trajectories, damage
a known subset in a known way, run a signal, and measure detection (does it rank the corrupted
ones worst?). This is the data-valuation field's standard known-answer check, and — unlike a
label-AUC — it exposes *blind spots*: a truncated demo is structurally bad yet geometrically
clean, so a directness/smoothness signal will (correctly, honestly) miss it.

Each corruption is deterministic given its ``seed`` (invariant 3) and returns a NEW
:class:`~robocurate.trajectory.Trajectory` tagged with ``meta.extra["corruption"]``; the input is
never mutated. Corruptions that change length (truncate, stall) slice/extend every feature column
together so the trajectory stays internally consistent.

The four kinds are chosen to span detectable-and-blind:
  * ``jitter`` — high-frequency per-step noise on the target feature (a *smoothness* defect:
    jerk / action-noise / SPARC should catch it; directness less so).
  * ``detour`` — a smooth out-and-back lateral excursion (a *directness* defect: path-efficiency
    should catch it; smoothness signals should not, since it adds no high-frequency content).
  * ``truncate`` — drop the last fraction of the episode (a *structural* defect that geometry
    signals are blind to — the kept portion is still smooth and direct).
  * ``stall`` — insert a held (repeated-frame) segment (a *duration/efficiency* defect; mostly a
    blind spot for geometry, caught by length).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt

from robocurate.trajectory import (
    Array,
    InMemoryFeatureStore,
    Trajectory,
    fingerprint_arrays,
)

F64 = npt.NDArray[np.float64]

CORRUPTIONS = ("jitter", "detour", "truncate", "stall")


def corrupt(
    traj: Trajectory,
    kind: str,
    *,
    feature: str,
    severity: float = 1.0,
    seed: int = 0,
) -> Trajectory:
    """Return a corrupted copy of ``traj`` (input untouched), tagged ``meta.extra["corruption"]``.

    Args:
        traj: The clean trajectory to damage.
        kind: One of :data:`CORRUPTIONS`.
        feature: Feature key the geometric corruptions (``jitter`` / ``detour``) act on (the
            others act on all columns). E.g. ``"action"`` or ``"observation.robot0_eef_pos"``.
        severity: Corruption strength (kind-specific; ``jitter`` noise scale, ``detour`` bump
            scale, ``truncate``/``stall`` the affected fraction of timesteps).
        seed: Deterministic RNG seed.
    """
    if kind not in CORRUPTIONS:
        raise ValueError(f"unknown corruption {kind!r}; known: {CORRUPTIONS}")
    rng = np.random.default_rng(seed)
    columns = {spec.key: np.asarray(traj.feature(spec.key)) for spec in traj.embodiment.features}

    # Capture the original control dt so a length-changing corruption can rebuild a strictly-
    # increasing timestamp (a stall holds position while time keeps advancing — a real pause, not
    # duplicate timestamps, which would just make derivative-based signals skip).
    ts = columns.get("timestamp")
    orig_dt = (
        float(np.median(np.diff(np.asarray(ts, dtype=np.float64).reshape(-1))))
        if ts is not None and np.asarray(ts).shape[0] > 1
        else None
    )

    if kind == "jitter":
        columns[feature] = _jitter(columns[feature], severity, rng)
    elif kind == "detour":
        columns[feature] = _detour(columns[feature], severity, rng)
    elif kind == "truncate":
        columns = _truncate(columns, severity)
    else:  # stall
        columns = _stall(columns, severity, rng)

    num_steps = int(next(iter(columns.values())).shape[0])
    if "timestamp" in columns and orig_dt is not None:
        columns["timestamp"] = (np.arange(num_steps) * orig_dt).astype(
            np.asarray(columns["timestamp"]).dtype
        )
    meta = replace(
        traj.meta,
        num_steps=num_steps,
        fingerprint=fingerprint_arrays(columns),
        extra={**dict(traj.meta.extra), "corruption": kind},
    )
    return Trajectory(meta, InMemoryFeatureStore(columns))


def _as2d(x: Array) -> F64:
    arr: F64 = np.asarray(x, dtype=np.float64)
    return arr.reshape(arr.shape[0], -1)


def _jitter(x: Array, severity: float, rng: np.random.Generator) -> Array:
    """Add per-step (high-frequency) Gaussian noise scaled to each column's spread."""
    flat = _as2d(x)
    scale = flat.std(axis=0)
    scale[scale == 0.0] = 1.0
    noised = flat + severity * 0.5 * scale * rng.standard_normal(flat.shape)
    out: Array = noised.reshape(np.asarray(x).shape).astype(np.asarray(x).dtype)
    return out


def _detour(x: Array, severity: float, rng: np.random.Generator) -> Array:
    """Add a smooth out-and-back lateral excursion: net displacement unchanged, path longer."""
    flat = _as2d(x).copy()
    num_steps, dim = flat.shape
    direction = rng.standard_normal(dim)
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 0 else np.ones(dim) / np.sqrt(dim)
    span = float(np.linalg.norm(flat[-1] - flat[0])) or float(np.linalg.norm(flat).mean()) or 1.0
    bump = np.sin(np.linspace(0.0, np.pi, num_steps))  # 0 at both ends -> net unchanged
    flat = flat + severity * 0.5 * span * np.outer(bump, direction)
    out: Array = flat.reshape(np.asarray(x).shape).astype(np.asarray(x).dtype)
    return out


def _truncate(columns: dict[str, Array], severity: float) -> dict[str, Array]:
    """Drop the last ``severity`` fraction of timesteps from every column (incomplete demo)."""
    num_steps = int(next(iter(columns.values())).shape[0])
    keep = max(2, round(num_steps * (1.0 - min(max(severity, 0.0), 0.9))))
    return {k: np.asarray(v)[:keep] for k, v in columns.items()}


def _stall(
    columns: dict[str, Array], severity: float, rng: np.random.Generator
) -> dict[str, Array]:
    """Insert a held (repeated-frame) segment mid-trajectory in every column (a hesitation)."""
    num_steps = int(next(iter(columns.values())).shape[0])
    hold = max(1, round(num_steps * min(max(severity, 0.0), 0.9)))
    at = int(rng.integers(1, num_steps)) if num_steps > 1 else 0
    out: dict[str, Array] = {}
    for k, v in columns.items():
        arr = np.asarray(v)
        repeated = np.repeat(arr[at : at + 1], hold, axis=0)
        out[k] = np.concatenate([arr[:at], repeated, arr[at:]], axis=0)
    return out


__all__ = ["CORRUPTIONS", "corrupt"]
