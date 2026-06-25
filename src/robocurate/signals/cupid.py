"""CUPID-inspired proxy-influence signal (Tier 2, optional ``influence`` extra).

Estimates each trajectory's **influence** on a downstream objective — the principled,
flagship notion behind CUPID — and keeps the helpful trajectories while flagging the
harmful/atypical ones. Two estimators (configurable ``mode``):

* ``"tracin"`` (default): TracInCP-style influence on the whole-dataset (consensus) BC
  objective — ``sum over training checkpoints of <grad of trajectory i's BC loss, grad of
  the full-dataset BC loss>``. Signed: positive aligns with the consensus (keep), negative
  opposes it — a contradictory demonstration (remove). ``higher_is_better=True``.
  Accumulating over checkpoints (rather than a single converged model, where well-fit
  gradients vanish) is what makes the estimate robust.
* ``"self_influence"``: ``sum over checkpoints of ||grad of trajectory i's BC loss||^2`` — an
  outlier/memorisation detector flagging atypical trajectories. ``higher_is_better=False``
  (more atypical ⇒ less keepable).

**Honesty note.** Faithful CUPID attributes influence on a *policy's* downstream task
performance, which needs the policy training loop (the experiment harness). This v1 measures
influence on a **proxy** behavior-cloning model (state → action MLP) trained on the data, so
"influence on downstream performance" becomes "influence on the behavior-cloning objective".
The full policy-attribution version lands with the harness.

Approximation choices (v1): the objective is the whole-dataset BC loss (each trajectory
contributes mildly to its own target gradient — a held-out / leave-one-out objective is a
deferred refinement); raw summed influence (the curator normalises per-signal anyway);
homogeneous state/action dimensions required (mixed-dim trajectories are scored as skips).
Tier 2 cost: one backward pass per trajectory per checkpoint. torch is imported lazily
(optional ``influence`` extra); deterministic on CPU given the seed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.torch_utils import resolve_device
from robocurate.trajectory import Array, FeatureRole, Trajectory

F64 = npt.NDArray[np.float64]

DEFAULT_HIDDEN_DIM = 32
DEFAULT_EPOCHS = 200
_MODEL_KEY = "influence"
_MODES = ("tracin", "self_influence")


def _require_torch() -> Any:
    """Import and return torch, with an actionable error if the extra is not installed."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the 'cupid' signal requires PyTorch, which is an optional dependency. Install "
            "it with `uv pip install 'robocurate[influence]'` (or `robocurate[all]`)."
        ) from exc
    return torch


class Cupid:
    """CUPID-inspired proxy-influence signal over a behavior-cloning proxy model.

    Args:
        mode: ``"tracin"`` (influence on the validation objective) or ``"self_influence"``
            (gradient-magnitude outlier detector).
        hidden_dim: Hidden width of the proxy BC MLP.
        epochs: Proxy training epochs.
        name: Override the signal name.
    """

    def __init__(
        self,
        *,
        mode: str = "tracin",
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        epochs: int = DEFAULT_EPOCHS,
        device: str | None = None,
        name: str = "cupid",
    ) -> None:
        _require_torch()  # fail fast with a clear message if the extra is missing
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.device = device  # None => auto (cuda if available, else cpu)
        higher_is_better = mode == "tracin"
        self._higher_is_better = higher_is_better
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER2_GPU_HEAVY,
            requires=frozenset(),  # runs on CPU; the real prerequisite is state+action data
            produces_per_transition=False,
            deterministic=True,
            description=(
                f"CUPID-inspired proxy influence ({mode}) on a behavior-cloning model "
                f"({'higher is more helpful' if higher_is_better else 'higher is more atypical'})."
            ),
        )

    # -- fit: train the proxy and compute every trajectory's influence ---------------

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        torch = _require_torch()
        samples: list[tuple[str, Array, Array]] = []
        excluded: dict[str, str] = {}
        state_dim: int | None = None
        action_dim: int | None = None
        for traj in trajectories:
            pair = _bc_pair(traj)
            if pair is None:
                excluded[traj.meta.fingerprint] = "trajectory lacks state and/or action features"
                continue
            state, action = pair
            if state_dim is None:
                state_dim, action_dim = state.shape[1], action.shape[1]
            if state.shape[1] != state_dim or action.shape[1] != action_dim:
                excluded[traj.meta.fingerprint] = (
                    "state/action dimensions differ from the dataset's proxy model"
                )
                continue
            samples.append((traj.meta.fingerprint, state, action))

        reason = _untrainable_reason(samples)
        if reason is not None or state_dim is None or action_dim is None:
            ctx.cache.put(
                _MODEL_KEY,
                {
                    "trained": False,
                    "reason": reason or "no usable trajectories",
                    "excluded": excluded,
                },
            )
            return

        samples.sort(key=lambda s: s[0])  # stable order for deterministic training
        mean, std = _state_norm(samples)
        device = resolve_device(self.device)
        model, checkpoints = _train_bc(
            samples,
            mean,
            std,
            state_dim,
            action_dim,
            self.hidden_dim,
            self.epochs,
            ctx.seed,
            device,
        )

        influence = self._compute_influence(torch, model, checkpoints, samples, mean, std, device)
        ctx.cache.put(
            _MODEL_KEY,
            {"trained": True, "influence": influence, "excluded": excluded, "mode": self.mode},
        )

    def _compute_influence(
        self,
        torch: Any,
        model: Any,
        checkpoints: list[Any],
        samples: list[tuple[str, Array, Array]],
        mean: F64,
        std: F64,
        device: str,
    ) -> dict[str, float]:
        # TracInCP: accumulate influence across training checkpoints rather than relying on a
        # single converged model (where well-fit trajectories have near-zero gradients).
        #   tracin: leave-one-out alignment with the rest of the data,
        #     influence_i = sum_t <g_i, G_t - g_i>  where G_t is the summed per-trajectory
        #     gradient at checkpoint t. Excluding self prevents a loud, large-gradient
        #     outlier from hijacking the objective and looking aligned with it; a demo that
        #     opposes the consensus of the others accumulates negative influence.
        #   self_influence: sum_t ||g_i||^2 (gradient magnitude, an outlier detector).
        influence: dict[str, float] = {fp: 0.0 for fp, _, _ in samples}
        for state_dict in checkpoints:
            model.load_state_dict(state_dict)
            grads = {
                fp: _grad_vector(torch, model, [(fp, s, a)], mean, std, device)
                for fp, s, a in samples
            }
            if self.mode == "self_influence":
                for fp, g_i in grads.items():
                    influence[fp] += float(torch.dot(g_i, g_i))
            else:
                g_sum = torch.stack(list(grads.values())).sum(dim=0)
                for fp, g_i in grads.items():
                    influence[fp] += float(torch.dot(g_i, g_sum) - torch.dot(g_i, g_i))
        return influence

    # -- score -----------------------------------------------------------------------

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        model = ctx.cache.get(_MODEL_KEY) if ctx.cache.has(_MODEL_KEY) else None
        return [self._score_one(traj, model) for traj in batch]

    def _score_one(self, traj: Trajectory, model: dict[str, Any] | None) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        if model is None or not model.get("trained"):
            excluded = (model or {}).get("excluded", {})
            reason = (
                excluded.get(fingerprint)
                or ((model or {}).get("reason") if model else None)
                or "proxy model was not trained"
            )
            return TrajectoryScore.skip(
                self.spec.name, fingerprint, reason=reason, higher_is_better=self._higher_is_better
            )
        if fingerprint in model.get("excluded", {}):
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason=model["excluded"][fingerprint],
                higher_is_better=self._higher_is_better,
            )
        if fingerprint not in model["influence"]:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="trajectory was not part of the fitted dataset",
                higher_is_better=self._higher_is_better,
            )

        value = float(model["influence"][fingerprint])
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=value,
            higher_is_better=self._higher_is_better,
            diagnostics={"mode": self.mode, "influence": value},
        )


# -- BC proxy helpers ----------------------------------------------------------------


def _bc_pair(traj: Trajectory) -> tuple[Array, Array] | None:
    """Return ``(state (T, Ds), action (T, Da))`` for behavior cloning, or ``None``."""
    action = traj.actions()
    proprio = traj.select_roles(FeatureRole.PROPRIO, FeatureRole.STATE)
    if action is None or not proprio:
        return None
    states = [np.asarray(v, dtype=np.float64).reshape(v.shape[0], -1) for v in proprio.values()]
    state = np.concatenate(states, axis=1)
    act = np.asarray(action, dtype=np.float64).reshape(action.shape[0], -1)
    if state.shape[0] != act.shape[0] or state.shape[0] == 0:
        return None
    return state, act


def _state_norm(train: list[tuple[str, Array, Array]]) -> tuple[F64, F64]:
    allstate: F64 = np.vstack([s for _, s, _ in train]).astype(np.float64)
    mean: F64 = allstate.mean(axis=0)
    std: F64 = np.where(allstate.std(axis=0) > 0.0, allstate.std(axis=0), 1.0)
    return mean, std


def _train_bc(
    train: list[tuple[str, Array, Array]],
    mean: F64,
    std: F64,
    state_dim: int,
    action_dim: int,
    hidden: int,
    epochs: int,
    seed: int,
    device: str,
) -> tuple[Any, list[Any]]:
    torch = _require_torch()
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(state_dim, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, action_dim),
    ).to(device)
    x = _standardize(train, mean, std)
    y = np.vstack([a for _, _, a in train])
    xt = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32).to(device)
    yt = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    # Snapshot a handful of checkpoints across training for TracInCP. Early/mid checkpoints
    # carry the informative gradients (they vanish near convergence for well-fit data).
    snapshot_at = {max(1, round(frac * epochs)) for frac in (0.1, 0.25, 0.5, 0.75)}
    checkpoints: list[Any] = []
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()
        if epoch in snapshot_at:
            checkpoints.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    model.eval()
    return model, checkpoints


def _standardize(samples: list[tuple[str, Array, Array]], mean: F64, std: F64) -> F64:
    """Stack and z-standardize the state rows of ``samples`` into one ``(N, Ds)`` matrix."""
    rows: F64 = np.vstack([np.asarray(s, dtype=np.float64) for _, s, _ in samples])
    return (rows - mean) / std


def _grad_vector(
    torch: Any,
    model: Any,
    samples: list[tuple[str, Array, Array]],
    mean: F64,
    std: F64,
    device: str,
) -> Any:
    """Flattened gradient of the mean BC loss over ``samples`` w.r.t. the model parameters."""
    x = _standardize(samples, mean, std)
    y = np.vstack([a for _, _, a in samples])
    xt = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32).to(device)
    yt = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32).to(device)
    loss_fn = torch.nn.MSELoss()
    model.zero_grad()
    loss = loss_fn(model(xt), yt)
    loss.backward()
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def _untrainable_reason(samples: list[tuple[str, Array, Array]]) -> str | None:
    if len(samples) < 3:
        return f"too few usable trajectories ({len(samples)}) to fit a proxy influence model"
    return None


__all__ = ["Cupid"]
