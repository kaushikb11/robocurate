"""Demo-SCORE-inspired learned quality classifier (Tier 1, optional ``demo-score`` extra).

Trains a small neural classifier to predict a trajectory's **quality** and scores each
trajectory by its predicted probability of being a good (successful) demonstration. A
demonstration labelled "success" that the model confidently predicts as a failure is
suspicious — likely mislabelled or low-quality — and ranks low, so curation drops it.

**Honesty note.** Faithful Demo-SCORE (Chen et al.) trains on a *policy's* progress
predictions across training checkpoints — it needs policy rollouts that a static dataset
does not contain. That full method belongs with the experiment harness (where policies
exist). This v1 is the feasible analogue: a learned quality classifier trained on the
dataset's own :class:`~robocurate.trajectory.SuccessLabel`\\ s. To avoid the classifier
trivially memorising labels, each labelled trajectory is scored with an **out-of-fold**
prediction (k-fold cross-validation); unlabelled trajectories are scored with a final model
trained on all labelled data.

Dependencies: this is the first signal behind an optional extra. The module imports cleanly
without PyTorch; torch is imported lazily, and constructing :class:`DemoScore` without it
raises a clear "install ``robocurate[demo-score]``" error. ``value`` is the predicted
P(quality), ``higher_is_better=True``. CPU-runnable; deterministic given the seed.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from robocurate.signals.base import (
    CostTier,
    SignalContext,
    SignalSpec,
    TrajectoryScore,
)
from robocurate.signals.redundancy import statistical_embedding
from robocurate.torch_utils import resolve_device
from robocurate.trajectory import Array, Trajectory

if TYPE_CHECKING:
    from robocurate.signals.redundancy import EmbeddingFn

DEFAULT_HIDDEN_DIM = 16
DEFAULT_EPOCHS = 150
DEFAULT_N_FOLDS = 5
_MODEL_KEY = "model"


def _require_torch() -> Any:
    """Import and return torch, with an actionable error if the extra is not installed."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the 'demo_score' signal requires PyTorch, which is an optional dependency. "
            "Install it with `uv pip install 'robocurate[demo-score]'` (or "
            "`robocurate[all]`)."
        ) from exc
    return torch


class DemoScore:
    """Learned quality classifier over trajectory embeddings, trained on success labels.

    Args:
        embedding: Maps a trajectory to a feature vector (defaults to
            :func:`~robocurate.signals.redundancy.statistical_embedding`). Pluggable, so a
            richer/learned embedding drops in without a core change.
        hidden_dim: Hidden width of the MLP classifier.
        epochs: Training epochs per model.
        n_folds: Cross-validation folds for the out-of-fold predictions.
        name: Override the signal name.
    """

    def __init__(
        self,
        *,
        embedding: EmbeddingFn | None = None,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        epochs: int = DEFAULT_EPOCHS,
        n_folds: int = DEFAULT_N_FOLDS,
        device: str | None = None,
        name: str = "demo_score",
    ) -> None:
        _require_torch()  # fail fast with a clear message if the extra is missing
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        self.embedding: EmbeddingFn = embedding if embedding is not None else statistical_embedding
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.n_folds = n_folds
        self.device = device  # None => auto (cuda if available, else cpu)
        self.spec = SignalSpec(
            name=name,
            version="0.1.0",
            cost_tier=CostTier.TIER1_GPU,
            requires=frozenset(),  # runs on CPU; the real prerequisite is labelled data
            produces_per_transition=False,
            deterministic=True,
            description=(
                "Demo-SCORE-inspired learned quality classifier; predicts P(good "
                "demonstration) from a trajectory embedding (higher is better)."
            ),
        )

    # -- fit: train the classifier(s) ------------------------------------------------

    def fit(self, trajectories: Iterable[Trajectory], ctx: SignalContext) -> None:
        embeddings: list[Array] = []
        labels: list[int] = []
        fingerprints: list[str] = []
        for traj in trajectories:
            emb = self.embedding(traj)
            label = traj.success()
            if emb is None or label is None or label.value is None:
                continue
            embeddings.append(np.asarray(emb, dtype=np.float64).reshape(-1))
            labels.append(1 if label.value else 0)
            fingerprints.append(traj.meta.fingerprint)

        reason = _untrainable_reason(labels, self.n_folds)
        if reason is not None:
            ctx.cache.put(_MODEL_KEY, {"trained": False, "reason": reason})
            return

        x = np.vstack(embeddings).astype(np.float64)
        y = np.asarray(labels, dtype=np.float64)
        mean = x.mean(axis=0)
        std = np.where(x.std(axis=0) > 0.0, x.std(axis=0), 1.0)
        x_std = (x - mean) / std

        device = resolve_device(self.device)
        oof = self._out_of_fold_predictions(x_std, y, fingerprints, seed=ctx.seed, device=device)
        final_model = _train_mlp(
            x_std, y, self.hidden_dim, self.epochs, seed=ctx.seed, device=device
        )
        ctx.cache.put(
            _MODEL_KEY,
            {
                "trained": True,
                "model": final_model,
                "mean": mean,
                "std": std,
                "oof": oof,
                "device": device,
                "embedding_dim": int(x.shape[1]),
            },
        )

    def _out_of_fold_predictions(
        self, x_std: Array, y: Array, fingerprints: list[str], *, seed: int, device: str
    ) -> dict[str, float]:
        # Deterministic fold assignment: a seeded permutation of a stable (sorted) order.
        order = np.argsort(fingerprints, kind="stable")
        rng = np.random.default_rng(seed)
        shuffled = order[rng.permutation(len(order))]
        folds = np.array_split(shuffled, min(self.n_folds, len(order)))

        oof: dict[str, float] = {}
        for fold in folds:
            if fold.size == 0:
                continue
            mask = np.ones(len(order), dtype=bool)
            mask[fold] = False
            model = _train_mlp(
                x_std[mask], y[mask], self.hidden_dim, self.epochs, seed=seed, device=device
            )
            preds = _predict_proba(model, x_std[fold], device)
            for idx, pred in zip(fold.tolist(), preds.tolist(), strict=True):
                oof[fingerprints[idx]] = float(pred)
        return oof

    # -- score -----------------------------------------------------------------------

    def score(self, batch: Sequence[Trajectory], ctx: SignalContext) -> list[TrajectoryScore]:
        model = ctx.cache.get(_MODEL_KEY) if ctx.cache.has(_MODEL_KEY) else None
        return [self._score_one(traj, model) for traj in batch]

    def _score_one(self, traj: Trajectory, model: dict[str, Any] | None) -> TrajectoryScore:
        fingerprint = traj.meta.fingerprint
        if model is None or not model.get("trained"):
            reason = (
                model.get("reason", "classifier was not trained")
                if model
                else "classifier was not trained"
            )
            return TrajectoryScore.skip(
                self.spec.name, fingerprint, reason=reason, higher_is_better=True
            )

        emb = self.embedding(traj)
        if emb is None:
            return TrajectoryScore.skip(
                self.spec.name,
                fingerprint,
                reason="trajectory has no embeddable features",
                higher_is_better=True,
            )

        used_oof = fingerprint in model["oof"]
        if used_oof:
            p_good = float(model["oof"][fingerprint])
        else:
            vec = (np.asarray(emb, dtype=np.float64).reshape(1, -1) - model["mean"]) / model["std"]
            p_good = float(_predict_proba(model["model"], vec, model.get("device", "cpu"))[0])

        label = traj.success()
        label_value = label.value if label is not None else None
        suspected_mislabel = label_value is True and p_good < 0.5
        return TrajectoryScore(
            signal=self.spec.name,
            trajectory_fingerprint=fingerprint,
            value=p_good,
            higher_is_better=True,
            diagnostics={
                "p_good": p_good,
                "label": label_value,
                "used_out_of_fold": bool(used_oof),
                "suspected_mislabel": bool(suspected_mislabel),
            },
        )


def _untrainable_reason(labels: list[int], n_folds: int) -> str | None:
    """Return why the classifier cannot be trained, or ``None`` if it can."""
    if len(labels) < n_folds:
        return f"too few labelled trajectories ({len(labels)}) to train a {n_folds}-fold classifier"
    if len(set(labels)) < 2:
        return "labelled data has only one class (need both success and failure examples)"
    return None


def _train_mlp(x: Array, y: Array, hidden: int, epochs: int, *, seed: int, device: str) -> Any:
    """Train a tiny MLP binary classifier deterministically; return the eval-mode model."""
    torch = _require_torch()
    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32).to(device)
    yt = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32).reshape(-1, 1).to(device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _predict_proba(model: Any, x: Array, device: str = "cpu") -> Array:
    """Return P(class=1) for each row of ``x``."""
    torch = _require_torch()
    with torch.no_grad():
        tensor = torch.tensor(np.ascontiguousarray(x), dtype=torch.float32).to(device)
        probs: Array = torch.sigmoid(model(tensor)).reshape(-1).cpu().numpy().astype(np.float64)
    return probs


__all__ = ["DemoScore"]
