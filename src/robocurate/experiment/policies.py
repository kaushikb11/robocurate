"""A real behavior-cloning policy (Tier: optional ``policy`` extra).

The first *real* component of the experiment harness: :class:`BCPolicy` trains an actual
torch MLP to predict actions from observations (behavior cloning) on a dataset's
``(state, action)`` pairs, and the resulting :class:`BCTrainedPolicy` acts in an
:class:`~robocurate.experiment.policy.Environment`. It implements the same
:class:`~robocurate.experiment.policy.Policy` / ``TrainedPolicy`` protocols as the fakes, so
it drops straight into the runner — turning the harness from "fake competence heuristic" into
"a real policy whose rollout success depends on what it was trained on".

A larger architecture (Diffusion Policy, an ACT/VLA fine-tune) implements the same protocol
later; BC is the simple, faithful, CPU-runnable starting point. torch is imported lazily, so
the module loads without the ``policy`` extra and constructing :class:`BCPolicy` without it
raises a clear install error. Deterministic given the seed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from robocurate.torch_utils import resolve_device
from robocurate.trajectory import Array, FeatureRole

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader
    from robocurate.experiment.policy import Observation
    from robocurate.trajectory import Trajectory

F64 = npt.NDArray[np.float64]

DEFAULT_HIDDEN_DIM = 64
DEFAULT_EPOCHS = 300
DEFAULT_LR = 0.01


def _require_torch() -> Any:
    """Import and return torch, with an actionable error if the extra is not installed."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "BCPolicy requires PyTorch, which is an optional dependency. Install it with "
            "`uv pip install 'robocurate[policy]'` (or `robocurate[all]`)."
        ) from exc
    return torch


class BCTrainedPolicy:
    """A trained BC policy: standardizes the observation and maps it to an action via the MLP."""

    def __init__(self, model: Any, mean: F64, std: F64, *, device: str = "cpu") -> None:
        self._model = model
        self._mean = mean
        self._std = std
        self._device = device

    def act(self, observation: Observation, *, seed: int) -> Array:
        torch = _require_torch()
        x = _concat_observation(observation)
        z = (x - self._mean) / self._std
        with torch.no_grad():
            tensor = (
                torch.tensor(np.ascontiguousarray(z), dtype=torch.float32)
                .reshape(1, -1)
                .to(self._device)
            )
            action: Array = self._model(tensor).reshape(-1).cpu().numpy().astype(np.float64)
        return action

    def predict(self, states: F64) -> F64:
        """Predict actions for a batch of ``(N, Ds)`` states (standardized with training stats).

        Used by the held-out-loss evaluator: a single batched forward over a whole validation
        split, equivalent to calling :meth:`act` per row but far faster.
        """
        torch = _require_torch()
        z = (np.asarray(states, dtype=np.float64) - self._mean) / self._std
        with torch.no_grad():
            tensor = torch.tensor(np.ascontiguousarray(z), dtype=torch.float32).to(self._device)
            out: F64 = self._model(tensor).cpu().numpy().astype(np.float64)
        return out


class BCPolicy:
    """Behavior-cloning policy: trains an MLP to predict actions from observations.

    Args:
        hidden_dim: Hidden width of the MLP.
        epochs: Training epochs.
        lr: Adam learning rate.
        name: Policy name (recorded in experiment arms).
    """

    def __init__(
        self,
        *,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        epochs: int = DEFAULT_EPOCHS,
        lr: float = DEFAULT_LR,
        device: str | None = None,
        name: str = "bc",
    ) -> None:
        _require_torch()  # fail fast with a clear message if the extra is missing
        self.name = name
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = device  # None => auto (cuda if available, else cpu)

    def train(self, train_set: DatasetReader, *, seed: int) -> BCTrainedPolicy:
        torch = _require_torch()
        device = resolve_device(self.device)
        states: list[F64] = []
        actions: list[F64] = []
        for traj in train_set:
            pair = _bc_pair(traj)
            if pair is None:
                continue
            state, action = pair
            states.append(state)
            actions.append(action)
        if not states:
            raise ValueError("BCPolicy.train: no usable (state, action) pairs in the training set")

        x = np.vstack(states)
        y = np.vstack(actions)
        mean = x.mean(axis=0)
        std = np.where(x.std(axis=0) > 0.0, x.std(axis=0), 1.0)
        x_std = (x - mean) / std

        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], self.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_dim, y.shape[1]),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        xt = torch.tensor(np.ascontiguousarray(x_std), dtype=torch.float32).to(device)
        yt = torch.tensor(np.ascontiguousarray(y), dtype=torch.float32).to(device)
        model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(xt), yt)
            loss.backward()
            optimizer.step()
        model.eval()
        return BCTrainedPolicy(model, mean, std, device=device)


def _bc_pair(traj: Trajectory) -> tuple[F64, F64] | None:
    """Return per-step ``(state (T, Ds), action (T, Da))`` for behavior cloning, or ``None``."""
    action = traj.actions()
    proprio = traj.select_roles(FeatureRole.PROPRIO, FeatureRole.STATE)
    if action is None or not proprio:
        return None
    states = [
        np.asarray(v, dtype=np.float64).reshape(v.shape[0], -1) for _, v in sorted(proprio.items())
    ]
    state = np.concatenate(states, axis=1)
    act = np.asarray(action, dtype=np.float64).reshape(action.shape[0], -1)
    if state.shape[0] != act.shape[0] or state.shape[0] == 0:
        return None
    return state, act


def _concat_observation(observation: Mapping[str, Array]) -> F64:
    """Concatenate observation values (sorted by key) into one vector, matching training."""
    parts = [np.asarray(observation[k], dtype=np.float64).reshape(-1) for k in sorted(observation)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


__all__ = ["BCPolicy", "BCTrainedPolicy"]
