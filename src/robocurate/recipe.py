"""Shareable curation recipes — save a :class:`Curator`'s config, reload it as a Curator.

A *recipe* is the fully-resolved, JSON-serializable configuration of a curation run: the
combiner and its weights, the budget, the selection mode, the validity gate, and the master
seed. Saving a recipe lets a user hand someone "the exact knobs I curated with", and loading
it reconstructs a ready-to-run :class:`~robocurate.curator.Curator` whose signals are rebuilt
from the combiner's weight keys via the signal registry.

Because a recipe captures the resolved :class:`~robocurate.curator.CurationConfig` (including
the seed) and the selection path is deterministic (Invariant 3), running a loaded recipe on
the same source dataset reproduces **byte-identical** selection decisions.

**Format: JSON.** RoboCurate's core has no YAML dependency (it must install clean on a no-GPU
laptop with a minimal dependency set), so recipes are plain JSON rather than YAML — adding a
runtime dependency just for recipe serialization is not worth it. The on-disk shape is exactly
``CurationConfig.to_dict()`` wrapped with a small envelope (a ``recipe_version`` and the
package ``code_version`` for provenance).

A recipe deliberately does **not** capture signal *hyper-parameters* beyond the signal name:
v1 signals are reconstructed from the registry with their defaults, which is what the combiner
weight keys name. Per-signal params are a documented follow-up, not silently dropped without
note here.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from robocurate.curator import (
    Budget,
    BudgetKind,
    CurationConfig,
    Curator,
    GateConfig,
    SelectionMode,
    WeightedSum,
)
from robocurate.manifest import code_version
from robocurate.signals import get as get_signal

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

RECIPE_VERSION = "1"


def _resolve_config(curator_or_config: Curator | CurationConfig) -> CurationConfig:
    """Return the resolved :class:`CurationConfig` for a :class:`Curator` or a config."""
    if isinstance(curator_or_config, CurationConfig):
        return curator_or_config
    # Reuse the curator's own snapshot logic so a saved recipe matches what the run records.
    return curator_or_config._config_snapshot()


def save_recipe(curator_or_config: Curator | CurationConfig, path: str | Path) -> None:
    """Write a shareable JSON recipe for ``curator_or_config`` to ``path``.

    Accepts either a live :class:`Curator` (its resolved config is snapshotted) or an
    already-resolved :class:`CurationConfig`.
    """
    from pathlib import Path as _Path

    config = _resolve_config(curator_or_config)
    document = {
        "recipe_version": RECIPE_VERSION,
        "code_version": code_version(),
        "config": config.to_dict(),
    }
    _Path(path).write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")


def load_recipe(path: str | Path) -> Curator:
    """Load a JSON recipe from ``path`` and reconstruct a ready-to-run :class:`Curator`.

    Signals are rebuilt from the combiner's weight keys via :func:`robocurate.signals.get`,
    and the budget, selection mode, validity gate, and seed are restored from the recipe.
    Running the returned curator on the same source dataset reproduces byte-identical
    decisions (Invariant 3).
    """
    from pathlib import Path as _Path

    document = json.loads(_Path(path).read_text(encoding="utf-8"))
    config = CurationConfig.from_dict(document["config"])
    return curator_from_config(config)


def curator_from_config(config: CurationConfig) -> Curator:
    """Reconstruct a live :class:`Curator` from a resolved :class:`CurationConfig`.

    The combiner is rebuilt from its serialized dict, signals are instantiated from the
    combiner's weight keys (their registry defaults), and budget / selection / gate / seed are
    restored verbatim.
    """
    combiner = _combiner_from_dict(config.combiner_dict)
    signal_names = _signal_names(config.combiner_dict)
    signals = [get_signal(name) for name in signal_names]
    return Curator(
        signals,
        combiner=combiner,
        budget=_budget_from_dict(config.budget),
        seed=config.seed,
        emit_baseline=config.emit_baseline,
        selection=SelectionMode(config.selection),
        gate=_gate_from_dict(config.gate_dict),
        batch_size=config.batch_size,
        drop_episode_indices=config.drop_episode_indices,
        keep_episode_indices=config.keep_episode_indices,
    )


def _signal_names(combiner_dict: Mapping[str, Any]) -> list[str]:
    """The signal names a recipe reconstructs: the combiner's weight keys, in sorted order."""
    weights = combiner_dict.get("weights", {})
    return sorted(weights)


def _combiner_from_dict(combiner_dict: Mapping[str, Any]) -> WeightedSum:
    name = combiner_dict.get("name", "weighted_sum")
    if name != "weighted_sum":
        raise ValueError(
            f"unknown combiner {name!r} in recipe; v1 recipes support 'weighted_sum' only"
        )
    weights = {str(k): float(v) for k, v in combiner_dict.get("weights", {}).items()}
    return WeightedSum(weights=weights)


def _budget_from_dict(budget: Budget | None) -> Budget | None:
    # CurationConfig.from_dict already reconstructs Budget; passthrough keeps a single source
    # of truth and tolerates a None (whole-dataset) budget.
    if budget is None:
        return None
    if not isinstance(budget, Budget):  # defensive: accept a raw dict too
        return Budget(kind=BudgetKind(budget["kind"]), value=float(budget["value"]))
    return budget


def _gate_from_dict(gate_dict: Mapping[str, Any] | None) -> GateConfig | None:
    if gate_dict is None:
        return None
    return GateConfig(
        signal=str(gate_dict["signal"]),
        reject_above=gate_dict.get("reject_above"),
        reject_below=gate_dict.get("reject_below"),
    )


__all__ = [
    "RECIPE_VERSION",
    "curator_from_config",
    "load_recipe",
    "save_recipe",
]
