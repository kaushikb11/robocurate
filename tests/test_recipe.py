"""Shareable recipes: save a Curator's config, reload it, reproduce identical decisions.

A recipe captures the resolved curation config (combiner + weights, budget, selection, gate,
seed). Loading it rebuilds a ready-to-run Curator whose signals come from the combiner's
weight keys. Because the selection path is deterministic (Invariant 3), a loaded recipe run on
the same source must produce byte-identical decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

from robocurate import signals
from robocurate.adapters import LeRobotReader
from robocurate.curator import (
    Budget,
    CurationConfig,
    Curator,
    GateConfig,
    SelectionMode,
    WeightedSum,
)
from robocurate.recipe import RECIPE_VERSION, load_recipe, save_recipe
from tests.synthetic import write_synthetic_lerobot_dataset


def _original_curator(seed: int = 5) -> Curator:
    return Curator(
        [signals.get("jerk"), signals.get("path_efficiency")],
        combiner=WeightedSum(weights={"jerk": 1.0, "path_efficiency": 2.0}),
        budget=Budget.fraction(0.5),
        seed=seed,
        selection=SelectionMode.TOP_K,
    )


def _decision_tuples(curator: Curator, src: Path) -> list[tuple[int, str, bool, str]]:
    result = curator.run(LeRobotReader(src))
    return [(d.episode_index, d.fingerprint, d.kept, d.reason) for d in result.decisions]


def test_recipe_roundtrip_reproduces_identical_decisions(tmp_path: Path) -> None:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=8)
    original = _original_curator()

    recipe_path = tmp_path / "recipe.json"
    save_recipe(original, recipe_path)
    reloaded = load_recipe(recipe_path)

    before = _decision_tuples(original, src)
    after = _decision_tuples(reloaded, src)
    # Byte-identical decisions: same kept/removed, same per-episode reason text (Invariant 3).
    assert before == after


def test_recipe_is_json_with_version_envelope(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.json"
    save_recipe(_original_curator(), recipe_path)

    document = json.loads(recipe_path.read_text())
    assert document["recipe_version"] == RECIPE_VERSION
    assert "code_version" in document
    cfg = document["config"]
    assert cfg["seed"] == 5
    assert cfg["combiner"]["weights"] == {"jerk": 1.0, "path_efficiency": 2.0}
    assert cfg["budget"] == {"kind": "fraction", "value": 0.5}


def test_recipe_restores_budget_selection_gate_seed(tmp_path: Path) -> None:
    curator = Curator(
        [signals.get("jerk"), signals.get("path_efficiency")],
        combiner=WeightedSum(weights={"jerk": 1.0, "path_efficiency": 1.0}),
        budget=Budget.count(3),
        seed=9,
        selection=SelectionMode.TOP_K,
        gate=GateConfig(signal="jerk", reject_above=1e9),  # never trips on toy data
    )
    recipe_path = tmp_path / "recipe.json"
    save_recipe(curator, recipe_path)
    reloaded = load_recipe(recipe_path)

    assert reloaded.seed == 9
    assert reloaded.budget == Budget.count(3)
    assert reloaded.selection is SelectionMode.TOP_K
    assert reloaded.gate == GateConfig(signal="jerk", reject_above=1e9)
    assert {s.spec.name for s in reloaded.signals} == {"jerk", "path_efficiency"}


def test_curation_config_from_dict_roundtrips() -> None:
    config = CurationConfig(
        combiner_dict={"name": "weighted_sum", "weights": {"jerk": 2.0}},
        budget=Budget.fraction(0.25),
        seed=3,
        emit_baseline=False,
        selection=SelectionMode.GREEDY_DEDUP.value,
        gate_dict={"signal": "jerk", "reject_above": 1.0, "reject_below": None},
        batch_size=16,
    )
    assert CurationConfig.from_dict(config.to_dict()) == config


def test_from_dict_back_compat_missing_optional_keys() -> None:
    # A minimal dict (older/partial recipe) still reconstructs with sensible defaults.
    config = CurationConfig.from_dict({"combiner": {"name": "weighted_sum", "weights": {}}})
    assert config.budget is None
    assert config.seed == 0
    assert config.selection == SelectionMode.TOP_K.value
    assert config.gate_dict is None
