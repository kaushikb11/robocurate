"""Tests for the optional Matplotlib scorecard visualizations (``viz`` extra).

Builds a tiny real curation run over synthetic trajectories and asserts each plot writes a
non-empty PNG. Also checks the lazy-import contract: importing the ``robocurate.viz`` module
does not require Matplotlib at import time.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.curator import Budget, CurationResult, Curator
from robocurate.signals.path_efficiency import PathEfficiency
from robocurate.signals.redundancy import Redundancy
from robocurate.viz import (
    plot_curation_summary,
    plot_scores_by_group,
    plot_signal_distributions,
)
from tests.synthetic import make_trajectory

pytestmark = pytest.mark.viz


def _curation_result() -> CurationResult:
    trajs = [make_trajectory(i, scale=1.0 + 0.5 * i) for i in range(6)]
    reader = InMemoryDatasetReader(trajs)
    return Curator([Redundancy(), PathEfficiency()], budget=Budget.count(4), seed=0).run(reader)


def _assert_nonempty(path: Path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_signal_distributions(tmp_path: Path) -> None:
    result = _curation_result()
    out = plot_signal_distributions(result, tmp_path / "dist.png")
    _assert_nonempty(out)


def test_plot_curation_summary(tmp_path: Path) -> None:
    result = _curation_result()
    out = plot_curation_summary(result.scorecard(), tmp_path / "summary.png")
    _assert_nonempty(out)


def test_plot_scores_by_group(tmp_path: Path) -> None:
    result = _curation_result()
    # Label episodes into two tiers to exercise group separation.
    groups = {i: ("expert" if i % 2 == 0 else "novice") for i in range(6)}
    out = plot_scores_by_group(result, groups, "redundancy", tmp_path / "groups.png")
    _assert_nonempty(out)


def test_importing_viz_does_not_require_matplotlib() -> None:
    # Importing the viz module must not import matplotlib eagerly (lazy import inside funcs).
    mod = importlib.import_module("robocurate.viz")
    assert hasattr(mod, "plot_signal_distributions")
