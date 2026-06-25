"""Contract tests for the standalone HTML curation report (:meth:`Scorecard.to_html`).

Two paths are covered explicitly:

* The report renders to a self-contained HTML document from a real curation run *without*
  Matplotlib (the ``viz`` extra is lazily imported and degrades to tables only).
* When the ``viz`` extra is present, the kept-vs-removed summary plot is base64-embedded as
  an ``<img>`` so the file stays self-contained.

A directly-constructed scorecard with an attached :class:`EffectReport` exercises the
downstream-effect section, including its CI bounds and per-task breakdown (invariant 6).
"""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from pathlib import Path

import pytest

from robocurate.adapters import LeRobotReader
from robocurate.curator import Budget, Curator, WeightedSum
from robocurate.manifest import BaselineRecord
from robocurate.metadata import DatasetFingerprint
from robocurate.scorecard import (
    EffectReport,
    QualitySummary,
    Scorecard,
    SignalReport,
    TrajectoryFlag,
)
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


def _run(tmp_path: Path) -> Scorecard:
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = Curator(
        [FakeActionMagnitudeSignal()],
        combiner=WeightedSum(),
        budget=Budget.fraction(0.5),
        seed=5,
    ).run(LeRobotReader(src))
    return result.scorecard()


def _card_with_effects() -> Scorecard:
    """A hand-built scorecard carrying an effect report and a special character in the id."""
    dataset = DatasetFingerprint(
        dataset_id="acme/teleop <v2> & more",
        source_format="lerobot_v2.1",
        content_hash="abc123",
        num_episodes=3,
    )
    flags = (
        TrajectoryFlag(0, "fp0", kept=True, reason="kept", signal_values={"sig": 0.1}),
        TrajectoryFlag(
            1, "fp1", kept=False, reason="low path efficiency", signal_values={"sig": 0.9}
        ),
        TrajectoryFlag(2, "fp2", kept=True, reason="kept", signal_values={"sig": 0.2}),
    )
    return Scorecard(
        schema_version="1",
        dataset=dataset,
        summary=QualitySummary(num_episodes=3, num_kept=2, num_removed=1),
        per_signal=(
            SignalReport(
                name="path_efficiency",
                description="straight-line vs travelled",
                higher_is_better=True,
                num_scored=3,
                num_skipped=0,
                minimum=0.1,
                median=0.2,
                maximum=0.9,
            ),
        ),
        flags=flags,
        baseline=BaselineRecord(method="equal_n_random", seed=5, n=2, kept_episode_indices=(0, 2)),
        effects=EffectReport(
            metric="success_rate",
            effect=0.12,
            ci_low=0.03,
            ci_high=0.21,
            per_task={"can": 0.18, "square": 0.06},
        ),
    )


def test_to_html_is_a_self_contained_document(tmp_path: Path) -> None:
    html = _run(tmp_path).to_html()
    assert html  # non-empty
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert "</html>" in html
    # No external assets: the styling is inlined.
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html


def test_to_html_contains_key_fields(tmp_path: Path) -> None:
    card = _run(tmp_path)
    html = card.to_html()

    assert "RoboCurate curation scorecard" in html
    assert card.dataset.dataset_id in html
    assert "fake_action_magnitude" in html
    assert "Quality summary" in html
    assert "Equal-N random baseline" in html
    assert "Removed episodes" in html
    # Every removed episode's reason appears (invariant 6: why-removed is explicit).
    for flag in (f for f in card.flags if not f.kept):
        assert flag.reason in html
    # No eval attached => no downstream gain claim.
    assert "makes no" in html and "claim" in html


def test_to_html_effects_section_includes_ci_and_per_task() -> None:
    html = _card_with_effects().to_html()
    assert "Downstream effect" in html
    assert "success_rate" in html
    assert "+0.120" in html  # the effect size
    assert "+0.030" in html and "+0.210" in html  # CI bounds
    assert "can" in html and "square" in html  # per-task breakdown


def test_to_html_escapes_special_characters() -> None:
    html = _card_with_effects().to_html()
    # The raw "<v2> &" must be escaped, not rendered as markup.
    assert "<v2>" not in html
    assert "&lt;v2&gt;" in html
    assert "&amp;" in html


def test_to_html_works_without_matplotlib(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    real_import = builtins.__import__

    def _no_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib disabled for this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_matplotlib)

    html = _run(tmp_path).to_html()
    assert html.lstrip().startswith("<!DOCTYPE html>")
    # Graceful degrade: tables present, but no embedded image.
    assert "<img" not in html
    assert "Signals" in html


@pytest.mark.viz
def test_to_html_embeds_plot_with_matplotlib(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    html = _run(tmp_path).to_html()
    assert '<img alt="Kept vs removed episodes"' in html
    assert 'src="data:image/png;base64,' in html


def test_per_task_mapping_is_accepted() -> None:
    # Guard the type contract: per_task is a Mapping; rendering must not require a dict subclass.
    effects: Mapping[str, float] = {"taskA": 0.5}
    report = EffectReport(metric="m", effect=0.5, ci_low=0.1, ci_high=0.9, per_task=effects)
    assert report.to_dict()["per_task"] == {"taskA": 0.5}
