"""Core-only (torch-free) tests for the benchmark formats: spec/result/leaderboard + resolve.

These exercise everything *except* training a policy:

* JSON round-trips for :class:`BenchmarkSpec`, :class:`BenchmarkResult`, and :class:`Leaderboard`
  (``from_dict(to_dict(x)) == x``);
* :func:`resolve_submission` for both an index-set JSON and a recipe JSON — the recipe path runs
  the cheap-signal :class:`~robocurate.curator.Curator`, confirming it needs no torch;
* leaderboard append + rank ordering, built from :class:`BenchmarkResult`s directly (no policy).

No ``ml`` marker: this whole module must pass in the core-only (no-torch) run.
"""

from __future__ import annotations

import json
from pathlib import Path

from robocurate.adapters.memory import InMemoryDatasetReader
from robocurate.benchmark.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    append_entry,
    load_leaderboard,
)
from robocurate.benchmark.runner import BenchmarkResult
from robocurate.benchmark.spec import BenchmarkSpec, build_spec
from robocurate.benchmark.submission import resolve_submission
from robocurate.curator import Budget, Curator
from robocurate.experiment.synthetic import make_identity_experiment_dataset
from robocurate.recipe import save_recipe
from robocurate.signals.jerk import Jerk


def _pool() -> InMemoryDatasetReader:
    return make_identity_experiment_dataset(num_helpful=8, num_harmful=2, seed=0)


def _make_result(name: str, submitted_mean: float) -> BenchmarkResult:
    """A directly-constructed BenchmarkResult (no torch) for leaderboard/format tests."""
    return BenchmarkResult(
        spec_version="0",
        metric="heldout_bc_loss",
        seeds=(0, 1, 2),
        pool={
            "dataset_id": "synthetic/identity",
            "source_format": "synthetic_identity_v0",
            "content_hash": "deadbeef",
            "num_episodes": 10,
        },
        submission_name=name,
        submission_kind="indices",
        num_kept=4,
        losses_by_arm={
            "submitted": [submitted_mean, submitted_mean, submitted_mean],
            "equal_n_random": [0.5, 0.5, 0.5],
            "full": [0.3, 0.3, 0.3],
        },
        mean_loss_by_arm={
            "submitted": {
                "mean": submitted_mean,
                "ci_low": submitted_mean,
                "ci_high": submitted_mean,
                "n": 3,
            },
            "equal_n_random": {"mean": 0.5, "ci_low": 0.5, "ci_high": 0.5, "n": 3},
            "full": {"mean": 0.3, "ci_low": 0.3, "ci_high": 0.3, "n": 3},
        },
        submitted_vs_equal_n={
            "effect": submitted_mean - 0.5,
            "ci_low": submitted_mean - 0.5,
            "ci_high": submitted_mean - 0.5,
            "n": 3,
            "separated": True,
        },
        code_version="0.0.1",
    )


def test_spec_json_round_trip() -> None:
    spec = build_spec(_pool(), eval_frac=0.25, seed=0)
    assert BenchmarkSpec.from_dict(spec.to_dict()) == spec
    # the eval split and train pool partition the pool's episodes
    assert set(spec.eval_split_indices).isdisjoint(spec.train_pool_indices)
    assert set(spec.eval_split_indices) | set(spec.train_pool_indices) == set(range(10))


def test_spec_split_is_deterministic() -> None:
    a = build_spec(_pool(), eval_frac=0.25, seed=0)
    b = build_spec(_pool(), eval_frac=0.25, seed=0)
    assert a.eval_split_indices == b.eval_split_indices
    assert a.train_pool_indices == b.train_pool_indices


def test_result_json_round_trip() -> None:
    result = _make_result("good", 0.1)
    assert BenchmarkResult.from_dict(result.to_dict()) == result


def test_leaderboard_json_round_trip() -> None:
    board = Leaderboard(
        version="0",
        entries=(
            LeaderboardEntry.from_result(_make_result("a", 0.2), name="a"),
            LeaderboardEntry.from_result(_make_result("b", 0.1), name="b"),
        ),
    )
    assert Leaderboard.from_dict(board.to_dict()) == board


def test_resolve_index_set(tmp_path: Path) -> None:
    pool = _pool()
    sub = tmp_path / "indices.json"
    sub.write_text(json.dumps({"kept_episode_indices": [0, 1, 2]}), encoding="utf-8")
    resolved = resolve_submission(sub, pool)
    assert resolved.kind == "indices"
    assert resolved.name == "indices"
    assert resolved.kept_episode_indices == (0, 1, 2)


def test_resolve_index_set_rejects_out_of_pool(tmp_path: Path) -> None:
    pool = _pool()
    sub = tmp_path / "bad.json"
    sub.write_text(json.dumps({"kept_episode_indices": [0, 999]}), encoding="utf-8")
    try:
        resolve_submission(sub, pool)
    except ValueError as exc:
        assert "not in the pool" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected a ValueError for an out-of-pool index")


def test_resolve_recipe_runs_cheap_curator_no_torch(tmp_path: Path) -> None:
    # A recipe submission resolves by running the cheap-signal curator — no torch required.
    pool = _pool()
    curator = Curator([Jerk()], budget=Budget.fraction(0.5), seed=0)
    recipe_path = tmp_path / "recipe.json"
    save_recipe(curator, recipe_path)
    resolved = resolve_submission(recipe_path, pool)
    assert resolved.kind == "recipe"
    # the curator kept some episodes, all within the pool
    assert len(resolved.kept_episode_indices) > 0
    assert set(resolved.kept_episode_indices) <= set(range(len(pool)))


def test_leaderboard_append_and_rank(tmp_path: Path) -> None:
    board_path = tmp_path / "lb.json"
    # Append worse first, then better — ranking must reorder by ascending mean loss.
    append_entry(
        board_path, _make_result("worse", 0.4), name="worse", created_utc="1970-01-01T00:00:00Z"
    )
    append_entry(
        board_path, _make_result("better", 0.1), name="better", created_utc="1970-01-01T00:00:00Z"
    )
    board = load_leaderboard(board_path)
    ranked = board.ranked()
    assert [e.name for e in ranked] == ["better", "worse"]
    # markdown always surfaces the references + the caveat (Invariants 5 and 6)
    md = board.to_markdown()
    assert "equal-N random" in md
    assert "full" in md
    assert "proxy" in md


def test_leaderboard_write_is_deterministic(tmp_path: Path) -> None:
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    append_entry(p1, _make_result("x", 0.2), name="x", created_utc="1970-01-01T00:00:00Z")
    append_entry(p2, _make_result("x", 0.2), name="x", created_utc="1970-01-01T00:00:00Z")
    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
