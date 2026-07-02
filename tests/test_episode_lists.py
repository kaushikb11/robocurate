"""User episode lists (`curate --drop-list/--keep-list`) and the rank→review→curate loop.

Pins the contract: a drop-/keep-list pre-filters the pool like the validity gate (removed
unconditionally, excluded from the valid pool AND the equal-N baseline pool, every removal
recorded with an explicit reason); signals are optional with a list (a pure list-based removal
is a valid curation — the safe alternative to in-place episode deletion); indices that match no
episode warn loudly (the mistyped-index safety rail); and the lists round-trip through
manifests/recipes so `verify` reproduces list-based runs byte-identically.

Also pins the recipe-weights regression: a recipe saved from a default-weight run must
reconstruct the signals that actually ran, not silently reload with zero signals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from robocurate.cli import main
from robocurate.curator import Curator
from robocurate.examples import write_demo_dataset
from robocurate.signals.jerk import Jerk


def _demo(tmp_path: Path) -> Path:
    src = tmp_path / "demo"
    write_demo_dataset(src)
    return src


def _manifest(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return data


def test_pure_drop_list_curation_no_signals(tmp_path: Path) -> None:
    src = _demo(tmp_path)
    drop_file = tmp_path / "drop.json"
    drop_file.write_text(json.dumps({"episode_indices": [1, 5]}), encoding="utf-8")
    out = tmp_path / "curated"

    rc = main(["curate", str(src), "--out", str(out), "--drop-list", str(drop_file)])
    assert rc == 0

    manifest = _manifest(out)
    removed = {d["episode_index"]: d["reason"] for d in manifest["decisions"] if not d["kept"]}
    assert set(removed) == {1, 5}
    assert all("drop-list" in reason for reason in removed.values())
    kept = sorted(d["episode_index"] for d in manifest["decisions"] if d["kept"])
    assert kept == [0, 2, 3, 4, 6, 7]
    # Invariant 5: the equal-N baseline pool also excludes the user-dropped episodes.
    assert not set(manifest["baseline"]["kept_episode_indices"]) & {1, 5}


def test_drop_list_accepts_bare_json_array(tmp_path: Path) -> None:
    src = _demo(tmp_path)
    drop_file = tmp_path / "drop.json"
    drop_file.write_text("[0, 2]", encoding="utf-8")
    out = tmp_path / "curated"
    assert main(["curate", str(src), "--out", str(out), "--drop-list", str(drop_file)]) == 0
    removed = sorted(d["episode_index"] for d in _manifest(out)["decisions"] if not d["kept"])
    assert removed == [0, 2]


def test_keep_list_restricts_the_pool(tmp_path: Path) -> None:
    src = _demo(tmp_path)
    keep_file = tmp_path / "keep.json"
    keep_file.write_text(json.dumps([0, 2, 4, 6]), encoding="utf-8")
    out = tmp_path / "curated"

    rc = main(
        [
            "curate",
            str(src),
            "--out",
            str(out),
            "--keep-list",
            str(keep_file),
            "--signals",
            "jerk",
            "--budget",
            "0.5",
        ]
    )
    assert rc == 0

    manifest = _manifest(out)
    kept = sorted(d["episode_index"] for d in manifest["decisions"] if d["kept"])
    # Budget 0.5 applies WITHIN the keep-list pool of 4 -> 2 kept, both from the list.
    assert len(kept) == 2
    assert set(kept) <= {0, 2, 4, 6}
    not_listed = [d for d in manifest["decisions"] if d["episode_index"] in (1, 3, 5, 7)]
    assert all("keep-list" in d["reason"] for d in not_listed)


def test_unknown_index_warns_instead_of_silently_ignoring(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = _demo(tmp_path)
    from robocurate.dataset import Dataset

    ds = Dataset.from_lerobot(src)
    with caplog.at_level(logging.WARNING, logger="robocurate.curator"):
        Curator([Jerk()], drop_episode_indices={1, 99}, seed=0).run(ds.reader)
    messages = [r.getMessage() for r in caplog.records]
    assert any("match no episode" in m and "99" in m for m in messages)


def test_drop_and_keep_lists_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        Curator([Jerk()], drop_episode_indices={1}, keep_episode_indices={2})

    src = _demo(tmp_path)
    lst = tmp_path / "l.json"
    lst.write_text("[1]", encoding="utf-8")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(
            [
                "curate",
                str(src),
                "--out",
                str(tmp_path / "x"),
                "--drop-list",
                str(lst),
                "--keep-list",
                str(lst),
            ]
        )


def test_recipe_conflicts_with_lists(tmp_path: Path) -> None:
    src = _demo(tmp_path)
    lst = tmp_path / "l.json"
    lst.write_text("[1]", encoding="utf-8")
    recipe = tmp_path / "r.json"
    recipe.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="mutually exclusive"):
        main(
            [
                "curate",
                str(src),
                "--out",
                str(tmp_path / "x"),
                "--recipe",
                str(recipe),
                "--drop-list",
                str(lst),
            ]
        )


def test_malformed_list_file_is_a_clear_error(tmp_path: Path) -> None:
    src = _demo(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text('{"episodes": [1]}', encoding="utf-8")  # wrong key
    with pytest.raises(SystemExit, match="episode_indices"):
        main(["curate", str(src), "--out", str(tmp_path / "x"), "--drop-list", str(bad)])


def test_drop_list_run_verifies_byte_identically(tmp_path: Path) -> None:
    """The manifest records the list, so `verify` reproduces a list-based run (Invariant 3)."""
    src = _demo(tmp_path)
    drop_file = tmp_path / "drop.json"
    drop_file.write_text("[1, 5]", encoding="utf-8")
    out = tmp_path / "curated"
    assert (
        main(
            [
                "curate",
                str(src),
                "--out",
                str(out),
                "--drop-list",
                str(drop_file),
                "--signals",
                "jerk",
                "--budget",
                "0.5",
            ]
        )
        == 0
    )
    assert main(["verify", str(src), str(out / "manifest.json")]) == 0


def test_rank_out_flags_roundtrips_into_drop_list(tmp_path: Path) -> None:
    """The review loop: rank --out-flags -> (human review) -> curate --drop-list."""
    src = _demo(tmp_path)
    flags = tmp_path / "flags.json"
    assert main(["rank", str(src), "--worst", "3", "--out-flags", str(flags)]) == 0
    payload = json.loads(flags.read_text(encoding="utf-8"))
    assert len(payload["episode_indices"]) == 3

    out = tmp_path / "curated"
    assert main(["curate", str(src), "--out", str(out), "--drop-list", str(flags)]) == 0
    removed = {d["episode_index"] for d in _manifest(out)["decisions"] if not d["kept"]}
    assert removed == set(payload["episode_indices"])


def test_cli_saved_recipe_reproduces_decisions(tmp_path: Path) -> None:
    """Regression: a recipe saved from a default-weight run must rebuild the signals that ran.

    Previously `curate --signals jerk --save-recipe` serialized the default WeightedSum with
    empty weights; reloading reconstructed ZERO signals and silently made different decisions.
    """
    src = _demo(tmp_path)
    recipe = tmp_path / "r.json"
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert (
        main(
            [
                "curate",
                str(src),
                "--out",
                str(out_a),
                "--signals",
                "jerk",
                "--budget",
                "0.5",
                "--save-recipe",
                str(recipe),
            ]
        )
        == 0
    )
    saved = json.loads(recipe.read_text(encoding="utf-8"))
    assert saved["config"]["combiner"]["weights"] == {"jerk": 1.0}

    assert main(["curate", str(src), "--out", str(out_b), "--recipe", str(recipe)]) == 0
    kept_a = sorted(d["episode_index"] for d in _manifest(out_a)["decisions"] if d["kept"])
    kept_b = sorted(d["episode_index"] for d in _manifest(out_b)["decisions"] if d["kept"])
    assert kept_a == kept_b


def test_drop_list_source_untouched(tmp_path: Path) -> None:
    src = _demo(tmp_path)
    from robocurate.adapters.lerobot import LeRobotReader

    before = LeRobotReader(src).fingerprint().content_hash
    drop_file = tmp_path / "drop.json"
    drop_file.write_text("[3]", encoding="utf-8")
    args = ["curate", str(src), "--out", str(tmp_path / "c"), "--drop-list", str(drop_file)]
    assert main(args) == 0
    assert LeRobotReader(src).fingerprint().content_hash == before
