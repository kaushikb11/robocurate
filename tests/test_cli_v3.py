"""End-to-end CLI tests on a LeRobotDataset **v3.0** source.

The CLI used to hardcode the v2.1 reader (``score``/``curate``/``baseline``/``inspect``/
``diff``/``verify`` crashed on a v3 directory) and ``CurationResult.save`` used to hardcode
the v2.1 writer (a v3 source silently downgraded). These tests pin the fixed contract:
every dataset-reading command auto-detects the on-disk version, and curating a v3 source
emits a v3 dataset — including the Stage-1 video-shard pass-through — never a downgrade.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from robocurate.adapters.lerobot_v3 import LeRobotReaderV3
from robocurate.cli import main
from tests.test_lerobot_v3_image import _write_synthetic_v3_with_video
from tests.test_lerobot_v3_reader import _write_synthetic_v3


def _curate_v3(src: Path, out: Path) -> None:
    rc = main(
        [
            "curate",
            str(src),
            "--out",
            str(out),
            "--signals",
            "jerk",
            "--budget",
            "0.5",
            "--json",
        ]
    )
    assert rc == 0


def test_cli_curate_v3_source_emits_v3_output(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_v3(src, lengths=[8, 8, 8, 8])
    source_fp_before = LeRobotReaderV3(src).fingerprint().content_hash

    out = tmp_path / "curated"
    _curate_v3(src, out)

    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert str(info["codebase_version"]).startswith("v3")  # not downgraded to v2.1
    assert len(LeRobotReaderV3(out)) == 2  # budget 0.5 of 4, reloads as v3
    assert (out / "manifest.json").is_file()
    # Invariant 1: the source is untouched.
    assert LeRobotReaderV3(src).fingerprint().content_hash == source_fp_before


def test_cli_curate_v3_passes_video_shards_through(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _, shard_bytes = _write_synthetic_v3_with_video(src, lengths=[8, 8, 8, 8])

    out = tmp_path / "curated"
    _curate_v3(src, out)

    # Every kept episode's referenced mp4 shard was copied byte-identical into the output.
    kept = [LeRobotReaderV3(out).read_episode(i) for i in range(2)]
    copied = sorted((out / "videos").rglob("*.mp4"))
    assert len(copied) == 2  # one shard per kept episode in this fixture
    copied_payloads = {p.read_bytes() for p in copied}
    assert copied_payloads <= set(shard_bytes.values())
    assert all(t.video_references() for t in kept)  # the reloaded episodes still see video


def test_cli_curate_no_videos_emits_honest_low_dim_output(tmp_path: Path) -> None:
    """--no-videos: no shards copied AND no video features declared — never a silent drop."""
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_v3_with_video(src, lengths=[8, 8, 8, 8])

    out = tmp_path / "curated"
    rc = main(
        [
            "curate",
            str(src),
            "--out",
            str(out),
            "--signals",
            "jerk",
            "--budget",
            "0.5",
            "--no-videos",
        ]
    )
    assert rc == 0
    assert not (out / "videos").exists()
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert not any(s.get("dtype") in ("video", "image") for s in info["features"].values())
    reloaded = LeRobotReaderV3(out)
    assert len(reloaded) == 2
    assert reloaded.read_episode(0).video_references() == {}


def test_default_write_with_missing_shards_errors_with_the_escape_hatch(tmp_path: Path) -> None:
    """A video-declaring source without shards on disk must fail loudly, naming --no-videos."""
    import shutil

    from robocurate.adapters.base import ValidationError
    from robocurate.curator import Budget, Curator
    from robocurate.signals.jerk import Jerk

    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_v3_with_video(src, lengths=[8, 8])
    shutil.rmtree(src / "videos")  # simulate a low-dim-only download of a video dataset

    result = Curator([Jerk()], budget=Budget.fraction(0.5), seed=0).run(LeRobotReaderV3(src))
    with pytest.raises(ValidationError, match="no-videos"):
        result.save(tmp_path / "curated")
    # The escape hatch produces a valid low-dim output from the same result.
    receipt = result.save(tmp_path / "curated_lowdim", write_videos=False)
    assert receipt.path.is_dir()


def test_cli_score_inspect_diff_verify_on_v3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_v3(src, lengths=[8, 8, 8, 8])
    out = tmp_path / "curated"
    _curate_v3(src, out)
    capsys.readouterr()  # drop curate output

    assert main(["score", str(src), "--signals", "jerk", "--json"]) == 0
    assert main(["inspect", str(src), "0", "--json"]) == 0
    capsys.readouterr()

    assert main(["diff", str(src), str(out), "--json"]) == 0
    diff_payload = json.loads(capsys.readouterr().out)
    assert diff_payload["n_source"] == 4
    assert diff_payload["n_curated"] == 2
    assert diff_payload["n_removed"] == 2

    # Invariant 3, user-facing: re-running the recorded manifest reproduces the decisions.
    assert main(["verify", str(src), str(out / "manifest.json"), "--json"]) == 0


def test_cli_baseline_on_v3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_v3(src, lengths=[8, 8, 8, 8])
    assert main(["baseline", str(src), "--n", "2", "--signals", "jerk"]) == 0


def test_all_skipped_signal_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A signal that skips every episode warns loudly instead of silently imputing neutral."""
    from robocurate.curator import Budget, Curator
    from robocurate.signals.jerk import Jerk
    from robocurate.signals.sim_validity import SimPhysicsValidity

    src = tmp_path / "src"
    src.mkdir()
    _write_synthetic_v3(src, lengths=[8, 8, 8, 8])
    reader = LeRobotReaderV3(src)  # real/teleop-shaped data: no sim state anywhere

    with caplog.at_level(logging.WARNING, logger="robocurate.curator"):
        Curator([Jerk(), SimPhysicsValidity()], budget=Budget.fraction(0.5), seed=0).run(reader)

    warnings = [r.getMessage() for r in caplog.records]
    assert any("sim_physics_validity" in w and "skipped all 4 episodes" in w for w in warnings)
    assert not any("jerk" in w for w in warnings)  # a scoring signal does not warn
