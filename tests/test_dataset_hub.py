"""``Dataset.from_lerobot`` Hub-id resolution (no network: ``huggingface_hub`` is faked).

Pins the contract: an existing local path always wins; a ``namespace/name`` id snapshot-downloads
low-dim files only unless ``include_videos`` is set; a Hub source records the *repo id* (not the
machine-specific cache path) as the dataset id; and a missing ``lerobot`` extra fails with an
actionable install hint, never a bare ModuleNotFoundError.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from robocurate.dataset import Dataset
from tests.test_lerobot_v3_reader import _write_synthetic_v3


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch, snapshot_root: Path, calls: list[dict[str, Any]]
) -> None:
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(
        *, repo_id: str, repo_type: str, allow_patterns: list[str] | None = None
    ) -> str:
        calls.append({"repo_id": repo_id, "repo_type": repo_type, "allow_patterns": allow_patterns})
        return str(snapshot_root)

    module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def test_local_dir_wins_even_when_id_shaped(tmp_path: Path) -> None:
    # A directory literally named "namespace/name" resolves as a path, never a download.
    src = tmp_path / "lerobot" / "local_ds"
    src.mkdir(parents=True)
    _write_synthetic_v3(src, lengths=[4, 4])
    ds = Dataset.from_lerobot(src)
    assert len(ds) == 2


def test_hub_id_downloads_low_dim_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "hub_cache"
    cache.mkdir()
    _write_synthetic_v3(cache, lengths=[4, 4, 4])
    calls: list[dict[str, Any]] = []
    _install_fake_hub(monkeypatch, cache, calls)

    ds = Dataset.from_lerobot("someuser/some_dataset")

    assert len(ds) == 3
    assert calls == [
        {
            "repo_id": "someuser/some_dataset",
            "repo_type": "dataset",
            "allow_patterns": ["meta/*", "meta/**", "data/**"],  # never the mp4 shards
        }
    ]
    # Provenance: the shareable repo id is the dataset id, not the local cache path.
    assert ds.fingerprint().dataset_id == "someuser/some_dataset"
    assert ds.read_episode(0).meta.source_dataset_id == "someuser/some_dataset"


def test_hub_id_include_videos_downloads_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "hub_cache"
    cache.mkdir()
    _write_synthetic_v3(cache, lengths=[4])
    calls: list[dict[str, Any]] = []
    _install_fake_hub(monkeypatch, cache, calls)

    Dataset.from_lerobot("someuser/some_dataset", include_videos=True)

    assert calls[0]["allow_patterns"] is None  # full snapshot, mp4 shards included


def test_hub_id_without_extra_raises_actionable_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # force the import failure
    with pytest.raises(ImportError, match=r"robocurate\[lerobot\]"):
        Dataset.from_lerobot("someuser/some_dataset")


def test_nonexistent_path_that_is_not_an_id_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="neither an existing local dataset directory"):
        Dataset.from_lerobot(tmp_path / "does_not_exist")


def test_needs_video_gates_on_the_image_requirement() -> None:
    from robocurate.cli import _needs_video
    from robocurate.signals.base import REQUIRES_IMAGE, CostTier, SignalSpec
    from robocurate.signals.jerk import Jerk

    class _FakeImageSignal:
        spec = SignalSpec(
            name="fake_image",
            version="0.0.1",
            cost_tier=CostTier.TIER0_CPU,
            requires=frozenset({REQUIRES_IMAGE}),
            produces_per_transition=False,
            deterministic=True,
            description="test stub",
        )

    assert _needs_video([Jerk()]) is False
    assert _needs_video([Jerk(), _FakeImageSignal()]) is True  # type: ignore[list-item]
