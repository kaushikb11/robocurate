"""Optional Hugging Face push-to-hub (behind the ``lerobot`` extra).

These tests never make a real network call. They assert two things that protect Invariant 1
and the optional-dependency contract:

* ``save(..., push_to_hub=None)`` is a pure no-op: no upload helper is ever invoked.
* ``maybe_push_to_hub`` uploads **only** the validated curated output directory (never the
  source), via a monkeypatched ``upload_folder`` whose call args are recorded and asserted.

The push path is marked ``@pytest.mark.lerobot`` because it is the optional-extra surface; it
still runs without the extra installed because every Hub call is monkeypatched.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from robocurate.adapters import LeRobotReader
from robocurate.curator import Budget, Curator
from tests.synthetic import FakeActionMagnitudeSignal, write_synthetic_lerobot_dataset


def _run_and_save(
    tmp_path: Path, *, push_to_hub: str | None, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[dict[str, Any]]]:
    """Curate a tiny synthetic dataset and save it, recording any ``upload_folder`` calls."""
    src = write_synthetic_lerobot_dataset(tmp_path / "src", num_episodes=6)
    result = Curator([FakeActionMagnitudeSignal()], budget=Budget.fraction(0.5), seed=1).run(
        LeRobotReader(src)
    )

    calls: list[dict[str, Any]] = []

    def _fake_upload_folder(**kwargs: Any) -> None:
        calls.append(kwargs)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.upload_folder = _fake_upload_folder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    out = tmp_path / "curated"
    result.save(out, push_to_hub=push_to_hub)
    return src, calls


@pytest.mark.lerobot
def test_save_without_push_is_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _src, calls = _run_and_save(tmp_path, push_to_hub=None, monkeypatch=monkeypatch)
    assert calls == []  # no upload attempted, no network
    assert (tmp_path / "curated" / "manifest.json").is_file()


@pytest.mark.lerobot
def test_push_uploads_only_the_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src, calls = _run_and_save(tmp_path, push_to_hub="acme/curated", monkeypatch=monkeypatch)
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["repo_id"] == "acme/curated"
    assert kwargs["repo_type"] == "dataset"
    # Invariant 1: the upload reads the curated OUTPUT, never the source.
    uploaded = Path(kwargs["folder_path"]).resolve()
    assert uploaded == (tmp_path / "curated").resolve()
    assert uploaded != src.resolve()


@pytest.mark.lerobot
def test_maybe_push_raises_clear_error_without_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robocurate.hub import maybe_push_to_hub

    out = tmp_path / "out"
    out.mkdir()

    # Simulate the extra being absent: importing huggingface_hub fails.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    real_import = builtins.__import__

    def _no_hub(name: str, *args: object, **kwargs: object) -> object:
        if name == "huggingface_hub" or name.startswith("huggingface_hub."):
            raise ImportError("huggingface_hub disabled for this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _no_hub)

    with pytest.raises(ImportError, match=r"robocurate\[lerobot\]"):
        maybe_push_to_hub(out, "acme/curated")


@pytest.mark.lerobot
def test_maybe_push_passes_output_dir_and_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from robocurate import hub

    out = tmp_path / "out"
    out.mkdir()

    calls: list[dict[str, Any]] = []

    def _fake_upload_folder(**kwargs: Any) -> None:
        calls.append(kwargs)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.upload_folder = _fake_upload_folder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    returned = hub.maybe_push_to_hub(out, "acme/curated", token="t0ken", private=True)
    assert returned == "acme/curated"
    assert len(calls) == 1
    kwargs = calls[0]
    assert Path(kwargs["folder_path"]).resolve() == out.resolve()
    assert kwargs["repo_id"] == "acme/curated"
    assert kwargs["repo_type"] == "dataset"
    assert kwargs["token"] == "t0ken"
    assert kwargs["private"] is True


@pytest.mark.lerobot
def test_maybe_push_missing_dir_raises(tmp_path: Path) -> None:
    from robocurate.hub import maybe_push_to_hub

    with pytest.raises(FileNotFoundError):
        maybe_push_to_hub(tmp_path / "does-not-exist", "acme/curated")
