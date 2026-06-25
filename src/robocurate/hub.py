"""Optional Hugging Face Hub publishing for a curated dataset (behind the ``lerobot`` extra).

Pushing to the Hub is strictly a *post-write* convenience: the curated dataset has already
been written to a local directory and validated (schema + checksum + round-trip) before any
upload happens. :func:`maybe_push_to_hub` reads **only** from that local output directory and
never touches the source dataset (Invariant 1): there is no code path here that can read the
source.

``huggingface_hub`` is an optional dependency (the ``lerobot`` extra). It is imported lazily so
the core still installs and runs clean on a no-GPU laptop with the minimal dependency set; a
missing dependency raises a clear, actionable :class:`ImportError` rather than failing obscurely
at call time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_INSTALL_HINT = (
    "pushing to the Hugging Face Hub requires the 'huggingface_hub' package, which is part of "
    "the optional 'lerobot' extra. Install it with: pip install 'robocurate[lerobot]'"
)


def maybe_push_to_hub(
    local_dir: str | Path,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Upload a curated dataset directory to the Hugging Face Hub as a dataset repo.

    Reads **only** from ``local_dir`` (the curated output) and uploads its contents to
    ``repo_id``; the source dataset is never read or written (Invariant 1). ``huggingface_hub``
    is imported lazily and a clear :class:`ImportError` is raised if the ``lerobot`` extra is
    not installed.

    Args:
        local_dir: The curated output directory to upload (must exist).
        repo_id: The target dataset repo id, e.g. ``"user/my-curated-dataset"``.
        token: An optional Hugging Face access token (falls back to the cached login).
        private: Whether to create the repo as private.

    Returns:
        The ``repo_id`` that was pushed to.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed (install the ``lerobot`` extra).
        FileNotFoundError: If ``local_dir`` does not exist.
    """
    folder = Path(local_dir)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"cannot push to the Hub: curated output directory {folder} does not exist"
        )
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(_INSTALL_HINT) from exc

    # Typed as Any at the call boundary: the upstream signatures evolve across versions.
    # ``private`` applies at repo *creation* time (``create_repo``), not on ``upload_folder``;
    # creating with ``exist_ok=True`` is idempotent, then we upload the validated output dir.
    hub_api: Any = huggingface_hub
    hub_api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    hub_api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    return repo_id


__all__ = ["maybe_push_to_hub"]
