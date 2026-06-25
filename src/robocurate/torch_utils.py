"""Small shared helpers for the optional torch-backed components.

Keeps device handling in one place so the learned signals (CUPID, Demo-SCORE) and the BC
policy run on a local GPU when one is present and on a Modal GPU worker automatically, while
still working on CPU. This is what lets the same job run on a laptop, a local GPU, or Modal
without code changes — only the resolved device differs.
"""

from __future__ import annotations


def resolve_device(device: str | None) -> str:
    """Resolve a torch device string.

    If ``device`` is given (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``) it is returned as-is.
    If ``None``, auto-detect: ``"cuda"`` when a CUDA device is available (a local or Modal
    GPU), else ``"cpu"``. Returns ``"cpu"`` when torch is not installed.

    Note: CPU runs are bit-deterministic given the seed; GPU runs are seeded but not
    guaranteed bit-reproducible (a known torch limitation), which the experiment's multi-seed
    bootstrap CIs already account for.
    """
    if device is not None:
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an optional extra
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = ["resolve_device"]
