"""Resolving a benchmark *submission* into a concrete episode selection.

A submission is a small JSON file in one of two kinds:

* a **recipe** (it has a ``recipe_version`` field) — the full curation config; resolving it
  runs the :class:`~robocurate.curator.Curator` on the pool and takes the kept episodes. The
  cheap heuristic signals need no torch, so a recipe submission resolves on a CPU laptop with
  the core install only.
* an **index-set** (it has a ``kept_episode_indices`` field) — a raw list of episode indices,
  the lowest-friction way to submit an arbitrary selection (e.g. a baseline, or a selection
  produced by an external tool).

Either way the result is a :class:`ResolvedSubmission`: a kind, a name, and the kept episode
indices, validated to be a subset of the pool's episodes. The runner restricts these to the
spec's train pool (eval episodes are never trainable) before scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader


@dataclass(frozen=True)
class ResolvedSubmission:
    """A submission resolved to a concrete episode selection.

    Attributes:
        kind: ``"recipe"`` (resolved by running the curator) or ``"indices"`` (a raw index set).
        name: A short human label for the submission (defaults to the file stem).
        kept_episode_indices: The episode indices the submission selects (subset of the pool).
    """

    kind: str
    name: str
    kept_episode_indices: tuple[int, ...]


def resolve_submission(path: str | Path, pool_reader: DatasetReader) -> ResolvedSubmission:
    """Resolve the submission at ``path`` against ``pool_reader`` into a selection.

    A recipe (has ``recipe_version``) is run through the curator; an index-set (has
    ``kept_episode_indices``) is read directly. The result is validated to be a subset of the
    pool's episode indices.
    """
    path = Path(path)
    name = _submission_name(path)
    document = json.loads(path.read_text(encoding="utf-8"))

    if "recipe_version" in document:
        # A recipe fixes the full curation config; running it on the pool yields the selection.
        # The cheap-signal curator needs no torch, so this resolves on a core install.
        from robocurate.recipe import load_recipe

        curator = load_recipe(path)
        result = curator.run(pool_reader)
        kept = tuple(result.kept_episode_indices)
        kind = "recipe"
    elif "kept_episode_indices" in document:
        kept = tuple(int(i) for i in document["kept_episode_indices"])
        kind = "indices"
    else:
        raise ValueError(
            f"submission {path} is neither a recipe (no 'recipe_version') nor an index-set "
            "(no 'kept_episode_indices'); one of those keys is required."
        )

    pool = set(range(len(pool_reader)))
    extra = sorted(set(kept) - pool)
    if extra:
        raise ValueError(
            f"submission {path} selects episodes not in the pool (0..{len(pool_reader) - 1}): "
            f"{extra}"
        )
    return ResolvedSubmission(kind=kind, name=name, kept_episode_indices=kept)


def _submission_name(path: Path) -> str:
    return path.stem


__all__ = ["ResolvedSubmission", "resolve_submission"]
