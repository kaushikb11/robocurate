"""The benchmark *spec* — the frozen, shareable definition of an open-benchmark task.

A :class:`BenchmarkSpec` is the "DataComp-for-robotics" task definition: a **fixed pool**
(captured by its :class:`~robocurate.metadata.DatasetFingerprint`), a **fixed held-out eval
split**, the complementary **train pool** a submission may select from, a **fixed BC training
config**, the **metric**, and the **seeds**. Freezing all of this is the whole point: "the data
is the submission" only means something when the train+eval split and the training recipe are
pinned, so two submissions differ *only* in which episodes they select.

The eval split is carved out deterministically (via the held-out evaluator's own
``_split_indices``), so the same pool + ``eval_frac`` + ``seed`` always yields the identical
held-out episodes — and that split is recorded on the spec so every submission is scored against
the *same* episodes.

**Honest caveat (read before interpreting any result):** the v0 metric is held-out BC loss, a
CPU-only *proxy* with a documented coverage bias toward the random control (see
:mod:`robocurate.experiment.heldout`). The unbiased arbiter is closed-loop rollout success; the
``metric`` field is a seam for a future ``rollout_success`` backend. This is scaffolding plus a
runnable synthetic proof, not "the benchmark the field has adopted".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from robocurate.experiment.heldout import _split_indices
from robocurate.manifest import code_version
from robocurate.metadata import DatasetFingerprint

if TYPE_CHECKING:
    from robocurate.adapters.base import DatasetReader

SPEC_VERSION = "0"

DEFAULT_METRIC = "heldout_bc_loss"
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

# The fixed BC training config a submission is scored under. Kept small so the whole
# benchmark runs on a CPU laptop (the `policy` extra) in seconds.
DEFAULT_TRAINING: dict[str, Any] = {"hidden_dim": 64, "epochs": 300, "lr": 0.01}


@dataclass(frozen=True)
class BenchmarkSpec:
    """The frozen definition of one open-benchmark task ("the data is the submission").

    Attributes:
        spec_version: Version of this spec schema.
        pool: Fingerprint of the fixed source pool a submission selects from.
        eval_split_indices: The fixed held-out eval episode indices (never trainable).
        train_pool_indices: The complementary pool episodes a submission may select.
        training: The fixed BC training config (``hidden_dim`` / ``epochs`` / ``lr``).
        metric: The scoring metric. ``"heldout_bc_loss"`` (the v0 proxy) today; a seam for a
            future ``"rollout_success"`` backend.
        seeds: The seeds each arm is trained under (results are aggregated across them).
        code_version: The package version that built the spec, for provenance.
    """

    spec_version: str
    pool: DatasetFingerprint
    eval_split_indices: tuple[int, ...]
    train_pool_indices: tuple[int, ...]
    training: Mapping[str, Any] = field(default_factory=lambda: dict(DEFAULT_TRAINING))
    metric: str = DEFAULT_METRIC
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    code_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "pool": _fingerprint_to_dict(self.pool),
            "eval_split_indices": list(self.eval_split_indices),
            "train_pool_indices": list(self.train_pool_indices),
            "training": dict(self.training),
            "metric": self.metric,
            "seeds": list(self.seeds),
            "code_version": self.code_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkSpec:
        return cls(
            spec_version=str(data["spec_version"]),
            pool=_fingerprint_from_dict(data["pool"]),
            eval_split_indices=tuple(int(i) for i in data["eval_split_indices"]),
            train_pool_indices=tuple(int(i) for i in data["train_pool_indices"]),
            training=dict(data.get("training", DEFAULT_TRAINING)),
            metric=str(data.get("metric", DEFAULT_METRIC)),
            seeds=tuple(int(s) for s in data.get("seeds", DEFAULT_SEEDS)),
            code_version=str(data.get("code_version", "")),
        )


def build_spec(
    reader: DatasetReader,
    *,
    eval_frac: float = 0.2,
    seed: int = 0,
    training: Mapping[str, Any] | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    metric: str = DEFAULT_METRIC,
) -> BenchmarkSpec:
    """Build a :class:`BenchmarkSpec` from a pool ``reader``.

    The held-out eval split is carved out deterministically by
    :func:`~robocurate.experiment.heldout._split_indices` (the same split the held-out evaluator
    uses), so the same pool + ``eval_frac`` + ``seed`` always pins the identical eval episodes.
    The complement is the train pool a submission may select from. The pool's content
    fingerprint is captured so a loaded spec verifies it is scored against the intended data.
    """
    train_pool, val = _split_indices(len(reader), eval_frac, seed)
    return BenchmarkSpec(
        spec_version=SPEC_VERSION,
        pool=reader.fingerprint(),
        eval_split_indices=tuple(val),
        train_pool_indices=tuple(train_pool),
        training=dict(training) if training is not None else dict(DEFAULT_TRAINING),
        metric=metric,
        seeds=seeds,
        code_version=code_version(),
    )


def _fingerprint_to_dict(fp: DatasetFingerprint) -> dict[str, Any]:
    return {
        "dataset_id": fp.dataset_id,
        "source_format": fp.source_format,
        "content_hash": fp.content_hash,
        "num_episodes": fp.num_episodes,
    }


def _fingerprint_from_dict(data: Mapping[str, Any]) -> DatasetFingerprint:
    return DatasetFingerprint(
        dataset_id=str(data["dataset_id"]),
        source_format=str(data["source_format"]),
        content_hash=str(data["content_hash"]),
        num_episodes=int(data["num_episodes"]),
    )


__all__ = [
    "DEFAULT_METRIC",
    "DEFAULT_SEEDS",
    "DEFAULT_TRAINING",
    "SPEC_VERSION",
    "BenchmarkSpec",
    "build_spec",
]
