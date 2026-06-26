"""The open-benchmark leaderboard — append-only, deterministic, honestly captioned.

A :class:`Leaderboard` is a list of :class:`LeaderboardEntry` records (one per scored
submission), persisted as deterministic JSON. Entries are ranked by **mean held-out loss
ascending** (lower is better). :meth:`Leaderboard.to_markdown` always shows the ``equal_n_random``
and ``full`` references alongside each submission, the paired effect-vs-baseline with its
``separated`` verdict, and the one-line coverage-bias caveat — so the table can never present a
selection's win without the fair comparison and the metric's known bias right next to it
(Invariants 5 and 6).

:func:`append_entry` loads-or-creates the leaderboard at a path, appends one result, and writes
it back with ``sort_keys=True`` for byte-stable output. The ``created_utc`` timestamp is a
parameter, not ``now()``, so a run is fully reproducible (and tests can pin it).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robocurate.benchmark.runner import BenchmarkResult

LEADERBOARD_VERSION = "0"

CAVEAT = (
    "Metric is held-out BC loss — a CPU proxy biased toward the equal-N random control "
    "(uniform eval coverage). Lower is better; a negative effect vs equal-N is a win, but "
    "rollout success is the future unbiased arbiter. This is v0 scaffolding, not a settled "
    "benchmark."
)


@dataclass(frozen=True)
class LeaderboardEntry:
    """One scored submission on the leaderboard.

    Attributes:
        name: The submission's name.
        kind: ``"recipe"`` or ``"indices"``.
        metric: The scoring metric (e.g. ``"heldout_bc_loss"``).
        num_kept: Train-pool episodes the submission selected.
        submitted_mean: The submission arm's mean held-out loss (the ranking key).
        equal_n_mean: The equal-N random control's mean held-out loss.
        full_mean: The full-train-pool reference's mean held-out loss.
        effect_vs_equal_n: Paired ``submitted - equal_n_random`` effect (negative == a win).
        effect_ci_low / effect_ci_high: Bootstrap CI bounds on that effect.
        separated: Whether the effect CI excludes zero (the separation is resolved).
        code_version: The package version that scored it.
        created_utc: Caller-supplied timestamp (never ``now()`` — reproducible).
    """

    name: str
    kind: str
    metric: str
    num_kept: int
    submitted_mean: float
    equal_n_mean: float
    full_mean: float
    effect_vs_equal_n: float
    effect_ci_low: float
    effect_ci_high: float
    separated: bool
    code_version: str
    created_utc: str | None = None

    @classmethod
    def from_result(
        cls, result: BenchmarkResult, *, name: str, created_utc: str | None = None
    ) -> LeaderboardEntry:
        eff = result.submitted_vs_equal_n
        return cls(
            name=name,
            kind=result.submission_kind,
            metric=result.metric,
            num_kept=result.num_kept,
            submitted_mean=float(result.mean_loss_by_arm["submitted"]["mean"]),
            equal_n_mean=float(result.mean_loss_by_arm["equal_n_random"]["mean"]),
            full_mean=float(result.mean_loss_by_arm["full"]["mean"]),
            effect_vs_equal_n=float(eff["effect"]),
            effect_ci_low=float(eff["ci_low"]),
            effect_ci_high=float(eff["ci_high"]),
            separated=bool(eff["separated"]),
            code_version=result.code_version,
            created_utc=created_utc,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "metric": self.metric,
            "num_kept": self.num_kept,
            "submitted_mean": self.submitted_mean,
            "equal_n_mean": self.equal_n_mean,
            "full_mean": self.full_mean,
            "effect_vs_equal_n": self.effect_vs_equal_n,
            "effect_ci_low": self.effect_ci_low,
            "effect_ci_high": self.effect_ci_high,
            "separated": self.separated,
            "code_version": self.code_version,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeaderboardEntry:
        return cls(
            name=str(data["name"]),
            kind=str(data["kind"]),
            metric=str(data["metric"]),
            num_kept=int(data["num_kept"]),
            submitted_mean=float(data["submitted_mean"]),
            equal_n_mean=float(data["equal_n_mean"]),
            full_mean=float(data["full_mean"]),
            effect_vs_equal_n=float(data["effect_vs_equal_n"]),
            effect_ci_low=float(data["effect_ci_low"]),
            effect_ci_high=float(data["effect_ci_high"]),
            separated=bool(data["separated"]),
            code_version=str(data["code_version"]),
            created_utc=data.get("created_utc"),
        )


@dataclass(frozen=True)
class Leaderboard:
    """A ranked, append-only collection of scored submissions."""

    version: str
    entries: tuple[LeaderboardEntry, ...] = ()

    def ranked(self) -> tuple[LeaderboardEntry, ...]:
        """Entries sorted by mean held-out loss ascending (lower is better), name as tiebreak."""
        return tuple(sorted(self.entries, key=lambda e: (e.submitted_mean, e.name)))

    def append(self, entry: LeaderboardEntry) -> Leaderboard:
        """Return a new leaderboard with ``entry`` appended (frozen — no in-place mutation)."""
        return Leaderboard(version=self.version, entries=(*self.entries, entry))

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "entries": [e.to_dict() for e in self.entries]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Leaderboard:
        return cls(
            version=str(data.get("version", LEADERBOARD_VERSION)),
            entries=tuple(LeaderboardEntry.from_dict(e) for e in data.get("entries", [])),
        )

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        """Render the ranked table, always with the equal-N + full references and the caveat."""
        lines: list[str] = []
        lines.append("# RoboCurate open-benchmark leaderboard (v0)")
        lines.append("")
        lines.append(f"> {CAVEAT}")
        lines.append("")
        ranked = self.ranked()
        if not ranked:
            lines.append("_No submissions yet._")
            lines.append("")
            return "\n".join(lines)
        lines.append(
            "| rank | submission | kind | N | submitted loss | equal-N random | full | "
            "effect vs equal-N | separated |"
        )
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
        for rank, e in enumerate(ranked, start=1):
            effect = f"{e.effect_vs_equal_n:+.4f} [{e.effect_ci_low:+.4f}, {e.effect_ci_high:+.4f}]"
            sep = "yes" if e.separated else "no"
            lines.append(
                f"| {rank} | {e.name} | {e.kind} | {e.num_kept} | "
                f"{e.submitted_mean:.4f} | {e.equal_n_mean:.4f} | {e.full_mean:.4f} | "
                f"{effect} | {sep} |"
            )
        lines.append("")
        lines.append(
            "_Lower loss is better. A win is a NEGATIVE effect vs the equal-N random control; "
            "`separated = yes` means the bootstrap CI excludes zero._"
        )
        lines.append("")
        return "\n".join(lines)


def load_leaderboard(path: str | Path) -> Leaderboard:
    """Load a leaderboard from ``path``, or return an empty one if the file does not exist."""
    p = Path(path)
    if not p.is_file():
        return Leaderboard(version=LEADERBOARD_VERSION)
    return Leaderboard.from_dict(json.loads(p.read_text(encoding="utf-8")))


def append_entry(
    path: str | Path,
    result: BenchmarkResult,
    *,
    name: str,
    created_utc: str | None = None,
) -> Leaderboard:
    """Append ``result`` (as an entry named ``name``) to the leaderboard at ``path``.

    Loads-or-creates the leaderboard, appends the entry, and writes it back as deterministic
    JSON (``sort_keys=True``). ``created_utc`` is a parameter (never ``now()``) so the write is
    reproducible. Returns the updated (in-memory) leaderboard.
    """
    board = load_leaderboard(path)
    entry = LeaderboardEntry.from_result(result, name=name, created_utc=created_utc)
    updated = board.append(entry)
    Path(path).write_text(updated.to_json(), encoding="utf-8")
    return updated


__all__ = [
    "CAVEAT",
    "LEADERBOARD_VERSION",
    "Leaderboard",
    "LeaderboardEntry",
    "append_entry",
    "load_leaderboard",
]
