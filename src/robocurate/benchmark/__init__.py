"""Open-benchmark v0 — "DataComp-for-robotics" scaffolding ("the data is the submission").

A thin CPU layer over the curation core and the held-out-loss evaluator: a frozen
:class:`BenchmarkSpec` pins a pool + a fixed held-out eval split + a fixed BC training config;
a *submission* is a selection (a recipe or a raw index-set); :func:`run_submission` scores it by
held-out BC loss against an equal-N random control; a :class:`Leaderboard` ranks submissions.

**Honest scope:** this is scaffolding plus a runnable proof on a synthetic dataset with a
*proxy* metric (held-out BC loss has a documented coverage bias toward the random control). It
is NOT "the benchmark the field has adopted"; the real pool + an unbiased rollout-success metric
+ a public leaderboard are the funded next step (see ``docs/BENCHMARK.md`` and ``docs/ROADMAP.md``).
"""

from __future__ import annotations

from robocurate.benchmark.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    append_entry,
    load_leaderboard,
)
from robocurate.benchmark.runner import BenchmarkResult, run_submission
from robocurate.benchmark.spec import BenchmarkSpec, build_spec
from robocurate.benchmark.submission import ResolvedSubmission, resolve_submission

__all__ = [
    "BenchmarkResult",
    "BenchmarkSpec",
    "Leaderboard",
    "LeaderboardEntry",
    "ResolvedSubmission",
    "append_entry",
    "build_spec",
    "load_leaderboard",
    "resolve_submission",
    "run_submission",
]
