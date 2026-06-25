## Summary

<!-- What does this change do, and why? Explain the motivation, not just the mechanics. -->

## Checklist

Please confirm each item (or note why it doesn't apply):

- [ ] **Tests added.** Where data I/O is touched, this includes both a **round-trip
      test** (load → curate → reload; source untouched + schema valid) and a
      **known-answer test** (tiny synthetic dataset where the bad trajectories are known
      and asserted to be flagged).
- [ ] **All checks green:** `uv run ruff check .`, `uv run ruff format --check .`,
      `uv run mypy src tests`, and `uv run pytest` all pass.
- [ ] **Source data stays READ-ONLY.** No code path writes back to, mutates, or deletes
      the user's source dataset — curation emits a new dataset plus a manifest.
- [ ] **Determinism preserved.** Any randomness in the selection path is explicitly
      seeded; same input + config + seed produces byte-identical decisions.
- [ ] **Fair comparison intact.** Any new selection method remains evaluable against an
      equal-size random baseline.
- [ ] **Docs updated** if the public API, CLI, or behavior changed (and `CHANGELOG.md`
      under `## [Unreleased]` if user-facing).

## Notes for reviewers

<!-- Anything you're unsure about, trade-offs you made, or context that would help review. -->
