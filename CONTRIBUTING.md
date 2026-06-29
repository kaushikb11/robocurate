# Contributing to RoboCurate

Thanks for taking the time to contribute. RoboCurate is a data-curation framework for
robot-learning / embodied-AI datasets, and it lives or dies on execution quality:
clean code, honest reporting, and reproducible results. Contributions of any size are
welcome — bug reports, docs, new signals, adapters, or test coverage.

This guide covers the dev loop, how CI is structured, how to add a new signal, and the
invariants every change must respect.

## Dev setup (uv-first)

RoboCurate is packaged and run with [uv](https://docs.astral.sh/uv/). The core installs
clean on a laptop with no GPU; heavy/learned signals live behind optional extras.

```bash
# Install everything (core + all optional extras + dev tools)
uv sync --all-extras --dev
```

Before opening a pull request, run the full local check. This is the same gate CI
enforces:

```bash
uv run ruff check . \
  && uv run ruff format --check . \
  && uv run mypy src tests \
  && uv run pytest
```

If `ruff format --check` fails, apply the fixes with `uv run ruff format .`.

## How CI is laid out

CI runs three independent jobs so the no-GPU promise stays honest:

- **`full`** — runs on Python 3.10, 3.11, and 3.12 with the torch-based extras
  (`demo-score`, `influence`, `policy`) plus `maniskill-demos` (h5py). It runs ruff
  (lint + format check), mypy, and the full pytest suite.
- **`rlds`** — a separate job on a single Python version that installs the heavy
  `rlds` extra (TensorFlow + tensorflow-datasets) and exercises the RLDS / Open X
  adapter against real TensorFlow.
- **`core-only`** — installs the package with **no extras and no torch**, then asserts
  that torch is genuinely absent, the package imports, and the cheap signals load and
  work. This guards the invariant that the core runs on a clean, no-GPU install.

When you add a feature, make sure it lands in the right job: anything that needs torch,
TensorFlow, or h5py belongs behind its extra and should be marked so the core-only run
skips it (see the pytest markers in `pyproject.toml`).

## Adding a new signal

Every quality signal is a plugin behind one contract — adding a signal never touches the
core engine.

1. Implement the `Signal` protocol (see `src/robocurate/signals/` for the contract and
   existing examples).
2. Register it via the `robocurate.signals` entry-point group in `pyproject.toml`:

   ```toml
   [project.entry-points."robocurate.signals"]
   my_signal = "robocurate.signals.my_signal:MySignal"
   ```

3. If your signal needs a heavy dependency (torch, TensorFlow, h5py, …), put that
   dependency behind a new optional extra and mark its tests so the core-only CI run
   skips them.
4. Colocate tests with the signal. A signal change that touches data I/O needs both a
   round-trip test and a known-answer test (see Testing below).

No signal should reach into engine internals. If you find yourself needing to, open an
issue first — the abstraction probably needs to change, and that's a discussion worth
having before code.

## Extending RoboCurate (custom signals & adapters)

A full, code-driven tutorial lives in [`docs/EXTENDING.md`](docs/EXTENDING.md): how to write
a custom signal (the `Signal` protocol, `fit`/`score`, `TrajectoryScore`, cost tiers and
requirements, skipping gracefully, determinism, and registering via entry points) and a
custom adapter (the read-only `DatasetReader` protocol). A minimal, runnable worked example
is in [`examples/custom_signal.py`](examples/custom_signal.py).

Before shipping a signal, verify it honors the contract with the **contract-checker** — it
runs a battery of structural and behavioral checks against any `Signal` and returns the list
of violations (empty means it passes):

```python
from robocurate.signals import check_signal_contract, assert_signal_contract

assert check_signal_contract(MySignal()) == []   # inspect violations, or
assert_signal_contract(MySignal())               # raise with the joined messages, for tests
```

## Invariants — please don't break these

These are the load-bearing guarantees the project makes. Every change is reviewed
against them:

1. **Source data is READ-ONLY.** Curation always emits a *new* dataset plus a manifest
   describing what was removed and why. There is no code path that writes back to the
   source.
2. **No silent data corruption, ever.** Every dataset write is validated against the
   schema and checksummed. A curated dataset that fails round-trip reload is a hard
   failure, not a warning.
3. **Deterministic outputs.** Given the same input dataset, config, and seed, curation
   produces byte-identical selection decisions. All randomness is seeded explicitly —
   no unseeded RNG anywhere in the selection path.
4. **Signals are plugins behind one contract.** Every signal implements the `Signal`
   protocol; new signals are addable without touching the core.
5. **Fair comparison is built in.** Any selection method must be evaluable against an
   equal-size random baseline. The dataset-size confound is the first thing reviewers
   attack, so the code makes the fair comparison trivial.
6. **Honesty in reporting.** Scorecards report effect sizes and uncertainty, never a
   single cherry-picked number. If a gain is task-dependent, the output says so.

## Testing

- **Tests are colocated** with the code they cover.
- **Any change touching data I/O needs two tests:**
  - a **round-trip test** — load → curate → reload, asserting the source is untouched
    and the curated dataset is schema-valid; and
  - a **known-answer test** — a tiny synthetic dataset where the bad trajectories are
    known, asserting they get flagged.
- Prefer a working vertical slice over broad, half-built coverage.

## Working on larger changes

For anything beyond a small, self-contained fix, a little process keeps reviews fast and
the design honest:

- **Plan before building.** On a multi-file change, outline the approach and the files
  you'll touch in the issue or PR description first, and get rough agreement before
  writing the code.
- **Freeze interfaces before building on them.** When a change introduces a new
  abstraction (a protocol, an adapter contract, a config surface), propose it and let it
  be reviewed *before* you implement features against it. Unreviewed interfaces are the
  expensive thing to undo.
- **Tests land in the same change as the code.** Don't defer them to a follow-up — a
  change touching data I/O is not complete without its round-trip and known-answer tests
  (see Testing).

## Pull requests

Keep commits small and reviewable — one logical change each, with the repo green
(tests + lint pass) after every commit. Write a commit message that explains the *why*.
The PR template walks through the invariant checklist; please fill it in honestly.

If you're unsure about an algorithm's detail, say so in the PR and cite what you're
basing it on rather than inventing specifics. We'd much rather have an honest "I'm not
sure this is right" than a confident guess.

Welcome aboard, and thank you.
