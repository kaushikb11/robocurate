# Getting started

The cold-start path: no GPU, no dataset download, under 30 seconds. It generates a tiny
synthetic dataset, curates it, and shows you the scorecard — so you can see the whole
loop before pointing RoboCurate at your own data.

## Run this first

```
uv sync
uv run python examples/make_demo_dataset.py ./demo_dataset
uv run robocurate curate ./demo_dataset --out ./demo_curated --signals jerk --budget 0.8
uv run robocurate report ./demo_curated
```

That's it. The `report` step prints a curation scorecard that looks like this:

```
# Curation scorecard — `./demo_dataset`

**2/8** episodes removed (25.0%); **6** kept.
Paired equal-N random baseline keeps **6** episodes (same size) for a confound-free comparison.

## Signals

| signal | scored | skipped | min | median | max | orientation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| jerk | 8 | 0 | 7.081 | 49.535 | 105.328 | higher=better |

## Removed episodes

- **episode 3** — removed: keep-score 0.0000 below budget 6
- **episode 5** — removed: keep-score 0.0589 below budget 6

_No downstream policy evaluation attached; this scorecard makes no claim about training gains._
```

## What you just saw

The demo dataset is a deliberate mix: half the episodes are smooth, direct reaches
("good"), and half are jittery, wandering ones ("bad"). The `jerk` signal scores the
roughness of each episode's action sequence, and `--budget 0.8` keeps the smoothest 80%.
The episodes it removed are the bad ones — by construction, so you can check the tool got
it right.

Three things were produced:

- **`./demo_curated/`** — a *new* LeRobotDataset directory with only the kept episodes.
  Your source `./demo_dataset/` is never touched: curation always emits a new dataset, and
  the writer refuses to write over a source (this is a hard invariant, not a default).
- **`./demo_curated/manifest.json`** — the auditable record of *what was removed and why*,
  one decision per source episode, plus the seed and config needed to reproduce the run.
- **The scorecard** — note the line about the **equal-N random baseline**. Any selection
  method has to beat a random subset of the *same size*, otherwise you can't tell whether a
  gain came from the signal or just from training on less data. The scorecard records that
  paired baseline so the fair comparison is always one step away, never an afterthought.

The demo is deterministic: the same seed gives byte-identical selection decisions every
time.

## A few more commands worth knowing

```
# Which quality signals can I run, and what does each one need installed?
uv run robocurate list-signals

# Is my dataset healthy before I curate? (read-only: schema, structural defects, coverage)
uv run robocurate validate ./demo_dataset       # alias: doctor

# Which episodes should I watch first? A ranked worst-episodes report — "watch these 3",
# not all 200 — with each line naming the signal(s) that flagged it. Read-only, and a
# diagnostic starting point, not proof an episode hurts training.
uv run robocurate rank ./demo_dataset --worst 3

# Why was one specific episode kept or removed? (reads the saved manifest)
uv run robocurate explain ./demo_curated 3
```

## Save a recipe, reproduce the run

A *recipe* captures the full run config — combiner, budget, selection, gate, seed — as a small
JSON file you can share. Re-running it on the same source reproduces byte-identical decisions:

```
# Curate, and save the exact knobs you used as a recipe.
uv run robocurate curate ./demo_dataset --out ./demo_curated \
    --signals jerk --budget 0.8 --save-recipe ./jerk_80.json

# Later (or on someone else's machine), reproduce it from the recipe alone.
uv run robocurate curate ./demo_dataset --out ./demo_curated_again --recipe ./jerk_80.json
```

`--recipe` is mutually exclusive with `--signals`/`--budget` (the recipe already fixes them).
A `curate` run also writes a Hugging Face `README.md` dataset card by default (`--no-card` to
skip), can emit a self-contained HTML scorecard with `--report-html report.html`, and can
publish the curated **output** to the Hub with `--push-to-hub user/my-curated-dataset` (this
reads only the validated output directory, never your source, and needs the `lerobot` extra).

## Next, on real data

Any dataset-reading command also accepts a Hugging Face Hub dataset id directly (needs the
`lerobot` extra: `uv sync --extra lerobot`). Only the low-dim files are downloaded — metadata
and parquet, never the mp4 video shards — so this is fast even for large video datasets:

```
uv run robocurate profile lerobot/svla_so101_pickplace
uv run robocurate validate lerobot/svla_so101_pickplace
```

When you want to see the signals run against a real public dataset with ground-truth
quality labels:

```
uv run --extra robomimic python experiments/robomimic_scorecard.py
```

This downloads the robomimic Multi-Human demonstrations (~50–120 MB on first run, into
`experiments/data/`) and measures how well each cheap signal recovers the dataset's known
operator-proficiency tiers.

One honest caveat: the AUC that experiment reports is a **diagnostic** — it tells you
whether a signal *separates* known-good from known-bad demos, which is necessary but not
sufficient. It is not proof that curating on that signal trains a better policy. Downstream
training validation is a separate, heavier step (see `experiments/README.md`).

## Where to go next

- **`docs/ARCHITECTURE.md`** — the abstractions and data flow (read before designing).
- **`docs/BLUEPRINT.md`** — the full technical blueprint (signals, experiment, roadmap).
- **`docs/ROADMAP.md`** — strategy and positioning.
- **`experiments/README.md`** — every experiment script, tagged by what it needs to run.
