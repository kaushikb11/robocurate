# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **LeRobotDataset v3.0 read adapter** (`LeRobotReaderV3`) — the current Hub default layout
  (multi-episode parquet shards + relational episode metadata), low-dim features, pyarrow-only,
  with version auto-detection in `Dataset.from_lerobot`. Validated on a real Hub dataset.
- **LeRobotDataset v3.0 image/video Stage-1 pass-through** — v3 writes now preserve source
  image/video frame data through curation (the kept episodes' frames survive the round trip).
- **`structural_validity` signal** — flags truncation / stall / non-finite defects the
  geometric signals miss (the 9th built-in signal).
- **Sim-free held-out BC-loss evaluator** — a CPU-only downstream cross-check of the curation.
- **CLI `list-signals`, `validate` (alias `doctor`), and `explain`** — list every loadable
  quality signal and its install extra; run a read-only dataset health check (schema,
  structural defects, coverage); and explain why one episode was kept or removed from a saved
  manifest.
- **HTML curation report** — `Scorecard.to_html()` and `curate --report-html PATH` write a
  self-contained HTML scorecard (no external assets).
- **Dataset card on save** — `CurationResult.save` writes a Hugging Face `README.md` dataset
  card by default (`curate --no-card` / `write_card=False` to opt out).
- **Manifest provenance** — every curated dataset records what was removed and why, the
  equal-N baseline, the config + seed + code version, and the parent manifest path when
  curating an already-curated dataset.
- **Shareable recipes** — `save_recipe` / `load_recipe` and `curate --save-recipe` /
  `--recipe` round-trip the full run config (combiner, budget, selection, gate, seed) as JSON
  so anyone can reproduce byte-identical decisions.
- **HuggingFace push-to-hub** (behind the `lerobot` extra) — `maybe_push_to_hub` and
  `save(..., push_to_hub=...)` / `curate --push-to-hub REPO_ID` upload the curated **output**
  directory to the Hub after the local write is validated (reads only the output, never the
  source; Invariant 1).
- **Known-answer downstream test and determinism CI** — a tiny synthetic dataset with known
  bad trajectories asserts they get flagged, and a CI check guards byte-identical selection
  decisions across runs.

## [0.0.1]

Initial pre-alpha release. The curation core.

### Added

- **Core abstractions** — the canonical internal trajectory representation and the
  `Signal` plugin protocol, so new quality signals are addable without touching the
  core.
- **Eight quality signals** — `jerk`, `action-noise`, `path-efficiency`,
  `spectral-smoothness`, `redundancy`, `sim-physics-validity`, the Demo-SCORE
  classifier, and the CUPID influence signal.
- **Adapters** for LeRobotDataset v2.1, RLDS / Open X, ManiSkill demonstrations, and
  robomimic.
- **Curator** with equal-N and length-matched random baselines, so every selection
  method can be compared fairly against the dataset-size confound.
- **Experiment harness** for the headline curation experiment.
- **CLI** with `curate`, `report`, and `diff` commands.
- **Optional visualization** helpers for scorecards (behind the `viz` extra).
