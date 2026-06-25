# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **LeRobotDataset v3.0 read adapter** (`LeRobotReaderV3`) — the current Hub default layout
  (multi-episode parquet shards + relational episode metadata), low-dim features, pyarrow-only,
  with version auto-detection in `Dataset.from_lerobot`. Validated on a real Hub dataset.
- **`structural_validity` signal** — flags truncation / stall / non-finite defects the
  geometric signals miss (the 9th built-in signal).
- **Sim-free held-out BC-loss evaluator** — a CPU-only downstream cross-check of the curation.

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
