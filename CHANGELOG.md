# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **The CLI now works end-to-end on LeRobot v3.0 datasets.** `score` / `curate` / `baseline` /
  `inspect` / `diff` / `verify` hardcoded the v2.1 reader (and crashed on a v3 directory), and
  `CurationResult.save` hardcoded the v2.1 writer — so "curating a v3 dataset emits a v3 dataset"
  did not hold through the public path. All dataset-reading commands now auto-detect the on-disk
  version, and `save` picks the writer matching the source (a v3 source writes v3, including the
  Stage-1 video-shard pass-through).
- **Per-episode fingerprints are content-only.** The positional bookkeeping columns
  (`episode_index`/`frame_index`/`index`/`task_index`), which real v3 datasets declare as
  features, are excluded from the content fingerprint — so a round-tripped episode keeps its
  fingerprint even after kept episodes are re-indexed, and `diff`/dedup/manifest matching stay
  correct.
- **A signal that skips every episode now warns loudly** (e.g. a sim-only signal on real data, or
  an image signal on a dataset with no decodable video) instead of silently contributing only
  neutral imputed scores.
- **A recipe saved from a default-weight run no longer silently reloads with zero signals.**
  Recipes reconstruct signals from the combiner's weight keys, but a run built with default
  (implicit 1.0) weights serialized as `weights: {}` — so `curate --signals jerk --save-recipe`
  followed by `--recipe` re-ran with *no* signals and different decisions, without any error.
  The config snapshot now records every running signal's weight explicitly (byte-identical
  behavior), so saved recipes and manifests are self-contained.

### Added

- **`curate --no-videos` — an explicitly low-dim output.** Skips the video-shard pass-through
  AND omits the video features from the output's declared schema, so the result claims exactly
  what it contains — never a silent drop. Also skips downloading mp4 shards for a Hub-id
  source. The default stays faithful: `curate` on a Hub id downloads the shards so the output
  preserves video, and a missing source shard is a hard error whose message names the escape
  hatch. (Found by running the real-Hub smoke test: a video-declaring dataset downloaded
  low-dim-only was previously impossible to curate.)
- **Hub-id ingestion.** `Dataset.from_lerobot` (and every dataset-reading CLI command) accepts a
  `namespace/name` Hugging Face Hub dataset id as well as a local directory (an existing local
  path always wins). Hub ids snapshot-download through the `huggingface_hub` cache — **low-dim
  files only** (metadata + parquet, never the mp4 video shards) unless an image signal is
  requested (`include_videos=True` in the API). The repo id, not the machine-specific cache
  path, is recorded as the dataset id so fingerprints and manifests from Hub sources are
  shareable. Needs the `lerobot` extra; without it, a Hub id fails with an actionable install
  hint.
- **`robocurate rank` — a ranked "worst N episodes" report (read-only).** Scores every episode
  with the cheap Tier-0 signals (or `--signals ...`), combines them with the same
  orientation-aware normalization the curator uses, and prints the `--worst N` (default 10)
  lowest keep-score episodes worst-first — each line naming the responsible signal(s) with the
  raw value and its position in the dataset's range, so the ranking is never a black box. Turns
  "watch 200 episodes" into "watch these 8" (the ask in huggingface/lerobot#3760) as a starting
  diagnosis for "why doesn't my policy work". Episodes that *every* requested signal skipped are
  listed as unscored with the skip reason, never silently ranked neutral; `--json` emits the full
  machine-readable payload. Backing it, `CurationResult.keep_scores` and
  `ScoreMatrix.normalized_signal_scores(...)` are new public accessors exposing the combined and
  per-signal keep-oriented scores a run actually used.
- **User episode lists: `curate --drop-list` / `--keep-list` + `rank --out-flags`.** Curate from
  a reviewed episode list: `--drop-list flags.json` removes the listed episodes unconditionally
  (`--keep-list` restricts the pool instead), with signals now *optional* — a pure list-based
  removal is a valid curation, i.e. the safe, non-destructive alternative to in-place episode
  deletion (new dataset + manifest, source untouched). Lists pre-filter like the validity gate:
  excluded from the valid pool AND the equal-N baseline pool, every removal recorded with an
  explicit reason, and indices that match no episode warn loudly (the mistyped-index safety
  rail). The lists are recorded in the manifest/recipe, so `verify` reproduces list-based runs
  byte-identically. `rank --out-flags flags.json` writes the ranked worst-N indices in the
  accepted format, closing the loop: rank → human review → `curate --drop-list`.
- **Failure-tolerant reading: a corrupt episode never has to abort a run.**
  `Curator(on_error=...)` (CLI `--on-error {abort,quarantine}` on `curate`/`score`) adds a
  reading-error policy. The default stays `"abort"` — a data-integrity tool must not silently
  tolerate corruption — but under `"quarantine"` an unreadable episode (corrupt parquet, missing
  file, crashed recording) is recorded as an unconditional removal in the decisions with reason
  `"quarantined: unreadable episode (<ExcType>: <msg>)"`, excluded from BOTH the valid pool and
  the equal-N baseline pool (mirroring the hard validity gate, Invariant 5), and summarized in
  one end-of-scoring warning — never a silent drop (Invariants 2/6). Scope is reading errors
  only: signals keep their skip-not-crash contract, and a raising signal remains a contract
  violation. Determinism is preserved (same input → same exceptions → byte-identical decisions),
  `on_error` round-trips through `CurationConfig`/recipes/manifests, and
  `robocurate validate`/`doctor` now reports an unreadable episode as a finding
  (`UnreadableEpisode`: index + error) instead of crashing the health check.
- **Four new CLI commands.** `profile` (a read-only dataset EDA report — episode-length and
  per-feature distributions, embodiment/task balance, success rate, and a nearest-neighbour
  diversity estimate; also `robocurate.dataset_profile`), `inspect <episode>` (one episode's
  per-signal value, orientation, diagnostics, and a per-transition min/median/max + worst-step
  trace), `compare <manifest_a> <manifest_b>` (diff two curation runs — kept-set sizes, Jaccard
  overlap, kept↔removed flips, per-signal-summary deltas), and `verify <dataset> <manifest>` (re-run
  a recorded run and assert the recomputed kept set + reasons are byte-identical — Invariant 3 made
  user-facing; non-zero exit on mismatch).
- **Configurable Zarr adapter** (`ZarrReader`, behind the `zarr` extra) — curate any
  one-group-per-episode Zarr store, mirroring the HDF5 adapter and reusing the same storage-agnostic
  schema (`ZarrSchema` is an alias of `HDF5Schema`). Read-only, deterministic natural-sort ordering,
  image-hint keys raise.
- **Expanded corruption suite + an honest detection-AUC blind-spot matrix.** Four new known-answer
  corruptions (`frame_skip`, `action_quantize`, `wrong_target_offset`, `dropped_dof`) plus a
  `detection_matrix(...)` that scores each cheap signal's detection-AUC per corruption and labels
  every cell *detects* / *blind* / *inverts*. Runnable via `experiments/blindspot_matrix.py`. The
  matrix is deliberately honest — e.g. it records that `dropped_dof` is a genuine blind spot for
  every cheap geometric signal and that `path_efficiency` *inverts* on `truncate` (Invariant 6).
- **Signal contract-checker + an extending guide.** `check_signal_contract(signal)` /
  `assert_signal_contract(signal)` run a black-box battery against any `Signal` (well-formed spec,
  one finite-or-skipped score per trajectory in order, per-transition shape, determinism) and return
  the violations — so a contributor can validate a custom signal in one line. A `docs/EXTENDING.md`
  tutorial (custom signals & adapters) and a worked `examples/custom_signal.py` accompany it.
- **Configurable generic HDF5 adapter** (`GenericHDF5Reader` + `HDF5Schema`, behind the `hdf5`
  extra) — curate any one-group-per-episode HDF5 robot dataset, not just robomimic's
  `data/demo_*` or ManiSkill's `traj_*`. A small frozen `HDF5Schema` describes where the pieces
  live (`episode_root`, `episode_pattern`, `action_path`, `obs_path`, optional
  `reward_path`/`success_path`/`timestamp_path`, group-vs-flat `obs` handling, per-key roles);
  `HDF5Schema.robomimic_like()` / `maniskill_like()` pin the two known layouts and double as
  worked examples. Read-only (`"r"`, invariant 1) and deterministic: episodes are natural-sorted
  by the trailing integer in the group name (else lexically), so the order and `fingerprint()`
  are stable. Image-hint obs keys raise rather than mis-handle pixels (low-dim v1 scope).
- **`COVERAGE` selection mode** (`SelectionMode.COVERAGE`, CLI `--selection coverage`) — a greedy
  submodular **facility-location** selector that keeps a representative, *diverse* subset best
  covering the embedding distribution, instead of just the top-scoring trajectories. Preserves
  rare-but-valid modes that `top_k` crowds out behind a dense high-scoring majority. CPU-only,
  reuses the same statistical embedding as `greedy_dedup`; `coverage_quality_weight` /
  `--coverage-quality-weight` tilts the objective from pure diversity toward keep-score. Keeps
  exactly the budgeted `k` and leaves the equal-N random baseline byte-identical across modes
  (Invariant 5). Deterministic: all ordering breaks ties by `(keep_score, fingerprint)`, no RNG.
- **Open-benchmark v0 scaffolding** (`robocurate.benchmark`, "DataComp-for-robotics") — the data
  is the submission: a frozen `BenchmarkSpec` pins a pool + a fixed held-out eval split + a fixed
  BC training config; a submission is a *selection* (a recipe or a raw index-set); `run_submission`
  scores it by held-out BC loss against an equal-N random control (Invariant 5) with bootstrap CIs
  and a `separated` verdict (Invariant 6); an append-only, deterministic `Leaderboard` ranks
  submissions and always shows the references + the proxy-metric caveat. Includes a `benchmark`
  CLI group (`init` / `run` / `leaderboard`), a runnable `examples/benchmark_identity.py` proof on
  the synthetic identity dataset, and `docs/BENCHMARK.md`. **Honest scope:** the held-out-loss
  metric is a CPU *proxy* with a documented coverage bias toward the random control; the real pool
  + an unbiased rollout-success metric (a `metric` seam) + a public leaderboard are the next step.
- **LeRobotDataset v3.0 read adapter** (`LeRobotReaderV3`) — the current Hub default layout
  (multi-episode parquet shards + relational episode metadata), low-dim features, pyarrow-only,
  with version auto-detection in `Dataset.from_lerobot`. Validated on a real Hub dataset.
- **LeRobotDataset v3.0 image/video Stage-1 pass-through** — v3 writes now preserve source
  image/video frame data through curation (the kept episodes' frames survive the round trip).
- **`structural_validity` signal** — flags truncation / stall / non-finite defects the
  geometric signals miss (the 9th built-in signal).
- **Three CPU image-quality signals** (behind the `video` extra) — `image_blur`
  (variance-of-Laplacian sharpness severity), `visual_stall` (fraction of adjacent frame pairs
  with a frozen camera, the image-space analogue of the structural stall check), and
  `visual_diversity` (image-space near-duplicate detection via k-NN distance in a cheap CPU
  appearance embedding). They decode frames on CPU via PyAV, advertise a `REQUIRES_IMAGE`
  requirement, and skip gracefully (recorded, never silent) on episodes without decodable video.
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
