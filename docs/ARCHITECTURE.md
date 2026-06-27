# ARCHITECTURE.md

> **Status: FROZEN (v1 core abstractions).** Reviewed and approved. Prompt 2 onward treats
> this file as truth. Changes require explicit re-review. The code in `src/robocurate/`
> implements exactly these abstractions, and the v1 signal suite (eight signals: jerk,
> action-noise, path-efficiency, spectral-smoothness, redundancy, sim physics-validity,
> Demo-SCORE, CUPID) is implemented on top of them. The abstractions themselves are
> unchanged from this document.

---

## 1. Canonical trajectory representation

**Decided design.** The core works in a `Trajectory` (one variable-length episode) that is
**columnar, typed, and lazy**, separating *schema* from *data*:

- `FeatureSpec(key, role, shape, dtype, units, names)` — describes one feature column. `role`
  is a `FeatureRole` (`IMAGE | STATE | PROPRIO | ACTION | REWARD | SUCCESS | TIME | EXTRA`),
  so signals request features by role generically. `units` and per-dim `names` are explicit
  and never inferred.
- `EmbodimentSpec(embodiment_id, features, control_hz)` — the obs/action space, carried with
  every trajectory so a `TrajectorySet` may be heterogeneous (mixed embodiments/spaces) with
  no global action-dim assumption.
- `Trajectory(meta, store)` exposes one uniform accessor `feature(key) -> np.ndarray` over a
  lazy `FeatureStore` backend, plus typed convenience views (`actions()`, `rewards()`,
  `timestamps()`, `success()`) that return `None` when absent rather than fabricating data.
- `timestamps()` is the authoritative control-rate source (`control_hz` is a hint); irregular
  teleop rates are first-class. `T` differs per trajectory; nothing pads to a global max.
- `SuccessLabel(value: bool|None, source, per_step)` is tri-state (success / fail / unknown).
- `TrajectoryMeta` carries `source_dataset_id`, `episode_index`, `embodiment`, content
  `fingerprint` (via `fingerprint_arrays`), `num_steps`, `source_format`, `success`, `extra`.
- Sim-only modalities (object poses, contact/penetration) are ordinary `EXTRA`/`STATE`
  features with explicit units — carried losslessly, not collapsed to a common denominator.

Exchange type is **NumPy** (`feature()` returns `np.ndarray`); the core has no torch
dependency. Streaming: the adapter yields trajectories one episode at a time and `feature()`
materializes lazily, so datasets larger than RAM stream episode-by-episode and cheap signals
never decode video.

**Rationale / rejected alternatives.** A uniform `feature(key)` table (vs fixed named typed
fields) means new modalities/embodiments need zero core changes (invariant 4); the cost is
that signals call `feature("observation.images.wrist")` rather than `traj.wrist_image`.
Lazy/streaming (vs eager) adds accessor indirection but is required for laptop-friendliness
and huge datasets. Carrying the full `EmbodimentSpec` per trajectory is mildly redundant for
homogeneous datasets but makes mixed-embodiment sets correct and the manifest self-describing
(metadata only, negligible cost). NumPy over torch keeps the core installable with no GPU;
torch tensors were rejected for the core for that reason.

---

## 2. The `Signal` protocol

**Decided design.** Two methods on a `Signal` Protocol:

- `fit(trajectories, ctx) -> None` — optional one-shot train/precompute over the whole
  dataset (Demo-SCORE classifier, CUPID embeddings); stateless heuristics no-op it.
- `score(batch, ctx) -> list[TrajectoryScore]` — one score per input, in order. Deterministic
  given `(batch, ctx, seed)` when `spec.deterministic`.

Supporting types: `SignalSpec(name, version, cost_tier, requires, produces_per_transition,
deterministic, description)`; `CostTier` (`TIER0_CPU | TIER1_GPU | TIER2_GPU_HEAVY`);
`requires: frozenset[str]` capability tokens (`"gpu"`, `"sim_state"`, `"encoder"`, or a
feature key by name); `TrajectoryScore(signal, trajectory_fingerprint, value,
higher_is_better, per_transition, skipped, skip_reason, diagnostics)` with a `.skip(...)`
constructor; `SignalContext(seed, device, cache, resources, dataset_meta, logger)` with a
`CacheHandle` (and `InMemoryCache`).

Unmet requirements → a **recorded skip** (`skipped=True, skip_reason=...`), never a crash and
never a silent removal. The engine — not the signal — owns batching, parallelism, requirement
gating, and caching. Plugins register via the `robocurate.signals` **entry-point** group (or
`signals.register`), so the community adds signals without touching core.

The shipped Tier-0 (`TIER0_CPU`, deterministic) plugins are `jerk`, `action_noise`,
`path_efficiency` (directness — net-displacement / total-path-length straightness index),
`spectral_smoothness` (SPARC, spectral arc length — high-frequency wobble in the speed profile,
the orthogonal complement to `path_efficiency`), `redundancy`, and `sim_physics_validity`; the
learned Tier-1/2 plugins are the Demo-SCORE-inspired classifier and the CUPID-inspired
proxy-influence signal. All implement this protocol with no engine-internal access.

**Rationale / rejected alternatives.** Separate `fit`+`score` (vs a single `__call__`) is the
seam that lets trained/influence signals fit the same contract as stateless ones; the no-op
cost is trivial. Engine-owned batching (vs signal-owned) keeps signals small and lets the
scheduler be optimized once. String capability tokens (vs a rich typed capability system)
were chosen for extensibility and lightness; they can harden later.

---

## 3. Dataset adapter layer

**Decided design.** Two **split** protocols make invariant 1 structural:

- `DatasetReader` — `__len__`, `__iter__` (lazy/streaming), `read_episode(i)`, `fingerprint()`,
  `meta`. **No write/save/mutate method exists**, so the source cannot be written through it.
- `DatasetWriter` — `write(trajectories, manifest) -> WriteReceipt`, `validate(path)`. The
  writer is constructed with a destination that must not exist and must not overlap the source
  (`SourceWriteError` otherwise). Every write ends with schema + per-file checksum +
  round-trip-reload validation; failure raises `ValidationError` and quarantines the partial
  output (renamed `*.invalid`) — never reported as success (invariant 2).

`LeRobotReader`/`LeRobotWriter` implement a faithful **minimal v2.1** layout (`meta/info.json`,
`meta/episodes.jsonl`, `meta/tasks.jsonl`, `data/chunk-000/episode_*.parquet`). Scalar and 1-D
vector features + standard bookkeeping columns are implemented; **image/video and the v3
layout are declared but raise a clear error** (`LeRobotVersion` enum pins the version). Full
parity (video, complete stats, an upstream-`lerobot` fast path) is a later rung.

`Manifest` (in `manifest.py`) records `schema_version`, source + output `DatasetFingerprint`,
the resolved `config_dict`, `seed`, `code_version`, the `SignalSpec`s, per-episode
`EpisodeDecision`s (kept/removed + reason + signal values), the equal-N `BaselineRecord`, and
per-file checksums.

**Rationale / rejected alternatives.** Splitting reader/writer (vs one `Adapter` with
`read`+`write`) turns "source read-only" into a type-level guarantee rather than a code-review
promise — the chosen design. The manifest holds `config_dict` (serialized) rather than
importing the engine, keeping it below the curator in the import graph. RLDS and raw sim
output slot in later by implementing the same two protocols. Targeting v2.1 first with v3
guarded was discussed; **both versions are declared in the surface** with v2.1 implemented and
v3 raising clearly, so the v3 path fills in with no interface change.

---

## 4. Curator / selector

**Decided design.** `Curator(signals, combiner, budget, seed, emit_baseline, ...)`:

1. Builds `SignalContext`; **gates** signals whose resource requirements (e.g. `gpu`) are
   unmet (logged skip); rejects non-deterministic signals from the selection path (invariant 3).
2. Calls each signal's `fit` once (fresh streaming iterator), then `score`s all episodes in
   batches into a `ScoreMatrix` (holds raw `TrajectoryScore`s + lightweight `TrajectoryRef`s,
   not heavy arrays).
3. A `Combiner` (Protocol; `WeightedSum` provided) normalizes each signal to `[0,1]` keep-
   oriented (respecting `higher_is_better`, NaN-skips imputed to neutral 0.5) and produces one
   keep-score per trajectory.
4. Optional **hard validity gate** (`GateConfig`): a pre-filter that thresholds one signal's
   value (e.g. `sim_physics_validity` > 0) and removes those trajectories *unconditionally*,
   before the budget. Gated-out episodes are excluded from the valid pool — and from the
   baseline pool — so invalid data is never kept and the curated-vs-random contrast stays a
   fair comparison on the valid data. No change to the frozen `Signal`/`TrajectoryScore`
   contract.
5. Selects within the valid pool by `SelectionMode`: **`TOP_K`** (highest keep-scores under
   `Budget`), **`GREEDY_DEDUP`** (keep one representative per near-duplicate cluster — the
   highest keep-score member — via a z-standardized embedding + `dedup_epsilon`, which top-K
   can't guarantee), or **`COVERAGE`** (greedy submodular facility-location over the same
   embedding — keep a diverse subset covering the whole distribution so rare-but-valid modes
   survive; `coverage_quality_weight` tilts diversity toward keep-score). All three keep exactly
   `K`. **Ties break by `(keep_score, fingerprint)`**, so identical `(dataset, config, seed)` →
   byte-identical decisions.
6. **Equal-N random baseline** (invariant 5) is emitted by default: a same-size (`N=K`) random
   subset drawn from the **valid pool** with an independent `SeedSequence([seed,
   _BASELINE_STREAM])` stream, recorded in `BaselineRecord`.

`CurationResult` holds kept/removed indices, per-episode `EpisodeDecision`s, the `ScoreMatrix`,
the baseline, and config; `.save(dest)` re-reads kept episodes from the source (streaming;
source untouched) and writes a validated new dataset + manifest; `.scorecard()` builds the
report.

**Rationale / rejected alternatives.** A combiner that emits only a relative keep-score (with
the curator applying the budget on top) keeps combiners simple; threshold/Pareto combiners are
declared as future `Combiner` implementations. Determinism via named `SeedSequence` streams +
fingerprint tie-breaks (vs a single global RNG) guarantees reproducibility and an independent
baseline draw. Baseline-on-by-default makes the confound comparison one field away. The hard
gate is a value-threshold pre-filter (vs adding a `valid` field to `TrajectoryScore`) so it
needs no change to the frozen contract and generalizes to any signal; greedy keep-one-
representative dedup is a curator selection mode (vs a `Signal`) because it is set-level, not
per-trajectory. Both gated and baseline arms exclude invalid data (vs gating only curated) so
the equal-N comparison isolates the selection method.

---

## 5. Scorecard / report

**Decided design.** `Scorecard(schema_version, dataset, summary, per_signal, flags, baseline,
effects)` with `to_json()` (stable, versioned), `to_markdown()` (terminal/human), and
`to_hf_dataset_card()`. Reports: `QualitySummary` (counts + % removed); `SignalReport` per
signal (scored/skipped counts, min/median/max, orientation); a `TrajectoryFlag` per episode
naming the signal value(s) and the **human reason for every removal** (invariant 6, no black
box). Reporting is **orientation-aware**: any per-signal diagnostic (e.g. skill-separation AUC
against ground-truth labels) respects each signal's `higher_is_better`, so a signal whose
keep-direction is wrong on a dataset is reported as such rather than masked. `EffectReport` (effect size + CI + per-task breakdown) is included **only** when a policy
eval is attached; without one the card explicitly makes no training-gain claim. Built from a
`CurationResult` via `build_scorecard`.

**Rationale / rejected alternatives.** Effect sizes always carry uncertainty and a per-task
breakdown, never a single cherry-picked number (invariant 6). The JSON schema is versioned so
the leaderboard / HF cards can depend on it. HF dataset-card output is a fragment appended to a
dataset's card.

---

## 6. CLI + Python API surface

**Decided design.** Package is **`robocurate`**. 5-line quickstart:

```python
from robocurate import Dataset, Curator, Budget, signals

ds = Dataset.from_lerobot("./aloha_sim_insertion")          # read-only facade
result = Curator([signals.get("jerk")], budget=Budget.fraction(0.8)).run(ds)
result.save("./aloha_curated")                              # new dataset + manifest
print(result.scorecard().to_markdown())                     # what/why + equal-N baseline
```

Power users use `LeRobotReader`/`LeRobotWriter`, custom `Combiner`s, `CurationConfig`, raw
`ScoreMatrix.to_numpy()`, and `emit_baseline`. CLI commands: `score` (report, no write),
`curate` (select + write), `baseline` (equal-N control), `report`, `diff`; all take `--seed`
and `--json`. A run is reproducible from config + seed + `code_version` (all in the manifest).
`Dataset` exposes only read access (no `write`), keeping the read-only guarantee visible at the
top of the API.

**Rationale / rejected alternatives.** `Dataset` delegates the full reader protocol so
`Curator.run(ds)` works directly. `report`/`diff` are wired as CLI surface but thin in the
skeleton (manifest re-loading and dataset diff land later). Signals resolve from the registry
by name; with none registered the CLI says so rather than fabricating a result.

---

## 7. Data flow (end to end)

```
source dataset
  └─ DatasetReader (read-only, streaming, version-pinned)        [validate schema on read]
       └─ Iterator[Trajectory]  (canonical, lazy, embodiment-aware)
            └─ Curator: gate → Signal.fit(all) → Signal.score(batch)   [seeded SignalContext]
                 └─ ScoreMatrix (per-traj value + optional per-transition; skips recorded)
                      └─ Combiner.combined_score → top-K under Budget   [tie-break by fingerprint]
                           ├─ equal-N random baseline (independent seed stream)   [invariant 5]
                           └─ CurationResult
                                ├─ DatasetWriter.write → NEW dataset
                                │     [schema + checksum + round-trip; source fingerprint
                                │      unchanged → invariants 1, 2] + Manifest
                                └─ Scorecard (json + markdown + HF card)           [invariant 6]
```

Streaming happens at the reader and in `Curator` batching; determinism enters at the seeded
context, the fingerprint tie-break, and the baseline stream; validation happens on read and on
write.

---

## 8. Expansion seams (recorded, not built)

- **Scoring not-yet-generated scenes:** `Trajectory` is produced by an adapter/iterator, so a
  future generator can be a `DatasetReader` yielding candidate (partial) trajectories; a
  "scene signal" declares `requires={"scene_spec"}` — no contract change. *Carries generality.*
- **Inline filtering of sim output during generation:** the `Curator` selects over an iterable,
  so the same selection applies to a streaming generator. *Carries generality; not built.*
- **Closed-loop regeneration from policy failures:** the `EffectReport` / eval hook is the seam
  where policy outcomes re-enter; the loop that drives generation from failures is **not**
  built and no abstraction commits to it yet.
- Deliberately **not** general yet: video/image I/O specifics, LeRobot v3 internals, and any
  GPU-signal scheduling beyond single-GPU batching.
