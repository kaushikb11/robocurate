# BLUEPRINT.md

The full technical blueprint for the curation framework. Open the section relevant to
the task at hand; don't load all of it every time. This is the detailed specification for
the signal algorithms and the experiment design.

---

## 0. What we're building (recap)

> Point it at any robot dataset and it tells you which trajectories are hurting your
> policy, and hands you back the clean subset that trains a better one.

LeRobotDataset-native, RLDS-compatible, works across real/teleop and sim data. Value is
execution quality, not idea novelty: the adjacent research (Demo-SCORE, CUPID, SCIZOR)
exists only as unmaintained single-paper repos. We are the clean, reliable, multi-source
tool people depend on.

---

## 1. The technical core — curation signals

### 1.1 State of the art (what we're building on)

- **Demo-SCORE** — trains a lightweight classifier to predict whether a demonstration is
  high-quality from policy-progress signals; filters low-quality demos. Real-data
  oriented. Moderate cost (train a small classifier).
- **CUPID** — data attribution / influence-function approach: estimates each
  trajectory's influence on downstream policy performance. The most principled signal and
  our intended flagship; also the most expensive. Works real + sim.
- **SCIZOR** — self-supervised filtering; removes suboptimal segments and redundant
  data without labels. Cheap-to-mid cost, real + sim.
- **DemInf** — mutual-information-based filtering between states and actions.
- **Re-Mix** — DoReMi-style domain reweighting to rebalance mixed datasets.
- **AWE / waypoint methods, L2D** — trajectory simplification / quality via waypoint
  structure.
- **robomimic data-quality findings** — empirical evidence that worse-than-random demos
  actively hurt, and that data composition dominates policy outcomes. This is the
  motivating evidence the whole tool rests on.
- **Influence / datamodels / TRAK** — general data attribution machinery adaptable to
  policy training.
- **Ensemble-variance / disagreement** — flag trajectories where an ensemble of policies
  or dynamics models disagree.
- **Heuristics** — action jerk/smoothness, action-noise, coverage/diversity,
  redundancy/dedup, embedding-based outlier detection.
- **Success verification** — VLM-based or reward-based relabeling of "successes" that
  weren't.

> For each signal, the literature reports downstream policy-success swings in the
> ~15–35% range from curation on some task suites — but the effect is task-dependent.
> Report effect sizes honestly and never imply a universal number.

### 1.2 Taxonomy of signals

**By what they detect**
- Suboptimal / low-quality trajectories (jerk, noise, Demo-SCORE, SCIZOR segments)
- Mislabeled / failed "successes" (success verification, reward checks)
- Redundant / duplicate / low-diversity bloat (dedup, coverage, redundancy removal)
- Out-of-distribution / harmful demos (embedding outliers, influence < 0)
- Physically invalid sim trajectories (collision/penetration checks — sim only)

**By cost tier**
- **Tier 0 — cheap, CPU, laptop-friendly:** jerk/smoothness, action-noise, episode
  length/coverage outliers, dedup by embedding distance, sim physics-validity checks.
- **Tier 1 — mid, single GPU:** Demo-SCORE-style learned quality classifier,
  embedding-based outlier/redundancy with a learned encoder, ensemble disagreement.
- **Tier 2 — expensive, single GPU + time:** CUPID-style influence / data attribution
  (the flagship, principled signal).

**Real vs sim applicability**
- Real + sim: jerk, noise, dedup, Demo-SCORE, embedding outliers, CUPID, success verify.
- Sim only: physics/collision/penetration validity (needs sim state or replay).
- The same `Signal` abstraction handles both; sim-only signals declare they require
  sim-state context and are skipped on real data with a clear message.

### 1.3 Recommended v1 signal set

Ship these (high effect-size, feasible, robust across real + sim):
1. **Jerk / smoothness heuristic** (Tier 0) — the reference vertical slice.
2. **Action-noise / outlier heuristic** (Tier 0).
3. **Redundancy / dedup** (Tier 0–1) — embedding-distance based.
4. **Demo-SCORE-style learned quality classifier** (Tier 1).
5. **CUPID-style influence** (Tier 2) — the flagship; gated behind a clear cost warning.
6. **Sim physics-validity** (Tier 0, sim only) — collision/penetration on replay.

Defer to later rungs: VLM-based semantic success verification, mutual-information
filtering, domain reweighting (Re-Mix), full datamodels.

For each v1 signal, implement with: explicit inputs (obs/actions/rewards/success/proprio),
the computation, per-trajectory and (where meaningful) per-transition outputs, cost tier,
and documented failure modes. (Detailed per-signal algorithm specs to be filled in as
each is implemented; propose precise computations and confirm before coding.)

### 1.4 The curation engine

- Per-signal scores normalize to a common scale, then combine via a user-chosen policy:
  threshold-per-signal, weighted sum, Pareto/multi-objective keep, or target-budget
  selection (keep top-K to a fixed budget).
- **Confound control is first-class:** every selection has a paired "equal-N random"
  selection of identical size, so the fair comparison is one flag away.
- Curation policies are explicit, inspectable config — never a hidden black box. The
  scorecard always explains why each removed trajectory was removed.

---

## 2. The bulletproof headline experiment

### 2.1 Design

- **Datasets/suites:** RoboMimic Multi-Human (real-ish teleop with known quality
  variation — directly tests "worse demos hurt") **and** a ManiSkill3 task suite (sim,
  GPU-parallel, cheap to scale seeds). Two sources prove the signals transfer.
- **Policies:** Diffusion Policy (single-GPU, the field's default BC baseline) as the
  primary, plus SmolVLA (cheap VLA fine-tune) to show the result holds for a VLA. Decide
  whether to add ACT for a third point if budget allows. This combination is what makes
  it credible without frontier compute.
- **Headline:** "curated subset trains a policy that beats full-data AND equal-size
  random selection by X% (±CI) success rate, across N tasks and S seeds, and the gain
  holds on held-out task variations."

### 2.2 The controls that make it unimpeachable

- **Full-data baseline** (train on everything).
- **Curated** (our selection at budget N).
- **Equal-N random** (random subset of the SAME size N) — kills the dataset-size
  confound; this is the control reviewers attack first.
- **Random-filter baseline.**
- **Multiple seeds** per condition; report variance + CIs.
- **Held-out generalization split** (unseen task variations).
- **Compute-matched** training across conditions.
- **Per-signal ablation** isolating each signal's contribution.

### 2.3 Budget and feasibility

- Diffusion Policy trains on a single A100-class GPU per run; ManiSkill3 gives
  GPU-parallel rollouts for cheap eval. Keep tasks/seeds in a range that fits a few GPUs
  over the experiment window (scope tasks × conditions × seeds to the budget; prefer more
  seeds on fewer tasks over thin coverage of many).
- Report honestly: if gains are task-dependent, show the per-task breakdown, not just the
  average.

### 2.4 What's impressive vs noise

- A few percent with overlapping CIs reads as noise — don't ship it as a headline.
- A clear, multi-seed, CI-separated gain over **equal-N random** (not just full-data) on
  multiple tasks is the citable result. The equal-N separation is what makes it real.

---

## 3. Architecture & API

### 3.1 Core abstractions

- **Trajectory representation** — canonical internal form; handles image+state obs,
  proprio, actions, rewards, success labels, variable rates, multi-embodiment,
  variable-length episodes; not a lossy lowest-common-denominator.
- **`Signal` protocol** — one contract for all signals; cheap CPU and expensive GPU
  signals both fit; community-extensible without touching core.
- **Dataset adapters** — LeRobotDataset (read+write) first; RLDS and raw sim output
  slot in later via the same interface. Source read-only; writes produce new dataset +
  manifest.
- **`Curator` / selector** — scores → selection; target-budget and equal-N-random modes.
- **Scorecard / report** — human- and machine-readable; explains every removal.

### 3.2 API surface (shape)

5-line quickstart (illustrative):
```python
from curate import Dataset, Curator, signals

ds = Dataset.from_lerobot("lerobot/some_dataset")
cur = Curator([signals.Jerk(), signals.Redundancy()], budget=0.8)
result = cur.run(ds)
result.save("./some_dataset_curated")   # new dataset + manifest; source untouched
```

Power-user API: explicit per-signal config, custom combiners, equal-N baseline emission,
streaming over datasets too large for RAM, and access to raw per-trajectory score arrays.

### 3.3 Adoption-driving outputs

- Per-dataset **scorecards** (readable + JSON), a **raw↔curated diff**, visualizations of
  flagged trajectories, and HF Hub dataset-card integration.

---

## 4. DX, docs & adoption

- **uv-first install**, zero dependency hell, cheap signals run on a laptop with no GPU.
- **First 5 minutes:** install → run on a public dataset → see a scorecard surface
  surprising bad/duplicate/mislabeled trajectories → save a curated subset.
- **Reliability:** schema validation + checksums on every write, deterministic/pinned
  outputs, fault-tolerant processing of huge datasets, streaming.
- **Wow demo:** one command on a popular public LeRobot/DROID/Open-X dataset that
  surfaces a striking count of bad/duplicate/mislabeled trajectories and trains a better
  policy on the curated subset. Pick the featured dataset for both virality and
  credibility.
- **Launch:** preprint + repo + reproducible demo thread + a small leaderboard.
- **Failure modes to engineer against:** signals that don't transfer across
  datasets/tasks (report per-task, don't overclaim); black-box distrust (always explain
  why-flagged); "what if it deletes good data" fear (source read-only, reversible,
  manifest); maintenance burden of many signals (strict plugin contract); steamroll risk
  if LeRobot adds basic scoring (stay ahead with depth — influence-based signals,
  neutrality, multi-sim — that they won't build).

---

## 5. Sequencing (~16 weeks, 4 engineers)

- **Weeks 1–3:** repo, uv, CI, core abstractions frozen, LeRobot adapter, one vertical
  slice (jerk) end to end + wow demo on a public dataset.
- **Weeks 4–7:** Tier 0/1 signals (noise, redundancy, Demo-SCORE), scorecards, streaming,
  RLDS read adapter.
- **Weeks 6–11 (overlap):** experiment harness with all controls; first headline runs.
- **Weeks 8–13:** CUPID-style influence signal (flagship); sim physics-validity signal;
  ManiSkill3 + RoboMimic integration.
- **Weeks 12–15:** full headline experiment, ablations, CIs; docs; visualizations.
- **Weeks 15–16:** launch (preprint + repo + demo + leaderboard).

Go/no-go checkpoints: frozen abstractions reviewed (wk3); vertical slice works on real
data (wk3); equal-N control separates for at least one signal (wk11); headline result
CI-separated on ≥2 tasks (wk14). If the equal-N separation never appears, stop and
re-examine before launch.

### Expansion ladder (build seams now, don't build the rungs yet)

curation core → filter sim output as it's generated → verify generated scenes before they
produce data → close the loop so policy failures drive what gets generated next.

> Caveats to keep visible: curation effect sizes are task-dependent and may not transfer;
> CUPID-style influence is compute-heavy and needs careful approximation to stay in
> moderate-compute budget; LeRobotDataset v3 format churn is a live maintenance risk —
> pin versions and test against migrations.