# RoboCurate — Roadmap & Positioning

> One line: **the open, reproducible, policy-aware data layer for robot learning** — curation
> now, verification of generated data next, an open benchmark as the wedge that ties it together.

This document is the strategic ladder. It is grounded in a mid-2026 landscape scan and a study of
how data-centric frameworks earn credibility, and it is deliberately honest about where the
framework is weak today and what it will *not* build. Sourced claims are dated; the rest is our
own assessment. The field moves fast (multiple data-curation papers per quarter in 2026) —
re-check the frontier each milestone.

## 1. The wedge

We win the way `uv`, `polars`, and `vllm` won: **execution and trust against a known problem**,
not idea novelty. The validation philosophy follows from this — a study of how data-centric
frameworks (DataComp, DataPerf, cleanlab, robomimic, lm-eval-harness) earn credibility: **validate
the harness, not the heuristic.** A curation framework earns credibility by being a correct, deterministic, reproducible
substrate that hosts *any* method on a fair benchmark and recovers others' published numbers — NOT
by proving its own default signals are SOTA (the lowest-trust, most-attacked claim form). Our own
signals ship as *baselines*; honest negatives about them are a credibility asset, not a failure.
Concretely, our defensible position is the combination almost no one else offers at once:

- **Policy-aware, not just heuristic.** The cheap geometric tier (smoothness, path-efficiency,
  dedup) is commoditized and, on its own, *weak* (see §2). The value is tying signals to
  **downstream policy impact** and reporting it honestly.
- **The fair comparison made trivial.** Adjacent prior art repeatedly skips the equal-size random
  baseline (Demo-SCORE, and CUPID, both omit it) — exactly our INVARIANT 5. Making "curated vs
  equal-N *and* equal-total-steps random, with effect sizes + uncertainty" the default is the
  honesty wedge.
- **Multi-source, read-only, reproducible, LeRobot/RLDS-native** — one contract over real + sim +
  generated data, source never mutated, deterministic, packaged and maintained (not a single-paper
  repo).

**Honest competitive reality.** The "only unmaintained single-paper
repos" framing is no longer fully true. The cheap-heuristic tier is taken and being commercialized
(HuggingFace `score_lerobot_episodes`, Oct 2025, folding into RobotData Studio). Funded platforms
use "verify/curate" language: Encord ($60M, Feb 2026), Foxglove data curation (Apr 2026), Config
($27M, May 2026), NVIDIA Cosmos Evaluator. **None is source-agnostic, policy-outcome-based,
deterministic, fully open, and benchmark-backed at once.** That precise intersection is the wedge —
but it is an *execution and adoption* race, not a standards moat. Speed and rigor matter.

## 2. Where we actually are (honest)

Built and tested (197 tests, mypy-strict, ruff-clean): a frozen core (canonical trajectory,
`Signal` protocol, adapters, curator, scorecard); **nine signals**; **LeRobotDataset v2.1 read+write
and v3.0 read+write** (the current Hub layout, validated on a real Hub dataset) plus RLDS/Open-X,
ManiSkill-demo, and robomimic adapters; an experiment harness with equal-N + length-matched controls
and paired CIs; a **CPU-only held-out-BC-loss evaluator**; an orientation-aware diagnostic; a
corruption known-answer test; a structural verifier that closes the geometry blind spot; and a
Modal-based downstream BC pipeline.

What the testing established:

- **Our cheap geometric signals are the commoditized tier, and they are weak.** The corruption test
  shows `path_efficiency`/`spectral_smoothness` *invert* on truncated demos (rank structurally broken
  data as higher quality; `structural_validity` then catches it). They are *features/baselines*,
  never standalone keep/drop filters.
- **AUC-vs-operator-tiers is a diagnostic, not validation** (CUPID: perceived quality can diverge from
  policy-maximizing data). The CPU held-out-loss proxy suggests the cheap signals don't beat random.
- **The rollout harness is not yet trustworthy.** It does not reproduce robomimic's published BC, and
  the gap grows with task difficulty (Lift 0.93/1.00, Can 0.36/0.86, Square 0.04/0.53 — a
  v1.5/robosuite-1.5 issue, not curation). So we deliberately do NOT report a curated-vs-random
  rollout delta; making the harness trustworthy is Rung-2 step 1.

The takeaway is not discouraging: per the validation philosophy in §1, the framework is validated by
the harness + known-answer + determinism + reproducibility + honest reporting — all of which we
have. The signals are baselines; the harness and the controls are the asset.

## 3. The ladder

Each rung names what we **own** and what we **deliberately don't**. Rungs are **earned, not
scheduled** — you only climb after the prior rung's success threshold is met. Horizons are
indicative, gated on passing the gate below them.

```
  ▲  earned, not scheduled — each rung climbs only after the one below passes its gate
  │
  4 │ Open data-engine harness     generate → verify → curate → retrain         [horizon]
  3 │ Verify the generated         physics + calibrated, hack-resistant checks  [later]
  2 │ Influence + open benchmark   reproduce a published method; the open       [next ·
  │                                "DataComp-for-robotics" leaderboard           compute-gated]
  1 │ Curation core  «you are here»   multi-source · deterministic · honest      [BUILT]
  └──────────────────────────────────────────────────────────────────────────────────────
     Rung 1 is the credible artifact you adopt today; Rungs 2–4 are what lab + compute
     support unlocks. None of the upper rungs is claimed as done — they are the direction.
```

### Rung 1 — Curation core *(now)*

- **Goal:** point at any robot dataset → it tells you which trajectories hurt your policy and hands
  back the clean subset + a manifest. Cheap signals + a Demo-SCORE-style learned classifier.
- **Own:** the multi-source read-only adapters, the `Signal` plugin contract, the
  curated-vs-controls harness, deterministic + honest reporting.
- **Don't:** treat geometric signals as the headline; claim novelty for the equal-N baseline.
- **Framework validation (the real Rung-1 bar — DONE):** a study of how data-centric
  *frameworks* earn credibility (DataComp, DataPerf, DCBench, cleanlab, robomimic) found they
  validate the **harness**, not "our method is SOTA": faithful multi-source I/O, deterministic +
  reproducible curation, **known-answer corruption recovery** (cleanlab's actual proof of
  correctness), a trivial equal-size-random fair comparison, and honest effect-size reporting. We
  have all of these (incl. real LeRobot v3 read/write validated on a Hub dataset; the corruption
  known-answer test with its honest blind-spots). **That is what makes the framework credible.**
- **Signal-efficacy is a SEPARATE, honest-open question (not the value prop).** Whether our *cheap
  geometric* signals beat random downstream is a per-signal claim the literature already doubts, and
  our evidence says they likely don't: the CPU held-out-BC-loss proxy shows curated ≈/worse than
  equal-N + length-matched random on can. Per invariant 6, we report that as a finding — it is the
  empirical motivation for the Rung-2 influence flagship, not a failure to paper over.
- **The rollout gate is NOT yet trustworthy (key finding):** our robomimic BC harness does *not*
  reproduce the published low_dim BC-MLP numbers, and the gap grows with task difficulty — Lift
  0.93/1.00, **Can 0.36/0.86, Square 0.04/0.53** (almost certainly the v1.5-dataset / robosuite-1.5
  controller-reconstruction issue, NOT curation). Until the harness reproduces a known baseline, a
  curated-vs-random rollout delta is measured on a broken instrument and means nothing — so we did
  NOT run it. Making the rollout harness trustworthy is Rung-2 work (§Rung 2, step 1).
- **Status:** framework BUILT and validated by the bar above. The rollout-win-for-our-own-signals
  experiment is deliberately NOT pursued (it conflates framework-validation with signal-efficacy and
  needs a trustworthy harness first). The harness diagnosis that established this is complete; GPU
  budget is reserved for the Rung-2 work that needs it.

### Rung 2 — Trustworthy harness + reproduce a published method + influence flagship + the open benchmark

*The "earn credibility" rung. The validation analysis sharpened the keystone:
credibility comes from being the reproducible harness that hosts others' methods and recovers
their published numbers — the cleanlab / lm-eval-harness / DataComp playbook — not from our own
heuristics winning. Four parts, in dependency order:*

- **Step 1 — A trustworthy downstream harness (the precondition the harness diagnosis exposed).** The
  robomimic rollout gate underperforms published BC (Can 0.36 vs 0.86; gap grows with difficulty),
  so it cannot yet be trusted to measure a curation delta. Fix it: pin **robomimic v0.2 + the
  original `offline_study` (mujoco-py) datasets** (or a task/policy/version where BC demonstrably
  reproduces the paper), proper seeds/rollouts/CIs. **Promote held-out-BC-loss (DataMIL-precedented)
  to the cheap PRIMARY metric; rollouts as confirmation.** Nothing else in Rung 2 is trustworthy
  without this.

- **Step 2 — Reproduce a published curation method INSIDE the framework (the keystone validation).**
  Implement CUPID (public code → lowest risk) or DataMIL (cheapest, validates our held-out path) as
  a `Signal` plugin and **recover its published gain** (CUPID's "<33% of data ≥ full-data"). One
  experiment proves, against an external anchor, that the adapters feed data faithfully, the plugin
  contract is real, the equal-N fair comparison works, and the harness end-to-end is correct — none
  of which a self-generated win for our own signals can prove.

- **Goal (a):** a real influence/attribution flagship. Ship CUPID as the reference, then a
  **QoQ-style trajectory-aggregated, noise/coverage-robust estimator** (arXiv 2603.09056, Mar 2026)
  as the actual flagship; keep an **ATHENA-style** acceleration path (arXiv 2606.16208, Jun 2026,
  ~313× speedup, billion-param VLAs) for if/when we reach VLA scale. Treat their effect sizes as
  author-reported and unreplicated.
- **Goal (b) — pulled earlier because it is the concrete category-defining wedge: "DataComp-for-
  robotics."** No open, reproducible benchmark exists where the *data* is the submission and the
  train+eval are fixed (vision/LLMs have DataComp/DCBench; robotics has only the inverse — fix data,
  vary model). The community is openly asking for it (RSS 2026 "Data-Centric Robotics" workshop,
  Jul 17 2026, proposes none). Build it on existing infra (LeRobotDataset + RoboCasa/RoboArena/
  LIBERO eval). This is the most achievable category-defining artifact within reach, and it sits
  directly on top of the controls we already have.
- **Own:** the open leaderboard/harness, the plugin signals (subsuming CUPID/SCIZOR/Demo-SCORE under
  one contract), the honest per-task + uncertainty reporting that single-number papers lack.
- **Don't:** compete on raw VLA scale; over-anchor on unreplicated 2026 influence numbers; try to
  prove our cheap heuristics are SOTA (Rung 1 settled that — report the honest negative and move on).
- **Success threshold:** the harness reproduces a published baseline number (step 1) AND reproduces a
  published curation gain inside it (CUPID's "<33% data ≥ full-data", step 2), CI-honest; ship a
  benchmark others actually submit to.
- **Honest dependency:** the reproduction can fail for boring version reasons (the same robosuite
  controller-reconstruction issue). Step 1 is non-negotiable and first; if a faithful reproduction still exceeds available
  compute, ship the harness + known-answer + held-out-loss and present the reproduction as costed,
  in-progress — still more credible than a noisy self-win on a broken instrument. Rung 2 is
  GPU-heavy — it is what lab/compute support funds; Rung 1 is the credible artifact you pitch with.

### Rung 3 — Verify the *generated* (not generate)

- **Goal:** "generated trajectory → verified, training-ready." A verification/curation layer over
  the *open* generators (RoboCasa, ManiSkill3, MimicGen/DexMimicGen, GR00T-Dreams). This is the
  repeatedly-admitted open bottleneck: every generator ships only a binary success-check its own
  authors call insufficient (MimicGen: *"developing better filtering mechanisms is left for future
  work"*; RoboCasa jerky/colliding demos; RoboTwin 2.0 *"lack automated quality control"*), and
  multiple papers show unfiltered synthetic data does not help or hurts.
- **Own:** source-agnostic, deterministic, policy-impact verification + a **standalone post-hoc
  physics/penetration/dynamics checker** over arbitrary datasets (genuine whitespace — nothing
  ships this), plus **calibrated, reward-hacking-resistant** success verification with *honestly
  reported human-agreement numbers* (no shipping tool publishes these).
- **Don't generate.** Generation is commoditized (MJX, Genesis, ManiSkill3 run on one 4090) and the
  frontier is lab-dominated ($11B+ PI, $39B Figure, NVIDIA vertically integrated). Treat generator
  output as a *source to curate*. Don't lean on VLM-as-judge as the moat — the API call is a
  one-liner; reliable, calibrated, hallucination/reward-hacking-resistant verification is the work.
- **Why it's reachable from strength:** it is a direct generalization of our `sim_physics_validity`
  signal and the corruption/verification machinery. The honest caveat: verification is partly
  commoditizing (sim success is a free env predicate; sim2real eval correlation is rising), so the
  durable core is narrow — real-world/sim-free verification, calibration, physics-checking — and
  contested. Moat = execution, not concept.
- **Horizon:** ~months 12–20, *after* Rungs 1–2 are real and trusted.

### Rung 4 — The open data-engine harness (not the autonomous fleet/loop)

- **Goal:** an open, reproducible integration of generate → verify → curate → eval → retrain — the
  *harness and honest scorekeeping* for the flywheel, LeRobot/RLDS-native.
- **Own:** the integration + reproducibility + openness layer (the part no incumbent has shipped
  openly), and the first failure-driven *curation* signal (generalize Demo-SCORE so a policy's
  failures drive what data is selected/flagged).
- **Don't:** build the fully-autonomous loop, a fleet, a generator, or a VLA. The full loop is owned
  by *nobody* today (every "data engine" is human-seeded, batch-triggered — GR00T, PI's RECAP,
  AutoRT, ARMADA), but its *components* are capital-bound (GR00T N1's final pretrain alone ≈ 50k
  H100-hrs; real fleets cost $1M+ to collect). A small team owns the harness, not the loop.
- **Horizon:** ~months 18–24+, as a frontier we grow into — pulled into existence by the users who
  climbed the earlier rungs with us, not attempted up front.

## 4. Non-goals (say no explicitly)

- We do **not** out-generate NVIDIA / Genesis / RoboCasa — generation is commoditized; we curate
  what it produces.
- We do **not** build a fleet, collect teleop at scale, or train frontier VLAs.
- We do **not** claim novelty for the equal-N baseline, or ship single-number results without
  per-task breakdown + uncertainty.
- We do **not** ship geometric signals as standalone keep/drop filters (the corruption test forbids
  it).
- We do **not** depend on Vulkan rendering on gVisor clouds (ManiSkill/SAPIEN); MuJoCo state-eval is
  the Modal-viable path.

## 5. The invariants are the moat

Everything above rests on the project invariants (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)),
and the landscape *validates*
them: source read-only; no silent corruption; deterministic; signals are plugins behind one
contract; **every selection is evaluable against an equal-size random baseline** (invariant 5);
**effect sizes + uncertainty, never a cherry-picked number** (invariant 6). The empirical record
justifies this — published single-number gains (+40%, +15.4%, 0→43%) routinely collapse under
per-task breakdown and small-N/no-CI scrutiny. Being the honest scorekeeper in a hype-heavy field
*is* the position.

## 6. Immediate next action

Rung 1 is built and validated by the right bar (§2). The LeRobot v3-native gap is closed; the
rollout-harness diagnosis is done (it doesn't reproduce published BC — Rung-2 step 1). So the
near-term, mostly-free work is:

1. **Polish Rung 1 for open-source release** — onboarding, docs, hygiene; present the framework
   honestly (harness + known-answer + held-out-loss; geometric signals as baselines with their
   honest-negative finding). This is the artifact we share with labs and collaborators.
2. **Add a known-answer DOWNSTREAM test on a working setup** ($0): inject known-bad demos into a task
   where BC actually trains and show curated beats equal-N on held-out loss — ground truth known, so
   a null is interpretable. (CPU; complements the corruption known-answer test.)
3. **Then Rung 2 (needs lab/compute support):** make the rollout harness trustworthy (pin
   robomimic v0.2 + offline_study datasets), reproduce a published method (CUPID/DataMIL) inside the
   framework, then the influence flagship + the open benchmark.

We deliberately don't spend GPU budget chasing a curated-vs-random rollout win for the cheap
signals: the harness isn't trustworthy yet, and the held-out proxy already indicates they likely
don't beat random.

## 7. Risks & watch-items

- **Competition / timing:** Encord, Foxglove, Config, NVIDIA Cosmos Evaluator, RobotData Studio, and
  **LeRobot's own in-progress curation pipeline**. Engage LeRobot maintainers early; win on
  execution + openness before the lane consolidates.
- **The premise itself is thin:** "synthetic data helps real policies" rests on small-N, no-CI,
  often physics-violating evidence (honest GR00T figure ≈ +5.8% avg, not the marketed +40%). Good
  for us — it makes honest verification valuable — but don't assume the tailwind is as strong as the
  marketing.
- **lerobot churn:** near-daily releases; v3 streaming/metadata APIs may shift. Pin versions; support
  v2.1 via the converter.
- **Sourcing caveat:** many 2026 figures (ATHENA, QoQ, RoboReward, DreamGen efficiencies) are recent
  preprints / vendor claims, read once, unreplicated. Re-verify before citing in anything public.

## 8. Get involved

RoboCurate is open and early, and the climb past Rung 1 is faster with partners. If any of these
fit your lab or team, please open a [GitHub issue or discussion](https://github.com/kaushikb11/robocurate/issues)
— we'd love to talk:

- **Compute / GPU sponsorship.** Rung 2 is the GPU-heavy part — making the rollout harness
  trustworthy (reproducing a published BC baseline) and reproducing a published curation method
  inside the framework. Rung 1 is the credible artifact today; compute is what closes the downstream
  gate. Modest, well-scoped credits go a long way here.
- **Real robot datasets to validate on.** Point RoboCurate at your own LeRobot / teleop / sim data
  and we'll help curate it and report the result honestly — including a clean negative. Real-data
  validation is exactly what hardens the tool.
- **Research collaboration.** Co-develop the influence flagship (CUPID → QoQ-style estimator) and the
  open "DataComp-for-robotics" benchmark — the category-defining artifact the field is openly asking
  for (RSS 2026 Data-Centric Robotics workshop).
- **Adoption + feedback.** Try it on a dataset, adopt it in your pipeline, and tell us where it
  breaks. Bug reports, new signals, and adapters are all welcome (see
  [`CONTRIBUTING.md`](../CONTRIBUTING.md)).
