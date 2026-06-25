# RoboCurate

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)
![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-orange)
[![CI](https://github.com/kaushikb11/robocurate/actions/workflows/ci.yml/badge.svg)](https://github.com/kaushikb11/robocurate/actions/workflows/ci.yml)

> Point it at any robot dataset and it tells you which trajectories are hurting your
> policy, and hands you back the clean subset that trains a better one.

RoboCurate is a data-curation framework for robot-learning / embodied-AI datasets. It is
[LeRobotDataset](https://github.com/huggingface/lerobot)-native — it reads and writes the
LeRobotDataset format so you incur near-zero switching cost — and it curates **both**
real/teleop data (Open X, DROID, LeRobot Hub) **and** simulation-generated data
(ManiSkill3, RoboCasa, RoboTwin).

> **Status: pre-alpha.** The framework is built and validated the way data-centric tools earn trust
> — faithful multi-source I/O (incl. real LeRobot v3), deterministic + reproducible curation,
> known-answer corruption recovery, a trivial equal-N fair comparison, and honest reporting — *not*
> by claiming our own signals are state-of-the-art. A trustworthy downstream rollout gate (and a
> real influence signal) is the next milestone; see [`docs/ROADMAP.md`](docs/ROADMAP.md). We're just
> getting started — here is an honest map of what's real today versus what's still ahead.

### Where this is going

RoboCurate is built as a 4-rung ladder, climbed in order — each rung earned only after the one
below it is real:

1. **Curation core** *(now, built)* — point at any robot dataset, get the clean subset + a manifest.
2. **Influence flagship + an open "DataComp-for-robotics" benchmark** *(next)* — a real
   policy-impact signal, and the open leaderboard the field is asking for.
3. **Verify the generated** *(later)* — a calibrated, physics-aware checker for the
   simulation-generated data that every generator ships without quality control.
4. **An open data-engine harness** *(horizon)* — reproducible generate → verify → curate → retrain.

The full strategy, the honest competitive picture, and where we're weak today are in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

### What you can run today on a laptop (no GPU)

- **Frozen core abstractions** — canonical trajectory, `Signal` protocol, adapters,
  curator, scorecard. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
- **Nine quality signals** (Tier 0→2): jerk, action-noise, path-efficiency (directness),
  spectral-smoothness (SPARC — spectral arc length), redundancy, structural-validity
  (truncation / stall / non-finite — the *structural* defects the geometric signals miss),
  sim physics-validity, a Demo-SCORE-inspired classifier, and CUPID-inspired proxy-influence.
  The cheap heuristic signals need only NumPy + PyArrow; the learned two live behind extras.
- **Honest self-checks you can run** — a known-answer corruption study (we inject defects we
  control and report each signal's blind spots — e.g. directness/smoothness *invert* on a
  truncated demo, which `structural-validity` then catches) and a sim-free held-out
  behavior-cloning-loss evaluator (a CPU-only downstream comparison of curated vs equal-N and
  length-matched random subsets — an independent cross-check of the GPU rollout gate).
- **Dataset adapters**: LeRobotDataset — **v3.0 read+write** (the current Hub default; low-dim
  features, version auto-detected, validated on a real Hub dataset) **and v2.1 read+write** — so
  curating a v3 dataset emits a v3 dataset. Plus RLDS / Open X-Embodiment, ManiSkill demonstrations,
  and robomimic. (v3 video-frame decode is a follow-up; low-dim curation needs only pyarrow.)
- **Curator + CLI**: target-budget top-K, equal-N random baseline, hard validity-gate,
  greedy keep-one-representative dedup. CLI `curate` / `report` / `diff`, plus `list-signals`
  (every loadable quality signal and its install extra), `validate` (alias `doctor` — a
  read-only dataset health check: schema, structural defects, coverage), and `explain` (why one
  episode was kept or removed, read from a saved manifest).
- **Shareable, reproducible curation runs.** Every `curate` write emits a provenance manifest
  (what was removed and why, the equal-N baseline, the config + seed + code version) and, by
  default, a Hugging Face `README.md` dataset card summarizing the run (`--no-card` to skip).
  `--save-recipe`/`--recipe` round-trip the full config as a JSON *recipe* so anyone can
  reproduce byte-identical decisions; `--report-html` writes a self-contained HTML scorecard;
  and `--push-to-hub <repo_id>` optionally publishes the curated **output** to the HF Hub after
  the local write is validated (reads only the output, never the source; needs the `lerobot`
  extra). v3 image/video frame data is preserved through curation (Stage-1 pass-through).
- **We test our own signals and honestly report their blind spots.** Two scripts you can run
  on real and synthetic data, framed as methodology rather than as headline numbers:
  - `experiments/robomimic_scorecard.py` — a **ground-truth diagnostic** on robomimic
    Multi-Human teleop: how well does each cheap signal track the dataset's operator-skill
    labels, against an equal-size random baseline? It is *orientation-aware* (respects each
    signal's `higher_is_better`) and reports every signal warts-and-all — including where one
    is flat or where its keep-direction is *backwards* on this data. It also runs a confound
    probe ("are we just keeping short episodes?").
  - `experiments/corruption_recovery.py` — a **known-answer test**: inject known-bad
    trajectories into a synthetic dataset and check the signals recover them. This is where we
    surface honest blind spots — e.g. directness and smoothness can invert on *truncated*
    demos and risk discarding rare recovery/corrective trajectories.
- **Honest caveat (a strength, not a footnote):** label-AUC is a *diagnostic*, not
  validation. Recovering operator labels need not mean better curation — CUPID found, on
  robomimic, that perceived quality can diverge from what maximizes policy success. The only
  real proof is the downstream gate, and we have **not** passed it yet (see below).

### Validated as machinery (synthetic data)

- **GPU pipeline on Modal** — curate → train → eval runs end-to-end. This is a *harness
  sanity check on a synthetic 16-demo `identity_synthetic` dataset*, confirming the plumbing
  works; it is **not** a real-data curation result and we report no metric from it. See
  `experiments/modal_app.py`.

### Pending — the real downstream gate (honestly not passed yet)

- **Downstream rollout validation is a Rung-2 capability, not a v1 claim.**
  `experiments/robomimic_bc_validation_modal.py` curates robomimic MH by a signal, trains a BC
  policy on Modal, and compares rollout success against **two** random controls — equal-N and
  length-matched — with paired CIs. The *pipeline* runs end-to-end, but the harness does **not yet
  reproduce robomimic's published BC numbers** (Can 0.36 vs 0.86, gap grows with task difficulty —
  a robosuite-1.5 / v1.5-dataset issue, not curation), so a curated-vs-random rollout delta would be
  measured on an untrustworthy instrument. We therefore do **not** report one. Making the rollout
  harness trustworthy (reproduce a published baseline, then a published *method* like CUPID/DataMIL
  inside it) is Rung-2 work — see [`docs/ROADMAP.md`](docs/ROADMAP.md). Our cheap signals' downstream
  efficacy is an honest open question; the CPU held-out-loss proxy already suggests they don't beat
  random — reported as a finding, not hidden. Platform note: robosuite/MuJoCo state-eval runs fine on
  Modal's gVisor (no Vulkan needed), unlike ManiSkill below.

### Blocked

- **ManiSkill3 sim-environment rollouts.** The integration (env, demo reader, image recipe) is
  code-complete but **never executed**: it is blocked by Modal's gVisor/Vulkan sandbox
  (diagnosis in `experiments/maniskill_modal.py`) and needs a non-gVisor GPU host (RunPod /
  Lambda / bare metal).

## Install

```bash
uv sync
```

The core installs clean on a laptop with no GPU — the cheap Tier-0 signals (jerk, action-noise,
path-efficiency, spectral-smoothness, redundancy, sim physics-validity) need only NumPy +
PyArrow. Learned signals and optional tooling live behind extras:

```bash
uv sync --extra demo-score   # Demo-SCORE-inspired classifier (torch, CPU-ok)
uv sync --extra influence    # CUPID-inspired proxy-influence signal (torch)
uv sync --extra policy       # the behavior-cloning policy for the experiment harness (torch)
uv sync --extra rlds         # read Open X-Embodiment / DROID RLDS datasets (tensorflow-datasets)
uv sync --extra viz          # scorecard plots (matplotlib): per-signal distributions,
                             # kept-vs-removed, per-signal values by operator tier
uv sync --all-extras         # everything
```

A learned signal is always discoverable by name; if its extra isn't installed, requesting it
returns a clear message telling you which extra to install — it never breaks the cheap
signals. The RLDS reader itself is TF-free (the `rlds` extra is only needed to *load* real
datasets via `RLDSReader.from_tfds`).

## Quickstart (target shape)

```python
from robocurate import Dataset, Curator, Budget, signals

ds = Dataset.from_lerobot("./aloha_sim_insertion")            # local LeRobotDataset dir
result = Curator([signals.Jerk()], budget=Budget.fraction(0.8)).run(ds)
result.save("./aloha_curated")            # new dataset + manifest; source untouched
print(result.scorecard().to_markdown())   # what was removed and why, + equal-N baseline
```

Or from the CLI:

```bash
robocurate curate ./aloha_sim_insertion --out ./aloha_curated --signals jerk --budget 0.8
```

## Guarantees

- **Source data is read-only.** Curation emits a *new* dataset plus a manifest describing
  what was removed and why. There is no code path that writes back to the source.
- **No silent data corruption.** Every write is validated against the LeRobotDataset
  schema and checksummed; a curated dataset that fails round-trip reload is a hard
  failure.
- **Deterministic outputs.** Same input + config + seed produces byte-identical selection
  decisions.
- **Honest reporting.** Scorecards report effect sizes and uncertainty, never a single
  cherry-picked number, and always explain *why* a trajectory was removed.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full project invariants.

## Get involved

RoboCurate is open and early. We're looking for **compute / GPU sponsorship** (to close the Rung-2
downstream gate), **real robot datasets** to validate curation on, **research collaboration** on the
influence flagship + the open benchmark, and **adoption + feedback**. If any of these fit your lab or
team, open a [GitHub issue or discussion](https://github.com/kaushikb11/robocurate/issues) — the
detail is in [`docs/ROADMAP.md`](docs/ROADMAP.md#8-get-involved).

## License

Apache-2.0.
