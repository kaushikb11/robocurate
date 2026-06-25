# Experiments

The validation harness behind RoboCurate's claims. Each script is tagged by where it can
actually run, so you know up front what it needs and what to expect.

- **`[laptop — no GPU]`** — runs on a plain laptop after `uv sync`; no GPU, no cloud.
- **`[Modal GPU]`** — runs on a Modal GPU worker (`modal token new` once,
  `robocurate[modal]` installed). CUDA / training works on Modal.
- **`[BLOCKED — diagnosis only]`** — cannot run on Modal as-is; kept here as a documented
  known-limitation with the root cause, so it isn't rediscovered. Needs a non-gVisor GPU
  host (RunPod / Lambda / bare metal).

The demo-dataset generator (`examples/make_demo_dataset.py`) lives under `examples/`, not
here — see `docs/GETTING_STARTED.md`.

## Laptop — no GPU

| File | What it does | Needs |
| --- | --- | --- |
| `robomimic_scorecard.py` | Runs the GPU-free signals over robomimic Multi-Human demos and reports each signal's rank-AUC against ground-truth operator-proficiency tiers, plus a curation-vs-random-baseline breakdown. The headline real-data scorecard. | `uv run --extra robomimic`; downloads ~50–120 MB into `experiments/data/` on first run. |
| `corruption_recovery.py` | Known-answer test: takes clean robomimic demos, injects defects into copies in a *known* way, and measures whether each signal ranks the corrupted copy as worse (a detection-AUC table per corruption × signal). Surfaces the blind spots label-AUC hides. | `uv run --extra robomimic`; reuses the data downloaded by `robomimic_scorecard.py`. |
| `plot_validation_results.py` | Plots downstream BC-validation results (curated signals vs an equal-N random baseline) from a saved results JSON. | `uv run --extra viz` (matplotlib); consumes the JSON emitted by `robomimic_bc_validation_modal.py`. |

## Modal GPU

| File | What it does | Needs |
| --- | --- | --- |
| `modal_app.py` | Runs a RoboCurate headline experiment (CUPID + BC) on a Modal GPU worker, shipping the experiment config to the worker and returning a report. | `modal run experiments/modal_app.py`; `modal token new` once, `robocurate[modal]` locally. |
| `robomimic_bc_validation_modal.py` | Downstream BC validation of a curation signal on robomimic MH: trains a policy on the curated subset vs an equal-N random baseline and compares. Uses state observations (CPU/CUDA physics, no Vulkan). | Modal GPU. **Reviewable / unrun — do not launch without sign-off (paid GPU compute).** |

## Blocked — diagnosis only

Both ManiSkill scripts are blocked on Modal because Modal runs containers under the
**gVisor** sandbox, whose `nvproxy` passes only the `compute`/`utility` GPU capabilities by
default. SAPIEN's renderer needs Vulkan, which requires the `graphics` capability — gated at
the runtime layer — so `vkCreateDevice` fails. CUDA works fine, which is why the
training/curation experiments above succeed; only the sim *renderer* is blocked. These need
a non-gVisor GPU host (RunPod / Lambda / bare metal).

| File | What it does | Status |
| --- | --- | --- |
| `maniskill_modal.py` | Rolls out a random policy in a state-obs ManiSkill3 task — a GPU/Vulkan smoke test. Prints a Vulkan diagnostic first so the failure mode is self-explanatory on Modal. | Blocked on Modal (gVisor/Vulkan); runs on a Vulkan-capable host. |
| `maniskill_experiment_modal.py` | The full ManiSkill3 vertical: download demos → curate → train BC → rollout. | Blocked on Modal (same gVisor/Vulkan limitation); needs a non-gVisor GPU host. |

## Supporting files

`data/` holds datasets downloaded by the laptop experiments (gitignored). The
`nvidia_*.json` files in this directory are the Vulkan ICD / EGL vendor config a non-gVisor
GPU host needs to initialise SAPIEN's renderer (referenced by the ManiSkill scripts).
