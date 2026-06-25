"""Downstream BC validation of a curation signal on robomimic MH — Modal GPU.

!!! REVIEWABLE / UNRUN. Do not launch without sign-off — this is paid GPU compute. !!!

This is the *real* ship-gate for a curation signal, the bar CUPID / SCIZOR / RINSE meet and
that an AUC-against-labels diagnostic does NOT: curate the dataset by the signal, train a BC
policy on the curated subset, and check it beats an **equal-N random subset** (and the full
set) on closed-loop success — multi-seed, with confidence intervals. Per Invariant 5,
the curated-vs-equal-N-random gap is the whole experiment; curated-vs-full confounds quality
with dataset size.

Why robomimic (not ManiSkill) here: ManiSkill/SAPIEN rendering is blocked on Modal by gVisor
(see experiments/maniskill_modal.py). robomimic evaluates in **robosuite/MuJoCo with low_dim
(state) observations**, whose rollouts are CPU/CUDA physics (mj_step) with **no Vulkan/OpenGL**
for state obs — so this plausibly runs on Modal where ManiSkill could not. THAT ASSUMPTION IS
THE FIRST THING TO VERIFY: the `robosuite_smoke` step below builds a state env and steps it; if
it fails (robosuite may create a GL context on env init), the eval needs a non-gVisor host just
like ManiSkill, and only the curation/training half runs here.

Design — curation expressed as robomimic filter keys (clean + read-only):
  1. Read robomimic MH with our RoboMimicReader; curate the *train* split by the chosen signal
     to a budget k -> a set of kept demo names.
  2. Write a NEW copy of the .hdf5 (the source is never mutated — invariant 1) that adds a
     mask/ filter key (`curated`, `random` = equal-N, or `random_steps` = length-matched)
     alongside `train`/`valid`. robomimic curation is *exactly* a filter key, so it slots into
     the native pipeline.
  3. Train robomimic BC (or BC-RNN) on each arm's filter key with rollouts enabled; report each
     curated subset's paired per-seed gap vs BOTH random controls, with uncertainty.

Task / policy:
  - **Square MH + BC-MLP** is the default informative test: published robomimic BC-MLP gets
    **~52.7% on Square MH** (Mandlekar et al. 2021, Table 1), leaving room for curation to move
    success (vs lift/can, which BC-MLP nearly saturates at 100% / 86%). This requires robomimic's
    FULL training budget (``num_epochs=2000``, epoch_every_n_steps=100 -> ~200k gradient steps).
  - **CORRECTION / lesson:** an earlier 600-epoch run undertrained Square MH to ~5% and was
    wrongly read as a BC-MLP "architecture floor." It was not — it was ~3x too few gradient
    steps. **Reproduce ~50% on full Square MH BEFORE trusting any curation comparison**; a gain
    measured at a broken floor is meaningless.
  - **BC-RNN-GMM** (``--policy rnn``) is robomimic's stronger policy (~78% on Square MH) — a
    higher ceiling, not a rescue from a floor.

Decision rule (write it down before running): the equal-N random baseline does NOT control for
trajectory length, and a directness signal shifts the kept set shorter (BC error compounds
~quadratically in horizon), so ALSO compare against a length-matched (equal-total-steps) random
baseline. Ship the signal only if curated beats BOTH random controls by a CI-separated margin,
with adequate seeds/rollouts. A win vs *full* alone, or a mere AUC-vs-labels number, is NOT
sufficient (AUC-vs-operator-tiers is a diagnostic, not the claim — CUPID shows perceived quality
can diverge from policy-maximizing data).

Open items to verify on a first CHEAP run (small epochs / few rollouts) before scaling:
  1. robosuite state-env init + step on Modal gVisor (the `robosuite_smoke` gate).
  2. The exact robomimic config fields / version (built below via config_factory) — robomimic's
     config schema shifts across releases; confirm the BC config keys resolve.
  3. Control mode / obs keys match between the dataset and the env robomimic builds from
     `env_args` embedded in the .hdf5 (robomimic handles this natively, but verify).
"""

from __future__ import annotations

import json
from typing import Any

import modal

# Cheapest sensible GPU: BC-MLP training is tiny and rollouts are CPU-bound MuJoCo, so the GPU
# only needs to exist, not be fast — a cheap GPU minimizes $/hr during the unavoidable CPU rollout
# wait. L4 (~$0.80/hr) is cheap + modern + has the CUDA the BC train needs; A10G is the proven
# fallback if L4 has any robosuite/driver issue.
GPU = "L4"
TASK = "square"  # hard enough that demo quality moves the needle (lift/can saturate)
SIGNAL = "path_efficiency"  # or "action_noise"; the candidate under test
BUDGET = 0.67  # keep the cleanest two-thirds
# Seeds/rollouts are entrypoint params (budget-fit defaults in main(); the rigorous bar is >=5
# seeds / >=200 rollouts, which exceeds the current compute budget).
HF_BASE = "https://huggingface.co/datasets/amandlek/robomimic/resolve/main/v1.5"

_IGNORE = [
    "**/.venv",
    "**/.git",
    "**/__pycache__",
    "**/*.egg-info",
    "**/dist",
    "**/build",
    "**/.mypy_cache",
    "**/.ruff_cache",
    "**/.pytest_cache",
    "**/data",  # don't ship the downloaded hdf5s into the image
]
# robomimic + robosuite + mujoco (CPU/CUDA physics; no Vulkan needed for state obs) + our pkg.
image = (
    modal.Image.debian_slim(python_version="3.10")
    # build-essential + cmake: egl_probe (a robomimic dep) compiles a wheel from source.
    # libglib2.0-0/libsm6/libxext6/libxrender1: opencv-python (pulled by robosuite's renderer)
    # needs them at import (libgthread-2.0.so.0 etc.) even for headless state-obs use.
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "libgl1",
        "libosmesa6",
        "libglfw3",
        "libglew-dev",
        "patchelf",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "torch>=2.2",
        "h5py>=3.0",
        # Match the dataset: the v1.5 low_dim hdf5s record env_version 1.5.1 and use the new
        # robosuite 1.5 composite-controller (BASIC) config, so they MUST be evaluated in
        # robosuite 1.5.1 (1.4.1 rejects kwargs like `lite_physics` and has a different
        # controller schema). The matching robomimic (v0.4+, with v1.5-dataset + robosuite-1.5
        # support and the mujoco_py-import guard) was never published to PyPI — install master.
        # mujoco 3.2.3 exactly: robosuite 1.5.1 requires mujoco>=3.2.3 and its OSC controller
        # calls the mj_fullM(m, dst, M) signature, which a later mujoco (3.3.x) changed to
        # mj_fullM(m, d, dst) — a newer mujoco breaks the mass-matrix step. 3.2.3 is the pin
        # that satisfies robosuite's floor while keeping the signature its code was written for.
        "robosuite==1.5.1",
        "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git",
        "mujoco==3.2.3",
        "egl_probe",
    )
    .env({"MUJOCO_GL": "osmesa"})  # headless software GL for any offscreen needs; state obs
    .add_local_dir(".", "/pkg", copy=True, ignore=_IGNORE)
    .run_commands(
        "pip install '/pkg[robomimic]'",
        # robomimic 0.3.0 has a stale, unconditional top-level `import mujoco_py` in
        # envs/env_robosuite.py (guarded with try/except on master, but not in the 0.3.0
        # release). robosuite 1.4.1 uses the new `mujoco` bindings and does NOT need the heavy
        # legacy mujoco_py — drop a tiny stub so the import and the single
        # `mujoco_py.builder.MujocoException` reference resolve without it.
        """python - <<'PY'
import os, site
stub = "class _B:\\n    class MujocoException(Exception):\\n        pass\\nbuilder = _B()\\n"
open(os.path.join(site.getsitepackages()[0], "mujoco_py.py"), "w").write(stub)
print("wrote mujoco_py stub")
PY""",
    )
)

app = modal.App("robocurate-robomimic-bc-validation", image=image)


def _ensure_dataset(task: str) -> str:
    """Download the MH low_dim hdf5 onto the worker; return its path."""
    import os
    import urllib.request

    root = "/root/robomimic_data"
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{task}_mh_low_dim_v15.hdf5")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{HF_BASE}/{task}/mh/low_dim_v15.hdf5", path)
    return path


def _build_signal(name: str) -> Any:
    """Construct the curation signal, configured for robomimic's true eef path."""
    if name == "path_efficiency":
        from robocurate.signals.path_efficiency import PathEfficiency

        return PathEfficiency(source="observation.robot0_eef_pos", motion="positions")
    from robocurate.signals import get as get_signal

    return get_signal(name)


def prepare_filter_keys(
    src_path: str, dst_path: str, signal: str, budget: float, seed: int
) -> dict[str, int]:
    """Write a copy of ``src_path`` adding `curated`/`random_equalN` mask filter keys.

    Curates the existing ``train`` split by ``signal`` to ``budget``; the equal-N random subset
    is drawn (seeded) from the same train pool so the comparison is confound-free. The source
    file is read-only and never modified (invariant 1); all new keys land in the copy.
    """
    import shutil

    import h5py
    import numpy as np

    from robocurate.adapters import RoboMimicReader
    from robocurate.curator import Budget, Curator

    # Restrict curation to the train split (so valid stays held out for all arms).
    with h5py.File(src_path, "r") as f:
        train_names = {x.decode() for x in np.asarray(f["mask"]["train"]).reshape(-1)}

    reader = RoboMimicReader(src_path)
    # demo name per episode index, then keep only the train-split demos for curation.
    name_of = {t.meta.episode_index: t.meta.extra["source_demo"] for t in reader}
    curator = Curator([_build_signal(signal)], budget=Budget.fraction(budget), seed=seed)
    result = curator.run(reader)
    kept = [
        name_of[d.episode_index]
        for d in result.decisions
        if d.kept and name_of[d.episode_index] in train_names
    ]

    rng = np.random.default_rng(seed)
    train_list = sorted(train_names, key=lambda n: int(n.split("_")[1]))
    random_keep = sorted(rng.choice(train_list, size=len(kept), replace=False).tolist())

    shutil.copy2(src_path, dst_path)
    with h5py.File(dst_path, "a") as f:
        mask = f["mask"]
        for key in ("curated", "random_equalN"):
            if key in mask:
                del mask[key]
        mask.create_dataset("curated", data=np.array([n.encode() for n in sorted(kept)]))
        mask.create_dataset("random_equalN", data=np.array([n.encode() for n in random_keep]))
    return {
        "curated": len(kept),
        "random_equalN": len(random_keep),
        "train_total": len(train_names),
    }


def robosuite_smoke(task: str) -> dict[str, Any]:
    """Build a robomimic env from the dataset's env_args and step it — the gVisor gate.

    Returns whether a state-obs robosuite/MuJoCo env initializes and steps on Modal. If this
    fails the way SAPIEN did, rollout eval needs a non-gVisor host (curation/training still run).
    """
    import h5py
    import numpy as np
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.obs_utils as ObsUtils

    path = _ensure_dataset(task)
    # robomimic requires its obs-modality registry initialized before env.get_observation;
    # the real training path does this from the config — for the standalone smoke we register
    # every low_dim obs key found in the dataset (no rgb keys in a low_dim dataset).
    with h5py.File(path, "r") as f:
        demo0 = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[0]
        low_dim_keys = list(f["data"][demo0]["obs"].keys())
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": low_dim_keys, "rgb": []}})

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=path)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=False)
    obs = env.reset()
    for _ in range(5):
        obs, *_ = env.step(np.zeros(env.action_dimension))
    return {
        "ok": True,
        "action_dim": int(env.action_dimension),
        "n_obs_keys": len(obs),
        "obs_keys": sorted(obs.keys())[:8],
    }


@app.function(timeout=1800)  # CPU only — the cheap gate; no GPU billing, no training.
def smoke(task: str = "lift") -> dict[str, Any]:
    """Cheapest first step: verify the two things that gate everything else.

    (1) the curation -> filter-key prep runs on the worker (our code + the dataset), and
    (2) a state-obs robosuite/MuJoCo env initializes and steps on Modal's gVisor sandbox —
    the open question, since this is what blocked ManiSkill. No GPU, no training. Uses lift
    (smallest download). Failures are returned, not raised, so one call reports the full state.
    """
    import tempfile
    import traceback

    src = _ensure_dataset(task)
    dst = tempfile.mktemp(suffix=".curated.hdf5")
    try:
        sizes = prepare_filter_keys(src, dst, signal=SIGNAL, budget=BUDGET, seed=0)
        prep_ok, prep_err = True, None
    except Exception:
        sizes, prep_ok, prep_err = None, False, traceback.format_exc()

    try:
        env_info, env_ok, env_err = robosuite_smoke(task), True, None
    except Exception:
        env_info, env_ok, env_err = None, False, traceback.format_exc()

    return {
        "task": task,
        "filter_key_prep_ok": prep_ok,
        "filter_key_sizes": sizes,
        "filter_key_error": prep_err,
        "robosuite_ok": env_ok,
        "robosuite_info": env_info,
        "robosuite_error": env_err,
    }


# robosuite low_dim obs keys for the policy (Lift/Can/Square share these; `object` dim varies).
LOW_DIM_OBS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def train_one(
    dataset_path: str,
    filter_key: str,
    output_dir: str,
    *,
    exp_name: str,
    seed: int,
    num_epochs: int,
    rollout_n: int,
    horizon: int,
    rate: int,
    policy: str = "mlp",
) -> float:
    """Train a BC-MLP on ``mask/<filter_key>`` with rollouts; return best rollout success rate.

    Uses robomimic master's API (verified against source): ``config.train.data`` is a list of
    dataset dicts; ``train()`` initializes obs utils itself and returns None; the best rollout
    success is read from the always-written ``last.pth`` checkpoint's
    ``variable_state["best_success_rate"]`` (-1.0 if no rollout ran).
    """
    import glob
    import os

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils
    from robomimic.config import config_factory
    from robomimic.scripts.train import train as robomimic_train

    config = config_factory(algo_name="bc")
    with config.values_unlocked():
        config.experiment.name = exp_name
        config.train.data = [{"path": dataset_path}]  # master wants a list of dataset dicts
        config.train.hdf5_filter_key = filter_key
        config.train.hdf5_validation_filter_key = None
        config.experiment.validate = False
        config.observation.modalities.obs.low_dim = LOW_DIM_OBS
        config.observation.modalities.obs.rgb = []
        config.algo.transformer.enabled = False
        if policy == "rnn":
            # BC-RNN-GMM — robomimic's strong policy on the hard tasks (e.g. Square MH), where
            # plain BC-MLP is near the floor. Field names verified against robomimic master
            # bc_config.py; rnn.horizon must match train.seq_length. (Slower/pricier than MLP.)
            config.algo.rnn.enabled = True
            config.algo.rnn.horizon = 10
            config.algo.rnn.hidden_dim = 1000
            config.algo.rnn.rnn_type = "LSTM"
            config.algo.gmm.enabled = True
            config.algo.actor_layer_dims = ()  # the RNN does the work; no extra MLP head
            config.train.seq_length = 10
        else:  # plain BC-MLP
            config.algo.rnn.enabled = False
            config.algo.gmm.enabled = False
        config.train.num_epochs = num_epochs
        config.experiment.epoch_every_n_steps = 100
        config.experiment.rollout.enabled = True
        config.experiment.rollout.n = rollout_n
        config.experiment.rollout.horizon = horizon
        config.experiment.rollout.rate = rate  # must be <= num_epochs or zero rollouts happen
        config.experiment.rollout.warmstart = 0
        config.experiment.rollout.terminate_on_success = True
        # CRITICAL cost fix: robomimic also rolls out whenever a checkpoint is saved, and
        # save.every_n_epochs defaults to 50 — which was firing a rollout every 50 epochs
        # (40 rollouts/seed over 2000 epochs, ~5 min each = HOURS), overriding `rate`. Disable
        # periodic + time saves so ONLY `rate` controls rollout frequency; last.pth (which carries
        # best_success_rate) is still written at the end regardless.
        config.experiment.save.enabled = True
        config.experiment.save.every_n_epochs = None
        config.experiment.save.every_n_seconds = None
        config.experiment.save.epochs = []
        config.experiment.render_video = False
        config.train.output_dir = output_dir
        config.train.seed = seed
        config.train.cuda = True

    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)
    config.lock()
    robomimic_train(config, device=device)

    run_dirs = sorted(glob.glob(os.path.join(os.path.expanduser(output_dir), exp_name, "*")))
    ckpt = FileUtils.load_dict_from_checkpoint(ckpt_path=os.path.join(run_dirs[-1], "last.pth"))
    best = ckpt["variable_state"]["best_success_rate"]  # {env_key: float}
    return float(max(best.values()))


@app.function(gpu=GPU, timeout=3600)
def cheap_train(task: str = "lift", signal: str = SIGNAL, budget: float = BUDGET) -> dict[str, Any]:
    """Cheap GPU discovery run: train ONE arm briefly + verify the rollout cadence is capped.

    Runs LONG ENOUGH (120 epochs > the 50-epoch save cadence) to confirm rollouts now follow
    ``rate`` only, not the checkpoint-save cadence. With rate=60 over 120 epochs there must be
    exactly TWO rollouts (epoch 60, 120); the earlier bug fired one every 50 epochs. The worker
    logs "Epoch N Rollouts took ..." once per rollout — count them. lift is BC-saturated, so this
    checks the machinery + cadence, not the curation effect.
    """
    import tempfile
    import traceback

    src = _ensure_dataset(task)
    dst = src.replace(".hdf5", ".curated.hdf5")
    sizes = prepare_filter_keys(src, dst, signal=signal, budget=budget, seed=0)
    out = tempfile.mkdtemp()
    try:
        success = train_one(
            dst,
            "curated",
            out,
            exp_name="cheap",
            seed=0,
            num_epochs=120,
            rollout_n=15,
            horizon=400,
            rate=60,  # expect exactly 2 rollouts (epoch 60, 120) if the save-trigger fix worked
        )
        return {"task": task, "filter_key_sizes": sizes, "curated_success": success, "ok": True}
    except Exception:
        return {
            "task": task,
            "filter_key_sizes": sizes,
            "error": traceback.format_exc(),
            "ok": False,
        }


def prepare_arm_filter_key(
    src_path: str, dst_path: str, *, arm: str, signal: str, budget: float, seed: int
) -> tuple[str, str]:
    """Return ``(dataset_path, filter_key)`` for one arm, writing a curated/random copy as needed.

    Arms:
      - ``full`` — the existing ``train`` split (no copy).
      - ``curated`` — curate the train split by ``signal`` (deterministic, seed-independent).
      - ``random`` — equal-N random subset (same demo COUNT as curated), fresh per ``seed``.
      - ``random_steps`` — length-matched random subset (same TOTAL transitions as curated, so the
        policy sees the same volume of data), fresh per ``seed``. This controls the horizon
        confound: directness skews the curated set shorter, and BC error compounds in horizon, so
        a curated-vs-equal-N win could be a length effect — this baseline removes that escape.
    Source is never mutated (invariant 1).
    """
    import shutil

    import h5py
    import numpy as np

    from robocurate.adapters import RoboMimicReader
    from robocurate.curator import Budget, Curator

    if arm == "full":
        return src_path, "train"

    # The random arms need the curated set's size/total-length to match, which is signal-
    # independent (a fraction of the same valid pool) — use a concrete signal, not the "-" marker.
    count_signal = signal if arm == "curated" else "path_efficiency"

    with h5py.File(src_path, "r") as f:
        train_names = {x.decode() for x in np.asarray(f["mask"]["train"]).reshape(-1)}
    reader = RoboMimicReader(src_path)
    name_of = {t.meta.episode_index: t.meta.extra["source_demo"] for t in reader}
    len_of = {t.meta.extra["source_demo"]: t.meta.num_steps for t in reader}
    result = Curator([_build_signal(count_signal)], budget=Budget.fraction(budget), seed=0).run(
        reader
    )
    curated = [
        name_of[d.episode_index]
        for d in result.decisions
        if d.kept and name_of[d.episode_index] in train_names
    ]
    pool = sorted(train_names, key=lambda n: int(n.split("_")[1]))

    if arm == "curated":
        keep, key = sorted(curated), "curated"
    elif arm == "random":
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(pool, size=len(curated), replace=False).tolist())
        key = "random"
    elif arm == "random_steps":
        # Add random demos (without replacement) until the cumulative transition count first
        # reaches the curated set's total — a length/volume-matched control of similar count.
        target = sum(len_of[n] for n in curated)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(pool).tolist()
        keep, total = [], 0
        for name in shuffled:
            if total >= target:
                break
            keep.append(name)
            total += len_of[name]
        keep, key = sorted(keep), "random_steps"
    else:
        raise ValueError(f"unknown arm {arm!r}")

    shutil.copy2(src_path, dst_path)
    with h5py.File(dst_path, "a") as f:
        if key in f["mask"]:
            del f["mask"][key]
        f["mask"].create_dataset(key, data=np.array([n.encode() for n in keep]))
    return dst_path, key


# Rollouts are CPU-bound MuJoCo physics, so a bigger GPU barely speeds them — the real win is
# fanning these single-arm jobs out across Modal workers (train_arm.map below) so they run
# concurrently. Use the cheap GPU (see GPU constant); one job per worker.
@app.function(gpu=GPU, timeout=10800)
def train_arm(combo: dict[str, Any]) -> dict[str, Any]:
    """One worker: prepare this arm's filter key, train BC + rollout, return its success rate."""
    import tempfile
    import traceback

    arm, signal, seed = combo["arm"], combo["signal"], combo["seed"]
    try:
        src = _ensure_dataset(combo["task"])
        dst = tempfile.mktemp(suffix=".hdf5")
        path, fkey = prepare_arm_filter_key(
            src, dst, arm=arm, signal=signal, budget=combo["budget"], seed=seed
        )
        success = train_one(
            path,
            fkey,
            tempfile.mkdtemp(),
            exp_name=f"{arm}_{signal}_{seed}",
            seed=seed,
            num_epochs=combo["num_epochs"],
            rollout_n=combo["rollout_n"],
            horizon=combo["horizon"],
            rate=combo["rate"],
            policy=combo.get("policy", "mlp"),
        )
        return {"arm": arm, "signal": signal, "seed": seed, "success": success, "ok": True}
    except Exception:
        return {
            "arm": arm,
            "signal": signal,
            "seed": seed,
            "error": traceback.format_exc()[-1500:],
            "ok": False,
        }


SIGNALS = ["path_efficiency", "action_noise"]  # signals under test (both recover operator skill)
# Two random controls: equal-N (same demo COUNT) and equal-steps (same TOTAL transitions, to
# control the horizon confound — curating by directness skews kept demos shorter, and BC error
# compounds ~quadratically in trajectory length).
RANDOM_ARMS = ["random", "random_steps"]


@app.local_entrypoint()
def run_baseline(task: str = TASK, seeds: int = 2) -> None:
    """M1 go/no-go: train ONLY the full arm; check it reproduces the published ~0.53 on Square MH.

    This is the cheap precondition (~$2-4) before any curation comparison. robomimic BC-MLP on
    Square MH low_dim reaches ~52.7% (Mandlekar 2021, Table 1) at the FULL 2000-epoch budget; an
    earlier 600-epoch run floored it at ~5% and was misread as an architecture limit. If this does
    not reach ~0.5, the pipeline is still broken — STOP and fix before spending on the comparison.
    """
    common = {
        "task": task,
        "budget": BUDGET,
        "policy": "mlp",
        "num_epochs": 2000,
        "rollout_n": 50,  # modest; this is a baseline sanity check, not the comparison
        "horizon": 400,
        "rate": 500,  # 4 eval checkpoints (500/1000/1500/2000); best_success = max over them
    }
    combos = [{**common, "arm": "full", "signal": "-", "seed": s} for s in range(seeds)]
    results = list(train_arm.map(combos))
    succ = [r["success"] for r in results if r.get("ok") and r["success"] >= 0]
    print(json.dumps(results, indent=2, sort_keys=True))
    # Published low_dim BC-MLP (Mandlekar 2021, Table 1) for the go/no-go reference.
    published = {"lift": 1.00, "can": 0.86, "square": 0.53}.get(task)
    if succ:
        mean = sum(succ) / len(succ)
        vals = sorted(round(v, 3) for v in succ)
        print(f"\nfull {task}-MH BC-MLP success: mean={mean:.3f} {vals}")
        if published is not None:
            ratio = mean / published
            verdict = "reproduces" if ratio >= 0.8 else "FAR BELOW (harness not trustworthy yet)"
            print(
                f"published ~{published:.2f} (Mandlekar Table 1) -> "
                f"ours is {ratio:.0%} of it: {verdict}"
            )
    else:
        print("\nNo successful runs — inspect the errors above before spending further.")


@app.local_entrypoint()
def main(
    task: str = TASK,
    policy: str = "mlp",
    seeds: int = 3,
    rollout_n: int = 100,
    rate: int = 700,
    as_json: bool = False,
) -> None:
    """Fan out the budget-fit validation (both signals) across Modal workers; print the gaps.

    Arms per seed: full (train split) + equal-N random + equal-steps (length-matched) random + a
    curated subset per signal. full/random arms are signal-independent, so they run once and are
    shared. All combos run concurrently via train_arm.map.

    Training uses robomimic's full budget (num_epochs=2000, ~200k gradient steps) — required to
    reach the published ~52.7% on Square MH. NOTE: defaults here (3 seeds, 100 rollouts, ~3 eval
    checkpoints) are BUDGET-FIT, not the rigorous bar — the result is *suggestive, not
    CI-definitive*; the rigorous version (>=5 seeds, >=200 rollouts) is the future ask once more
    compute exists. ``policy`` is ``mlp`` (default) or ``rnn`` (BC-RNN-GMM, a higher ceiling).
    """
    common = {
        "task": task,
        "budget": BUDGET,
        "policy": policy,
        "num_epochs": 2000,  # robomimic default; less undertrains (earlier 600-epoch floored ~5%)
        "rollout_n": rollout_n,
        "horizon": 400,
        "rate": rate,  # rollouts now follow rate only (periodic-save trigger disabled in train_one)
    }
    combos: list[dict[str, Any]] = []
    for seed in range(seeds):
        combos.append({**common, "arm": "full", "signal": "-", "seed": seed})
        for rarm in RANDOM_ARMS:
            combos.append({**common, "arm": rarm, "signal": "-", "seed": seed})
        for sig in SIGNALS:
            combos.append({**common, "arm": "curated", "signal": sig, "seed": seed})

    results = list(train_arm.map(combos))

    # Always persist the raw results locally (gitignored) so one run yields both the readable
    # summary below and a JSON the plotter (plot_validation_results.py) can render.
    from pathlib import Path

    out_json = Path(__file__).parent / "data" / f"validation_{task}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"(raw results saved to {out_json})")

    if as_json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    ok = [r for r in results if r.get("ok")]
    fails = [r for r in results if not r.get("ok")]

    def by_seed(arm: str, signal: str | None = None) -> dict[int, float]:
        return {
            r["seed"]: r["success"]
            for r in ok
            if r["arm"] == arm and (signal is None or r["signal"] == signal)
        }

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def paired_delta(curated: dict[int, float], control: dict[int, float]) -> tuple[float, float]:
        """Mean and ~95% (t, df=n-1, ~2*SE) half-width of the per-seed paired difference."""
        deltas = [curated[s] - control[s] for s in curated if s in control]
        if not deltas:
            return float("nan"), float("nan")
        m = mean(deltas)
        if len(deltas) < 2:
            return m, float("nan")
        var = sum((d - m) ** 2 for d in deltas) / (len(deltas) - 1)
        return m, 2.0 * (var / len(deltas)) ** 0.5  # rough 95% half-width; n is small, report it

    print(f"\nrobomimic/{task} MH — downstream BC validation ({policy.upper()}, {seeds} seeds)")
    arms = {
        "full": "full (all train)",
        "random": "equal-N random",
        "random_steps": "len-matched random",
    }
    seeded = {arm: by_seed(arm) for arm in arms}
    for arm, label in arms.items():
        vals = list(seeded[arm].values())
        print(f"  {label:22s} {mean(vals):.3f}  {sorted(round(v, 3) for v in vals)}")
    print("\n  curated vs each control (paired per-seed Δ, mean ± ~95% half-width):")
    for sig in SIGNALS:
        cur = by_seed("curated", sig)
        print(
            f"  {'curated:' + sig:22s} {mean(list(cur.values())):.3f}  "
            f"{sorted(round(v, 3) for v in cur.values())}"
        )
        for arm in ("random", "random_steps"):
            d, hw = paired_delta(cur, seeded[arm])
            sep = "separated" if d - hw > 0 else "NOT separated"
            print(f"      vs {arms[arm]:20s} Δ = {d:+.3f} ± {hw:.3f}  ({sep})")
    print(
        f"\nShip a signal only if curated beats BOTH random controls with the CI excluding 0. "
        f"With {seeds} seeds (budget-fit) the CI is wide — treat this as SUGGESTIVE, not "
        "definitive, and confirm the baseline reproduces the published ~0.53 on Square MH first."
    )
    if fails:
        print(f"\n{len(fails)} run(s) failed:")
        for r in fails:
            print(f"  {r['arm']}/{r['signal']}/seed{r['seed']}: {r['error'].splitlines()[-1]}")


@app.local_entrypoint()
def run_smoke(task: str = "lift") -> None:
    """Cheap CPU gate: does the curation prep + robosuite state-eval work on Modal?"""
    result = smoke.remote(task=task)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["robosuite_ok"] and result["filter_key_prep_ok"]:
        print("\nSMOKE PASSED — robosuite state-eval runs on Modal; safe to wire the cheap train.")
    else:
        print("\nSMOKE FAILED — see the error above; rollout eval may need a non-gVisor host.")


@app.local_entrypoint()
def run_cheap_train(task: str = "lift") -> None:
    """Cheap GPU discovery run: verify robomimic BC train + rollout + success extraction works."""
    result = cheap_train.remote(task=task)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("ok"):
        s = result["curated_success"]
        verdict = "no rollout ran" if s < 0 else f"curated success={s:.3f}"
        print(f"\nCHEAP TRAIN OK — pipeline works on Modal GPU ({verdict}).")
    else:
        print("\nCHEAP TRAIN FAILED — see the traceback above.")
