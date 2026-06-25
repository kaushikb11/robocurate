"""Full ManiSkill3 headline experiment on a Modal GPU: demos -> curate -> train -> rollout.

This is the real vertical. On a GPU worker it: (1) downloads ManiSkill demonstrations for a
task and replays them to flat ``state`` observations; (2) reads them with
:class:`~robocurate.adapters.maniskill_demos.ManiSkillDemoReader`; (3) curates with CUPID;
(4) trains the BC policy on the curated vs control subsets; (5) evaluates each in
:class:`~robocurate.experiment.maniskill.ManiSkillEnvironment` for REAL success rates — the
curated-vs-equal-N headline, now with real physics instead of the fake env.

It runs the same ``run_config`` path verified locally; only the dataset (real demos) and the
environment (real ManiSkill) differ from the synthetic demo.

!!! BLOCKED ON MODAL — needs a non-gVisor GPU host (RunPod/Lambda/bare metal). !!!
The sim renderer cannot initialise on Modal: gVisor's nvproxy blocks the Vulkan ``graphics``
capability, so SAPIEN's ``vkCreateDevice`` fails (CUDA works, rendering doesn't). See the full
diagnosis in ``experiments/maniskill_modal.py``. Run this on a host with the standard
nvidia-container-runtime and full driver capabilities. The remaining iteration points there:
  1. The demo download + replay step — ``download_demo`` / ``replay_trajectory`` flags and the
     output path can vary by ManiSkill version (we glob for the file and fail loudly).
  2. The control mode must match between the replayed demos and the env (both read
     ``control_mode``), or the trained policy's actions won't fit the env's action space.
"""

from __future__ import annotations

import json
from typing import Any

import modal

GPU = "A10G"
TASK = "PickCube-v1"
CONTROL_MODE = "pd_joint_delta_pos"

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
]
# Same Vulkan-enabled CUDA image as the smoke app (SAPIEN inits the renderer even for state
# obs), plus robocurate[all] (torch) + h5py for the demo reader.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        # graphics (SAPIEN/Vulkan) + build tools (mani_skill builds toppra from source)
        "libvulkan1",
        "vulkan-tools",
        "libegl1",
        "libgl1",
        "libglvnd0",
        "libxext6",
        "git",
        "build-essential",
        "clang",
        "cmake",
        "libglib2.0-0",
        "libsm6",  # OpenCV (cv2) runtime libs pulled in by mani_skill.envs
    )
    .env({"NVIDIA_DRIVER_CAPABILITIES": "all", "MS_SKIP_ASSET_DOWNLOAD_PROMPT": "1"})
    .pip_install("torch>=2.2", "mani_skill>=3.0.0b", "h5py>=3.0")
    .add_local_dir(".", "/pkg", copy=True, ignore=_IGNORE)
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d",
        "cp /pkg/experiments/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json",
        "pip install '/pkg[all]'",
    )
)

app = modal.App("robocurate-maniskill-experiment", image=image)


def _prepare_demos(task: str, control_mode: str) -> str:
    """Download + replay ManiSkill demos to flat state observations; return the .h5 path.

    Commands per the ManiSkill3 docs: download motionplanning demos (generated with
    ``pd_joint_pos`` / no obs), then replay them to ``state`` obs + the chosen control mode.
    The output is ``trajectory.state.<control_mode>.physx_cpu.h5`` next to the input.
    """
    import glob
    import os
    import subprocess

    demos_root = os.path.join(os.path.expanduser("~"), ".maniskill", "demos", task)
    subprocess.run(["python", "-m", "mani_skill.utils.download_demo", task], check=True)
    raws = glob.glob(f"{demos_root}/**/trajectory.h5", recursive=True)
    if not raws:
        raise FileNotFoundError(f"no raw demos for {task} after download; inspect {demos_root}")
    subprocess.run(
        [
            "python",
            "-m",
            "mani_skill.trajectory.replay_trajectory",
            "--traj-path",
            raws[0],
            "--use-first-env-state",
            "-o",
            "state",
            "-c",
            control_mode,
            "-b",
            "physx_cpu",
            "--num-procs",
            "4",
            "--save-traj",
        ],
        check=True,
    )
    matches = glob.glob(f"{demos_root}/**/trajectory.state.*.h5", recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"no replayed state demos found for {task}; inspect {demos_root} and the replay flags"
        )
    return matches[0]


@app.function(gpu=GPU, timeout=3600)
def run_maniskill_experiment(
    task: str = TASK,
    control_mode: str = CONTROL_MODE,
    budget: float = 0.5,
    seeds: int = 3,
    eval_episodes: int = 20,
) -> dict[str, Any]:
    """Demos -> curate -> train BC -> evaluate in ManiSkill; return the report dict.

    Defaults are deliberately small for a cheap first run: ``eval_episodes`` is the number of
    parallel envs, and evaluation currently steps the policy per-env in Python (one forward
    pass per env per step), so scale up only after a small run confirms wall-clock and that
    the GPU fits the env count. A batched-action rollout is the follow-up optimization.
    """
    from robocurate.experiment import ExperimentConfig, run_config

    demo_path = _prepare_demos(task, control_mode)
    config = ExperimentConfig(
        dataset={"kind": "maniskill_demos", "params": {"path": demo_path}},
        signals=[{"name": "cupid", "params": {"mode": "tracin"}}],
        budget={"kind": "fraction", "value": budget},
        policy={"name": "bc", "params": {"epochs": 300}},
        environment={
            "name": "maniskill",
            "params": {"task_id": task, "obs_mode": "state", "control_mode": control_mode},
        },
        seeds=list(range(seeds)),
        eval_episodes=eval_episodes,
        include_ablations=True,
    )
    return run_config(config).to_dict()


@app.local_entrypoint()
def main(task: str = TASK, as_json: bool = False) -> None:
    """Launch the full ManiSkill headline experiment on Modal and print the result."""
    result = run_maniskill_experiment.remote(task=task)
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    headline = result["headline"]["curated_vs_equal_n_random"]
    print(f"ManiSkill experiment: {result['dataset_id']}  (GPU={GPU})")
    if headline is not None:
        verdict = "separated" if headline["separated"] else "not separated"
        print(
            f"Curated vs equal-N random success: {headline['effect']:+.3f} "
            f"(95% CI [{headline['ci_low']:+.3f}, {headline['ci_high']:+.3f}]) — {verdict}"
        )
    for arm in result["arms"]:
        s = arm["success"]
        print(f"  {arm['name']:18} {s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
