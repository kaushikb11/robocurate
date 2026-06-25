"""Roll out a random policy in a state-obs ManiSkill3 task — GPU smoke test.

!!! KNOWN-BLOCKED ON MODAL — DO NOT EXPECT THIS TO WORK ON MODAL. !!!

Investigation result (kept here so it isn't rediscovered): Modal runs containers on the
**gVisor** sandbox, whose ``nvproxy`` passes only the ``compute``/``utility`` GPU
capabilities by default. **Vulkan needs the ``graphics`` capability**, gated by a host-side
runsc flag (``--nvproxy-allowed-driver-capabilities``) that only Modal can set. So on Modal
the NVIDIA driver libraries are all present (``libGLX_nvidia.so.0`` + companions resolve),
but ``vkCreateDevice`` is blocked at the runtime layer → SAPIEN fails with
``vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed`` even in state-obs mode
(SAPIEN always builds a Vulkan ``RenderSystem``). CUDA works fine, which is why the curation /
BC-training experiments (``modal_app.py``) succeed — only the sim renderer is blocked.

So: ManiSkill rendering needs a **non-gVisor GPU host** — RunPod / Lambda / bare metal with
the standard ``nvidia-container-runtime`` and full driver capabilities. The image below (CUDA
base + NVIDIA Vulkan ICD + Optimus layer + glvnd EGL vendor file) is the correct recipe for
such a host; it is retained for that purpose and as the documented diagnosis. The function
prints a Vulkan diagnostic first so the failure mode is self-explanatory if run on Modal.

Sources: gVisor GPU docs (capabilities default + "Vulkan requires graphics"); Modal uses
gVisor (their infra writeups); SAPIEN/ManiSkill issues #270/#922.

Usage on a Vulkan-capable host (needs ``modal token new`` only if run via Modal):

    modal run experiments/maniskill_modal.py            # PickCube-v1 smoke
"""

from __future__ import annotations

import json

import modal

GPU = "A10G"

# ManiSkill3 image. SAPIEN initialises its Vulkan render system even for state-only obs, so the
# full GPU graphics stack is required: a CUDA base image, the NVIDIA Vulkan ICD, the Vulkan/GL
# apt libs, and NVIDIA_DRIVER_CAPABILITIES=all so Modal exposes the graphics driver libs.
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
maniskill_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        # graphics (SAPIEN/Vulkan) + build tools (mani_skill builds toppra from source).
        # Modal DOES provide libGLX_nvidia.so.0 (the NVIDIA Vulkan driver), so we point the
        # loader at the NVIDIA ICD + Optimus layer below; mesa is kept only for vulkaninfo.
        "libvulkan1",
        "vulkan-tools",
        "mesa-vulkan-drivers",
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
    # pip_install BEFORE the volatile Vulkan config so the slow mani_skill build stays cached
    # across iterations on the ICD/layer/env settings.
    .pip_install("torch>=2.2", "mani_skill>=3.0.0b")
    .add_local_dir(".", "/pkg", copy=True, ignore=_IGNORE)
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d /etc/vulkan/implicit_layer.d "
        "/usr/share/glvnd/egl_vendor.d",
        "cp /pkg/experiments/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json",
        "cp /pkg/experiments/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json",
        "cp /pkg/experiments/nvidia_egl_vendor.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "pip install /pkg",
    )
    .env(
        {
            "MS_SKIP_ASSET_DOWNLOAD_PROMPT": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            # NVIDIA ICD only (the CUDA-matched device SAPIEN needs), and the glvnd EGL vendor
            # file SAPIEN's _vulkan_tricks looks for. Correct for a non-gVisor host; on Modal
            # the graphics ioctls are still blocked by gVisor nvproxy (see module docstring).
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        }
    )
)

app = modal.App("robocurate-maniskill", image=maniskill_image)


@app.function(gpu=GPU, timeout=900)
def maniskill_smoke(
    task_id: str = "PickCube-v1", episodes: int = 16, seed: int = 0
) -> dict[str, object]:
    """Roll out a random policy in ``task_id`` (state obs) and return the success rate.

    Builds the env once (no separate probe), runs for the task's ``max_episode_steps``.
    Prints a Vulkan diagnostic first so even a failure tells us the GPU/Vulkan state on Modal.
    """
    import subprocess

    from robocurate.experiment.maniskill import ManiSkillEnvironment

    diag = subprocess.run(
        [
            "bash",
            "-lc",
            "echo '== libGLX_nvidia =='; (ldconfig -p | grep -i libGLX_nvidia || echo MISSING); "
            "echo '== ldd libGLX_nvidia (missing companion libs?) =='; "
            "(ldd /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>&1 | grep -i 'not found' "
            "|| echo 'all deps resolved'); "
            "echo '== libnvidia-* present =='; "
            "(ls /usr/lib/x86_64-linux-gnu/libnvidia-* 2>&1 | xargs -n1 basename 2>/dev/null "
            "| tr '\\n' ' ' || echo none); echo; "
            "echo '== VK_ICD_FILENAMES =='; echo \"$VK_ICD_FILENAMES\"; "
            "echo '== vulkaninfo --summary (does the NVIDIA GPU appear?) =='; "
            "(vulkaninfo --summary 2>&1 | grep -iE 'deviceName|driverName|deviceType|GPU' "
            "| head -20 || true)",
        ],
        capture_output=True,
        text=True,
    )
    print(diag.stdout, diag.stderr)

    env = ManiSkillEnvironment(task_id=task_id, obs_mode="state")
    result = env.smoke_rollout(episodes=episodes, seed=seed)
    return {
        "task": task_id,
        "episodes": result.n_episodes,
        "success_rate": result.success_rate,
    }


@app.local_entrypoint()
def main(task_id: str = "PickCube-v1", episodes: int = 16) -> None:
    """Launch the ManiSkill smoke rollout on Modal and print the result."""
    result = maniskill_smoke.remote(task_id=task_id, episodes=episodes)
    print(json.dumps(result, indent=2))
    print(
        f"\nManiSkill ran on Modal ({GPU}): {result['task']} "
        f"random-policy success_rate={result['success_rate']}"
    )
