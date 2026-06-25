"""Run a RoboCurate headline experiment on a Modal GPU.

This is the Modal execution backend for the experiment harness. It ships an
:class:`~robocurate.experiment.ExperimentConfig` (as a plain dict) to a GPU worker, which
runs the exact ``run_config`` path verified locally — only the resolved device differs
(``cuda`` on the worker, so the torch-backed CUPID + BC components use the GPU automatically).

The job builds its synthetic dataset on the worker, so there is zero data plumbing for the
first run. Real datasets (a Modal Volume / HF download) are a follow-up.

Usage (needs ``modal token new`` once, and ``robocurate[modal]`` installed locally):

    modal run experiments/modal_app.py                 # default CUPID + BC config
    modal run experiments/modal_app.py --as-json       # machine-readable report

To curate without running the full experiment, ``run_experiment`` takes any config dict.
"""

from __future__ import annotations

import json
from typing import Any

import modal

GPU = "A10G"

# The image installs robocurate[all] (numpy + pyarrow + torch; not the heavy TF rlds extra)
# from the local project, so the package metadata — and thus the signal entry points — are
# present on the worker. copy=True makes the source available at build time for pip install.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(
        ".",
        "/pkg",
        copy=True,
        ignore=[
            "**/.venv",
            "**/.git",
            "**/__pycache__",
            "**/*.egg-info",
            "**/dist",
            "**/build",
            "**/.mypy_cache",
            "**/.ruff_cache",
            "**/.pytest_cache",
        ],
    )
    .run_commands("pip install '/pkg[all]'")
)

app = modal.App("robocurate-experiment", image=image)

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {"kind": "identity_synthetic", "params": {"num_helpful": 12, "num_harmful": 4}},
    "signals": [{"name": "cupid", "params": {"mode": "tracin"}}],
    "budget": {"kind": "fraction", "value": 0.5},
    "combiner": None,
    "policy": {"name": "bc", "params": {"epochs": 250}},
    "environment": {"name": "fake", "params": {}},
    "seed": 0,
    "seeds": [0, 1, 2, 3, 4],
    "eval_episodes": 200,
    "include_ablations": True,
    "include_random_filter": True,
    "stats_seed": 0,
}


@app.function(gpu=GPU, timeout=1800)
def run_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Run one experiment from a config dict on the GPU worker; return the report dict."""
    from robocurate.experiment import ExperimentConfig, run_config

    report = run_config(ExperimentConfig.from_dict(config))
    return report.to_dict()


@app.local_entrypoint()
def main(as_json: bool = False) -> None:
    """Launch the default experiment on Modal and print the headline result."""
    result = run_experiment.remote(DEFAULT_CONFIG)
    if as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    headline = result["headline"]["curated_vs_equal_n_random"]
    print(f"Experiment: {result['dataset_id']}  ({len(result['seeds'])} seeds, GPU={GPU})")
    if headline is not None:
        verdict = "separated" if headline["separated"] else "not separated"
        print(
            f"Curated vs equal-N random: {headline['effect']:+.3f} "
            f"(95% CI [{headline['ci_low']:+.3f}, {headline['ci_high']:+.3f}]) — {verdict}"
        )
    for arm in result["arms"]:
        s = arm["success"]
        print(f"  {arm['name']:18} {s['mean']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
