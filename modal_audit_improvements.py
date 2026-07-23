"""
Comprehensive Modal runner for the MICCAI audit improvements.

Runs the following tasks on an H200 GPU:
  - task11_sae.py       (SAE + Artifact Verification with FDR)
  - task16_lvlm_steering.py (LVLM Robustness: ActAdd + CAA + Entanglement)
  - task25_sota_linear_probing.py (SOTA FM Linear Probing)
  - task26_large_scale_fairness_patching.py (Fairness Patching + Power Analysis)
  - task27_causal_mediation_grounding.py (Causal Mediation + Token Nulling)

Usage:
  modal run modal_audit_improvements.py           # run all tasks
  modal run modal_audit_improvements.py --task 11 # run only task 11
  modal run modal_audit_improvements.py --dry-run # dry-run tasks 16/25
"""

import modal
import subprocess
import os
import sys

app = modal.App("miccai-audit-improvements")

vol_checkpoints = modal.Volume.from_name("miccai-checkpoints", create_if_missing=True)
vol_results     = modal.Volume.from_name("miccai-results",     create_if_missing=True)
vol_hf_cache    = modal.Volume.from_name("miccai-hf-cache",    create_if_missing=True)
vol_data        = modal.Volume.from_name("miccai-data",        create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "torchvision", "torchaudio",
        "tqdm", "pandas", "numpy",
        "scikit-learn", "transformers", "accelerate",
        "pillow", "scipy", "hf_transfer",
        "timm", "scikit-image",
        "statsmodels", "open_clip_torch",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_TOKEN": "YOUR_HF_TOKEN"
    })
    .workdir("/root/project")
    .add_local_dir(
        "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT",
        remote_path="/root/project",
        ignore=[
            "results", "logs", ".git", "__pycache__",
            "data",
            "*.log", "*.tar.gz", "*.zip", "*.pdf",
            "modal_*_out.txt",
        ],
    )
    .add_local_dir(
        "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data",
        remote_path="/root/project/data",
    )
)


def _run(script: str, extra_args: list[str] = None) -> bool:
    """Helper: run a script and return True on success."""
    cmd = ["python3", script] + (extra_args or [])
    print(f"\n{'=' * 60}\nRunning: {' '.join(cmd)}\n{'=' * 60}")
    result = subprocess.run(cmd, cwd="/root/project")
    if result.returncode != 0:
        print(f"ERROR: {script} exited with code {result.returncode}")
        return False
    return True


@app.function(
    gpu="H200",
    image=image,
    timeout=7200,   # 2 hours
    volumes={
        "/root/project/results":     vol_results,
        "/root/.cache/huggingface":  vol_hf_cache,
    },
)
def run_task11():
    """Task 11: SAE Training + Biopsy Artifact Verification (FDR-corrected)."""
    ok = _run("task11_sae.py")
    vol_results.commit()
    return ok


@app.function(
    gpu="H200",
    image=image,
    timeout=14400,  # 4 hours (multiple LLM models)
    volumes={
        "/root/project/results":    vol_results,
        "/root/.cache/huggingface": vol_hf_cache,
    },
)
def run_task16(dry_run: bool = False):
    """Task 16: LVLM Robustness (ActAdd + CAA + entanglement diagnostic)."""
    extra = ["--dry-run"] if dry_run else []
    ok = _run("task16_lvlm_steering.py", extra)
    vol_results.commit()
    return ok


@app.function(
    gpu="H200",
    image=image,
    timeout=7200,
    volumes={
        "/root/project/results":    vol_results,
        "/root/.cache/huggingface": vol_hf_cache,
    },
)
def run_task25(dry_run: bool = False):
    """Task 25: SOTA Dermatology FM Linear Probing."""
    extra = ["--dry-run"] if dry_run else []
    ok = _run("task25_sota_linear_probing.py", extra)
    vol_results.commit()
    return ok


@app.function(
    gpu="H200",
    image=image,
    timeout=7200,
    volumes={
        "/root/project/results":     vol_results,
    },
)
def run_task26():
    """Task 26: Large-Scale Fairness Patching + Formal Power Analysis."""
    ok = _run("task26_large_scale_fairness_patching.py")
    vol_results.commit()
    return ok


@app.function(
    gpu="H200",
    image=image,
    timeout=7200,
    volumes={
        "/root/project/results":     vol_results,
    },
)
def run_task27():
    """Task 27: Causal Mediation Analysis + Token/Span Nulling."""
    ok = _run("task27_causal_mediation_grounding.py")
    vol_results.commit()
    return ok


@app.local_entrypoint()
def main(task: str = "all", dry_run: bool = False):
    """
    Args:
      --task    : one of '11', '16', '25', '26', '27', 'all' (default: all)
      --dry-run : pass dry-run to tasks 16 and 25
    """
    print(f"Starting MICCAI Audit Improvements (task={task}, dry_run={dry_run})")

    task_map = {
        "11": ("run_task11",  run_task11,  {}),
        "16": ("run_task16",  run_task16,  {"dry_run": dry_run}),
        "25": ("run_task25",  run_task25,  {"dry_run": dry_run}),
        "26": ("run_task26",  run_task26,  {}),
        "27": ("run_task27",  run_task27,  {}),
    }

    tasks_to_run = list(task_map.keys()) if task == "all" else [task]
    results = {}

    for t in tasks_to_run:
        if t not in task_map:
            print(f"Unknown task: {t}. Choose from {list(task_map.keys())} or 'all'.")
            continue
        name, fn, kwargs = task_map[t]
        print(f"\n>>> Launching {name}...")
        try:
            ok = fn.remote(**kwargs)
            results[t] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"  ERROR launching {name}: {e}")
            results[t] = "ERROR"

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    for t, status in results.items():
        name = task_map[t][0]
        print(f"  {name:<20s}: {status}")
