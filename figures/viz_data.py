"""Data loading helpers for figure scripts (read-only, no project modifications)."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from viz_style import CLASS_NAMES, DATA_DIR, FIGURES_DIR, MODELS, PROJECT_ROOT

PROGRESS_JSON = PROJECT_ROOT / "results" / "experiment_progress_v3.json"
LOG_CANDIDATES = [
    PROJECT_ROOT / "experiment_run_full.log",
    PROJECT_ROOT / "experiment_run.log",
]


def load_progress() -> dict:
    """Load aggregated multi-seed results."""
    if not PROGRESS_JSON.exists():
        raise FileNotFoundError(
            f"Missing {PROGRESS_JSON}. Run run_experiment.py first."
        )
    with open(PROGRESS_JSON) as f:
        return json.load(f)


def metric_stats(progress: dict, model: str, metric: str) -> Tuple[float, float]:
    """Return (mean, std) for a scalar metric across seeds."""
    values = progress["results"][model][metric]
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def all_metric_stats(progress: dict, metric: str) -> Dict[str, Tuple[float, float]]:
    return {m: metric_stats(progress, m, metric) for m in MODELS}


def load_json_data(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python figures/export_figure_data.py"
        )
    with open(path) as f:
        return json.load(f)


def load_npz_data(name: str) -> dict:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python figures/export_figure_data.py"
        )
    return dict(np.load(path, allow_pickle=True))


def parse_training_logs() -> Dict[str, List[dict]]:
    """
    Parse training epoch summaries from experiment logs.
    Returns {run_name: [{epoch, train_loss, train_acc, val_loss, val_acc}, ...]}.
    """
    pattern_run = re.compile(r"Starting Training Run: (.+)")
    pattern_epoch = re.compile(r"Epoch (\d+) Summary:")
    pattern_train = re.compile(r"Train Loss:\s+([\d.]+)\s+\|\s+Train Acc:\s+([\d.]+)%")
    pattern_val = re.compile(r"Val Loss:\s+([\d.]+)\s+\|\s+Val Acc:\s+([\d.]+)%")

    runs: Dict[str, List[dict]] = {}
    current_run: Optional[str] = None
    current_epoch: Optional[int] = None
    pending: dict = {}

    for log_path in LOG_CANDIDATES:
        if not log_path.exists():
            continue
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                m = pattern_run.search(line)
                if m:
                    current_run = m.group(1).strip()
                    runs.setdefault(current_run, [])
                    continue
                m = pattern_epoch.search(line)
                if m and current_run:
                    current_epoch = int(m.group(1))
                    pending = {"epoch": current_epoch}
                    continue
                m = pattern_train.search(line)
                if m and current_run and current_epoch:
                    pending["train_loss"] = float(m.group(1))
                    pending["train_acc"] = float(m.group(2))
                    continue
                m = pattern_val.search(line)
                if m and current_run and current_epoch and pending:
                    pending["val_loss"] = float(m.group(1))
                    pending["val_acc"] = float(m.group(2))
                    runs[current_run].append(pending)
                    pending = {}
                    current_epoch = None

    # Deduplicate epochs per run (logs may repeat)
    for run_name, epochs in runs.items():
        seen = {}
        for ep in epochs:
            seen[ep["epoch"]] = ep
        runs[run_name] = [seen[k] for k in sorted(seen)]

    return runs


def run_name_to_model_seed(run_name: str) -> Tuple[str, int]:
    """Map 'Late_Fusion_seed_42' -> ('Late Fusion', 42)."""
    mapping = {
        "Late_Fusion": "Late Fusion",
        "GMU_Baseline": "GMU Baseline",
        "Cross_Attn_VtoT": "Cross-Attn V→T",
        "Cross_Attn_TtoV": "Cross-Attn T→V",
        "Image_Only": "Image-Only",
        "Text_Only": "Text-Only",
    }
    for key, model in mapping.items():
        if run_name.startswith(key + "_seed_"):
            seed = int(run_name.split("_seed_")[-1])
            return model, seed
    raise ValueError(f"Unknown run name: {run_name}")


def aggregate_training_by_model(runs: dict, seed: int = 42) -> Dict[str, List[dict]]:
    """Get training curves for each model at a given seed."""
    out: Dict[str, List[dict]] = {}
    for run_name, epochs in runs.items():
        try:
            model, s = run_name_to_model_seed(run_name)
        except ValueError:
            continue
        if s == seed and epochs:
            out[model] = epochs
    return out


def sanitize_model_key(model: str) -> str:
    return model.replace(" ", "_")


def ci_95(values: List[float]) -> Tuple[float, float, float]:
    """Return mean, lower, upper for 95% CI (t-distribution, small n)."""
    from scipy import stats
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    if n < 2:
        return mean, mean, mean
    sem = stats.sem(arr)
    h = sem * stats.t.ppf(0.975, n - 1)
    return mean, mean - h, mean + h
