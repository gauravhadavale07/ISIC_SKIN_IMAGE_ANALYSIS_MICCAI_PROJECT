import json
import numpy as np

def metric_stats(progress, model, metric):
    values = progress["results"][model][metric]
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))

with open("results/experiment_progress.json") as f:
    progress = json.load(f)

MODELS = ['Late Fusion', 'GMU Baseline', 'Cross-Attention T→V', 'Cross-Attention V→T']
cka_means = [metric_stats(progress, m, "Linear_CKA")[0] for m in MODELS]
print("cka_means:", cka_means)
