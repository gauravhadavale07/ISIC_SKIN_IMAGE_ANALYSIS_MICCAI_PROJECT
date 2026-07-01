#!/usr/bin/env python3
"""Figure 4: Overall performance comparison (grouped bar chart with error bars)."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import all_metric_stats, load_progress
from viz_style import MODEL_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure

METRICS = [
    ("Accuracy", "Accuracy"),
    ("AUROC", "AUROC"),
    ("Precision (Macro)", "Precision"),
    ("Recall (Macro)", "Recall"),
    ("F1 (Macro)", "Macro F1"),
]


def main():
    apply_style()
    progress = load_progress()

    fig, ax = plt.subplots(figsize=(12, 6))
    n_metrics = len(METRICS)
    n_models = len(MODELS)
    x = np.arange(n_metrics)
    width = 0.24

    for i, model in enumerate(MODELS):
        means, stds = [], []
        for key, _ in METRICS:
            m, s = all_metric_stats(progress, key)[model]
            means.append(m)
            stds.append(s)
        offset = (i - 1) * width
        ax.bar(
            x + offset, means, width, yerr=stds,
            label=MODEL_SHORT[model], color=MODEL_COLORS[model],
            capsize=4, error_kw=dict(linewidth=1.2, capthick=1.2),
            edgecolor="white", linewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in METRICS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Architecture", frameon=True, edgecolor="#CCCCCC")
    ax.set_title(
        "OOD Performance on PAD-UFES-20 (Mean ± Std, 6 Seeds)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "fig04_overall_performance")


if __name__ == "__main__":
    main()
