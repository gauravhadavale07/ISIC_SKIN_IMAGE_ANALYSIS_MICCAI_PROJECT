#!/usr/bin/env python3
"""Figure 5: Blank-text ablation figure."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import load_progress, metric_stats
from viz_style import MODEL_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure


def main():
    apply_style()
    progress = load_progress()

    metrics = ["Real_Accuracy", "Blank_Accuracy", "Blank_Accuracy_Drop"]
    labels = ["Real Accuracy", "Blank Accuracy", "Accuracy Drop"]
    x = np.arange(len(MODELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_sub = ["#4477AA", "#CC6677", "#228833"]

    for j, (metric, label, color) in enumerate(zip(metrics, labels, colors_sub)):
        means = [metric_stats(progress, m, metric)[0] for m in MODELS]
        stds = [metric_stats(progress, m, metric)[1] for m in MODELS]
        ax.bar(
            x + (j - 1) * width, means, width, yerr=stds,
            label=label, color=color, alpha=0.85 if j < 2 else 1.0,
            capsize=4, edgecolor="white",
        )

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "Blank-Text Ablation Audit (PAD-UFES-20, Mean ± Std)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(frameon=True, edgecolor="#CCCCCC")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "fig05_blank_text_ablation")


if __name__ == "__main__":
    main()
