#!/usr/bin/env python3
"""Figure 11: Per-class precision, recall, F1, and accuracy."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import load_json_data
from viz_style import CLASS_NAMES, MODELS, MODEL_COLORS, MODEL_SHORT, apply_style, save_figure


def main():
    apply_style()
    pcm = load_json_data("per_class_metrics.json")

    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1 Score"),
        ("per_class_accuracy", "Class Accuracy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    x = np.arange(len(CLASS_NAMES))
    width = 0.24

    for ax, (key, title) in zip(axes, metrics):
        for i, model in enumerate(MODELS):
            vals = pcm[model][key]
            ax.bar(
                x + (i - 1) * width, vals, width,
                label=MODEL_SHORT[model], color=MODEL_COLORS[model],
                edgecolor="white",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylabel(title)
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        if ax is axes[0]:
            ax.legend(frameon=True, edgecolor="#CCCCCC")

    fig.suptitle(
        "Per-Class Metrics on PAD-UFES-20 (Seed 42)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig11_per_class_metrics")


if __name__ == "__main__":
    main()
