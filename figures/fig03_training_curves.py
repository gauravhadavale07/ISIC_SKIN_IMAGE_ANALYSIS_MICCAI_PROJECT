#!/usr/bin/env python3
"""Figure 3: Training curves (loss and accuracy) for all models."""

import json

import matplotlib.pyplot as plt
import numpy as np

from viz_data import aggregate_training_by_model, parse_training_logs
from viz_style import DATA_DIR, MODEL_COLORS, MODELS, apply_style, save_figure


def main():
    apply_style()

    curves_path = DATA_DIR / "training_curves.json"
    if curves_path.exists():
        with open(curves_path) as f:
            curves = json.load(f)
    else:
        curves = aggregate_training_by_model(parse_training_logs(), seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("train_loss", "Train Loss", axes[0, 0]),
        ("val_loss", "Validation Loss", axes[0, 1]),
        ("train_acc", "Train Accuracy (%)", axes[1, 0]),
        ("val_acc", "Validation Accuracy (%)", axes[1, 1]),
    ]

    for model in MODELS:
        if model not in curves:
            continue
        epochs = curves[model]
        xs = [e["epoch"] for e in epochs]
        for key, ylabel, ax in panels:
            ys = [e[key] for e in epochs]
            ax.plot(xs, ys, marker="o", label=model, color=MODEL_COLORS[model], linewidth=2)

    for _, ylabel, ax in panels:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC")
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(
        "Training Curves on MILK10k (Seed 42, Representative Run)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig03_training_curves")


if __name__ == "__main__":
    main()
