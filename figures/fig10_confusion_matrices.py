#!/usr/bin/env python3
"""Figure 10: Confusion matrices for each model."""

import json

import matplotlib.pyplot as plt
import numpy as np

from viz_data import load_json_data
from viz_style import CLASS_NAMES, MODELS, MODEL_SHORT, apply_style, save_figure, DATA_DIR


def main():
    apply_style()
    cm_path = DATA_DIR / "confusion_matrices.json"
    if cm_path.exists():
        cms = load_json_data("confusion_matrices.json")
    else:
        raise FileNotFoundError("Run export_figure_data.py first.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmax = max(max(sum(row) for row in cms[m]) for m in MODELS)

    for ax, model in zip(axes, MODELS):
        cm = np.array(cms[model], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, where=row_sums > 0)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{MODEL_SHORT[model]}\n(row-normalized)", fontweight="bold")

        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = int(cm[i, j])
                color = "white" if cm_norm[i, j] > 0.55 else "#333333"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04)
    cbar.set_label("Recall (per true class)")
    fig.suptitle(
        "Confusion Matrices on PAD-UFES-20 (Seed 42)",
        fontsize=14, fontweight="bold", y=1.05,
    )
    fig.tight_layout()
    save_figure(fig, "fig10_confusion_matrices")


if __name__ == "__main__":
    main()
