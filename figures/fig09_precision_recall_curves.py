#!/usr/bin/env python3
"""Figure 9: Precision-Recall curves (macro-averaged) for all models."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

from viz_data import load_npz_data, sanitize_model_key
from viz_style import CLASS_NAMES, MODELS, MODEL_COLORS, MODEL_SHORT, apply_style, save_figure


def macro_pr(y_true, y_prob, n_classes):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    precisions, recalls = [], []
    for i in range(n_classes):
        p, r, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        precisions.append(p)
        recalls.append(r)
    mean_r = np.linspace(0, 1, 200)
    interp_p = []
    for p, r in zip(precisions, recalls):
        interp_p.append(np.interp(mean_r, r[::-1], p[::-1]))
    mean_p = np.mean(interp_p, axis=0)
    pr_auc = auc(mean_r, mean_p)
    return mean_r, mean_p, pr_auc


def main():
    apply_style()
    data = load_npz_data("ood_predictions.npz")
    n_classes = len(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(8, 8))
    for model in MODELS:
        key = sanitize_model_key(model)
        y_true = data[f"{key}_y_true"]
        y_prob = data[f"{key}_y_prob"]
        recall, precision, pr_auc = macro_pr(y_true, y_prob, n_classes)
        ax.plot(
            recall, precision, color=MODEL_COLORS[model], linewidth=2.5,
            label=f"{MODEL_SHORT[model]} (AUC = {pr_auc:.3f})",
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(
        "Macro-Averaged Precision-Recall Curves (PAD-UFES-20, Seed 42)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=True, edgecolor="#CCCCCC")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "fig09_precision_recall_curves")


if __name__ == "__main__":
    main()
