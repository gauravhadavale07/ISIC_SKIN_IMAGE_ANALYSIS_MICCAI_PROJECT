#!/usr/bin/env python3
"""Figure 8: ROC curves (macro-averaged one-vs-rest) for all models."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from viz_data import load_npz_data, sanitize_model_key
from viz_style import CLASS_NAMES, MODELS, MODEL_COLORS, MODEL_SHORT, apply_style, save_figure


def macro_roc(y_true, y_prob, n_classes):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fpr_dict, tpr_dict = {}, {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr_dict[i], tpr_dict[i] = fpr, tpr
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= n_classes
    roc_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, roc_auc


def main():
    apply_style()
    data = load_npz_data("ood_predictions.npz")
    n_classes = len(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(8, 8))
    for model in MODELS:
        key = sanitize_model_key(model)
        y_true = data[f"{key}_y_true"]
        y_prob = data[f"{key}_y_prob"]
        fpr, tpr, roc_auc = macro_roc(y_true, y_prob, n_classes)
        ax.plot(
            fpr, tpr, color=MODEL_COLORS[model], linewidth=2.5,
            label=f"{MODEL_SHORT[model]} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.5, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        "Macro-Averaged ROC Curves (PAD-UFES-20, Seed 42)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", frameon=True, edgecolor="#CCCCCC")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "fig08_roc_curves")


if __name__ == "__main__":
    main()
