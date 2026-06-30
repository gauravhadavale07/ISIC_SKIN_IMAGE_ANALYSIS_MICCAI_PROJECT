#!/usr/bin/env python3
"""Figure 19: Visual vs. fusion feature norm comparison."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import load_progress, metric_stats
from viz_style import MODEL_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure


def main():
    apply_style()
    progress = load_progress()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(MODELS))
    width = 0.35

    vis_means = [metric_stats(progress, m, "Vis_Feat_Norm")[0] for m in MODELS]
    vis_stds = [metric_stats(progress, m, "Vis_Feat_Norm")[1] for m in MODELS]
    fus_means = [metric_stats(progress, m, "Fused_Feat_Norm")[0] for m in MODELS]
    fus_stds = [metric_stats(progress, m, "Fused_Feat_Norm")[1] for m in MODELS]

    ax.bar(
        x - width / 2, vis_means, width, yerr=vis_stds,
        label="Visual CLS Norm", color="#4477AA", capsize=4, edgecolor="white",
    )
    ax.bar(
        x + width / 2, fus_means, width, yerr=fus_stds,
        label="Fused Feature Norm", color="#CC6677", capsize=4, edgecolor="white",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax.set_ylabel("Mean L2 Feature Norm")
    ax.set_title(
        "Visual vs. Fused Feature Norms (PAD-UFES-20, Mean ± Std)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(frameon=True, edgecolor="#CCCCCC")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "fig19_feature_norm_comparison")


if __name__ == "__main__":
    main()
