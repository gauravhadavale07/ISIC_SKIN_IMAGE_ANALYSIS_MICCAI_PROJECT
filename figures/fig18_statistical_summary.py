#!/usr/bin/env python3
"""Figure 18: Statistical summary forest plot with confidence intervals."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import ci_95, load_progress
from viz_style import MODEL_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure

METRICS = [
    ("Accuracy", "Accuracy", 1.0),
    ("AUROC", "AUROC", 1.0),
    ("F1 (Macro)", "Macro F1", 1.0),
    ("CFR", "CFR (%)", 100.0),
    ("Mean_Delta_P", "Mean ΔP (pp)", 1.0),
    ("Linear_CKA", "Linear CKA", 1.0),
]


def main():
    apply_style()
    progress = load_progress()

    fig, axes = plt.subplots(1, len(MODELS), figsize=(14, 8), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]

    y_labels = [m[1] for m in METRICS]
    y_pos = np.arange(len(METRICS))

    for ax, model in zip(axes, MODELS):
        means, los, his = [], [], []
        for key, label, scale in METRICS:
            vals = [v * scale for v in progress["results"][model][key]]
            if key == "Accuracy":
                vals = [v * 100 for v in vals]
            mean, lo, hi = ci_95(vals)
            means.append(mean)
            los.append(lo)
            his.append(hi)

        for i, (m, lo, hi) in enumerate(zip(means, los, his)):
            ax.plot([lo, hi], [i, i], color=MODEL_COLORS[model], linewidth=2.5)
            ax.plot(m, i, "o", color=MODEL_COLORS[model], markersize=9)
            ax.text(hi + 0.02 * max(his), i, f"{m:.2f}", va="center", fontsize=9)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels if ax is axes[0] else [])
        ax.set_title(MODEL_SHORT[model], fontweight="bold", color=MODEL_COLORS[model])
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.invert_yaxis()

    fig.suptitle(
<<<<<<< HEAD
        "Statistical Summary Forest Plot (95% CI, 5 Seeds)",
=======
        "Statistical Summary Forest Plot (95% CI, 3 Seeds)",
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig18_statistical_summary")


if __name__ == "__main__":
    main()
