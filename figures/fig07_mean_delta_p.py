#!/usr/bin/env python3
"""Figure 7: Mean ΔP comparison with confidence intervals."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import ci_95, load_progress
from viz_style import MODEL_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure


def main():
    apply_style()
    progress = load_progress()

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(MODELS))

    for i, model in enumerate(MODELS):
        values = progress["results"][model]["Mean_Delta_P"]
        mean, lo, hi = ci_95(values)
        ax.bar(i, mean, color=MODEL_COLORS[model], edgecolor="white", linewidth=1.2, alpha=0.9)
        ax.errorbar(
            i, mean, yerr=[[mean - lo], [hi - mean]],
            fmt="none", color="#333333", capsize=8, linewidth=2,
        )
        ax.text(i, hi + 0.5, f"{mean:.1f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax.set_ylabel("Mean Probability Shift ΔP (pp)")
    ax.set_title(
        "Mean ΔP Under Counterfactual Override\n(95% CI, 5 Seeds)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "fig07_mean_delta_p")


if __name__ == "__main__":
    main()
