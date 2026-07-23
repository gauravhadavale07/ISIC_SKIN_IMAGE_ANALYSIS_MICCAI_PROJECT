#!/usr/bin/env python3
"""Figure 6: Counterfactual Flip Rate (CFR) bar chart."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import load_progress, metric_stats
from viz_style import MODEL_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure


def main():
    apply_style()
    progress = load_progress()

    means = [metric_stats(progress, m, "CFR")[0] for m in MODELS]
    stds = [metric_stats(progress, m, "CFR")[1] for m in MODELS]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        range(len(MODELS)), means, yerr=stds,
        color=[MODEL_COLORS[m] for m in MODELS],
        capsize=6, edgecolor="white", linewidth=1.2,
        error_kw=dict(linewidth=1.5, capthick=1.5),
    )
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax.set_ylabel("Counterfactual Flip Rate (%)")
    ax.set_title(
        "CFR Under Semantic Counterfactual Override\n(Mean ± Std, 5 Seeds)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{m:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    fig.tight_layout()
    save_figure(fig, "fig06_counterfactual_flip_rate")


if __name__ == "__main__":
    main()
