#!/usr/bin/env python3
"""Figure 20: Final multi-panel publication summary figure."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import all_metric_stats, load_progress, metric_stats
from viz_style import MODEL_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure


def main():
    apply_style()
    progress = load_progress()

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    panels = [
        (gs[0, 0], "Accuracy", "Accuracy", 100, False),
        (gs[0, 1], "AUROC", "AUROC", 1, False),
        (gs[0, 2], "Blank_Accuracy_Drop", "Blank Acc. Drop (pp)", 1, False),
        (gs[1, 0], "CFR", "CFR (%)", 1, False),
        (gs[1, 1], "Mean_Delta_P", "Mean ΔP (pp)", 1, False),
        (gs[1, 2], "Linear_CKA", "Linear CKA", 1, True),
    ]

    for spec, metric, ylabel, scale, invert_y in panels:
        ax = fig.add_subplot(spec)
        means = []
        stds = []
        for model in MODELS:
            m, s = metric_stats(progress, model, metric)
            if metric == "Accuracy":
                m, s = m * 100, s * 100
            means.append(m)
            stds.append(s)

        x = np.arange(len(MODELS))
        bars = ax.bar(
            x, means, yerr=stds, capsize=5,
            color=[MODEL_COLORS[m] for m in MODELS],
            edgecolor="white", linewidth=1.2,
            error_kw=dict(linewidth=1.3, capthick=1.3),
        )
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT[m] for m in MODELS], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold", fontsize=12)
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")

        if metric == "Linear_CKA":
            ax.axhline(0.95, color="#B85450", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.set_ylim(0.78, 1.0)

        for bar, val in zip(bars, means):
            fmt = f"{val:.2f}" if metric != "Linear_CKA" else f"{val:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.15 + 0.5,
                fmt, ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    fig.suptitle(
        "Mechanistic Audit of Multimodal Grounding in Dermoscopic Foundation Models\n"
        "(PAD-UFES-20 OOD, Mean ± Std over 5 Seeds)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    save_figure(fig, "fig20_publication_summary")


if __name__ == "__main__":
    main()
