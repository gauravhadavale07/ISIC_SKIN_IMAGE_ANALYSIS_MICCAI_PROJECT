#!/usr/bin/env python3
"""Figure 17: Calibration / reliability diagrams with ECE."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import load_json_data
from viz_style import MODELS, MODEL_COLORS, MODEL_SHORT, apply_style, save_figure


def main():
    apply_style()
    cal = load_json_data("calibration.json")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1.5, label="Perfect calibration")

    for model in MODELS:
        d = cal[model]
        ax.plot(
            d["bin_confidence"], d["bin_accuracy"],
            marker="o", linewidth=2.5, color=MODEL_COLORS[model],
            label=f"{MODEL_SHORT[model]} (ECE={d['ece']:.3f})",
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Reliability Diagrams (PAD-UFES-20, Seed 42)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=True, edgecolor="#CCCCCC")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_figure(fig, "fig17_calibration_plot")


if __name__ == "__main__":
    main()
