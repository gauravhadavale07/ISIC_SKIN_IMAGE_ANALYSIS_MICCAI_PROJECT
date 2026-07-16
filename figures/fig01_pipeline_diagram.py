#!/usr/bin/env python3
"""Figure 1: Overall experimental pipeline diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from viz_style import apply_style, save_figure


def draw_box(ax, x, y, w, h, text, color="#E8EEF7", edge="#4477AA", fontsize=11):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor=edge, linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold")


def draw_arrow(ax, x1, y1, x2, y2):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.8, color="#333333",
    )
    ax.add_patch(arr)


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 14))

    steps = [
        (0.88, "MILK10k Dataset\n(Training + Validation)", "#D5E8D4", "#228833"),
        (0.76, "Multimodal Training\n(Frozen ViT-B/16 + Bio_ClinicalBERT)", "#FFF2CC", "#D6B656"),
        (0.64, "Fusion Architectures\nLate Fusion  |  GMU  |  Cross-Attention", "#DAE8FC", "#4477AA"),
        (0.52, "PAD-UFES-20\n(Zero-Shot OOD Evaluation)", "#F8CECC", "#B85450"),
        (0.40, "Standard OOD Metrics\nAccuracy · AUROC · F1 · Confusion Matrix", "#E1D5E7", "#9673A6"),
        (0.28, "Blank-Text Ablation Audit\nModality dependence probe", "#FFE6CC", "#D79B00"),
        (0.16, "Counterfactual Semantic Audit\nCFR · Mean ΔP", "#FFE6CC", "#D79B00"),
        (0.04, "CKA Geometric Audit\n+ Statistical Multi-Seed Analysis", "#F5F5F5", "#666666"),
    ]

    cx = 0.5
    bw, bh = 0.72, 0.085

    for i, (y, text, fc, ec) in enumerate(steps):
        draw_box(ax, cx, y, bw, bh, text, color=fc, edge=ec, fontsize=10 if i > 0 else 11)

    for i in range(len(steps) - 1):
        y_top = steps[i][0] - bh / 2 - 0.01
        y_bot = steps[i + 1][0] + bh / 2 + 0.01
        draw_arrow(ax, cx, y_top, cx, y_bot)

  # Side annotation for frozen backbones
    ax.text(
        0.95, 0.70, "Frozen\nBackbones",
        ha="right", va="center", fontsize=9, style="italic", color="#555555",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CCCCCC"),
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.0)
    ax.axis("off")
    ax.set_title(
        "Mechanistic Audit Pipeline for Multimodal Dermoscopic Foundation Models",
        fontsize=14, fontweight="bold", pad=16,
    )

    save_figure(fig, "fig01_pipeline_diagram")


if __name__ == "__main__":
    main()
