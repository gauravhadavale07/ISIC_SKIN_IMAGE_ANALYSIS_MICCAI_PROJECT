#!/usr/bin/env python3
"""Figure 2: Architecture comparison block diagrams."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from viz_style import MODEL_COLORS, apply_style, save_figure


def block(ax, x, y, w, h, text, color="#E8EEF7"):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01",
        facecolor=color, edgecolor="#333333", linewidth=1.2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="bold")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5),
    )


def draw_late_fusion(ax):
    ax.set_title("Late Fusion", fontsize=13, fontweight="bold", color=MODEL_COLORS["Late Fusion"])
    block(ax, 0.05, 0.72, 0.28, 0.14, "Image\nViT-B/16 (frozen)", "#DAE8FC")
    block(ax, 0.05, 0.48, 0.28, 0.14, "Text\nBio_ClinicalBERT\n(frozen)", "#D5E8D4")
    block(ax, 0.42, 0.55, 0.18, 0.24, "CLS\n768-d", "#FFF2CC")
    block(ax, 0.42, 0.30, 0.18, 0.18, "Pooler\n768-d", "#FFF2CC")
    block(ax, 0.68, 0.42, 0.26, 0.22, "Concat\n1536-d", "#F8CECC")
<<<<<<< HEAD
    block(ax, 0.68, 0.12, 0.26, 0.22, "MLP\n512 -> 6 classes", "#E1D5E7")
=======
    block(ax, 0.68, 0.12, 0.26, 0.22, "MLP\n512 → 6 classes", "#E1D5E7")
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
    arrow(ax, 0.33, 0.79, 0.42, 0.67)
    arrow(ax, 0.33, 0.55, 0.42, 0.39)
    arrow(ax, 0.60, 0.67, 0.68, 0.53)
    arrow(ax, 0.60, 0.39, 0.68, 0.53)
    arrow(ax, 0.81, 0.42, 0.81, 0.34)


def draw_gmu(ax):
    ax.set_title("GMU Baseline", fontsize=13, fontweight="bold", color=MODEL_COLORS["GMU Baseline"])
    block(ax, 0.05, 0.72, 0.28, 0.14, "Image\nViT CLS 768-d", "#DAE8FC")
    block(ax, 0.05, 0.48, 0.28, 0.14, "Text\nPooler 768-d", "#D5E8D4")
    block(ax, 0.42, 0.62, 0.20, 0.16, "Tanh\ntransform", "#FFF2CC")
    block(ax, 0.42, 0.38, 0.20, 0.16, "Tanh\ntransform", "#FFF2CC")
    block(ax, 0.68, 0.55, 0.24, 0.18, "Gate σ\nfrom concat", "#F8CECC")
    block(ax, 0.68, 0.20, 0.24, 0.22, "z·h_v + (1-z)·h_t", "#E1D5E7")
    block(ax, 0.68, 0.02, 0.24, 0.14, "Classifier\n6 classes", "#FFE6CC")
    arrow(ax, 0.33, 0.79, 0.42, 0.70)
    arrow(ax, 0.33, 0.55, 0.42, 0.46)
    arrow(ax, 0.62, 0.70, 0.68, 0.64)
    arrow(ax, 0.62, 0.46, 0.68, 0.64)
    arrow(ax, 0.80, 0.55, 0.80, 0.42)
    arrow(ax, 0.80, 0.20, 0.80, 0.16)


def draw_cross_attn(ax):
    ax.set_title("Cross-Attention", fontsize=13, fontweight="bold", color=MODEL_COLORS["Cross-Attention"])
    block(ax, 0.05, 0.72, 0.28, 0.14, "Image\nPatch seq 197×768", "#DAE8FC")
    block(ax, 0.05, 0.48, 0.28, 0.14, "Text\nToken seq 128×768", "#D5E8D4")
    block(ax, 0.42, 0.50, 0.28, 0.28, "Multi-Head\nCross-Attention\nQ=visual, K/V=text", "#F8CECC")
    block(ax, 0.76, 0.55, 0.20, 0.18, "Residual +\nLayerNorm", "#FFF2CC")
<<<<<<< HEAD
    block(ax, 0.76, 0.22, 0.20, 0.22, "Multimodal\nCLS -> 6 classes", "#E1D5E7")
=======
    block(ax, 0.76, 0.22, 0.20, 0.22, "Multimodal\nCLS → 6 classes", "#E1D5E7")
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
    arrow(ax, 0.33, 0.79, 0.42, 0.64)
    arrow(ax, 0.33, 0.55, 0.42, 0.58)
    arrow(ax, 0.70, 0.64, 0.76, 0.64)
    arrow(ax, 0.86, 0.55, 0.86, 0.44)


def main():
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    draw_late_fusion(axes[0])
    draw_gmu(axes[1])
    draw_cross_attn(axes[2])
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    fig.suptitle(
        "Multimodal Fusion Architectures Under Comparison",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fig02_architecture_comparison")


if __name__ == "__main__":
    main()
