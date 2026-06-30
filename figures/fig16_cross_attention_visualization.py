#!/usr/bin/env python3
"""Figure 16: Cross-attention visualization over lesion images."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from viz_data import load_json_data, load_npz_data
from viz_style import PROJECT_ROOT, apply_style, save_figure, DATA_DIR


def main():
    apply_style()
    attn_path = DATA_DIR / "attention_maps.npz"
    cases_path = DATA_DIR / "counterfactual_cases_Cross-Attention.json"
    if not attn_path.exists():
        raise FileNotFoundError("Run export_figure_data.py first.")

    attn = np.load(attn_path)["weights"]
    with open(cases_path) as f:
        cases = json.load(f)[:min(4, len(attn))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, case, weights in zip(axes, cases, attn):
        img_path = Path(case["filepath"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        img = mpimg.imread(str(img_path))

        ax.imshow(img)
        # Overlay mean attention as horizontal bar strip (text token salience)
        w = weights / (weights.max() + 1e-8)
        bar = np.tile(w[:32], (16, 1))  # show first 32 text tokens
        ax.imshow(bar, extent=[0, img.shape[1], img.shape[0] - 30, img.shape[0]],
                  alpha=0.55, cmap="hot", aspect="auto")
        ax.set_title(
            f"True: {case['true_label']} → Pred: {case['real_pred']}",
            fontsize=11, fontweight="bold",
        )
        ax.axis("off")

    fig.suptitle(
        "Cross-Attention CLS→Text Token Salience (PAD-UFES-20)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig16_cross_attention_visualization")


if __name__ == "__main__":
    main()
