#!/usr/bin/env python3
"""Figure 15: Counterfactual case studies."""

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from viz_style import DATA_DIR, PROJECT_ROOT, apply_style, save_figure


def load_cases():
    path = DATA_DIR / "counterfactual_cases_Cross-Attention.json"
    if not path.exists():
        path = DATA_DIR / "counterfactual_cases.json"
    if not path.exists():
        raise FileNotFoundError("Run export_figure_data.py first.")
    with open(path) as f:
        return json.load(f)


def main():
    apply_style()
    cases = load_cases()[:4]
    n = len(cases)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3.8 * n))
    if n == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        ax.axis("off")
        img_path = Path(case["filepath"])
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        if img_path.exists():
            img = mpimg.imread(str(img_path))
            ax_img = ax.inset_axes([0.0, 0.05, 0.22, 0.9])
            ax_img.imshow(img)
            ax_img.axis("off")
            ax_img.set_title("Lesion", fontsize=10, fontweight="bold")

        hist = textwrap.fill(case["clinical_history"][:200], width=55)
        cf_txt = textwrap.fill(case["cf_text"], width=55)
        info = (
            f"True: {case['true_label']}   |   "
            f"Original Pred: {case['real_pred']} (p={case['real_prob']:.2f})\n\n"
            f"Clinical History:\n{hist}\n\n"
            f"Counterfactual Text:\n{cf_txt}\n\n"
            f"New Pred: {case['cf_pred']}  |  "
            f"ΔP on original class: {case['delta_p']:.3f}  |  "
            f"Flipped: {'Yes' if case['flipped'] else 'No'}"
        )
        ax.text(0.26, 0.5, info, va="center", ha="left", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", facecolor="#F8F8F8", edgecolor="#CCCCCC"))

    fig.suptitle(
        "Counterfactual Case Studies (Cross-Attention, PAD-UFES-20)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "fig15_counterfactual_case_studies")


if __name__ == "__main__":
    main()
