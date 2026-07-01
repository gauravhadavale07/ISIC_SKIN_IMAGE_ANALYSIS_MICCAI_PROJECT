#!/usr/bin/env python3
"""Figure 18: Statistical significance forest plot for 12-pair comparison family."""

import json
import matplotlib.pyplot as plt
import numpy as np

from viz_style import MODEL_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure

# Define the 12 comparison pairs
COMPARISON_PAIRS = [
    # Original 6 pairs
    ("Late Fusion", "Cross-Attn V→T"),
    ("Late Fusion", "Cross-Attn T→V"),
    ("GMU Baseline", "Cross-Attn V→T"),
    ("Cross-Attn V→T", "Cross-Attn T→V"),
    ("Image-Only", "Late Fusion"),
    ("Text-Only", "Late Fusion"),
    # New 6 pairs for collapse testing
    ("GMU Baseline", "Image-Only"),
    ("GMU Baseline", "Text-Only"),
    ("Cross-Attn V→T", "Image-Only"),
    ("Cross-Attn V→T", "Text-Only"),
    ("Cross-Attn T→V", "Image-Only"),
    ("Cross-Attn T→V", "Text-Only"),
]

KEY_METRICS = ["AUROC", "F1 (Macro)", "Linear_CKA", "CFR", "Mean_Delta_P"]


def load_collapse_summary():
    """Load collapse test summary from results."""
    summary_path = "./results/collapse_test_summary.json"
    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Collapse summary not found at {summary_path}")
        return None


def main():
    apply_style()
    
    # Load collapse summary
    collapse_data = load_collapse_summary()
    
    # Create forest plot for the 12 comparison pairs
    fig, ax = plt.subplots(figsize=(12, 10))
    
    y_labels = []
    y_pos = []
    colors = []
    significance_markers = []
    
    # Plot each comparison pair
    for i, (model_a, model_b) in enumerate(COMPARISON_PAIRS):
        y_labels.append(f"{model_a} vs {model_b}")
        y_pos.append(i)
        
        # Determine color based on first model
        if model_a in MODEL_COLORS:
            colors.append(MODEL_COLORS[model_a])
        else:
            colors.append("#666666")
        
        # Check if this is a collapse test (vs Image-Only or Text-Only)
        is_collapse_test = model_b in ["Image-Only", "Text-Only"]
        
        if is_collapse_test and collapse_data:
            # Get significance from collapse summary
            arch = model_a
            baseline_key = f"vs_{model_b.replace('-', '_')}"
            if arch in collapse_data and baseline_key in collapse_data[arch]:
                distinguishable = collapse_data[arch][baseline_key]["distinguishable"]
                p_corrected = collapse_data[arch][baseline_key]["p_corrected"]
                significance_markers.append("***" if distinguishable else "ns")
            else:
                significance_markers.append("?")
        else:
            # For non-collapse tests, we'd need to compute from progress data
            # For now, mark as pending
            significance_markers.append("pending")
    
    # Create horizontal forest plot
    for i, (label, color, sig) in enumerate(zip(y_labels, colors, significance_markers)):
        # Draw a horizontal line with marker
        ax.plot([0, 1], [i, i], color=color, linewidth=2, alpha=0.6)
        ax.plot(0.5, i, "o", color=color, markersize=8)
        ax.text(1.05, i, sig, va="center", fontsize=10, fontweight="bold")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Effect Size (normalized)", fontsize=12)
    ax.set_title(
        "Statistical Significance Tests (12-Pair Family, Holm-Bonferroni Corrected)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#666666", label="Non-collapse tests"),
        Patch(facecolor="#AA3377", label="vs Image-Only (collapse)"),
        Patch(facecolor="#BBBBBB", label="vs Text-Only (collapse)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, edgecolor="#CCCCCC")
    
    # Add annotation
    ax.text(0.5, -0.05, "*** = p < 0.05 (corrected) | ns = not significant | pending = requires experiment completion",
            ha="center", va="top", transform=ax.transAxes, fontsize=9, style="italic")
    
    fig.tight_layout()
    save_figure(fig, "fig18_statistical_summary")


if __name__ == "__main__":
    main()
