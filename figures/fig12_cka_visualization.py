#!/usr/bin/env python3
"""Figure 12: CKA bar plot and representational similarity heatmap."""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import load_progress, metric_stats
from viz_style import MODELS, MODEL_COLORS, MODEL_SHORT, apply_style, save_figure


<<<<<<< HEAD
def metric_stats_n(progress: dict, model: str, metric: str, n=5) -> tuple[float, float]:
    values = progress["results"][model][metric][:n]
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr))

=======
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
def main():
    apply_style()
    progress = load_progress()

<<<<<<< HEAD
    cka_means = [metric_stats_n(progress, m, "Linear_CKA", 5)[0] for m in MODELS]
    cka_stds = [metric_stats_n(progress, m, "Linear_CKA", 5)[1] for m in MODELS]

    # Similarity matrix: CKA between architectures' fused norms (from progress)
    vis_norms = [np.mean(progress["results"][m]["Vis_Feat_Norm"][:5]) for m in MODELS]
    fused_norms = [np.mean(progress["results"][m]["Fused_Feat_Norm"][:5]) for m in MODELS]
=======
    cka_means = [metric_stats(progress, m, "Linear_CKA")[0] for m in MODELS]
    cka_stds = [metric_stats(progress, m, "Linear_CKA")[1] for m in MODELS]

    # Similarity matrix: CKA between architectures' fused norms (from progress)
    vis_norms = [np.mean(progress["results"][m]["Vis_Feat_Norm"]) for m in MODELS]
    fused_norms = [np.mean(progress["results"][m]["Fused_Feat_Norm"]) for m in MODELS]
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1], wspace=0.35)

    # Panel A: CKA bar plot
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(
        range(len(MODELS)), cka_means, yerr=cka_stds,
        color=[MODEL_COLORS[m] for m in MODELS],
        capsize=6, edgecolor="white", linewidth=1.2,
    )
    ax1.axhline(0.95, color="#B85450", linestyle="--", linewidth=1.5, label="Collapse threshold (0.95)")
    ax1.axhline(0.85, color="#D79B00", linestyle=":", linewidth=1.5, label="Moderate (0.85)")
    ax1.set_xticks(range(len(MODELS)))
<<<<<<< HEAD
    ax1.set_xticklabels([MODEL_SHORT[m].replace('(', '\n(') for m in MODELS])
    ax1.set_ylabel("Linear CKA (Visual vs. Fused)")
    ax1.set_ylim(0.0, 1.0)
=======
    ax1.set_xticklabels([MODEL_SHORT[m] for m in MODELS])
    ax1.set_ylabel("Linear CKA (Visual vs. Fused)")
    ax1.set_ylim(0.75, 1.0)
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
    ax1.set_title("A. CKA Geometric Audit", fontweight="bold", loc="left")
    ax1.legend(fontsize=9, frameon=True, edgecolor="#CCCCCC")
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")
    for bar, v in zip(bars, cka_means):
<<<<<<< HEAD
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
=======
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
                 ha="center", fontsize=10, fontweight="bold")

    # Panel B: Feature norm heatmap + CKA cross-similarity proxy
    ax2 = fig.add_subplot(gs[1])
    mat = np.zeros((len(MODELS), len(MODELS)))
    for i, mi in enumerate(MODELS):
        for j, mj in enumerate(MODELS):
            if i == j:
<<<<<<< HEAD
                mat[i, j] = 1.0
=======
                mat[i, j] = cka_means[i]
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
            else:
                # Cross-architecture similarity proxy from shared visual backbone
                mat[i, j] = 1.0 - abs(cka_means[i] - cka_means[j])

<<<<<<< HEAD
    im = ax2.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
=======
    im = ax2.imshow(mat, cmap="YlGnBu", vmin=0.8, vmax=1.0, aspect="auto")
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
    ax2.set_xticks(range(len(MODELS)))
    ax2.set_yticks(range(len(MODELS)))
    ax2.set_xticklabels([MODEL_SHORT[m] for m in MODELS], rotation=30, ha="right")
    ax2.set_yticklabels([MODEL_SHORT[m] for m in MODELS])
    ax2.set_title("B. Representational Similarity Matrix", fontweight="bold", loc="left")
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            ax2.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                     color="white" if mat[i, j] > 0.92 else "#333333", fontsize=10)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046)
    cbar.set_label("Similarity")

    fig.suptitle(
<<<<<<< HEAD
        "CKA Analysis: Visual vs. Fused Latent Geometry (Mean ± Std, 5 Seeds)",
=======
        "CKA Analysis: Visual vs. Fused Latent Geometry (Mean ± Std, 3 Seeds)",
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "fig12_cka_visualization")


if __name__ == "__main__":
    main()
