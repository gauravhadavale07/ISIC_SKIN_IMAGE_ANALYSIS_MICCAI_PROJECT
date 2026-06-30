#!/usr/bin/env python3
"""Figure 13: t-SNE projection of fused feature embeddings."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from viz_data import load_npz_data, sanitize_model_key
from viz_style import CLASS_NAMES, CLASS_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure


def main():
    apply_style()
    data = load_npz_data("fused_features.npz")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    for ax, model in zip(axes, MODELS):
        key = sanitize_model_key(model)
        feats = data[key]
        # Subsample for speed if large
        n = min(1500, len(feats))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(feats), n, replace=False)
        X = feats[idx]

        # Get labels from OOD predictions file
        ood = load_npz_data("ood_predictions.npz")
        labels = ood[f"{key}_y_true"][idx]

        tsne = TSNE(n_components=2, perplexity=35, random_state=42, n_iter=1000, init="pca")
        emb = tsne.fit_transform(X)

        for c in range(len(CLASS_NAMES)):
            mask = labels == c
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                s=18, alpha=0.7, edgecolors="none",
            )
        ax.set_title(MODEL_SHORT[model], fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.2, linestyle="--")

    axes[0].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        frameon=True, edgecolor="#CCCCCC", title="Class",
    )
    fig.suptitle(
        "t-SNE of Fused Embeddings (PAD-UFES-20, Seed 42)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig13_tsne_embeddings")


if __name__ == "__main__":
    main()
