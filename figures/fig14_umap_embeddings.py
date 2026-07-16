#!/usr/bin/env python3
"""Figure 14: UMAP projection of fused feature embeddings.

Falls back to sklearn SpectralEmbedding if umap-learn is not installed.
"""

import matplotlib.pyplot as plt
import numpy as np

from viz_data import load_npz_data, sanitize_model_key
from viz_style import CLASS_NAMES, CLASS_COLORS, MODELS, MODEL_SHORT, apply_style, save_figure


def embed_2d(X):
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
        return reducer.fit_transform(X), "UMAP"
    except ImportError:
        from sklearn.manifold import SpectralEmbedding
        print("  Note: umap-learn not found; using SpectralEmbedding fallback.")
        reducer = SpectralEmbedding(n_components=2, random_state=42)
        return reducer.fit_transform(X), "Spectral Embedding"


def main():
    apply_style()
    data = load_npz_data("fused_features.npz")
    ood = load_npz_data("ood_predictions.npz")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    method_name = "UMAP"

    for ax, model in zip(axes, MODELS):
        key = sanitize_model_key(model)
        feats = data[key]
        n = min(1500, len(feats))
        rng = np.random.default_rng(42)
        idx = rng.choice(len(feats), n, replace=False)
        X = feats[idx]
        labels = ood[f"{key}_y_true"][idx]

        emb, method_name = embed_2d(X)

        for c in range(len(CLASS_NAMES)):
            mask = labels == c
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=CLASS_COLORS[c], label=CLASS_NAMES[c],
                s=18, alpha=0.7, edgecolors="none",
            )
        ax.set_title(MODEL_SHORT[model], fontweight="bold")
        ax.set_xlabel(f"{method_name} 1")
        ax.set_ylabel(f"{method_name} 2")
        ax.grid(True, alpha=0.2, linestyle="--")

    axes[0].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1),
        frameon=True, edgecolor="#CCCCCC", title="Class",
    )
    fig.suptitle(
        f"{method_name} of Fused Embeddings (PAD-UFES-20, Seed 42)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "fig14_umap_embeddings")


if __name__ == "__main__":
    main()
