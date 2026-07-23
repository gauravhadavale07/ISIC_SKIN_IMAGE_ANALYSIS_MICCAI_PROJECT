import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

def main():
    # Load screening results
    res_1337_path = os.path.join(cfg.paths.results_dir, "task32_art_screening_results_seed1337.csv")
    res_42_path = os.path.join(cfg.paths.results_dir, "task32_art_screening_results_seed42.csv")
    
    if not os.path.exists(res_1337_path) or not os.path.exists(res_42_path):
        print("Screening results not found. Please pull them from the modal volume first.")
        return
        
    df_1337 = pd.read_csv(res_1337_path)
    df_42 = pd.read_csv(res_42_path)
    
    # Merge on feature name
    df_merged = pd.merge(df_1337, df_42, on="feature", suffixes=("_1337", "_42"))
    
    # Filter features that are significant in both runs
    df_sig = df_merged[(df_merged["significant_1337"] == True) & (df_merged["significant_42"] == True)].copy()
    
    if len(df_sig) == 0:
        print("No robust significant features found across both seeds.")
        return
        
    # Calculate combined robustness score (e.g. geometric mean of fusion scores)
    df_sig["robust_fusion_score"] = (df_sig["fusion_score_1337"] * df_sig["fusion_score_42"]) ** 0.5
    df_sig = df_sig.sort_values(by="robust_fusion_score", ascending=False).reset_index(drop=True)
    
    print(f"Found {len(df_sig)} robust interaction features.")
    print("Top 10 Robust Features:")
    print(df_sig[["feature", "robust_fusion_score", "F_stat_1337", "F_stat_42"]].head(10))
    
    # Load activations for seed 1337 (we just need one valid run's activations to visualize the pattern)
    acts_path = os.path.join(cfg.paths.results_dir, "task32_top50_activations_seed1337.csv")
    df_acts = pd.read_csv(acts_path)
    
    # Create visualization dir
    viz_dir = os.path.join(cfg.paths.results_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    top_features = df_sig["feature"].head(10).tolist()
    
    sns.set_theme(style="whitegrid")
    
    for feat in top_features:
        if feat not in df_acts.columns:
            print(f"Warning: {feat} not found in activations CSV. Skipping plot.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # We want to plot the mean activation for groups A, B, C, D
        # Group definitions:
        # A: Malignant Image + Malignant Text
        # B: Malignant Image + Benign Text
        # C: Benign Image + Benign Text
        # D: Benign Image + Malignant Text
        
        sns.barplot(
            data=df_acts,
            x="group", 
            y=feat,
            order=["A", "B", "C", "D"],
            palette=["#d62728", "#ff9896", "#98df8a", "#2ca02c"],
            capsize=0.1
        )
        
        plt.title(f"Feature {feat} Activation by Interaction Group", fontsize=14, pad=15)
        plt.ylabel("Mean Activation", fontsize=12)
        plt.xlabel("Group", fontsize=12)
        
        # Add a custom legend or text to explain the groups
        group_text = (
            "A: Malignant Image + Malignant Text\n"
            "B: Malignant Image + Benign Text\n"
            "C: Benign Image + Benign Text\n"
            "D: Benign Image + Malignant Text"
        )
        plt.text(0.02, 0.95, group_text, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        out_plot = os.path.join(viz_dir, f"{feat}_interaction_plot.png")
        plt.savefig(out_plot, dpi=300)
        plt.close()
        
    print(f"\nGenerated bar plots for top {len(top_features)} robust features in {viz_dir}")

if __name__ == "__main__":
    main()
