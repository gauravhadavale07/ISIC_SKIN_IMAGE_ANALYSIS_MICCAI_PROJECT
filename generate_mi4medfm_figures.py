import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

def generate_causal_patching_plot(results_dir, output_dir):
    csv_path = os.path.join(results_dir, "task8_activation_patching.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Architecture", y="Patch Recovery %", hue="Architecture", palette="viridis")
    plt.title("Causal Logit Recovery via Activation Patching", fontsize=14, fontweight="bold")
    plt.ylabel("Real Logit Recovery (%)", fontsize=12)
    plt.xlabel("Architecture", fontsize=12)
    plt.ylim(0, 105)
    
    for i, row in df.iterrows():
        plt.text(i, row["Patch Recovery %"] + 2, f"{row['Patch Recovery %']:.1f}%", ha='center', fontsize=11)
        
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_causal_patching.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def generate_sae_money_plot(results_dir, output_dir):
    csv_path = os.path.join(results_dir, "task11_sae_features.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty: return
    
    # Feature 1: Highest positive correlation with Malignancy
    feat1 = df.iloc[0]
    
    # Feature 2: Highly activating feature that fires rarely (likely spurious/biopsy)
    # We look for something with high max_act but low L0_Count
    df_sorted_l0 = df.sort_values(by="L0_Count")
    feat2 = df_sorted_l0.iloc[0]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    plt.suptitle("SAE Feature Visualizations (Top-5 Activating Images)", fontsize=16, fontweight="bold")
    
    # Row 1
    paths1 = feat1["Top5_Paths"].split(";")
    for i, p in enumerate(paths1[:5]):
        if os.path.exists(p):
            axes[0, i].imshow(Image.open(p).convert("RGB"))
        axes[0, i].axis("off")
    axes[0, 2].set_title(f"Feature {feat1['Feature_ID']}: Highly Correlated with Malignancy (r={feat1['Correlation_with_Malignancy']:.2f})", fontsize=12, pad=15)
    
    # Row 2
    paths2 = feat2["Top5_Paths"].split(";")
    for i, p in enumerate(paths2[:5]):
        if os.path.exists(p):
            axes[1, i].imshow(Image.open(p).convert("RGB"))
        axes[1, i].axis("off")
    axes[1, 2].set_title(f"Feature {feat2['Feature_ID']}: Potential Spurious/Artifact Feature (Fires {feat2['L0_Count']} times)", fontsize=12, pad=15)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_sae_features.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

def generate_steering_table(results_dir, output_dir):
    csv_path = os.path.join(results_dir, "task9_attention_steering.csv")
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    # The baseline was printed, but diff_from_baseline is recorded
    # We can reconstruct baseline from Acc - Diff
    baseline_acc = df.iloc[0]["Accuracy"] - df.iloc[0]["Diff_from_Baseline"]
    
    df_sorted = df.sort_values(by="Diff_from_Baseline")
    worst_head = df_sorted.iloc[0] # Ablating this hurt the most
    
    table_tex = f"""\\begin{{table}}[h]
\\centering
\\caption{{Attention Head Steering Intervention on Cross-Attention (T→V)}}
\\label{{tab:attention_steering}}
\\begin{{tabular}}{{lc}}
\\toprule
\\textbf{{Intervention}} & \\textbf{{Accuracy}} \\\\
\\midrule
Baseline & {baseline_acc*100:.2f}\\% \\\\
Ablate Text-Sensitive Head {worst_head['Head']} & {worst_head['Accuracy']*100:.2f}\\% ($\\downarrow {abs(worst_head['Diff_from_Baseline'])*100:.2f}$\\%) \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out_path = os.path.join(output_dir, "tab_attention_steering.tex")
    with open(out_path, "w") as f:
        f.write(table_tex)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    results_dir = "./results"
    output_dir = "./figures/output"
    os.makedirs(output_dir, exist_ok=True)
    
    generate_causal_patching_plot(results_dir, output_dir)
    generate_sae_money_plot(results_dir, output_dir)
    generate_steering_table(results_dir, output_dir)
