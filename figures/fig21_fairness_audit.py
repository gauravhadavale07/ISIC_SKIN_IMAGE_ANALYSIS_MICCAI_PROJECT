import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Adjust path to find config and viz modules if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from figures.viz_style import setup_style, get_color_palette
except ImportError:
    # Fallback if viz_style doesn't exist or isn't accessible
    def setup_style():
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    def get_color_palette():
        return sns.color_palette("deep")

def plot_fairness_audit():
    setup_style()
    palette = get_color_palette()
    
    csv_path = "../results/task6_ddi_stratified_audit_rigorous.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Filter to only the strata (exclude 'Overall')
    df_strata = df[df['Skin_Tone_Bin'] != 'Overall'].copy()
    
    # Rename for cleaner legend
    model_order = [
        "Image-Only", 
        "Late Fusion", 
        "GMU", 
        "Cross-Attention (T->V)", 
        "Cross-Attention (V->T)"
    ]
    tone_order = ["FST I/II (Light)", "FST III/IV (Medium)", "FST V/VI (Dark)"]

    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        data=df_strata, 
        x="Architecture", 
        y="AUROC", 
        hue="Skin_Tone_Bin",
        order=model_order,
        hue_order=tone_order,
        palette="rocket_r", # A nice gradient for skin tone concept
        edgecolor='black',
        linewidth=1
    )
    
    # Add error bars
    # seaborn barplot does not natively take custom lower/upper bounds easily, 
    # so we overlay errorbar using matplotlib
    
    # Group coordinates
    x_coords = [p.get_x() + p.get_width() / 2.0 for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    
    # The patches are ordered by hue, then by x.
    # So all FST I/II patches first, then FST III/IV, then FST V/VI.
    n_models = len(model_order)
    
    for hue_idx, tone in enumerate(tone_order):
        tone_df = df_strata[df_strata['Skin_Tone_Bin'] == tone].set_index("Architecture")
        for model_idx, model in enumerate(model_order):
            patch_idx = hue_idx * n_models + model_idx
            x = x_coords[patch_idx]
            y = y_coords[patch_idx]
            
            if model in tone_df.index:
                row = tone_df.loc[model]
                y_err_lower = y - row['AUROC_CI_Lower']
                y_err_upper = row['AUROC_CI_Upper'] - y
                
                ax.errorbar(x, y, yerr=[[y_err_lower], [y_err_upper]], 
                            color='black', capsize=5, capthick=1.5, elinewidth=1.5)

    plt.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Random Chance (0.5)')
    
    plt.title('Demographic Bias: Skin-Tone Stratification Audit (AUROC)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Architecture', fontsize=14, labelpad=10)
    plt.ylabel('Macro AUROC (OVR)', fontsize=14, labelpad=10)
    plt.ylim(0.4, 0.85)
    plt.xticks(rotation=15)
    
    plt.legend(title="Fitzpatrick Skin Type (FST)", title_fontsize='12', fontsize='11', loc='upper right')
    plt.tight_layout()
    
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/fig21_fairness_audit.png", dpi=300, bbox_inches='tight')
    plt.savefig("output/fig21_fairness_audit.pdf", bbox_inches='tight')
    plt.close()
    print("Generated figures/output/fig21_fairness_audit.png and .pdf")

if __name__ == "__main__":
    plot_fairness_audit()
