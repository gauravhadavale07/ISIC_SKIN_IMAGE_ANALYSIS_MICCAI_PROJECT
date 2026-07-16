import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Adjust path to find config and viz modules if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from figures.viz_style import setup_style, get_color_palette
except ImportError:
    def setup_style():
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    def get_color_palette():
        return sns.color_palette("deep")

def parse_delta(delta_str):
    return float(delta_str.replace("pp", "").replace("+", ""))

def parse_ci(ci_str):
    # e.g., "[-0.09, +0.59]"
    match = re.match(r"\[([+-]?\d+\.\d+),\s*([+-]?\d+\.\d+)\]", ci_str)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0

def plot_holm_significance():
    setup_style()
    palette = get_color_palette()
    
    csv_path = "../results/task1_metadata_shuffle_significance.csv"
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Process data
    architectures = df['Architecture'].tolist()
    deltas = df['Delta'].apply(parse_delta).tolist()
    
    ci_lows = []
    ci_highs = []
    for ci_str in df['95% CI']:
        low, high = parse_ci(ci_str)
        ci_lows.append(low)
        ci_highs.append(high)
        
    p_vals = df['p-value (Holm_str)'].tolist()
    
    # Setup Forest Plot
    plt.figure(figsize=(10, 5))
    
    y_pos = range(len(architectures))
    
    # Plot central point (delta)
    plt.scatter(deltas, y_pos, color=palette[0], s=100, zorder=3)
    
    # Plot error bars (CI bounds)
    for i in range(len(architectures)):
        plt.plot([ci_lows[i], ci_highs[i]], [y_pos[i], y_pos[i]], color=palette[0], linewidth=3, zorder=2)
        plt.plot([ci_lows[i], ci_lows[i]], [y_pos[i]-0.15, y_pos[i]+0.15], color=palette[0], linewidth=2, zorder=2)
        plt.plot([ci_highs[i], ci_highs[i]], [y_pos[i]-0.15, y_pos[i]+0.15], color=palette[0], linewidth=2, zorder=2)
        
        # Annotate p-value
        plt.text(ci_highs[i] + 0.2, y_pos[i], f"Holm p = {p_vals[i]}", va='center', fontsize=11, fontweight='bold' if '<' in str(p_vals[i]) else 'normal')

    # Add vertical line at 0 (No effect)
    plt.axvline(0, color='red', linestyle='--', linewidth=2, zorder=1)
    
    plt.yticks(y_pos, architectures, fontsize=12)
    plt.xlabel('Accuracy Drop on Lexical Shuffle ($\Delta$ Percentage Points)', fontsize=14, labelpad=10)
    plt.title('Effect Size of Textual Ablation (100k-Resample 95% CIs)', fontsize=16, fontweight='bold', pad=15)
    
    # Invert Y axis so first model is on top
    plt.gca().invert_yaxis()
    
    # Dynamic X limits to fit annotations
    plt.xlim(min(ci_lows) - 0.5, max(ci_highs) + 1.5)
    
    plt.tight_layout()
    
    os.makedirs("output", exist_ok=True)
    plt.savefig("output/fig22_holm_significance.png", dpi=300, bbox_inches='tight')
    plt.savefig("output/fig22_holm_significance.pdf", bbox_inches='tight')
    plt.close()
    print("Generated figures/output/fig22_holm_significance.png and .pdf")

if __name__ == "__main__":
    plot_holm_significance()
