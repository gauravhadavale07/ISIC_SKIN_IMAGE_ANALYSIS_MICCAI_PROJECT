import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

df = pd.read_csv('../results/task12_contrastive_steering.csv')

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)

# X axis = Match Neutral (shortcut suppression rate)
# Y axis = Accuracy (baseline retention)
# Points colored by alpha

sc = plt.scatter(df['match_neutral'], df['accuracy'], c=df['alpha'], cmap='coolwarm', s=150, edgecolor='k', linewidth=1)
plt.colorbar(sc, label=r'Steering Intensity ($\alpha$)')

# Annotate the points
for i in range(len(df)):
    alpha = df['alpha'].iloc[i]
    if alpha in [-1.0, 0.0, 1.0, 2.0, 3.0]:
        plt.annotate(f"$\\alpha={alpha}$", (df['match_neutral'].iloc[i], df['accuracy'].iloc[i]),
                     xytext=(10, -10), textcoords='offset points', fontsize=12)

plt.xlabel("Shortcut Neutralization Rate (Match vs Ablation)")
plt.ylabel("Overall Accuracy")
plt.title("Pareto Frontier: Activation Addition Steering")
plt.tight_layout()

os.makedirs('output', exist_ok=True)
plt.savefig('output/fig_steering_pareto.pdf', bbox_inches='tight', dpi=300)
plt.savefig('output/fig_steering_pareto.png', bbox_inches='tight', dpi=300)
print("Saved fig_steering_pareto.pdf")
