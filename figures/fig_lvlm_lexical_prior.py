import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('figures', exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

# Empirical single-sample diagnostic values from results/task16_single_sample_diagnostic.csv.
tokens = ["Absolute top token: 'The'", "MEL bucket max: 'M'", "NEV bucket max: 'N'"]
logits = [15.16, 13.10, 11.34]

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#95a5a6', '#e74c3c', '#3498db']
bars = ax.barh(tokens, logits, color=colors, edgecolor='white', linewidth=1.5)

# Invert y-axis to have largest at the top
ax.invert_yaxis()

# Annotate values
for bar, logit in zip(bars, logits):
    ax.text(logit + 0.2, bar.get_y() + bar.get_height()/2, f'{logit:.2f}',
            va='center', fontsize=12, fontweight='bold', color='#333333')

# Vertical line highlighting the dropoff
ax.axvline(x=10, color='gray', linestyle='--', alpha=0.5, zorder=0)

ax.set_xlabel('Next-token logit at baseline ($\\alpha=0.0$)', fontsize=14, fontweight='bold')
ax.set_title('Single-Sample LVLM Logit Teardown', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('figures/fig_lvlm_lexical_prior.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_lvlm_lexical_prior.png', dpi=300, bbox_inches='tight')
print("Saved fig_lvlm_lexical_prior")
