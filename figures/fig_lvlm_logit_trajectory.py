import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Set style for MI4MedFM theme (clean, high contrast, clinical)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

# Empirical Data from the deep diagnostic run
alphas = [0.0, 2.0, 50.0]
mel_logits = [13.10, 11.02, 3.07]
nev_logits = [11.34, 9.75, 4.68]

# Setup Figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(alphas[:2], mel_logits[:2], color='#e74c3c', marker='o', linewidth=3, markersize=10, label="Malignant ('M')")
ax.plot(alphas[:2], nev_logits[:2], color='#3498db', marker='s', linewidth=3, markersize=10, label="Benign ('N')")

# Plot the destruction point (alpha=50) with dashed lines connecting to show the jump
ax.plot(alphas[1:], mel_logits[1:], color='#e74c3c', linestyle='--', linewidth=2, alpha=0.5)
ax.plot(alphas[1:], nev_logits[1:], color='#3498db', linestyle='--', linewidth=2, alpha=0.5)

# Plot the final points as different tokens
ax.plot([50.0], [3.07], color='#e74c3c', marker='X', markersize=12)
ax.plot([50.0], [4.68], color='#3498db', marker='X', markersize=12)

# Annotations for tokens
ax.text(0.0, 13.3, "'M'", color='#e74c3c', fontsize=12, ha='center', va='bottom', fontweight='bold')
ax.text(0.0, 11.1, "'N'", color='#3498db', fontsize=12, ha='center', va='top', fontweight='bold')

ax.text(2.0, 11.2, "'M'", color='#e74c3c', fontsize=12, ha='center', va='bottom', fontweight='bold')
ax.text(2.0, 9.5, "'N'", color='#3498db', fontsize=12, ha='center', va='top', fontweight='bold')

ax.text(50.0, 2.7, "'Pos'", color='#e74c3c', fontsize=12, ha='center', va='top', fontweight='bold')
ax.text(50.0, 5.0, "'Normal'", color='#3498db', fontsize=12, ha='center', va='bottom', fontweight='bold')

# The absolute top token annotation
ax.annotate("Absolute Top Token:\n'The'", xy=(1.0, 14.5), xycoords='data', ha='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

ax.annotate("Absolute Top Token:\n'achuset' (Hallucination)", xy=(35.0, 8.5), xycoords='data', ha='center', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdeded", ec="#e74c3c", alpha=0.9))

# Shading regions
ax.axvspan(-2, 10, color='#f2f4f4', alpha=0.5, label='Coherent Steering Region')
ax.axvspan(10, 55, color='#fdeded', alpha=0.5, label='Semantic Destruction Region')

# Gap annotation
ax.annotate('', xy=(0.0, 11.34), xytext=(0.0, 13.10),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax.text(0.5, 12.22, r'$\Delta = 1.76$', va='center', fontsize=12, fontweight='bold')

ax.set_xlim(-2, 55)
ax.set_ylim(0, 16)
ax.set_xlabel(r'Intervention Strength ($\alpha$)', fontsize=14, fontweight='bold')
ax.set_ylabel('Target Token Logit', fontsize=14, fontweight='bold')
ax.set_title('LVLM ActAdd Trajectory: Intervention vs. Hallucination Threshold', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', frameon=True, shadow=True, fancybox=True)

plt.tight_layout()
plt.savefig('figures/fig_lvlm_logit_trajectory.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig_lvlm_logit_trajectory.png', dpi=300, bbox_inches='tight')
print("Saved fig_lvlm_logit_trajectory")
