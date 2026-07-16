import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

os.makedirs("figures", exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.2

results_path = Path("results/task16_results.csv")
df = pd.read_csv(results_path)
primary = df[df["condition"] == "Primary (MEL->NEV)"].copy()
random_control = df[df["condition"] == "Random Control"].copy()

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    primary["alpha"],
    primary["mean_mel_minus_nev_margin"],
    color="#8e44ad",
    marker="o",
    linewidth=3,
    markersize=9,
    label="ActAdd MEL->NEV",
)
ax.plot(
    random_control["alpha"],
    random_control["mean_mel_minus_nev_margin"],
    color="#7f8c8d",
    marker="s",
    linewidth=2.5,
    markersize=8,
    linestyle="--",
    label="Random direction control",
)

ax.axhline(0, color="#e74c3c", linewidth=2, alpha=0.85, label="MEL/NEV boundary")
ax.fill_between(primary["alpha"], 0, primary["mean_mel_minus_nev_margin"], color="#8e44ad", alpha=0.08)

for _, row in primary.iterrows():
    ax.text(
        row["alpha"],
        row["mean_mel_minus_nev_margin"] + 0.08,
        f"{row['flip_rate']:.0%} flips",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#4a235a",
    )

ax.set_xlabel(r"Intervention strength ($\alpha$)", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean MEL-minus-NEV margin", fontsize=14, fontweight="bold")
ax.set_title("Task 16 H200 Sweep: Active Margin Shift, No Boundary Crossing", fontsize=15, fontweight="bold")
ax.set_xticks(primary["alpha"].tolist())
ax.set_ylim(min(-0.1, primary["mean_mel_minus_nev_margin"].min() - 0.25), max(random_control["mean_mel_minus_nev_margin"].max() + 0.35, 3.2))
ax.legend(loc="upper right", frameon=True, shadow=True)

caption_note = "Primary arm: N=50 baseline-correct MEL; reverse NEV arm N=0"
ax.text(
    0.01,
    0.02,
    caption_note,
    transform=ax.transAxes,
    fontsize=10,
    color="#333333",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb", alpha=0.9),
)

plt.tight_layout()
plt.savefig("figures/fig_lvlm_margin_scatter.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig_lvlm_margin_scatter.png", dpi=300, bbox_inches="tight")
print("Saved fig_lvlm_margin_scatter")
