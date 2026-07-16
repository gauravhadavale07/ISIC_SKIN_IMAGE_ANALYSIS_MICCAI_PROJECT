import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 1. Load results and compute effect size
df = pd.read_csv("results/task13_lvlm_audit.csv")
real = df['Real_log_prob'].values
shuff = df['Shuffled_log_prob'].values

diff = real - shuff
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)
cohens_d = mean_diff / std_diff

flip_pct = np.mean(shuff < real) * 100

print(f"Cohen's d (Real vs Shuffled): {cohens_d:.4f}")
print(f"Percentage of images where Shuffled log-prob < Real log-prob: {flip_pct:.2f}%")

# 2. Scope check
pad_df = pd.read_csv("pad_ufes_20_test.csv")
print(f"\nTotal rows in pad_ufes_20_test.csv: {len(pad_df)}")
print("Let's look at clinical_history lengths.")

real_lengths = pad_df['clinical_history'].apply(lambda x: len(str(x))).values
neutral_length = len("Patient with no clinical history available.")
print(f"Average Real text length (chars): {np.mean(real_lengths):.2f}")
print(f"Neutral text length (chars): {neutral_length}")

# Correlate length with Real log prob
corr, p = pearsonr(real_lengths, real)
print(f"Pearson correlation (Real Text Length vs Real Log Prob): r={corr:.4f}, p={p:.4e}")

