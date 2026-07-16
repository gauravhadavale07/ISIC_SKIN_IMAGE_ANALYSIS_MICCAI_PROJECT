import pandas as pd
import numpy as np

df = pd.read_csv('results/task13_lvlm_audit.csv')
real_probs = df['Real_log_prob']
shuffled_probs = df['Shuffled_log_prob']
neutral_probs = df['Neutral_log_prob']

diffs = real_probs - shuffled_probs
percent_positive = (diffs > 0).mean() * 100

cohens_d = diffs.mean() / diffs.std()

print(f'Percent Real > Shuffled: {percent_positive:.2f}%')
print(f"Cohen's d: {cohens_d:.4f}")
print(f'Mean Real: {real_probs.mean():.4f}')
print(f'Mean Shuffled: {shuffled_probs.mean():.4f}')
print(f'Mean Neutral: {neutral_probs.mean():.4f}')
