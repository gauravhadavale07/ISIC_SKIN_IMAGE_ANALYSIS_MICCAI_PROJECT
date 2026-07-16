import pandas as pd
import numpy as np
from scipy import stats

def compute_ratio_of_means(group):
    baseline_disparity = group['light_true_logit'].mean() - group['dark_true_logit'].mean()
    main_shift = group['main_light_to_dark_true_logit'].mean() - group['dark_true_logit'].mean()
    return (main_shift / baseline_disparity) * 100

def bootstrap_ci_ratio(df_head, n_iterations=10000, seed=42):
    rng = np.random.default_rng(seed)
    ratios = []
    n = len(df_head)
    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        sample = df_head.iloc[indices]
        ratio = compute_ratio_of_means(sample)
        ratios.append(ratio)
    
    ratios = np.array(ratios)
    return np.percentile(ratios, [2.5, 97.5])

def main():
    df = pd.read_csv('results/task15_ddi_demographic_patching.csv')
    
    # Check Head 5 specifically, or we can check the baseline disparity for any head (the baseline is independent of head)
    # The baseline disparity is the same for all heads in the DataFrame for the same pair, so we just take head 0's rows.
    df_base = df[df['head'] == 0]
    
    light_logits = df_base['light_true_logit'].values
    dark_logits = df_base['dark_true_logit'].values
    
    differences = light_logits - dark_logits
    mean_diff = differences.mean()
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(light_logits, dark_logits)
    
    # Wilcoxon signed-rank test
    w_stat, w_p_val = stats.wilcoxon(light_logits, dark_logits)
    
    print("=== Baseline Disparity Significance (N=59) ===")
    print(f"Mean Difference (Light - Dark): {mean_diff:.4f}")
    print(f"Paired t-test p-value: {p_val:.4e}")
    print(f"Wilcoxon signed-rank p-value: {w_p_val:.4e}")
    
    # Bootstrap CI for baseline disparity
    rng = np.random.default_rng(42)
    boot_diffs = []
    for _ in range(10000):
        sample = rng.choice(differences, size=len(differences), replace=True)
        boot_diffs.append(sample.mean())
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"95% CI for Baseline Disparity: [{ci_lower:.4f}, {ci_upper:.4f}]\n")
    
    # Now for Head 5 recovery CI
    print("=== Head 5 Recovery Significance ===")
    df_head5 = df[df['head'] == 5]
    ci_rec_lower, ci_rec_upper = bootstrap_ci_ratio(df_head5, n_iterations=10000)
    point_estimate = compute_ratio_of_means(df_head5)
    print(f"Point Estimate: {point_estimate:.2f}%")
    print(f"95% CI for Ratio-of-Means Recovery: [{ci_rec_lower:.2f}%, {ci_rec_upper:.2f}%]")

if __name__ == '__main__':
    main()
