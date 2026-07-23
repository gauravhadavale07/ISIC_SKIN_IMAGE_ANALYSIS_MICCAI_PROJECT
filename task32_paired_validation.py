import os
import sys
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

def main():
    acts_path = os.path.join(cfg.paths.results_dir, "task32_top50_activations_seed1337.csv")
    if not os.path.exists(acts_path):
        print(f"Error: {acts_path} not found.")
        return
        
    df = pd.read_csv(acts_path)
    
    # Top features to validate
    features = ['feat_819', 'feat_3276', 'feat_810', 'feat_4463']
    
    print("==================================================")
    print("Paired Validation for Top Multimodal Interaction Features")
    print("==================================================\n")
    
    # We will test A > B (Melanoma Images) and C > D (Benign Images)
    
    # Process Melanoma Images (Groups A and B)
    df_mel = df[df['group'].isin(['A', 'B'])].copy()
    
    for feat in features:
        if feat not in df.columns:
            continue
            
        print(f"--- Analyzing {feat} on Melanoma Images (A vs B) ---")
        
        # Pivot to get paired activations per sample_id
        pivot = df_mel.pivot(index='sample_id', columns='group', values=feat).dropna()
        
        # Compute difference A - B
        pivot['diff'] = pivot['A'] - pivot['B']
        
        total_pairs = len(pivot)
        a_greater_b = (pivot['diff'] > 0).sum()
        a_equal_b = (pivot['diff'] == 0).sum()
        a_less_b = (pivot['diff'] < 0).sum()
        
        pct_greater = (a_greater_b / total_pairs) * 100
        
        # Paired t-test and Wilcoxon signed-rank test
        t_stat, p_t = stats.ttest_rel(pivot['A'], pivot['B'])
        w_stat, p_w = stats.wilcoxon(pivot['A'], pivot['B'], zero_method='zsplit')
        
        print(f"Total Paired Images: {total_pairs}")
        print(f"A > B: {a_greater_b} ({pct_greater:.1f}%)")
        print(f"A == B: {a_equal_b} ({(a_equal_b/total_pairs)*100:.1f}%)")
        print(f"A < B: {a_less_b} ({(a_less_b/total_pairs)*100:.1f}%)")
        print(f"Mean Act(A): {pivot['A'].mean():.4f}")
        print(f"Mean Act(B): {pivot['B'].mean():.4f}")
        print(f"Paired T-Test p-value: {p_t:.2e}")
        print(f"Wilcoxon p-value: {p_w:.2e}\n")

if __name__ == "__main__":
    main()
