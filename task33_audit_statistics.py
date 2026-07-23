import os
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

def art_from_scratch(df, feat):
    """Recomputes ART strictly from scratch to verify the original logic."""
    mu_i = df.groupby('image_class')[feat].transform('mean')
    mu_j = df.groupby('text_class')[feat].transform('mean')
    mu_all = df[feat].mean()
    
    y_aligned = df[feat] - mu_i - mu_j + mu_all
    ranked = y_aligned.rank()
    
    df_temp = pd.DataFrame({
        'ranked': ranked,
        'image': df['image_class'],
        'text': df['text_class']
    })
    
    model = ols('ranked ~ C(image) * C(text)', data=df_temp).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    return anova.loc['C(image):C(text)', 'F'], anova.loc['C(image):C(text)', 'PR(>F)']

def kruskal_wallis_groups(df, feat):
    """Basic non-parametric test across the 4 experimental groups."""
    groups = [group[feat].values for name, group in df.groupby('group')]
    # If all values are 0 or identically distributed, this returns H=0, p=1
    try:
        stat, p = stats.kruskal(*groups)
        return stat, p
    except ValueError:
        return 0.0, 1.0

def permutation_interaction_test(df, feat, n_perms=1000):
    """
    Computes an interaction F-statistic on the RAW data (not ranked).
    Then shuffles the values and recomputes to build a null distribution.
    This does not rely on the ART assumptions.
    """
    def get_f(df_in, y_col):
        model = ols(f'{y_col} ~ C(image_class) * C(text_class)', data=df_in).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        return anova.loc['C(image_class):C(text_class)', 'F']
        
    try:
        obs_f = get_f(df, feat)
    except:
        return 0.0, 1.0
        
    df_perm = df.copy()
    count = 0
    for _ in range(n_perms):
        df_perm[feat] = np.random.permutation(df[feat].values)
        try:
            perm_f = get_f(df_perm, feat)
            if perm_f >= obs_f:
                count += 1
        except:
            pass
            
    return obs_f, count / n_perms

def main():
    print("--- Step 5 & 6: Independent Statistical Reproduction ---")
    acts_csv_path = os.path.join(cfg.paths.results_dir, "task32_top50_activations_seed1337.csv")
    df = pd.read_csv(acts_csv_path)
    
    features_to_test = ['feat_819', 'feat_810', 'feat_3276', 'feat_4463']
    
    print("\nResults:")
    print(f"{'Feature':<10} | {'ART_F':<10} | {'ART_p':<10} | {'KW_p':<10} | {'Perm_F':<10} | {'Perm_p':<10} | {'NonZeros':<8}")
    print("-" * 80)
    
    np.random.seed(1337)
    for feat in features_to_test:
        non_zeros = (df[feat] > 0).sum()
        art_f, art_p = art_from_scratch(df, feat)
        kw_stat, kw_p = kruskal_wallis_groups(df, feat)
        perm_f, perm_p = permutation_interaction_test(df, feat, n_perms=50)
        
        print(f"{feat:<10} | {art_f:<10.2f} | {art_p:<10.2e} | {kw_p:<10.2e} | {perm_f:<10.2f} | {perm_p:<10.2e} | {non_zeros:<8}")

    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
