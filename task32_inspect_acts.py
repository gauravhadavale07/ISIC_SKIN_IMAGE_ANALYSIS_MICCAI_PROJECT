import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

def main():
    acts_path = os.path.join(cfg.paths.results_dir, "task32_top50_activations_seed1337.csv")
    df = pd.read_csv(acts_path)
    
    features = ['feat_819', 'feat_3276', 'feat_810', 'feat_4463', 'feat_3642', 'feat_4120']
    
    for feat in features:
        if feat not in df.columns:
            continue
            
        print(f"\n--- {feat} ---")
        non_zero = df[df[feat] > 0]
        print(f"Total non-zero activations across all groups: {len(non_zero)} / {len(df)}")
        if len(non_zero) > 0:
            print("Value counts by group:")
            print(non_zero['group'].value_counts())
            print("\nTop 5 values:")
            print(non_zero[['sample_id', 'group', feat]].sort_values(by=feat, ascending=False).head(5))

if __name__ == "__main__":
    main()
