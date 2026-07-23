import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from task33_audit_extraction import build_2x2_dataframe

def main():
    print("--- Step 2 & 7: Verification of Sample Alignment and Paired Structure ---\n")
    
    # Rebuild inputs
    df_inputs = build_2x2_dataframe(cfg.paths.pad_ufes_csv)
    
    # Load stored results
    acts_csv_path = os.path.join(cfg.paths.results_dir, "task32_top50_activations_seed1337.csv")
    df_stored = pd.read_csv(acts_csv_path)
    
    # 1. Verify rows per image
    counts = df_stored['sample_id'].value_counts()
    print(f"Total unique images: {len(counts)}")
    print(f"Row count distribution per image:\n{counts.value_counts()}\n")
    
    # The user asked to confirm "four rows A,B,C,D" but the implementation only created 2 rows per image
    # (A,B for Malignant, C,D for Benign). Let's explicitly check and report this.
    
    # Join the inputs to the stored outputs
    df_merged = df_stored.merge(df_inputs, on=['sample_id', 'group'], how='left')
    
    print("Verifying identical image paths within pairs...")
    errors = 0
    checked = 0
    for sample_id, group_df in df_merged.groupby('sample_id'):
        paths = group_df['img_path'].unique()
        if len(paths) != 1:
            print(f"ERROR: sample_id {sample_id} has multiple image paths! {paths}")
            errors += 1
        checked += 1
        
    print(f"Verified {checked} image pairs. Image Path Mismatches: {errors}\n")
    
    print("10 Random Paired Examples:")
    # Print 10 random examples
    sample_ids = df_merged['sample_id'].unique()
    import numpy as np
    np.random.seed(1337)
    random_samples = np.random.choice(sample_ids, 10, replace=False)
    
    for sid in random_samples:
        subset = df_merged[df_merged['sample_id'] == sid]
        print(f"\nSample ID: {sid}")
        print(f"Image Path: {subset['img_path'].iloc[0]}")
        for _, row in subset.iterrows():
            print(f"  Group: {row['group']}, Text: '{row['text'][:50]}...', "
                  f"ImgClass: {row['image_class_x']}, TxtClass: {row['text_class_x']}")
            
if __name__ == "__main__":
    main()
