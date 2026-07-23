import pandas as pd
import numpy as np

def run_sanity_check():
    df = pd.read_csv("results/task31_feature1449_annotations.csv")
    
    print("=== SANITY CHECK: FEATURE 1449 ===\n")
    print(f"Number of sampled images: {len(df)}")
    
    n_present = len(df[df['artifact_present'] == 1])
    n_absent = len(df[df['artifact_present'] == 0])
    print(f"Number with artifact_present = 1: {n_present}")
    print(f"Number with artifact_present = 0: {n_absent}")
    
    print("\nFirst 10 (activation, artifact_present) pairs:")
    for _, row in df.head(10).iterrows():
        print(f"  Activation: {row['feature1449_activation']:.4f}, Artifact: {row['artifact_present']}")
        
    print("\nOverall Feature 1449 Activation stats in the 100 samples:")
    non_zero = len(df[df['feature1449_activation'] > 0])
    print(f"Number of non-zero activations in sample: {non_zero}")
    print(f"Max activation in sample: {df['feature1449_activation'].max():.4f}")
    
    acts_present = df[df['artifact_present'] == 1]['feature1449_activation'].values
    acts_absent = df[df['artifact_present'] == 0]['feature1449_activation'].values
    
    print(f"\nMedian activation computed directly:")
    print(f"  Artifact present (N={len(acts_present)}): {np.median(acts_present) if len(acts_present)>0 else 0:.4f}")
    print(f"  Artifact absent (N={len(acts_absent)}): {np.median(acts_absent) if len(acts_absent)>0 else 0:.4f}")
    
    print(f"\nGroup sizes used in Mann-Whitney test:")
    print(f"  Group 1 (Present): {len(acts_present)}")
    print(f"  Group 2 (Absent): {len(acts_absent)}")

if __name__ == "__main__":
    run_sanity_check()
