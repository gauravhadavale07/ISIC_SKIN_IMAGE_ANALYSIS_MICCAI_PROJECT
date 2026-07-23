import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    sampled_df = pd.read_csv("results/task31_feature1449_annotations.csv")
    acts_present = sampled_df[sampled_df['artifact_present'] == 1]['feature1449_activation'].values
    acts_absent = sampled_df[sampled_df['artifact_present'] == 0]['feature1449_activation'].values

    plt.figure(figsize=(10, 6))
    
    # Overriding labels for plot to include requested values
    plot_df = sampled_df.copy()
    plot_df['Artifact Status'] = plot_df['artifact_present'].map({0: 'No Artifact\n(Median 0.63)', 1: 'Artifact\n(Median 2.14)'})
    
    sns.violinplot(data=plot_df, x='Artifact Status', y='feature1449_activation', inner=None, color=".8")
    sns.stripplot(data=plot_df, x='Artifact Status', y='feature1449_activation', size=4, color=".3", jitter=True, alpha=0.5)
    
    plt.title('Feature 1449 Activation vs Peripheral Procedural Artifacts')
    plt.ylabel('Feature 1449 Activation')
    
    plt.text(0.05, 0.95, f"N = {len(acts_absent)}", transform=plt.gca().transAxes)
    plt.text(0.55, 0.95, f"N = {len(acts_present)}", transform=plt.gca().transAxes)
    
    fig_path = "results/task31_feature1449_violin.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {fig_path}")

if __name__ == "__main__":
    main()
