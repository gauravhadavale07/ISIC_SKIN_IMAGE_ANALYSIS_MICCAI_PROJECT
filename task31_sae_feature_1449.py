import os
import sys
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlavaForConditionalGeneration, AutoProcessor
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import extract_all_activations, TopKSAE

def main():
    print("=" * 70)
    print("TASK 31: FEATURE 1449 ARTIFACT VALIDATION (CASE-CONTROL)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.milk10k_csv,
        img_dir=cfg.paths.milk10k_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )

    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)

    seed = 1337
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_{seed}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    base_model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    base_model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))

    print("\nExtracting 768-D dense activations...")
    X, y, filepaths_list = extract_all_activations(base_model, loader, device)

    if not filepaths_list:
        filepaths_list = dataset.df['filepath'].values.tolist()
        
    # Load SAE weights
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    sae_path = os.path.join(cfg.paths.results_dir, "sae_weights.pth")
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    X_device = X.to(device)
    with torch.no_grad():
        _, sparse_acts = sae(X_device)
    sparse_acts = sparse_acts.cpu().numpy()

    # Get feature 1449 activations
    feat_1449 = sparse_acts[:, 1449]

    # Create dataframe for all images
    df_all = pd.DataFrame({
        'filepath': filepaths_list,
        'label': y.numpy(),
        'feature1449_activation': feat_1449
    })

    df_all = df_all[df_all['filepath'].apply(lambda x: os.path.exists(str(x)))]
    
    print("\nExecuting Activation-Stratified Case-Control Sampling...")
    active_df = df_all[df_all['feature1449_activation'] > 0]
    inactive_df = df_all[df_all['feature1449_activation'] == 0]
    
    print(f"Total active images found: {len(active_df)}")
    
    # Sample an equal number of inactive controls
    control_df = inactive_df.sample(len(active_df), random_state=42)
    
    sampled_df = pd.concat([active_df, control_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total validation subset size: {len(sampled_df)} ({len(active_df)} Active vs {len(control_df)} Inactive)")

    # --- LVLM Annotation ---
    model_id = "llava-hf/llava-1.5-7b-hf"
    print(f"\nLoading LVLM for annotation: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    lvlm = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(device)
    lvlm.eval()

    print("\nStarting blind annotation...")
    artifact_types = []
    artifact_present = []

    with torch.no_grad():
        for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
            img_path = str(row['filepath'])
            img = Image.open(img_path).convert('RGB')
            
            prompt = "USER: <image>\nDoes this clinical photograph contain any visible peripheral procedural artifacts such as a ruler, measurement scale, skin marking, pen mark, surgical marker, or adhesive label? Answer with exactly one word from the following list: 'none', 'ruler', 'marker', 'label', or 'other'.\nASSISTANT:"
            
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
            
            generate_ids = lvlm.generate(**inputs, max_new_tokens=5, temperature=0.1)
            output_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            ans = output_text.split("ASSISTANT:")[-1].strip().lower()
            
            found_type = "none"
            for t in ["none", "ruler", "marker", "label", "other", "scale", "pen", "ink"]:
                if t in ans:
                    if t in ["scale", "ruler"]: found_type = "ruler"
                    elif t in ["pen", "ink", "marker"]: found_type = "marker"
                    elif t == "label": found_type = "other"
                    else: found_type = t
                    break
            
            if found_type == "none":
                artifact_present.append(0)
                artifact_types.append("none")
            else:
                artifact_present.append(1)
                artifact_types.append(found_type)

    sampled_df['artifact_present'] = artifact_present
    sampled_df['artifact_type'] = artifact_types
    sampled_df['is_active'] = (sampled_df['feature1449_activation'] > 0).astype(int)

    # Save CSV
    out_csv = "results/task31_feature1449_annotations.csv"
    sampled_df.to_csv(out_csv, index=False)
    print(f"\nSaved annotations to {out_csv}")

    print("\n--- Primary Statistical Analysis: Contingency Table ---")
    active_artifact = len(sampled_df[(sampled_df['is_active'] == 1) & (sampled_df['artifact_present'] == 1)])
    active_no_artifact = len(sampled_df[(sampled_df['is_active'] == 1) & (sampled_df['artifact_present'] == 0)])
    inactive_artifact = len(sampled_df[(sampled_df['is_active'] == 0) & (sampled_df['artifact_present'] == 1)])
    inactive_no_artifact = len(sampled_df[(sampled_df['is_active'] == 0) & (sampled_df['artifact_present'] == 0)])
    
    table = [[active_artifact, active_no_artifact], [inactive_artifact, inactive_no_artifact]]
    print(f"|          | Artifact | No Artifact |")
    print(f"| -------- | -------: | ----------: |")
    print(f"| Active   | {active_artifact:8} | {active_no_artifact:11} |")
    print(f"| Inactive | {inactive_artifact:8} | {inactive_no_artifact:11} |")

    res = stats.fisher_exact(table)
    odds_ratio, p_value = res.statistic, res.pvalue
    
    # 95% CI for Odds Ratio
    # SE(ln(OR)) = sqrt(1/a + 1/b + 1/c + 1/d)
    try:
        se_ln_or = np.sqrt(1/max(active_artifact, 1e-5) + 1/max(active_no_artifact, 1e-5) + 1/max(inactive_artifact, 1e-5) + 1/max(inactive_no_artifact, 1e-5))
        ci_lower = np.exp(np.log(odds_ratio) - 1.96 * se_ln_or)
        ci_upper = np.exp(np.log(odds_ratio) + 1.96 * se_ln_or)
    except:
        ci_lower, ci_upper = float('inf'), float('inf')

    print(f"\nFisher's Exact Test:")
    print(f"Odds Ratio = {odds_ratio:.2f}")
    print(f"95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"p = {p_value:.4e}")

    print("\n--- Secondary Analysis: Activation Magnitude ---")
    acts_with_artifact = sampled_df[sampled_df['artifact_present'] == 1]['feature1449_activation'].values
    acts_no_artifact = sampled_df[sampled_df['artifact_present'] == 0]['feature1449_activation'].values

    if len(acts_with_artifact) > 0 and len(acts_no_artifact) > 0:
        u_stat, p_val_mw = stats.mannwhitneyu(acts_with_artifact, acts_no_artifact, alternative='two-sided')
        print(f"Mann-Whitney Test comparing activation among Artifact vs No-Artifact groups:")
        print(f"  U statistic: {u_stat:.2f}")
        print(f"  p-value: {p_val_mw:.4e}")
    else:
        print("Not enough data points in one or both groups to run Mann-Whitney U test.")

    print("\n--- Generating Visualization ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Boxplot
    sns.boxplot(data=sampled_df, x='artifact_present', y='feature1449_activation', ax=axes[0], palette="Set2")
    axes[0].set_xticklabels(['No Artifact', 'Artifact'])
    axes[0].set_xlabel("Peripheral Procedural Artifacts")
    axes[0].set_ylabel("Feature 1449 Activation")
    axes[0].set_title("Activation Magnitude Distribution")

    # Panel 2: Stacked Bar Chart
    active_totals = active_artifact + active_no_artifact
    inactive_totals = inactive_artifact + inactive_no_artifact
    
    pct_active_artifact = (active_artifact / active_totals) * 100 if active_totals > 0 else 0
    pct_active_no_artifact = (active_no_artifact / active_totals) * 100 if active_totals > 0 else 0
    pct_inactive_artifact = (inactive_artifact / inactive_totals) * 100 if inactive_totals > 0 else 0
    pct_inactive_no_artifact = (inactive_no_artifact / inactive_totals) * 100 if inactive_totals > 0 else 0

    bars1 = [pct_active_artifact, pct_inactive_artifact]
    bars2 = [pct_active_no_artifact, pct_inactive_no_artifact]
    names = ['Active (>0)', 'Inactive (0)']

    axes[1].bar(names, bars1, color='#d95f02', edgecolor='white', label='Artifact')
    axes[1].bar(names, bars2, bottom=bars1, color='#1b9e77', edgecolor='white', label='No Artifact')

    axes[1].set_ylabel("Percentage (%)")
    axes[1].set_title("Proportion of Artifact Presence")
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    fig_path = "results/task31_feature1449_validation.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {fig_path}")

if __name__ == "__main__":
    main()
