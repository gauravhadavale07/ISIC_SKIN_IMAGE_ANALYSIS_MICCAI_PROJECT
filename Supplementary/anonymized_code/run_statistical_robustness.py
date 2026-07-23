import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def task20_stats():
    from config import cfg

    print("="*50)
    print("TASK 20: STATISTICAL ROBUSTNESS (N=200)")
    print("="*50)
    task20_path = os.path.join(cfg.paths.extended_tables_dir, "task20_visual_biopsy_leak.csv")
    if not os.path.exists(task20_path):
        task20_path = os.path.join(cfg.paths.results_dir, "task20_visual_biopsy_leak.csv")
    if not os.path.exists(task20_path):
        print(f"File not found: {task20_path}")
        return
        
    df = pd.read_csv(task20_path)
    drop = df["activation_drop"]
    mean_drop = drop.mean()
    n = len(drop)
    
    t_stat, p_val = stats.ttest_1samp(drop, 0.0)
    se = drop.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    
    print(f"Mean Absolute Activation Drop: {mean_drop:.4f}")
    print(f"Paired t-test p-value (vs 0 drop): {p_val:.4e}")
    print(f"95% CI: [{mean_drop - h:.4f}, {mean_drop + h:.4f}]")
    
    rel_drop = df["relative_drop_pct"].dropna()
    mean_rel = rel_drop.mean()
    n_rel = len(rel_drop)
    
    t_stat_rel, p_val_rel = stats.ttest_1samp(rel_drop, 0.0)
    se_rel = rel_drop.std(ddof=1) / np.sqrt(n_rel)
    h_rel = se_rel * stats.t.ppf((1 + 0.95) / 2., n_rel-1)
    
    print(f"Mean Relative Drop: {mean_rel:.2f}%")
    print(f"Paired t-test p-value: {p_val_rel:.4e}")
    print(f"95% CI: [{mean_rel - h_rel:.2f}%, {mean_rel + h_rel:.2f}%]")
    print("="*50)

def task23_stats():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import cfg
    from dataset import MultimodalDermatologyDataset, get_transforms
    from task23_adversarial_demographic_swap import load_model, SEED
    
    print("\nRunning Task 23 inference to get paired margins...")
    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    try:
        model = load_model(device, SEED)
    except FileNotFoundError:
        model = load_model(device, cfg.seeds[0])
        
    df = pd.read_csv(cfg.paths.pad_ufes_csv)
    df_mel = df[df['diagnostic'].astype(str).str.upper() == 'MEL'].head(50)
    df_nev = df[df['diagnostic'].astype(str).str.upper() == 'NEV'].head(50)
    df_subset = pd.concat([df_mel, df_nev]).reset_index(drop=True)

    profile_A = "Male, age 85, presents with a lesion on the face."
    profile_B = "Female, age 18, presents with a lesion on the abdomen."

    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    temp_csv_A = os.path.join(cfg.paths.results_dir, "temp_pad_ufes_A.csv")
    temp_csv_B = os.path.join(cfg.paths.results_dir, "temp_pad_ufes_B.csv")

    df_subset['clinical_history'] = profile_A
    df_subset.to_csv(temp_csv_A, index=False)
    dataset_A = MultimodalDermatologyDataset(csv_file=temp_csv_A, img_dir=cfg.paths.pad_ufes_img_dir, tokenizer=tokenizer, transform=get_transforms())
    loader_A = DataLoader(dataset_A, batch_size=32, shuffle=False, num_workers=4)

    df_subset['clinical_history'] = profile_B
    df_subset.to_csv(temp_csv_B, index=False)
    dataset_B = MultimodalDermatologyDataset(csv_file=temp_csv_B, img_dir=cfg.paths.pad_ufes_img_dir, tokenizer=tokenizer, transform=get_transforms())
    loader_B = DataLoader(dataset_B, batch_size=32, shuffle=False, num_workers=4)

    mel_idx = cfg.data.LABEL_MAP['MEL']
    nev_idx = cfg.data.LABEL_MAP['NEV']

    def get_margins(loader):
        margins = []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                output = model(imgs, input_ids, attn_mask)
                logits = output[0] if isinstance(output, tuple) else output
                margin = logits[:, mel_idx] - logits[:, nev_idx]
                margins.extend(margin.cpu().tolist())
        return np.array(margins)

    margins_A = get_margins(loader_A)
    margins_B = get_margins(loader_B)
    
    margin_shift = margins_A - margins_B
    mean_shift = np.mean(margin_shift)
    
    n = len(margin_shift)
    t_stat, p_val = stats.ttest_rel(margins_A, margins_B)
    se = np.std(margin_shift, ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf((1 + 0.95) / 2., n-1)
    
    print("\n" + "="*50)
    print("TASK 23: STATISTICAL ROBUSTNESS (N=100)")
    print("="*50)
    print(f"Mean Margin Shift (A - B): {mean_shift:.4f}")
    print(f"Paired t-test p-value: {p_val:.4e}")
    print(f"95% CI of Margin Shift: [{mean_shift - h:.4f}, {mean_shift + h:.4f}]")
    print("="*50)
    
    try:
        os.remove(temp_csv_A)
        os.remove(temp_csv_B)
    except:
        pass

if __name__ == '__main__':
    task20_stats()
    task23_stats()
