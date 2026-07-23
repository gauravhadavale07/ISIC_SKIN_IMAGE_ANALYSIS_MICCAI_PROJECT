import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionClassifier, CrossAttentionT2VClassifier

# We'll use the central 10 seeds
SEEDS = cfg.seeds

ARCHITECTURES = {
    "Late Fusion": (LateFusionClassifier, "Late_Fusion"),
    "GMU Baseline": (GMUClassifier, "GMU_Baseline"),
    "Cross-Attention (V->T)": (CrossAttentionClassifier, "Cross-Attention_V→T"),
    "Cross-Attention T->V": (CrossAttentionT2VClassifier, "Cross-Attention_T→V"),
}

class ShuffledTextDataset(MultimodalDermatologyDataset):
    def __init__(self, csv_file, img_dir, tokenizer, shuffle_seed, transform=None):
        super().__init__(csv_file=csv_file, img_dir=img_dir, tokenizer=tokenizer, transform=transform)
        # Shuffle the clinical_history
        rng = np.random.default_rng(shuffle_seed)
        hist = self.df['clinical_history'].values.copy()
        rng.shuffle(hist)
        self.df['clinical_history'] = hist

def get_accuracy(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            logits, _, _ = model(imgs, input_ids, attn_mask)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return np.mean(all_preds == all_labels)

def hedges_g_paired(a, b):
    diffs = np.array(b) - np.array(a)
    sd = np.std(diffs, ddof=1)
    if sd == 0:
        d = float('inf') if diffs.mean() != 0 else 0.0
    else:
        d = float(diffs.mean() / sd)
    n = len(diffs)
    J = 1 - (3 / (4 * (n - 1) - 1)) if n > 1 else 1.0
    return d * J

def main():
    print("=" * 70)
    print(f"TASK 1: METADATA-SHUFFLE STATISTICAL SIGNIFICANCE (N={len(SEEDS)} Seeds)")
    print("=" * 70)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    # Prepare standard dataset
    real_ds = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )
    real_loader = DataLoader(real_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)
    
    # Prepare one shuffled dataset to test structural semantic blindness
    s_ds = ShuffledTextDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        shuffle_seed=42, # Single fixed text shuffle
        transform=get_transforms()
    )
    s_loader = DataLoader(s_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0)

    results = []
    
    for arch_name, (model_cls, prefix) in ARCHITECTURES.items():
        print(f"\nEvaluating: {arch_name}")
        real_accs = []
        shuffled_accs = []
        
        for seed in SEEDS:
            ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"{prefix}_seed_{seed}", "best_model.pth")
            if not os.path.exists(ckpt_path):
                print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
                continue
                
            model = model_cls().to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
            model.eval()
            
            r_acc = get_accuracy(model, real_loader, device)
            s_acc = get_accuracy(model, s_loader, device)
            
            real_accs.append(r_acc)
            shuffled_accs.append(s_acc)
            print(f"  Seed {seed}: Real {r_acc*100:.2f}%, Shuffle {s_acc*100:.2f}%")
            
            del model
            torch.cuda.empty_cache()

        if len(real_accs) < 2:
            print(f"  Not enough seeds evaluated for {arch_name}.")
            continue
            
        real_mean = np.mean(real_accs)
        real_std = np.std(real_accs)
        shuf_mean = np.mean(shuffled_accs)
        shuf_std = np.std(shuffled_accs)
        
        # Paired test
        t, p_val = stats.ttest_rel(real_accs, shuffled_accs)
        g = hedges_g_paired(shuffled_accs, real_accs) # diff = real - shuffled
        delta = real_mean - shuf_mean
        
        # Standard error of differences
        diffs = np.array(real_accs) - np.array(shuffled_accs)
        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        ci_lower = delta - stats.t.ppf(0.975, len(diffs)-1) * se
        ci_upper = delta + stats.t.ppf(0.975, len(diffs)-1) * se
        
        results.append({
            "Architecture": arch_name,
            "Real Acc (mean±std)": f"{real_mean*100:.2f}±{real_std*100:.2f}%",
            "Shuffle Acc (mean±std)": f"{shuf_mean*100:.2f}±{shuf_std*100:.2f}%",
            "Delta": f"{delta*100:+.2f}pp",
            "95% CI": f"[{ci_lower*100:+.2f}, {ci_upper*100:+.2f}]",
            "p-value (raw)": p_val,
            "Hedges g": g
        })

    if not results:
        return

    # Holm-Bonferroni Correction
    from statsmodels.stats.multitest import multipletests
    p_vals_raw = [r['p-value (raw)'] for r in results]
    _, p_vals_corrected, _, _ = multipletests(p_vals_raw, alpha=0.05, method='holm')
    
    for i, r in enumerate(results):
        r['p-value (raw_str)'] = f"{r['p-value (raw)']:.5f}"
        r['p-value (Holm_str)'] = f"{p_vals_corrected[i]:.5f}"

    # Output formatting
    df_res = pd.DataFrame(results)
    
    print("\n" + "="*115)
    print(f"TASK 1 RESULTS: METADATA-SHUFFLE STATISTICAL SIGNIFICANCE (HOLM-BONFERRONI CORRECTED, N={len(SEEDS)})")
    print("="*115)
    
    header = f"{'Architecture':<25} | {'Real Acc':<20} | {'Shuffle Acc':<20} | {'Delta':<8} | {'95% CI':<15} | {'p-val (Holm)':<12} | {'Hedges g':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['Architecture']:<25} | {r['Real Acc (mean±std)']:<20} | {r['Shuffle Acc (mean±std)']:<20} | {r['Delta']:<8} | {r['95% CI']:<15} | {r['p-value (Holm_str)']:<12} | {r['Hedges g']:.3f}"
        print(row)
        
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.paths.results_dir, "task1_metadata_shuffle_significance.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

if __name__ == "__main__":
    main()
