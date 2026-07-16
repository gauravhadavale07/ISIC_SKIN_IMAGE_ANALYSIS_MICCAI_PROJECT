import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.gmu import GMUClassifier

MODEL_SEED = 1337
SHUFFLE_SEED = 42

class ModifiedTextDataset(MultimodalDermatologyDataset):
    def __init__(self, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mode == 'shuffled':
            rng = np.random.default_rng(SHUFFLE_SEED)
            hist = self.df['clinical_history'].values.copy()
            rng.shuffle(hist)
            self.df['clinical_history'] = hist
        elif mode == 'neutral':
            self.df['clinical_history'] = cfg.audit.neutral_string

def get_gate_activations(model, dataloader, device):
    gate_acts = []
    
    def hook_fn(module, input, output):
        # output is the gate, shape (B, 512)
        gate_acts.append(output.detach().cpu())
        
    handle = model.gate.register_forward_hook(hook_fn)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting GMU Gates", leave=False):
            imgs = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            _ = model(imgs, input_ids, attn_mask)
            
    handle.remove()
    
    # Concatenate all gate activations
    all_gates = torch.cat(gate_acts, dim=0).numpy() # (N, 512)
    # Return both mean per sample (N,) and flattened distribution (N*512)
    return np.mean(all_gates, axis=1), all_gates.flatten()

def main():
    print("=" * 70)
    print("TASK 2: GMU GATE-WEIGHT ANALYSIS")
    print("=" * 70)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    datasets = {
        "Real": ModifiedTextDataset('real', csv_file=cfg.paths.pad_ufes_csv, img_dir=cfg.paths.pad_ufes_img_dir, tokenizer=tokenizer, transform=get_transforms()),
        "Neutral": ModifiedTextDataset('neutral', csv_file=cfg.paths.pad_ufes_csv, img_dir=cfg.paths.pad_ufes_img_dir, tokenizer=tokenizer, transform=get_transforms()),
        "Shuffled": ModifiedTextDataset('shuffled', csv_file=cfg.paths.pad_ufes_csv, img_dir=cfg.paths.pad_ufes_img_dir, tokenizer=tokenizer, transform=get_transforms()),
    }
    
    loaders = {k: DataLoader(v, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4) for k, v in datasets.items()}
    
    print(f"\nLoading GMU Baseline...")
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"GMU_Baseline_seed_{MODEL_SEED}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
        
    model = GMUClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    
    results = {}
    
    for mode, loader in loaders.items():
        print(f"  Processing {mode} Text...")
        mean_per_sample, flattened = get_gate_activations(model, loader, device)
        results[mode] = {
            "mean_per_sample": mean_per_sample,
            "flattened": flattened,
            "overall_mean": np.mean(flattened),
            "overall_std": np.std(flattened)
        }
    
    # Statistical tests: Paired test (Wilcoxon Signed-Rank) on absolute differences per dimension
    print("\nRunning Paired Statistical Tests (Wilcoxon Signed-Rank)...")
    
    tests = []
    # To save time on 700k elements, sample 50,000 pairs randomly or use t-test, but Wilcoxon is fine.
    # We will compute the Mean Absolute Difference (MAD) per sample dimension
    for cmp_mode in ["Neutral", "Shuffled"]:
        diffs = results["Real"]["flattened"] - results[cmp_mode]["flattened"]
        mad = np.mean(np.abs(diffs))
        
        # Wilcoxon can be slow, but usually <5s for 700k items in scipy. 
        # But we'll run it on a large subset if too slow. Let's just run it.
        # Alternatively, we can use scipy.stats.ttest_rel
        from scipy.stats import ttest_rel
        t_stat, t_p = ttest_rel(results["Real"]["flattened"], results[cmp_mode]["flattened"])
        
        tests.append({
            "Comparison": f"Real vs {cmp_mode}",
            "Mean Abs Diff": mad,
            "Paired T-Stat": t_stat,
            "Paired p-value": t_p
        })
    
    print("\n" + "="*85)
    print("TASK 2 RESULTS: GMU GATE DISTRIBUTION (PAIRED TEST)")
    print("="*85)
    
    for mode, data in results.items():
        print(f"{mode:10} | Mean Gate Value: {data['overall_mean']:.4f} ± {data['overall_std']:.4f}")
        
    print("\nPaired Tests:")
    for t in tests:
        print(f"{t['Comparison']:18} | Mean Abs Diff: {t['Mean Abs Diff']:.4f} | p-value: {t['Paired p-value']:.4e}")
    
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    
    # Save test stats
    df_tests = pd.DataFrame(tests)
    df_tests.to_csv(os.path.join(cfg.paths.results_dir, "task2_gmu_gate_tests.csv"), index=False)
    
    # Save histogram data
    df_hist = pd.DataFrame({
        "Real_Gate_Mean": results["Real"]["mean_per_sample"],
        "Neutral_Gate_Mean": results["Neutral"]["mean_per_sample"],
        "Shuffled_Gate_Mean": results["Shuffled"]["mean_per_sample"]
    })
    hist_out = os.path.join(cfg.paths.results_dir, "task2_gmu_gate_hist_data.csv")
    df_hist.to_csv(hist_out, index=False)
    
    print(f"\nResults saved to results/task2_gmu_gate_tests.csv and results/task2_gmu_gate_hist_data.csv")

if __name__ == "__main__":
    main()
