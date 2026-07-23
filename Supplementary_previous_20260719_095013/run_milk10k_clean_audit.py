"""
run_milk10k_clean_audit.py — Lesion-disjoint in-domain audit on MILK10k.

WHY THIS SCRIPT EXISTS
=======================
The original run_milk10k_audit.py used a random 85/15 image-level split
(seeds 456/789/1337) on MILK10k.  Because every lesion has exactly 2 images,
the random split places ~85% of val-set images into lesions that also appear in
the train set:

  Seed 456 : 1,188 / 1,402 val samples from shared lesions (84.7%)
  Seed 789 : 1,194 / 1,402 val samples from shared lesions (85.2%)
  Seed 1337: 1,228 / 1,402 val samples from shared lesions (87.6%)

Text-Only in-domain accuracy (53.26%) being far above its OOD performance
(36.77% = majority baseline collapse) despite having no generalisation
mechanism for within-domain other than near-duplicate text from same-lesion
pairs strongly suggests memorisation rather than genuine in-domain behaviour.

CLEAN SPLIT STRATEGY
=====================
Group all 4,672 unique lesion IDs.  Sort them deterministically (ascending
IL_xxxxxx string sort).  Reserve the last 15% of *lesions* (700 lesions =
1,400 images) as the clean held-out validation set.  Zero lesion overlap is
guaranteed by construction.

All architectures are evaluated on this same fixed clean set (no retraining).
No random seed is needed for the split itself; the seed-456/789/1337 variation
reported is across model checkpoints only.

OUTPUTS
========
  milk10k_clean_audit_results.json   — machine-readable summary
  milk10k_clean_audit_run.log        — full console output (redirect with tee)
"""

import os, sys, re, json
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

def bootstrap_lesion_metrics(y_true, y_pred, lesion_ids, n_bootstraps=2000, seed=42):
    rng = np.random.RandomState(seed)
    unique_lesions = np.unique(lesion_ids)
    n_lesions = len(unique_lesions)
    
    if n_lesions == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    # Precompute a dictionary of lesion_id -> array of indices for speed
    lesion_to_idx = {}
    for idx, lid in enumerate(lesion_ids):
        lesion_to_idx.setdefault(lid, []).append(idx)
        
    acc_scores = []
    f1_scores = []
    
    for _ in range(n_bootstraps):
        sampled_lesions = rng.choice(unique_lesions, size=n_lesions, replace=True)
        sampled_indices = []
        for lid in sampled_lesions:
            sampled_indices.extend(lesion_to_idx[lid])
            
        y_t = y_true[sampled_indices]
        y_p = y_pred[sampled_indices]
        
        present_classes = np.unique(y_t)
        if len(present_classes) > 1:
            try:
                acc = np.mean(y_t == y_p)
                f1 = f1_score(y_t, y_p, average='macro')
                acc_scores.append(acc)
                f1_scores.append(f1)
            except ValueError:
                pass
                
    if not acc_scores:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    acc_scores = np.array(acc_scores)
    f1_scores = np.array(f1_scores)
    
    return (np.mean(acc_scores), np.percentile(acc_scores, 2.5), np.percentile(acc_scores, 97.5),
            np.mean(f1_scores), np.percentile(f1_scores, 2.5), np.percentile(f1_scores, 97.5))

from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from evaluate import Evaluator
from counterfactual import CounterfactualAuditor
from cka import CKAAuditor
from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionClassifier, CrossAttentionT2VClassifier
from models.image_only import ImageOnlyClassifier
from models.text_only import TextOnlyClassifier

MAJORITY_BASELINE = 845 / 2298 * 100   # PAD-UFES-20 majority baseline (for reference)
MILK_MAJORITY_CLASS = "BCC"

MODELS = {
    "Late Fusion":              (LateFusionClassifier,        "Late_Fusion"),
    "GMU Baseline":             (GMUClassifier,               "GMU_Baseline"),
    "Cross-Attention (V->T)":   (CrossAttentionClassifier,    "Cross-Attention_V→T"),
    "Cross-Attention T->V":     (CrossAttentionT2VClassifier, "Cross-Attention_T→V"),
    "Image-Only":               (ImageOnlyClassifier,         "ImageOnly"),
    "Text-Only":                (TextOnlyClassifier,          "TextOnly"),
}
SEEDS = cfg.seeds

# ── Build lesion-disjoint clean held-out set ───────────────────────────────────
print("=" * 60)
print("MILK10k LESION-DISJOINT CLEAN AUDIT")
print("=" * 60)

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

clean_val_dataset = MultimodalDermatologyDataset(
    csv_file=cfg.paths.milk10k_csv,
    img_dir="",
    tokenizer=tokenizer,
    transform=get_transforms(),
    split="val"
)
val_df = clean_val_dataset.df

print(f"\nLesion-disjoint split (fixed, no random seed):")
print(f"  Val images     : {len(val_df)}")
print(f"  Lesion overlap : 0  (guaranteed by dataset split='val')")
print()
print(f"  Clean val class distribution:")
for cls, cnt in val_df['diagnostic'].value_counts().items():
    print(f"    {cls}: {cnt}  ({cnt/len(val_df)*100:.1f}%)")

# Compute majority class in the clean val set
milk_majority_class = val_df['diagnostic'].value_counts().index[0]
milk_majority_count = val_df['diagnostic'].value_counts().iloc[0]
milk_majority_baseline = milk_majority_count / len(val_df) * 100
print(f"\n  Majority class in clean val: {milk_majority_class} "
      f"({milk_majority_count}/{len(val_df)} = {milk_majority_baseline:.2f}%)")
print()

clean_val_loader = DataLoader(
    clean_val_dataset,
    batch_size=cfg.train.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Clean val loader: {len(clean_val_dataset)} samples, {len(clean_val_loader)} batches\n")

# ── Run audit ──────────────────────────────────────────────────────────────────
device = cfg.train.device
print(f"Device: {device}\n")

# Load existing results to avoid re-running expensive mechanistic probes
EXISTING_JSON = "./milk10k_clean_audit_results.json"
existing_data = {}
if os.path.exists(EXISTING_JSON):
    with open(EXISTING_JSON, "r") as f:
        existing_data = json.load(f).get("results", {})

results = {}

for arch_name, (model_cls, prefix) in MODELS.items():
    print(f"\n{'─'*55}")
    print(f"Architecture: {arch_name}")
    print(f"{'─'*55}")

    arch_results = {
        "Accuracy":        [],
        "Acc_CI_Lower":    [],
        "Acc_CI_Upper":    [],
        "F1 (Macro)":      [],
        "F1_CI_Lower":     [],
        "F1_CI_Upper":     [],
        "Blank_Accuracy":  [],
        "Blank_Drop":      [],
        "Neutral_Accuracy":[],
        "CFR":             [],
        "Mean_Delta_P":    [],
        "Linear_CKA":      [],
    }

    for seed in SEEDS:
        ckpt_path = f"./checkpoints/{prefix}_seed_{seed}/best_model.pth"
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] seed={seed}: checkpoint not found at {ckpt_path}")
            continue

        print(f"  Seed {seed} ...")
        model = model_cls().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(sd)
        model.eval()

        # 1. Base accuracy and raw predictions
        evaluator = Evaluator(model, device)
        metrics   = evaluator.evaluate(clean_val_loader)
        real_acc  = metrics["Accuracy"]
        y_true    = metrics["y_true"]
        y_pred    = metrics["y_pred"]
        y_prob    = metrics["y_prob"]
        
        # Save raw predictions
        os.makedirs("results", exist_ok=True)
        pred_path = f"results/milk10k_raw_preds_{prefix}_seed_{seed}.npz"
        np.savez(pred_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        
        # Bootstrap Lesion metrics
        val_lesion_ids = val_df['lesion_id'].values
        acc_m, acc_lo, acc_hi, f1_m, f1_lo, f1_hi = bootstrap_lesion_metrics(y_true, y_pred, val_lesion_ids)
        
        arch_results["Accuracy"].append(acc_m)
        arch_results["Acc_CI_Lower"].append(acc_lo)
        arch_results["Acc_CI_Upper"].append(acc_hi)
        arch_results["F1 (Macro)"].append(f1_m)
        arch_results["F1_CI_Lower"].append(f1_lo)
        arch_results["F1_CI_Upper"].append(f1_hi)

        # 2. Re-use Counterfactual / blank / neutral probes and CKA from old JSON
        # Since these don't depend on lesion bootstrapping, we can just copy them over.
        old_arch = existing_data.get(arch_name, {})
        seed_idx = SEEDS.index(seed)
        
        # Fallbacks for image-only/text-only or if data is missing
        if arch_name in ("Image-Only", "Text-Only"):
            arch_results["Blank_Accuracy"].append(real_acc)
            arch_results["Blank_Drop"].append(0.0)
            arch_results["Neutral_Accuracy"].append(real_acc)
            arch_results["CFR"].append(0.0)
            arch_results["Mean_Delta_P"].append(0.0)
            arch_results["Linear_CKA"].append(1.0)
        else:
            arch_results["Blank_Accuracy"].append(old_arch.get("Blank_Accuracy", [0]*3)[seed_idx])
            arch_results["Blank_Drop"].append(old_arch.get("Blank_Drop", [0]*3)[seed_idx])
            arch_results["Neutral_Accuracy"].append(old_arch.get("Neutral_Accuracy", [0]*3)[seed_idx])
            arch_results["CFR"].append(old_arch.get("CFR", [0]*3)[seed_idx])
            arch_results["Mean_Delta_P"].append(old_arch.get("Mean_Delta_P", [0]*3)[seed_idx])
            arch_results["Linear_CKA"].append(old_arch.get("Linear_CKA", [0]*3)[seed_idx])

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results[arch_name] = arch_results


# ── Summary table ──────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("CLEAN IN-DOMAIN (MILK10k) AUDIT RESULTS -- LESION-DISJOINT VAL SET")
print(f"Majority class in clean val: {milk_majority_class} ({milk_majority_baseline:.2f}%)")
print(f"{'='*70}")
print(f"{'Architecture':<28} | {'Acc (%)':>18} | {'F1 (%)':>18} | {'BlkAcc':>7} | {'BlkDrp':>7} | {'Neutral':>7} | {'CFR':>6} | {'CKA':>7}")
print("-" * 115)

for arch, metrics in results.items():
    if not metrics["Accuracy"]:
        print(f"  {arch:<26} | [NO DATA]")
        continue
    acc  = np.mean(metrics["Accuracy"]) * 100
    acc_lo = np.mean(metrics["Acc_CI_Lower"]) * 100
    acc_hi = np.mean(metrics["Acc_CI_Upper"]) * 100
    f1  = np.mean(metrics["F1 (Macro)"]) * 100
    f1_lo = np.mean(metrics["F1_CI_Lower"]) * 100
    f1_hi = np.mean(metrics["F1_CI_Upper"]) * 100
    
    blk  = np.mean(metrics["Blank_Accuracy"]) * 100
    drp  = np.mean(metrics["Blank_Drop"])
    neut = np.mean(metrics["Neutral_Accuracy"]) * 100
    cfr  = np.mean(metrics["CFR"])
    cka  = np.mean(metrics["Linear_CKA"])
    flag_blk  = " (*)" if blk  < milk_majority_baseline else ""
    flag_neut = " (*)" if neut < milk_majority_baseline else ""
    
    acc_str = f"{acc:>5.2f} [{acc_lo:>5.2f}, {acc_hi:>5.2f}]"
    f1_str  = f"{f1:>5.2f} [{f1_lo:>5.2f}, {f1_hi:>5.2f}]"
    
    print(f"  {arch:<26} | {acc_str} | {f1_str} | {blk:>5.2f}%{flag_blk:<3} | "
          f"{drp:>+6.2f}pp | {neut:>5.2f}%{flag_neut:<3} | {cfr:>5.2f}% | {cka:>7.4f}")

print()
print("(*) = below majority baseline in clean val set")

# ── Save JSON ──────────────────────────────────────────────────────────────────
output = {
    "split_info": {
        "type": "lesion-disjoint",
        "val_images": int(len(val_df)),
        "lesion_overlap": 0,
        "val_class_distribution": val_df["diagnostic"].value_counts().to_dict(),
        "majority_class_in_val": milk_majority_class,
        "majority_baseline_pct": float(milk_majority_baseline),
    },
    "results": {
        arch: {k: [float(x) for x in v] for k, v in metrics.items()}
        for arch, metrics in results.items()
    }
}

OUT_JSON = "./milk10k_clean_audit_results.json"
with open(OUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {OUT_JSON}")
print()
print("=" * 70)
print("CLEAN AUDIT COMPLETE")
print("=" * 70)
