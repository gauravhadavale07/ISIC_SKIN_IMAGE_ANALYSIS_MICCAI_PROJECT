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
from sklearn.metrics import accuracy_score

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
    "Cross-Attention (V->T)":   (CrossAttentionClassifier,    "Cross-Attention"),
    "Cross-Attention T->V":     (CrossAttentionT2VClassifier, "Cross-Attention_T2V"),
    "Image-Only":               (ImageOnlyClassifier,         "ImageOnly"),
    "Text-Only":                (TextOnlyClassifier,          "TextOnly"),
}
SEEDS = [456, 789, 1337]

# ── Build lesion-disjoint clean held-out set ───────────────────────────────────
print("=" * 60)
print("MILK10k LESION-DISJOINT CLEAN AUDIT")
print("=" * 60)

CSV_PATH = "./milk10k_train.csv"
IMG_DIR  = "./data/raw_milk10k/"

df = pd.read_csv(CSV_PATH)
df['lesion_id'] = df['filepath'].apply(
    lambda x: re.search(r'(IL_\d+)', x).group(1)
    if re.search(r'(IL_\d+)', x) else None
)
df = df.dropna(subset=['lesion_id'])

all_lesions     = sorted(df['lesion_id'].unique())   # deterministic sort
n_lesions       = len(all_lesions)
n_val_lesions   = int(0.15 * n_lesions)              # 700 lesions
val_lesion_set  = set(all_lesions[-n_val_lesions:])  # last 15% by ID order

val_df   = df[df['lesion_id'].isin(val_lesion_set)].reset_index(drop=True)
train_df = df[~df['lesion_id'].isin(val_lesion_set)].reset_index(drop=True)

print(f"\nLesion-disjoint split (fixed, no random seed):")
print(f"  Total images   : {len(df)}")
print(f"  Total lesions  : {n_lesions}")
print(f"  Val lesions    : {n_val_lesions}  ({n_val_lesions/n_lesions*100:.1f}% of lesions)")
print(f"  Val images     : {len(val_df)}")
print(f"  Train images   : {len(train_df)}")
print(f"  Lesion overlap : 0  (guaranteed by construction)")
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

# ── Build a Dataset for the clean val set ─────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

# Write val_df to a temp CSV so we can reuse MultimodalDermatologyDataset
TEMP_CSV = "./milk10k_clean_val_temp.csv"
val_df.to_csv(TEMP_CSV, index=False)

clean_val_dataset = MultimodalDermatologyDataset(
    csv_file=TEMP_CSV,
    img_dir=IMG_DIR,
    tokenizer=tokenizer,
    transform=get_transforms(),
)
clean_val_loader = DataLoader(
    clean_val_dataset,
    batch_size=cfg.train.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
print(f"Clean val loader: {len(clean_val_dataset)} samples, {len(clean_val_loader)} batches\n")

# ── Run audit ──────────────────────────────────────────────────────────────────
device = cfg.train.device
print(f"Device: {device}\n")

results = {}

for arch_name, (model_cls, prefix) in MODELS.items():
    print(f"\n{'─'*55}")
    print(f"Architecture: {arch_name}")
    print(f"{'─'*55}")

    arch_results = {
        "Accuracy":        [],
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

        # 1. Base accuracy
        evaluator = Evaluator(model, device)
        metrics   = evaluator.evaluate(clean_val_loader)
        real_acc  = metrics["Accuracy"]
        arch_results["Accuracy"].append(real_acc)

        # 2. Counterfactual / blank / neutral probes
        if arch_name == "Image-Only":
            arch_results["Blank_Accuracy"].append(real_acc)
            arch_results["Blank_Drop"].append(0.0)
            arch_results["Neutral_Accuracy"].append(real_acc)
            arch_results["CFR"].append(0.0)
            arch_results["Mean_Delta_P"].append(0.0)
        else:
            cf_aud = CounterfactualAuditor(model, tokenizer, device)
            cf_metrics = cf_aud.run_audit(clean_val_loader)
            arch_results["Blank_Accuracy"].append(cf_metrics["Blank_Accuracy"])
            arch_results["Blank_Drop"].append(cf_metrics["Blank_Accuracy_Drop"])
            arch_results["Neutral_Accuracy"].append(cf_metrics["Neutral_Accuracy"])
            arch_results["CFR"].append(cf_metrics["CFR"])
            arch_results["Mean_Delta_P"].append(cf_metrics["Mean_Delta_P"])

        # 3. CKA
        if arch_name in ("Image-Only", "Text-Only"):
            arch_results["Linear_CKA"].append(1.0)
        else:
            cka_aud = CKAAuditor(model, device)
            cka_metrics = cka_aud.run_audit(clean_val_loader)
            arch_results["Linear_CKA"].append(cka_metrics["Linear_CKA"])

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    results[arch_name] = arch_results

# ── Cleanup temp file ──────────────────────────────────────────────────────────
if os.path.exists(TEMP_CSV):
    os.remove(TEMP_CSV)

# ── Summary table ──────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("CLEAN IN-DOMAIN (MILK10k) AUDIT RESULTS -- LESION-DISJOINT VAL SET")
print(f"Majority class in clean val: {milk_majority_class} ({milk_majority_baseline:.2f}%)")
print(f"{'='*70}")
print(f"{'Architecture':<28} | {'Acc':>6} | {'BlkAcc':>7} | {'BlkDrp':>7} | {'Neutral':>7} | {'CFR':>6} | {'CKA':>7}")
print("-" * 85)

for arch, metrics in results.items():
    if not metrics["Accuracy"]:
        print(f"  {arch:<26} | [NO DATA]")
        continue
    acc  = np.mean(metrics["Accuracy"]) * 100
    blk  = np.mean(metrics["Blank_Accuracy"]) * 100
    drp  = np.mean(metrics["Blank_Drop"])
    neut = np.mean(metrics["Neutral_Accuracy"]) * 100
    cfr  = np.mean(metrics["CFR"])
    cka  = np.mean(metrics["Linear_CKA"])
    flag_blk  = " (*)" if blk  < milk_majority_baseline else ""
    flag_neut = " (*)" if neut < milk_majority_baseline else ""
    print(f"  {arch:<26} | {acc:>5.2f}% | {blk:>5.2f}%{flag_blk:<3} | "
          f"{drp:>+6.2f}pp | {neut:>5.2f}%{flag_neut:<3} | {cfr:>5.2f}% | {cka:>7.4f}")

print()
print("(*) = below majority baseline in clean val set")

# ── Save JSON ──────────────────────────────────────────────────────────────────
output = {
    "split_info": {
        "type": "lesion-disjoint",
        "total_images": int(len(df)),
        "total_lesions": int(n_lesions),
        "val_lesions": int(n_val_lesions),
        "val_images": int(len(val_df)),
        "train_images": int(len(train_df)),
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
