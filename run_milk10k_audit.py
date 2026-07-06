import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from dataset import MultimodalDermatologyDataset, get_transforms
from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionClassifier, CrossAttentionT2VClassifier
from models.image_only import ImageOnlyClassifier
from models.text_only import TextOnlyClassifier
from config import cfg

from evaluate import Evaluator
from counterfactual import CounterfactualAuditor
from cka import CKAAuditor

# Architecture mapping
MODELS = {
    "Late Fusion": (LateFusionClassifier, "Late_Fusion"),
    "GMU Baseline": (GMUClassifier, "GMU_Baseline"),
    "Cross-Attention (V->T)": (CrossAttentionClassifier, "Cross-Attention"),
    "Cross-Attention T->V": (CrossAttentionT2VClassifier, "Cross-Attention_T2V"),
    "Image-Only": (ImageOnlyClassifier, "ImageOnly"),
    "Text-Only": (TextOnlyClassifier, "TextOnly")
}

SEEDS = [456, 789, 1337]

def build_milk10k_val_loader(seed):
    dataset = MultimodalDermatologyDataset(
        csv_file="./milk10k_train.csv",
        img_dir="./data/raw_milk10k/",
        transform=get_transforms(),
        is_ood=False
    )
    
    n = len(dataset)
    train_size = int(0.85 * n)
    val_size = n - train_size
    
    gen = torch.Generator().manual_seed(seed)
    _, val_sub = random_split(dataset, [train_size, val_size], generator=gen)
    
    loader = DataLoader(
        val_sub,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return loader

print("="*60)
print("TIER 1.1: MILK10k IN-DOMAIN AUDIT")
print("="*60)

results = {}

print("Loading tokenizer...")
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

for arch_name, (model_cls, prefix) in MODELS.items():
    print(f"\nEvaluating: {arch_name}")
    print("-" * 40)
    
    arch_results = {
        "Accuracy": [],
        "Blank_Accuracy": [],
        "Blank_Drop": [],
        "Neutral_Accuracy": [],
        "CFR": [],
        "Mean_Delta_P": [],
        "Linear_CKA": []
    }
    
    for seed in SEEDS:
        print(f"  Seed: {seed}")
        ckpt_path = f"./checkpoints/{prefix}_seed_{seed}/best_model.pth"
        if not os.path.exists(ckpt_path):
            print(f"    Missing checkpoint!")
            continue
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_cls().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        model.eval()
        
        val_loader = build_milk10k_val_loader(seed)
        
        # 1. Base Evaluator
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(val_loader)
        real_acc = metrics['Accuracy']
        arch_results['Accuracy'].append(real_acc)
        
        # 2. Counterfactual Auditor
        # Skip CF/Blank text for ImageOnly
        if arch_name == "Image-Only":
            arch_results['Blank_Accuracy'].append(real_acc)
            arch_results['Blank_Drop'].append(0.0)
            arch_results['Neutral_Accuracy'].append(real_acc)
            arch_results['CFR'].append(0.0)
            arch_results['Mean_Delta_P'].append(0.0)
        else:
            cf_auditor = CounterfactualAuditor(model, tokenizer, device)
            cf_metrics = cf_auditor.run_audit(val_loader)
            
            blank_acc = cf_metrics['Blank_Accuracy']
            arch_results['Blank_Accuracy'].append(blank_acc)
            arch_results['Blank_Drop'].append(cf_metrics['Blank_Accuracy_Drop'])
            arch_results['Neutral_Accuracy'].append(cf_metrics['Neutral_Accuracy'])
            
            # Using diagnostic_swap for CFR
            cfr = cf_metrics['CFR']
            arch_results['CFR'].append(cfr)
            arch_results['Mean_Delta_P'].append(cf_metrics['Mean_Delta_P'])
            
        # 3. CKA Auditor
        if arch_name in ["Image-Only", "Text-Only"]:
             arch_results['Linear_CKA'].append(1.0)
        else:
             cka_auditor = CKAAuditor(model, device)
             cka_metrics = cka_auditor.run_audit(val_loader)
             arch_results['Linear_CKA'].append(cka_metrics['Linear_CKA'])
             
    results[arch_name] = arch_results

print("\n" + "="*60)
print("IN-DOMAIN (MILK10k) AUDIT RESULTS SUMMARY (Mean across 3 seeds)")
print("="*60)
print(f"{'Architecture':<25} | {'Acc':<6} | {'BlkAcc':<6} | {'BlkDrp':<6} | {'CFR':<6} | {'CKA':<6}")
print("-" * 65)
for arch, metrics in results.items():
    if not metrics['Accuracy']: continue
    acc = np.mean(metrics['Accuracy']) * 100
    blk = np.mean(metrics['Blank_Accuracy']) * 100
    drp = np.mean(metrics['Blank_Drop'])
    cfr = np.mean(metrics['CFR'])
    cka = np.mean(metrics['Linear_CKA'])
    print(f"{arch:<25} | {acc:5.2f}% | {blk:5.2f}% | {drp:5.2f}% | {cfr:5.2f}% | {cka:5.4f}")
