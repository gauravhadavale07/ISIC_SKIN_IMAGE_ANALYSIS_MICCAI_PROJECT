"""
run_lexical_control_audit.py — Lexical Control Audit on PAD-UFES-20.

Validates whether multimodal models actually use the semantic information
(Age, Sex, Location) in the clinical text, or if they just rely on the lexical
presence of text tokens to satisfy the attention mechanism.

Method:
- Shuffles the `clinical_history` column randomly across the entire dataset.
- The lexical distribution is perfectly preserved (all valid strings).
- The semantic link to the image/label is destroyed.

Outputs:
- Lexical Control Accuracy for each architecture.
- If (Real_Acc - Lexical_Control_Acc) is large, the model reads semantics.
- If Lexical_Control_Acc == Real_Acc, the model is blind to text meaning.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from evaluate import Evaluator
from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionClassifier, CrossAttentionT2VClassifier

MODELS = {
    "Late Fusion":              (LateFusionClassifier,        "Late_Fusion"),
    "GMU Baseline":             (GMUClassifier,               "GMU_Baseline"),
    "Cross-Attention (V->T)":   (CrossAttentionClassifier,    "Cross-Attention"),
    "Cross-Attention T->V":     (CrossAttentionT2VClassifier, "Cross-Attention_T2V"),
}
SEEDS = [456, 789, 1337]

print("=" * 60)
print("LEXICAL CONTROL AUDIT (PAD-UFES-20)")
print("=" * 60)

# Load real dataset progress for baseline comparison
try:
    with open("./results/experiment_progress.json") as f:
        progress = json.load(f)
except FileNotFoundError:
    print("Error: ./results/experiment_progress.json not found.")
    sys.exit(1)

# 1. Prepare Lexically Controlled Dataset
df = pd.read_csv(cfg.paths.pad_ufes_csv)

# Defensive scrubbing matching dataset.py
df = df[df['diagnostic'].astype(str).str.upper() != 'NAN'].reset_index(drop=True)

# Randomly shuffle clinical history
np.random.seed(42) # Deterministic shuffle
df['clinical_history'] = np.random.permutation(df['clinical_history'].values)

temp_csv = "./pad_ufes_20_lexical_control.csv"
df.to_csv(temp_csv, index=False)

tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
control_dataset = MultimodalDermatologyDataset(
    csv_file=temp_csv,
    img_dir=cfg.paths.pad_ufes_img_dir,
    tokenizer=tokenizer,
    transform=get_transforms()
)
control_loader = DataLoader(
    control_dataset,
    batch_size=cfg.train.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = cfg.train.device
results = {}

for arch_name, (model_cls, prefix) in MODELS.items():
    print(f"\nEvaluating: {arch_name}")
    arch_accs = []
    
    for seed in SEEDS:
        ckpt_path = f"{cfg.paths.checkpoint_dir}/{prefix}_seed_{seed}/best_model.pth"
        if not os.path.exists(ckpt_path): continue
        
        model = model_cls().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        model.eval()
        
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(control_loader)
        arch_accs.append(metrics["Accuracy"] * 100) # Convert to pp
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    results[arch_name] = arch_accs

if os.path.exists(temp_csv): os.remove(temp_csv)

print("\n" + "=" * 60)
print("LEXICAL CONTROL RESULTS (vs Real vs Blank)")
print("=" * 60)
print(f"{'Architecture':<28} | {'Real':>7} | {'Blank':>7} | {'Shuffle':>7} | {'Lexical_Drop':>12}")
print("-" * 75)

for arch, accs in results.items():
    if not accs: continue
    shuf_acc = np.mean(accs)
    
    # Get baseline numbers from experiment_progress.json
    base = progress["results"].get(arch, {})
    real_acc = np.mean(base.get("Real_Accuracy", [0]))
    blnk_acc = np.mean(base.get("Blank_Accuracy", [0]))
    
    lexical_drop = real_acc - shuf_acc
    
    print(f"{arch:<28} | {real_acc:>6.2f}% | {blnk_acc:>6.2f}% | {shuf_acc:>6.2f}% | {lexical_drop:>+9.2f}pp")

print("=" * 60)

with open("./results/lexical_control_results.json", "w") as f:
    json.dump(results, f, indent=2)
