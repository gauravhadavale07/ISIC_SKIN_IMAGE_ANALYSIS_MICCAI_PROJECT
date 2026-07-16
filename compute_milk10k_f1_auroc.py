import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dataset import MultimodalDermatologyDataset, get_transforms
from config import cfg

from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionClassifier, CrossAttentionT2VClassifier
from models.image_only import ImageOnlyClassifier
from models.text_only import TextOnlyClassifier
from evaluate import Evaluator

MODELS = {
    "Image-Only":               (ImageOnlyClassifier,         "ImageOnly"),
    "Text-Only":                (TextOnlyClassifier,          "TextOnly"),
    "Late Fusion":              (LateFusionClassifier,        "Late_Fusion"),
    "GMU Baseline":             (GMUClassifier,               "GMU_Baseline"),
    "Cross-Attention (V->T)":   (CrossAttentionClassifier,    "Cross-Attention"),
    "Cross-Attention T->V":     (CrossAttentionT2VClassifier, "Cross-Attention_T2V"),
}

SEEDS = cfg.seeds

# Prepare dataset
tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
TEMP_CSV = "./milk10k_clean_val_temp.csv"
IMG_DIR = "./data/raw_milk10k/"

dataset = MultimodalDermatologyDataset(csv_file=TEMP_CSV, img_dir=IMG_DIR, tokenizer=tokenizer, transform=get_transforms())
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

device = cfg.train.device

print("Calculating AUROC and F1-Macro on clean MILK10k split...")
for arch_name, (model_cls, prefix) in MODELS.items():
    accs, f1s, aurocs = [], [], []
    for seed in SEEDS:
        ckpt_path = f"./checkpoints/{prefix}_seed_{seed}/best_model.pth"
        if not os.path.exists(ckpt_path):
            continue
        model = model_cls().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(loader)
        y_true = metrics["y_true"]
        y_pred = metrics["y_pred"]
        y_prob = metrics["y_prob"]
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        
        # Determine present classes for AUROC
        unique_classes = np.unique(y_true)
        try:
            if len(unique_classes) == 6:
                auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            else:
                # Need to handle missing classes in ROC AUC
                y_prob_filtered = y_prob[:, unique_classes]
                y_prob_filtered = y_prob_filtered / y_prob_filtered.sum(axis=1, keepdims=True)
                auroc = roc_auc_score(y_true, y_prob_filtered, multi_class='ovr', average='macro', labels=unique_classes)
        except Exception as e:
            auroc = float('nan')
            
        accs.append(acc)
        f1s.append(f1)
        aurocs.append(auroc)
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    if accs:
        print(f"{arch_name}: Acc={np.mean(accs):.4f}, F1={np.mean(f1s):.4f}, AUROC={np.mean(aurocs):.4f}")
    else:
        print(f"{arch_name}: No checkpoints found.")
