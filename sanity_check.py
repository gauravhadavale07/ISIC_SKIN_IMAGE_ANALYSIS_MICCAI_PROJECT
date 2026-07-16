import sys, os, json, re
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from evaluate import Evaluator
from models.late_fusion import LateFusionClassifier
import pandas as pd

def main():
    print("Running sanity check for Late Fusion seed 456...")
    device = cfg.train.device
    
    with open("./milk10k_clean_audit_results.json", "r") as f:
        existing_data = json.load(f)["results"]
        
    old_acc = existing_data["Late Fusion"]["Accuracy"][0]
    
    # We don't have F1 in the old JSON, but we can compute what the F1 WOULD be
    # Let's just compare accuracy first.
    print(f"Old accuracy from JSON: {old_acc:.4f}")
    
    # Load data exactly like run_milk10k_clean_audit.py
    df = pd.read_csv("./milk10k_train.csv")
    df['lesion_id'] = df['filepath'].apply(
        lambda x: re.search(r'(IL_\d+)', x).group(1)
        if re.search(r'(IL_\d+)', x) else None
    )
    df = df.dropna(subset=['lesion_id'])
    
    val_lesion_set = df['lesion_id'].unique()
    val_lesion_set = np.sort(val_lesion_set)
    n_lesions = len(val_lesion_set)
    n_val_lesions = int(n_lesions * 0.15)
    val_lesion_set = val_lesion_set[-n_val_lesions:]
    
    val_df = df[df['lesion_id'].isin(val_lesion_set)].reset_index(drop=True)
    val_df.to_csv("./milk10k_clean_val_temp.csv", index=False)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    clean_val_dataset = MultimodalDermatologyDataset(
        csv_file="./milk10k_clean_val_temp.csv",
        img_dir="./data/raw_milk10k/",
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
    
    model = LateFusionClassifier().to(device)
    ckpt = torch.load("./checkpoints/Late_Fusion_seed_456/best_model.pth", map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()
    
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(clean_val_loader)
    
    new_acc = metrics["Accuracy"]
    y_true = metrics["y_true"]
    y_pred = metrics["y_pred"]
    new_f1 = f1_score(y_true, y_pred, average='macro')
    
    delta_acc = new_acc - old_acc
    print(f"New accuracy computed: {new_acc:.4f}")
    print(f"Delta (New - Old): {delta_acc:.6f}")
    
    if abs(delta_acc) < 1e-4:
        print("SUCCESS: Accuracy matches exactly.")
    else:
        print("WARNING: Accuracy mismatch detected!")

if __name__ == "__main__":
    main()
