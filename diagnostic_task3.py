import os
import sys
import re
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from transforms import get_eval_transforms
from evaluate import Evaluator

from models.late_fusion import LateFusionClassifier
from models.cross_attention import CrossAttentionT2VClassifier

SEED = 42
MODEL_SEED = 1337

def main():
    print("=" * 70)
    print("DIAGNOSTIC: EVALUATING EXISTING CHECKPOINTS ON TASK 3 SPLIT")
    print("=" * 70)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    # EXACT SAME SPLIT LOGIC AS TASK 3
    csv_path = cfg.paths.milk10k_csv
    img_dir = "./data/raw_milk10k/"
    
    df = pd.read_csv(csv_path)
    df['lesion_id'] = df['filepath'].apply(
        lambda x: re.search(r'(IL_\d+)', x).group(1)
        if re.search(r'(IL_\d+)', x) else None
    )
    df = df.dropna(subset=['lesion_id'])
    
    all_lesions = sorted(df['lesion_id'].unique())
    n_lesions = len(all_lesions)
    n_val_lesions = int(0.15 * n_lesions)
    val_lesion_set = set(all_lesions[-n_val_lesions:])
    
    val_df = df[df['lesion_id'].isin(val_lesion_set)].reset_index(drop=True)
    
    val_csv = "./temp_diagnostic_val.csv"
    val_df.to_csv(val_csv, index=False)
    
    val_ds = MultimodalDermatologyDataset(csv_file=val_csv, img_dir=img_dir, tokenizer=tokenizer, transform=get_eval_transforms())
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)
    
    from models.image_only import ImageOnlyClassifier
    from models.text_only import TextOnlyClassifier
    
    architectures = {
        "Image-Only": (ImageOnlyClassifier, "ImageOnly"),
        "Text-Only": (TextOnlyClassifier, "TextOnly"),
        "Late Fusion": (LateFusionClassifier, "Late_Fusion"),
        "Cross-Attention (T->V)": (CrossAttentionT2VClassifier, "Cross-Attention_T2V")
    }
    
    for arch_name, (ModelClass, prefix) in architectures.items():
        print(f"\nEvaluating {arch_name} on the Task 3 Lesion-Disjoint Split...")
        # Evaluate seed 1337 for simplicity, or 456
        ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"{prefix}_seed_{456}", "best_model.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"{prefix}_seed_{1337}", "best_model.pth")
            
        if not os.path.exists(ckpt_path):
            print(f"  [ERROR] Checkpoint not found: {ckpt_path}")
            continue
            
        model = ModelClass().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        model.eval()
        
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(val_loader)
        
        print(f"  {arch_name} Accuracy: {metrics['Accuracy']*100:.2f}% | F1-Macro: {metrics['F1 (Macro)']*100:.2f}%")
        
    if os.path.exists(val_csv): os.remove(val_csv)

if __name__ == "__main__":
    main()
