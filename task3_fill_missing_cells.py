import os
import sys
import re
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset
from transforms import get_train_transforms, get_eval_transforms
from trainer import MultimodalTrainer, set_seed
from evaluate import Evaluator

from models.late_fusion import LateFusionClassifier
from models.cross_attention import CrossAttentionT2VClassifier

SEED = 42

def main():
    print("=" * 70)
    print("TASK 3: FILL MISSING ARCHITECTURE x DATASET CELLS (MILK10k CLEAN)")
    print("=" * 70)

    device = cfg.train.device
    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    # Build lesion-disjoint split
    csv_path = cfg.paths.milk10k_csv
    img_dir = "" # Paths in csv are absolute in fast_build.py
    
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
    train_df = df[~df['lesion_id'].isin(val_lesion_set)].reset_index(drop=True)
    
    train_csv = "./temp_milk10k_clean_train.csv"
    val_csv = "./temp_milk10k_clean_val.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"Lesion-disjoint split created.")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    train_ds = MultimodalDermatologyDataset(csv_file=train_csv, img_dir=img_dir, tokenizer=tokenizer, transform=get_train_transforms())
    val_ds = MultimodalDermatologyDataset(csv_file=val_csv, img_dir=img_dir, tokenizer=tokenizer, transform=get_eval_transforms())
    
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)
    
    architectures = {
        "Late Fusion": LateFusionClassifier,
        "Cross-Attention (T->V)": CrossAttentionT2VClassifier
    }
    
    results = []
    
    for arch_name, ModelClass in architectures.items():
        print(f"\n{'-'*60}")
        print(f"Training and Evaluating: {arch_name}")
        print(f"{'-'*60}")
        
        run_name = f"task3_clean_{arch_name.replace(' ', '_').replace('->', '2')}"
        
        model = ModelClass()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=cfg.train.learning_rate, 
            weight_decay=cfg.train.weight_decay
        )
        
        total_steps = len(train_loader) * cfg.train.epochs
        warmup_steps = int(cfg.train.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            run_name=run_name
        )
        
        trainer.fit()
        
        # Load best model
        best_model_path = os.path.join(cfg.paths.checkpoint_dir, run_name, "best_model.pth")
        ckpt = torch.load(best_model_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        
        # Evaluate
        evaluator = Evaluator(model, device)
        metrics = evaluator.evaluate(val_loader)
        
        results.append({
            "Architecture": arch_name,
            "Accuracy": metrics["Accuracy"] * 100,
            "AUROC": metrics["AUROC"],
            "Macro F1": metrics["F1 (Macro)"]
        })
        
        del model
        torch.cuda.empty_cache()
    
    # Cleanup temp CSVs
    if os.path.exists(train_csv): os.remove(train_csv)
    if os.path.exists(val_csv): os.remove(val_csv)
    
    print("\n" + "="*85)
    print("TASK 3 RESULTS: MISSING CELLS EVALUATED ON LESION-DISJOINT MILK10k")
    print("="*85)
    
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.paths.results_dir, "task3_missing_cells.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")

if __name__ == "__main__":
    main()
