import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import re
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.image_only import ImageOnlyClassifier
from models.late_fusion import LateFusionClassifier
from models.cross_attention import CrossAttentionT2VClassifier

def extract_features_and_labels(model, dataloader, device, feature_extractor_fn):
    model.eval()
    features = []
    locations = []
    
    # Regex to extract location from text: "Patient, age X, presents with a lesion on the {location}."
    loc_regex = re.compile(r"lesion on the (.*?)\.")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            # Extract clinical history text to get the target location
            for text in batch["clinical_history"]:
                match = loc_regex.search(text)
                if match:
                    locations.append(match.group(1).strip().lower())
                else:
                    locations.append("unknown")
            
            feats = feature_extractor_fn(model, imgs, input_ids, attn_mask)
            features.append(feats.cpu().numpy())
            
    return np.vstack(features), np.array(locations)

def get_image_only_feats(model, imgs, input_ids, attn_mask):
    return model.vision_encoder(imgs)

def get_late_fusion_feats(model, imgs, input_ids, attn_mask):
    vision_feat = model.vision_encoder(imgs)
    text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    text_feat = text_outputs.last_hidden_state[:, 0, :]
    return torch.cat([vision_feat, text_feat], dim=1)

def get_t2v_feats(model, imgs, input_ids, attn_mask):
    vision_seq = model.vision_encoder.forward_features(imgs)
    text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    text_seq = text_outputs.last_hidden_state
    
    attn_output, _ = model.cross_attn(
        query=text_seq,
        key=vision_seq,
        value=vision_seq,
        need_weights=False
    )
    return attn_output.mean(dim=1)

def main():
    print("=" * 70)
    print("TASK 10: LINEAR PROBING FOR CONCEPT ALIGNMENT")
    print("=" * 70)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    # We will use the PAD-UFES-20 test set
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )
    
    # We need to return clinical_history from the dataset to parse the location
    # Let's monkeypatch __getitem__ to also return the raw text
    original_getitem = dataset.__getitem__
    def new_getitem(self, idx):
        item = original_getitem(idx)
        item["clinical_history"] = self.df.iloc[idx]['clinical_history']
        return item
    MultimodalDermatologyDataset.__getitem__ = new_getitem
    
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)
    
    models_to_test = {
        "Image-Only Baseline": (ImageOnlyClassifier(), get_image_only_feats, 1337),
        "Late Fusion": (LateFusionClassifier(), get_late_fusion_feats, 1337),
        "Cross-Attention T->V": (CrossAttentionT2VClassifier(), get_t2v_feats, 1337)
    }
    
    results = []
    
    for name, (model, feat_fn, seed) in models_to_test.items():
        print(f"\\nEvaluating {name}...")
        
        if "T->V" in name:
            ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_{seed}", "best_model.pth")
        elif "Image-Only" in name:
            ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"ImageOnly_seed_2024", "best_model.pth")
        else:
            ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"{name.replace(' ', '_')}_seed_{seed}", "best_model.pth")
            
        if not os.path.exists(ckpt_path):
            print(f"Error: Checkpoint not found at {ckpt_path}")
            continue
            
        model.to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        
        X, y = extract_features_and_labels(model, loader, device, feat_fn)
        
        # Filter out unknown locations
        valid_idx = y != "unknown"
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        
        # Train Linear Probe (Logistic Regression)
        # We will do a simple 5-fold cross validation or just train/test on a random split
        # Since we just want to know if the representation linearly separates concepts, we can use 5-fold CV
        from sklearn.model_selection import cross_val_score
        
        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        scores = cross_val_score(clf, X_valid, y_valid, cv=5, scoring='accuracy')
        
        avg_acc = np.mean(scores)
        std_acc = np.std(scores)
        
        print(f"  Location Linear Probe Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        
        results.append({
            "Architecture": name,
            "Probe_Accuracy": avg_acc,
            "Probe_Std": std_acc
        })
        
    df_results = pd.DataFrame(results)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    df_results.to_csv(os.path.join(cfg.paths.results_dir, "task10_linear_probing.csv"), index=False)
    print(f"\\nResults saved to results/task10_linear_probing.csv")

if __name__ == "__main__":
    main()
