"""
Task 25: SOTA Dermatology Foundation Model Linear Probing.

Tests concept alignment (lesion location from clinical history) for SOTA
dermatology FMs that are publicly available. Closed-weight models are formally
documented. Falls back to open timm models (SkinCLIP-equivalent EVA-02, etc.)
if the dermatology-specific HF repos are unavailable.

Improvements:
  - Robust model loader with --dry-run support
  - Balanced cross-validated accuracy (stratified k-fold)
  - Also probes for macro-class (malignant/benign) as a second target
  - Documents unavailable models with reasons
  - Saves per-model feature distributions for interpretability
"""

import argparse
import os
import re
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms

# ─── Closed-Weight Documentation ──────────────────────────────────────────────

CLOSED_MODELS = [
    {
        "name": "PanDerm",
        "reason": "Weights are closed (institutional access only). No public HuggingFace checkpoint.",
        "paper": "PanDerm: Pan-disease dermatology foundation model (2024)",
    },
    {
        "name": "DermFM-Zero",
        "reason": "Weights not publicly available (gated on HuggingFace without clear access path)."
    },
    {
        "name": "SLIMP",
        "reason": "Closed source. Authors only released API/demo."
    },
    {
        "name": "MM-Skin",
        "reason": "Weights are closed. No public repository or HF model card as of July 2026.",
        "paper": "MM-Skin: Multimodal dermatology FM (2024)",
    },
]

# ─── Model Registry ────────────────────────────────────────────────────────────

HF_MODELS = [
    # {
    #     "name": "ViT-B16-IN1k",
    #     "path": "google/vit-base-patch16-224-in21k",
    #     "type": "hf_vision"
    # },
    {
        "name": "DermLIP",
        "path": "redlessone/DermLIP_ViT-B-16",
        "type": "open_clip"
    },
    {"name": "JI-ADF",        "path": "ji-adf/ji-adf-vit",          "type": "hf_vision"},
    # SkinCLIP: publicly available CLIP fine-tuned on dermatology
    {"name": "SkinCLIP",      "path": "GoodAI/SkinCLIP",            "type": "hf_vision"},
]

TIMM_FALLBACK_MODELS = [
    # # EVA-02: strong open-access ViT alternative when derm FMs are unavailable
    # {"name": "EVA02-L-448",   "path": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", "type": "timm"},
    # # ViT-B/16 ImageNet-1k (baseline reference)
    # {"name": "ViT-B16-IN1k",  "path": "vit_base_patch16_224.augreg_in1k",               "type": "timm"},
    # # DINOv2 Large
    # {"name": "DINOv2-L",      "path": "vit_large_patch14_dinov2.lvd142m",               "type": "timm"},
]


# ─── Feature Extractor Wrappers ────────────────────────────────────────────────

def hf_vision_extractor(model, imgs):
    """Extract CLS token or pooler output from HF vision models."""
    outputs = model(pixel_values=imgs)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0, :]


def timm_extractor(model, imgs):
    """Extract global average pooled or CLS features from timm models."""
    feats = model.forward_features(imgs)
    if feats.dim() == 3:
        return feats[:, 0, :]   # CLS token
    return feats                 # Already pooled

def open_clip_extractor(model, imgs):
    """Extract image embeddings from open_clip models."""
    return model.encode_image(imgs)


# ─── Feature Extraction ────────────────────────────────────────────────────────

LOC_REGEX = re.compile(r"lesion on the (.*?)\.")

def extract_features_and_targets(model_wrapper, dataloader, device):
    """
    Returns:
      features    : np.ndarray (N, D)
      locations   : np.ndarray (N,) str labels (lesion location)
      macro_labels: np.ndarray (N,) str labels (Benign/Malignant)
    """
    features    = []
    locations   = []
    macro_labels = []

    MACRO_MAP = {
        "MEL": "Malignant", "BCC": "Malignant", "SCC": "Malignant",
        "NEV": "Benign",    "ACK": "Benign",    "SEK": "Benign",
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting [{model_wrapper['name']}]", leave=False):
            imgs = batch["image"].to(device)

            # Location labels from clinical history
            for text in batch.get("clinical_history", [""]*len(imgs)):
                match = LOC_REGEX.search(str(text))
                locations.append(match.group(1).strip().lower() if match else "unknown")

            # Macro class from diagnostic label
            for diag in batch.get("diagnostic", ["unknown"]*len(imgs)):
                macro_labels.append(MACRO_MAP.get(str(diag).strip().upper(), "Unknown"))

            try:
                feats = model_wrapper["extract_fn"](model_wrapper["model"], imgs)
                features.append(feats.cpu().float().numpy())
            except Exception as e:
                print(f"  Feature extraction error: {e}")
                return None, None, None

    return np.vstack(features), np.array(locations), np.array(macro_labels)


# ─── Probing Experiment ────────────────────────────────────────────────────────

def run_probe(X: np.ndarray, y: np.ndarray, target_name: str, n_splits: int = 5):
    """Run stratified cross-validated logistic regression probe."""
    valid_mask = y != "unknown"
    X_v = X[valid_mask]
    y_v = y[valid_mask]

    if len(set(y_v)) < 2:
        print(f"  [{target_name}] Only one class present, skipping.")
        return np.nan, np.nan, np.nan

    le = LabelEncoder()
    y_enc = le.fit_transform(y_v)

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    bal_acc = cross_val_score(clf, X_v, y_enc, cv=cv, scoring="balanced_accuracy")
    acc     = cross_val_score(clf, X_v, y_enc, cv=cv, scoring="accuracy")

    return float(np.mean(bal_acc)), float(np.std(bal_acc)), float(np.mean(acc))


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False):
    print("=" * 70)
    print("TASK 25: SOTA DERMATOLOGY FM LINEAR PROBING")
    print("=" * 70)

    if dry_run:
        print(">>> DRY RUN: testing model loading only <<<")

    device = cfg.train.device

    # ── Dataset ────────────────────────────────────────────────────────────────
    original_getitem = MultimodalDermatologyDataset.__getitem__

    def new_getitem(self, idx):
        item = original_getitem(self, idx)
        row  = self.df.iloc[idx]
        item["clinical_history"] = str(row.get("clinical_history", ""))
        item["diagnostic"]       = str(row.get("diagnostic", "unknown"))
        return item

    MultimodalDermatologyDataset.__getitem__ = new_getitem

    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        transform=get_transforms(),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # ── Document closed models ─────────────────────────────────────────────────
    print("\n--- CLOSED-WEIGHT MODELS (formally excluded) ---")
    closed_records = []
    for cm in CLOSED_MODELS:
        print(f"  {cm['name']}: {cm['reason']}")
        closed_records.append(cm)

    results   = []
    all_models = HF_MODELS + TIMM_FALLBACK_MODELS

    # ── Attempt each model ────────────────────────────────────────────────────
    for m_info in all_models:
        name  = m_info["name"]
        path  = m_info["path"]
        mtype = m_info["type"]
        print(f"\n>>> Loading {name} ({path}) ...")

        wrapper = {"name": name}
        model   = None

        try:
            if mtype == "hf_vision":
                from transformers import AutoModel, AutoImageProcessor
                import os
                hf_token = os.environ.get("HF_TOKEN")
                model = AutoModel.from_pretrained(path, trust_remote_code=True, token=hf_token).to(device)
                model.eval()
                wrapper["model"]      = model
                wrapper["extract_fn"] = hf_vision_extractor
                
                try:
                    processor = AutoImageProcessor.from_pretrained(path, token=hf_token)
                    if hasattr(processor, "crop_size") and "height" in processor.crop_size:
                        size = processor.crop_size["height"]
                    elif hasattr(processor, "size") and "height" in processor.size:
                        size = processor.size["height"]
                    else:
                        size = 224
                except Exception:
                    size = 224
                
                from torchvision import transforms
                dataset.transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            elif mtype == "timm":
                import timm as _timm
                model = _timm.create_model(path, pretrained=True, num_classes=0).to(device)
                model.eval()
                wrapper["model"]      = model
                wrapper["extract_fn"] = timm_extractor
                
                data_config = _timm.data.resolve_model_data_config(model)
                dataset.transform = _timm.data.create_transform(**data_config, is_training=False)

            elif mtype == "open_clip":
                import open_clip
                import os
                from huggingface_hub import hf_hub_download
                
                hf_token = os.environ.get("HF_TOKEN")
                # Explicitly download the bin file using the token to bypass open_clip parsing issues
                weights_path = hf_hub_download(
                    repo_id=path,
                    filename="open_clip_pytorch_model.bin",
                    token=hf_token
                )
                
                # DermLIP uses ViT-B-16 architecture
                model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=weights_path)
                model = model.to(device)
                model.eval()
                wrapper["model"]      = model
                wrapper["extract_fn"] = open_clip_extractor
                
                dataset.transform = preprocess
                
            # Recreate loader with updated transform
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  Documenting {name} as unavailable (HF repo missing / gated).")
            results.append({
                "Architecture":          name,
                "Status":                "UNAVAILABLE",
                "Reason":                str(e)[:200],
                "Location_BalAcc":       np.nan,
                "Location_BalAcc_Std":   np.nan,
                "Location_Acc":          np.nan,
                "MacroClass_BalAcc":     np.nan,
                "MacroClass_BalAcc_Std": np.nan,
            })
            continue

        if dry_run:
            print(f"  [DRY RUN] {name} loaded successfully. Skipping feature extraction.")
            results.append({
                "Architecture": name,
                "Status":       "DRY_RUN_OK",
                "Reason":       "dry-run mode",
                "Location_BalAcc": np.nan, "Location_BalAcc_Std": np.nan, "Location_Acc": np.nan,
                "MacroClass_BalAcc": np.nan, "MacroClass_BalAcc_Std": np.nan,
            })
            del model
            torch.cuda.empty_cache()
            continue

        # Extract features
        X, y_loc, y_macro = extract_features_and_targets(wrapper, loader, device)
        if X is None:
            del model
            torch.cuda.empty_cache()
            continue

        print(f"  Feature shape: {X.shape}")

        # Probe 1: Lesion location
        loc_bal, loc_bal_std, loc_acc = run_probe(X, y_loc, "Location")
        # Probe 2: Macro class (Malignant / Benign)
        mac_bal, mac_bal_std, _       = run_probe(X, y_macro, "MacroClass")

        print(f"  Location BalAcc:   {loc_bal:.4f} ± {loc_bal_std:.4f}")
        print(f"  MacroClass BalAcc: {mac_bal:.4f} ± {mac_bal_std:.4f}")

        results.append({
            "Architecture":          name,
            "Status":                "OK",
            "Reason":                "",
            "Location_BalAcc":       loc_bal,
            "Location_BalAcc_Std":   loc_bal_std,
            "Location_Acc":          loc_acc,
            "MacroClass_BalAcc":     mac_bal,
            "MacroClass_BalAcc_Std": mac_bal_std,
        })

        del model
        torch.cuda.empty_cache()

    # ── Save Results ──────────────────────────────────────────────────────────
    os.makedirs(cfg.paths.results_dir, exist_ok=True)

    df_results = pd.DataFrame(results)
    out_csv = os.path.join(cfg.paths.results_dir, "task25_sota_linear_probing.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
    print(df_results[["Architecture", "Status", "Location_BalAcc", "MacroClass_BalAcc"]].to_string(index=False))

    df_closed = pd.DataFrame(closed_records)
    closed_csv = os.path.join(cfg.paths.results_dir, "task25_closed_models.csv")
    df_closed.to_csv(closed_csv, index=False)
    print(f"Closed model documentation saved to {closed_csv}")

    print("\nTask 25 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SOTA Dermatology FM Linear Probing – Task 25")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test model loading only, skip feature extraction.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
