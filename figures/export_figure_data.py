#!/usr/bin/env python3
"""
Standalone data-export script for publication figures.

Reads existing checkpoints and PAD-UFES-20 test set; writes artifacts to
figures/data/. Does NOT modify any existing project files.

Usage (from project root):
    python figures/export_figure_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports (read-only use of existing modules)
# ---------------------------------------------------------------------------
FIGURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURES_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg  # noqa: E402
from dataset import MultimodalDermatologyDataset  # noqa: E402
from transforms import get_eval_transforms  # noqa: E402
from models.late_fusion import LateFusionClassifier  # noqa: E402
from models.gmu import GMUClassifier  # noqa: E402
from models.cross_attention import CrossAttentionV2TClassifier, CrossAttentionT2VClassifier  # noqa: E402
from models.image_only import ImageOnlyClassifier  # noqa: E402
from models.text_only import TextOnlyClassifier  # noqa: E402
from counterfactual import CounterfactualAuditor  # noqa: E402
from viz_data import aggregate_training_by_model, parse_training_logs  # noqa: E402
from viz_style import CLASS_NAMES, DATA_DIR, MODELS  # noqa: E402

REPRESENTATIVE_SEED = 42
ARCHITECTURES = {
    "Late Fusion": LateFusionClassifier,
    "GMU Baseline": GMUClassifier,
    "Cross-Attn V→T": CrossAttentionV2TClassifier,
    "Cross-Attn T→V": CrossAttentionT2VClassifier,
    "Image-Only": ImageOnlyClassifier,
    "Text-Only": TextOnlyClassifier,
}


def load_model(model_cls, seed: int, device):
    folder_map = {
        "LateFusionClassifier": "Late_Fusion",
        "GMUClassifier": "GMU_Baseline",
        "CrossAttentionV2TClassifier": "Cross_Attn_VtoT",
        "CrossAttentionT2VClassifier": "Cross_Attn_TtoV",
        "ImageOnlyClassifier": "Image_Only",
        "TextOnlyClassifier": "Text_Only",
    }
    folder = folder_map[model_cls.__name__]
    ckpt_path = PROJECT_ROOT / "checkpoints" / f"{folder}_seed_{seed}" / "best_model.pth"
    model = model_cls().to(device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_ood_inference(model, loader, device):
    all_labels, all_probs, all_preds = [], [], []
    all_fused, all_vis = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="OOD inference", leave=False):
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            logits, fused, vis = model(imgs, ids, mask)
            probs = torch.softmax(logits.float(), dim=1)
            preds = torch.argmax(logits, dim=1)

            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_fused.append(fused.float().cpu().numpy())
            all_vis.append(vis.float().cpu().numpy())

    return {
        "y_true": np.concatenate(all_labels),
        "y_prob": np.concatenate(all_probs),
        "y_pred": np.concatenate(all_preds),
        "fused_feat": np.concatenate(all_fused),
        "vis_feat": np.concatenate(all_vis),
    }


def export_counterfactual_cases(model, loader, tokenizer, device, dataset_df, n_cases=6):
    auditor = CounterfactualAuditor(model, tokenizer, device)
    cases = []
    global_idx = 0

    with torch.no_grad():
        for batch in loader:
            if len(cases) >= n_cases:
                break
            imgs = batch["image"].to(device)
            real_ids = batch["input_ids"].to(device)
            real_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            real_logits, _, _ = model(imgs, real_ids, real_mask)
            real_probs = torch.softmax(real_logits.float(), dim=1)
            _, real_preds = torch.max(real_logits, 1)

            cf_ids, cf_mask = auditor._get_override_tensors(labels)
            cf_logits, _, _ = model(imgs, cf_ids, cf_mask)
            cf_probs = torch.softmax(cf_logits.float(), dim=1)
            _, cf_preds = torch.max(cf_logits, 1)

            B = imgs.size(0)
            for i in range(B):
                if len(cases) >= n_cases:
                    break
                if global_idx >= len(dataset_df):
                    break
                row = dataset_df.iloc[global_idx]
                global_idx += 1

                flipped = int(real_preds[i] != cf_preds[i])
                if not flipped and len(cases) < n_cases // 2:
                    continue

                real_p = real_probs[i, real_preds[i]].item()
                cf_p = cf_probs[i, real_preds[i]].item()

                cases.append({
                    "filepath": row["filepath"],
                    "clinical_history": row["clinical_history"],
                    "true_label": CLASS_NAMES[labels[i].item()],
                    "real_pred": CLASS_NAMES[real_preds[i].item()],
                    "cf_pred": CLASS_NAMES[cf_preds[i].item()],
                    "cf_text": (
                        cfg.audit.benign_override
                        if auditor.is_malignant[labels[i]].item()
                        else cfg.audit.malignant_override
                    ),
                    "real_prob": real_p,
                    "cf_prob_on_real_class": cf_p,
                    "delta_p": abs(real_p - cf_p),
                    "flipped": bool(flipped),
                })
    return cases


def export_attention_maps(model, loader, device, n_samples=8):
    """Extract cross-attention weights for representative samples."""
    if not hasattr(model, "cross_attn"):
        return None

    maps = []
    with torch.no_grad():
        for batch in loader:
            if len(maps) >= n_samples:
                break
            imgs = batch["image"].to(device)
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            v_seq = model.vision_encoder.forward_features(imgs)
            t_out = model.text_encoder(input_ids=ids, attention_mask=mask)
            t_seq = t_out.last_hidden_state
            key_padding_mask = (mask == 0)

            _, attn_weights = model.cross_attn(
                query=v_seq,
                key=t_seq,
                value=t_seq,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            # attn_weights: (B, 197, 128) — CLS row = patch 0 attends to text
            cls_attn = attn_weights[:, 0, :].cpu().numpy()
            maps.append(cls_attn)
    return np.concatenate(maps, axis=0)[:n_samples]


def compute_calibration(y_true, y_prob, n_bins=15):
  confidences = y_prob.max(axis=1)
  predictions = y_prob.argmax(axis=1)
  accuracies = (predictions == y_true).astype(float)
  bins = np.linspace(0, 1, n_bins + 1)
  bin_centers, bin_acc, bin_conf, counts = [], [], [], []
  ece = 0.0
  for i in range(n_bins):
      lo, hi = bins[i], bins[i + 1]
      mask = (confidences > lo) & (confidences <= hi)
      if mask.sum() == 0:
          continue
      acc = accuracies[mask].mean()
      conf = confidences[mask].mean()
      w = mask.mean()
      ece += w * abs(acc - conf)
      bin_centers.append((lo + hi) / 2)
      bin_acc.append(acc)
      bin_conf.append(conf)
      counts.append(int(mask.sum()))
  return {
      "bin_centers": [float(x) for x in bin_centers],
      "bin_accuracy": [float(x) for x in bin_acc],
      "bin_confidence": [float(x) for x in bin_conf],
      "counts": counts,
      "ece": float(ece),
  }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exporting figure data on {device}...")

    # --- Training curves from logs (no GPU needed) ---
    runs = parse_training_logs()
    curves = aggregate_training_by_model(runs, seed=REPRESENTATIVE_SEED)
    with open(DATA_DIR / "training_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print("  training_curves.json")

    # --- OOD dataset ---
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    test_ds = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        tokenizer=tokenizer,
        transform=get_eval_transforms(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=2
    )
    test_df = test_ds.df

    ood = {}
    fused_bank = {}
    vis_bank = {}
    per_class = {}
    calibration = {}
    confusion = {}

    for model_name, model_cls in ARCHITECTURES.items():
        print(f"  Exporting {model_name} (seed {REPRESENTATIVE_SEED})...")
        model = load_model(model_cls, REPRESENTATIVE_SEED, device)
        res = run_ood_inference(model, test_loader, device)

        ood[model_name] = {
            "y_true": res["y_true"],
            "y_prob": res["y_prob"],
            "y_pred": res["y_pred"],
        }
        fused_bank[model_name] = res["fused_feat"]
        vis_bank[model_name] = res["vis_feat"]

        cm = confusion_matrix(
            res["y_true"], res["y_pred"], labels=list(range(len(CLASS_NAMES)))
        )
        confusion[model_name] = cm

        prec, rec, f1, sup = precision_recall_fscore_support(
            res["y_true"], res["y_pred"], labels=list(range(len(CLASS_NAMES))),
            zero_division=0,
        )
        per_class[model_name] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist(),
            "support": sup.tolist(),
            "per_class_accuracy": [
                cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.0
                for i in range(len(CLASS_NAMES))
            ],
        }
        calibration[model_name] = compute_calibration(res["y_true"], res["y_prob"])

        if "Cross-Attn" in model_name:
            attn = export_attention_maps(model, test_loader, device)
            if attn is not None:
                np.savez_compressed(DATA_DIR / "attention_maps.npz", weights=attn)

        if "Cross-Attn" in model_name:
            cases = export_counterfactual_cases(
                model, test_loader, tokenizer, device, test_df, n_cases=6
            )
            with open(DATA_DIR / "counterfactual_cases.json", "w") as f:
                json.dump(cases, f, indent=2)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    np.savez_compressed(
        DATA_DIR / "ood_predictions.npz",
        **{f"{k.replace(' ', '_')}_y_true": v["y_true"] for k, v in ood.items()},
        **{f"{k.replace(' ', '_')}_y_prob": v["y_prob"] for k, v in ood.items()},
        **{f"{k.replace(' ', '_')}_y_pred": v["y_pred"] for k, v in ood.items()},
    )
    np.savez_compressed(
        DATA_DIR / "fused_features.npz",
        **{k.replace(" ", "_"): v for k, v in fused_bank.items()},
        **{f"{k.replace(' ', '_')}_vis": v for k, v in vis_bank.items()},
    )

    with open(DATA_DIR / "per_class_metrics.json", "w") as f:
        json.dump(per_class, f, indent=2)
    with open(DATA_DIR / "confusion_matrices.json", "w") as f:
        json.dump({k: v.tolist() for k, v in confusion.items()}, f, indent=2)
    with open(DATA_DIR / "calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)

    # Counterfactual cases for all models
    for model_name, model_cls in ARCHITECTURES.items():
        model = load_model(model_cls, REPRESENTATIVE_SEED, device)
        cases = export_counterfactual_cases(
            model, test_loader, tokenizer, device, test_df, n_cases=4
        )
        with open(DATA_DIR / f"counterfactual_cases_{model_name.replace(' ', '_')}.json", "w") as f:
            json.dump(cases, f, indent=2)
        del model

    print(f"\nAll artifacts saved to {DATA_DIR}/")


if __name__ == "__main__":
    main()
