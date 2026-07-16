"""
update_figure_data.py — Runs evaluation on OOD test set for all 4 models across seeds 456, 789, 1337,
then aggregates confusion matrices, per-class metrics, and ECE calibration data,
saving them to figures/data/ for final report generation.
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from config import cfg
from dataset import MultimodalDermatologyDataset
from transforms import get_eval_transforms
from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionV2TClassifier, CrossAttentionT2VClassifier

DEVICE = cfg.train.device
SEEDS  = [456, 789, 1337]
CLASS_NAMES = ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']

ARCHITECTURES = {
    "Late Fusion": LateFusionClassifier,
    "GMU Baseline": GMUClassifier,
    "Cross-Attention": CrossAttentionV2TClassifier,
    "Cross-Attention T→V": CrossAttentionT2VClassifier
}

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
    print("Evaluating all models for figure data generation...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    test_ds = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir="",
        tokenizer=tokenizer,
        transform=get_eval_transforms()
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    per_class_all = {}
    confusion_all = {}
    calibration_all = {}

    for model_name, model_cls in ARCHITECTURES.items():
        print(f"  Processing {model_name}...")
        
        folder_name = model_name.replace(" ", "_")
        if model_name == "Cross-Attention":
            folder_name = "Cross-Attention"
        elif model_name == "Cross-Attention T→V":
            folder_name = "Cross-Attention_T2V"

        # Accumulate predictions across seeds
        y_true_list = []
        y_prob_list = []
        y_pred_list = []

        for seed in SEEDS:
            ckpt_path = f"./checkpoints/{folder_name}_seed_{seed}/best_model.pth"
            model = model_cls().to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            model.eval()

            all_labels, all_probs, all_preds = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    imgs = batch["image"].to(DEVICE)
                    ids = batch["input_ids"].to(DEVICE)
                    mask = batch["attention_mask"].to(DEVICE)
                    labels = batch["label"]

                    logits, _, _ = model(imgs, ids, mask)
                    probs = torch.softmax(logits.float(), dim=1)
                    preds = torch.argmax(logits, dim=1)

                    all_labels.append(labels.numpy())
                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())

            y_true_list.append(np.concatenate(all_labels))
            y_prob_list.append(np.concatenate(all_probs))
            y_pred_list.append(np.concatenate(all_preds))

        # Use seed 456 as the representative for raw confusion matrices and per-class metrics
        # to match the single-matrix reporting expectation, but use the actual matched seeds.
        rep_idx = 0 # seed 456
        y_true = y_true_list[rep_idx]
        y_prob = y_prob_list[rep_idx]
        y_pred = y_pred_list[rep_idx]

        # ECE averaged over seeds
        ece_vals = [compute_calibration(y_true_list[i], y_prob_list[i])["ece"] for i in range(len(SEEDS))]
        avg_ece = float(np.mean(ece_vals))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
        confusion_all[model_name] = cm.tolist()

        prec, rec, f1, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
            zero_division=0
        )

        per_class_all[model_name] = {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "f1": f1.tolist(),
            "support": sup.tolist(),
            "per_class_accuracy": [
                float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0.0
                for i in range(len(CLASS_NAMES))
            ],
        }

        # Save calibration details for the representative seed
        cal_data = compute_calibration(y_true, y_prob)
        cal_data["ece"] = avg_ece # Override with average ECE across seeds for better statistics
        calibration_all[model_name] = cal_data

    # Write out JSON files
    os.makedirs("./figures/data/", exist_ok=True)
    with open("./figures/data/per_class_metrics.json", "w") as f:
        json.dump(per_class_all, f, indent=2)
    with open("./figures/data/confusion_matrices.json", "w") as f:
        json.dump(confusion_all, f, indent=2)
    with open("./figures/data/calibration.json", "w") as f:
        json.dump(calibration_all, f, indent=2)

    print("Successfully updated figures/data/ JSON files with all 4 models.")

if __name__ == "__main__":
    main()
