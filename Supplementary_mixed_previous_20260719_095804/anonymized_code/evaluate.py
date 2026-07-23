import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.amp import autocast
from tqdm import tqdm
from typing import Dict, Any
from config import cfg

class Evaluator:
    """
    Clinical standard statistical evaluation engine.
    Computes rigorous metrics required for MICCAI/ISIC submissions.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, Any]:
        """
        Runs a full forward pass over the dataloader and calculates clinical metrics.
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []  # FIX: now holds the FULL (N, num_classes) probability matrix, not just class-1

        pbar = tqdm(dataloader, desc="Evaluating Metrics")

        for batch in pbar:
            imgs = batch["image"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # FIX: device-aware autocast (matches cka.py / counterfactual.py),
            # and now respects cfg.train.use_amp instead of always defaulting on.
            with autocast(device_type=self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda"):
                logits, _, _ = self.model(imgs, input_ids, attn_mask)

                # Convert logits to probabilities via Softmax
                probs = torch.softmax(logits, dim=1)

                # Hard predictions via argmax
                _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # FIX: keep the FULL probability row per sample (shape: num_classes),
            # not just probs[:, 1]. The original code assumed binary
            # classification and extracted only "probability of class 1",
            # which is meaningless for this 6-class problem (MEL=0..SEK=5) and
            # was the direct cause of roc_auc_score() raising ValueError on
            # every call below, silently caught and replaced with a hardcoded
            # 0.5 placeholder.
            all_probs.append(probs.float().cpu().numpy())

        # Convert to numpy arrays for sklearn
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.concatenate(all_probs, axis=0)  # (N, num_classes)

        # ---------------------------------------------------------
        # CALCULATE CLINICAL METRICS
        # ---------------------------------------------------------
        acc = accuracy_score(y_true, y_pred)

        # Macro averaging treats all classes equally regardless of support
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # FIX: multi-class macro AUROC (one-vs-rest) over the full probability
        # matrix. The original binary call `roc_auc_score(y_true, y_prob)`
        # raised ValueError on EVERY call against 6-class labels — caught and
        # silently replaced with a hardcoded 0.5, which is not even a
        # meaningful "chance" baseline for 6-class macro-OVR AUROC. That
        # constant is also exactly why a real multi-seed run showed
        # `AUROC: 0.5000 ± 0.0000` with zero variance across 3 different
        # seeds — a direct symptom of the fallback firing every time, which
        # in turn made the AUROC paired t-test come back `t=nan, p=nan`.
        # `labels=` is passed explicitly to pin probability-column order to
        # class index, regardless of which classes happen to appear in this
        # particular y_true (defends against the same kind of silent
        # index-misalignment bug fixed elsewhere in this pipeline).
        try:
            auroc = roc_auc_score(
                y_true, y_prob,
                multi_class='ovr',
                average='macro',
                labels=list(range(cfg.model.num_classes))
            )
        except ValueError as e:
            # Genuine edge case (e.g. a class entirely absent from this split).
            # Surface it loudly instead of silently injecting a fake constant.
            print(f"⚠️  AUROC could not be computed ({e}). Reporting NaN instead of a placeholder.")
            auroc = float('nan')

        cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.model.num_classes)))

        results = {
            "Accuracy": acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1 (Macro)": f1,
            "AUROC": auroc,
            "Confusion Matrix": cm.tolist(),  # Convert to list for easy JSON serialization later
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob
        }

        return results

    def print_report(self, results: Dict[str, Any], prefix: str = ""):
        """Utility to elegantly print the metrics."""
        # FIX: class order now derived from cfg.data.LABEL_MAP instead of a
        # stale "(TN, FP | FN, TP)" binary-confusion-matrix label, which no
        # longer described this 6x6 matrix.
        class_order = [name for name, _ in sorted(cfg.data.LABEL_MAP.items(), key=lambda kv: kv[1])]

        print(f"\n📊 --- {prefix} Evaluation Report ---")
        print(f"Accuracy:  {results['Accuracy'] * 100:.2f}%")
        print(f"AUROC:     {results['AUROC']:.4f} (macro, one-vs-rest)")
        print(f"F1 Score:  {results['F1 (Macro)']:.4f} (Macro)")
        print(f"Precision: {results['Precision (Macro)']:.4f}")
        print(f"Recall:    {results['Recall (Macro)']:.4f}")
        print(f"Confusion Matrix (rows=true, cols=pred, class order: {class_order}):")
        for row in results['Confusion Matrix']:
            print(f"  {row}")
        print("-" * 35)