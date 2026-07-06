import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.amp import autocast
from tqdm import tqdm
from transformers import AutoTokenizer
from config import cfg
from typing import Dict, Any


class CounterfactualAuditor:
    """
    Executes the mechanistic behavioral audit to detect Modality Collapse.
    Evaluates Blank-Text Ablation, Counterfactual Flip Rate (CFR), and Mean Probability Shift (Delta P).
    """
    def __init__(self, model: torch.nn.Module, tokenizer: AutoTokenizer, device: torch.device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # Pre-tokenize the blank string (structural baseline)
        self.blank_tokens = self.tokenizer(
            cfg.audit.blank_string,
            padding="max_length",
            truncation=True,
            max_length=cfg.data.max_text_len,
            return_tensors="pt"
        )

        # Pre-tokenize the neutral string (non-empty baseline)
        self.neutral_tokens = self.tokenizer(
            cfg.audit.neutral_string,
            padding="max_length",
            truncation=True,
            max_length=cfg.data.max_text_len,
            return_tensors="pt"
        )

        # Pre-tokenize the semantic overrides
        self.benign_cf = self.tokenizer(
            cfg.audit.benign_override,
            padding="max_length",
            truncation=True,
            max_length=cfg.data.max_text_len,
            return_tensors="pt"
        )

        self.malignant_cf = self.tokenizer(
            cfg.audit.malignant_override,
            padding="max_length",
            truncation=True,
            max_length=cfg.data.max_text_len,
            return_tensors="pt"
        )

        # =====================================================================
        # FIX (critical): build a full 6-class -> binary (0=benign / 1=malignant)
        # lookup table, instead of the original code's implicit binary
        # assumption.
        #
        # ORIGINAL BUG: _get_override_tensors() only wrote into cf_ids/cf_mask
        # for `labels == 0` and `labels == 1`. The real dataset is 6-class
        # (cfg.data.LABEL_MAP: MEL=0, BCC=1, SCC=2, ACK=3, NEV=4, SEK=5). Any
        # sample with label in {2, 3, 4, 5} — SCC/ACK/NEV/SEK, which is 61% of
        # PAD-UFES-20 (1,401 / 2,298 samples) — never got cf_ids/cf_mask
        # written, so they stayed at their torch.zeros_like(...) initial
        # value: an all-zero token sequence. That's functionally a second
        # blank-text probe, not a semantic counterfactual, and for
        # CrossAttentionClassifier an all-zero attention_mask means EVERY key
        # position is masked with -inf before softmax -> NaN -> exactly the
        # `Mean_Delta_P: nan ± nan` observed in a real run. It also explains
        # why CFR (61.85%) landed almost exactly at the broken-sample
        # proportion (60.97%): most of that signal was "model reacts to being
        # given blank input", which the blank-text probe already measures
        # separately and correctly.
        #
        # FIX: this lookup is built ONCE here from the single source of truth
        # in config.py (cfg.data.LABEL_MAP + cfg.data.LABEL_MAPPING), so every
        # one of the 6 classes is routed to its TRUE semantic-opposite cohort.
        # =====================================================================
        num_classes = len(cfg.data.LABEL_MAP)
        is_malignant = torch.zeros(num_classes, dtype=torch.bool)
        for class_name, class_idx in cfg.data.LABEL_MAP.items():
            is_malignant[class_idx] = bool(cfg.data.LABEL_MAPPING[class_name])
        self.is_malignant = is_malignant.to(device)  # (6,) bool, indexed by label id

    def _get_override_tensors(self, labels: torch.Tensor):
        """
        Dynamically selects the contradictory text tensor for each sample in the batch.

        Routing rule (FIX: now covers ALL 6 classes, not just labels 0/1):
            - Ground truth is in the MALIGNANT cohort (MEL/BCC/SCC) -> inject
              the BENIGN override text (the contradiction).
            - Ground truth is in the BENIGN cohort (ACK/NEV/SEK)    -> inject
              the MALIGNANT override text (the contradiction).

        .contiguous() ensures non-contiguous expand() views don't cause downstream .view() errors.
        """
        B = labels.size(0)

        # Expand pre-tokenized overrides to match batch size
        # .contiguous() converts the non-contiguous expand() view to a proper tensor
        b_ids  = self.benign_cf["input_ids"].expand(B, -1).contiguous().to(self.device)
        b_mask = self.benign_cf["attention_mask"].expand(B, -1).contiguous().to(self.device)

        m_ids  = self.malignant_cf["input_ids"].expand(B, -1).contiguous().to(self.device)
        m_mask = self.malignant_cf["attention_mask"].expand(B, -1).contiguous().to(self.device)

        # Create empty tensors to hold the dynamic batch
        cf_ids  = torch.zeros_like(b_ids)
        cf_mask = torch.zeros_like(b_mask)

        # FIX: route via the full 6-class -> binary lookup instead of the
        # original `labels == 1` / `labels == 0` checks. is_malignant_batch[i]
        # is True iff sample i's ground-truth class (any of the 6) belongs to
        # the malignant cohort (MEL/BCC/SCC).
        is_malignant_batch = self.is_malignant[labels]  # (B,) bool

        # Malignant ground truth -> inject BENIGN override
        cf_ids[is_malignant_batch]  = b_ids[is_malignant_batch]
        cf_mask[is_malignant_batch] = b_mask[is_malignant_batch]

        # Benign ground truth -> inject MALIGNANT override
        cf_ids[~is_malignant_batch]  = m_ids[~is_malignant_batch]
        cf_mask[~is_malignant_batch] = m_mask[~is_malignant_batch]

        return cf_ids, cf_mask

    @torch.no_grad()
    def run_audit(self, dataloader) -> Dict[str, Any]:
        """
        Executes the three-probe audit over the OOD dataloader.

        Probes:
            1. Real Text Baseline  — normal inference, establishes transfer accuracy
            2. Blank Text Ablation — replaces text with empty string; accuracy drop
                                     measures how much the model relies on text
            3. Counterfactual Override — injects semantically opposite text;
                                         CFR and Delta P measure text influence strength
        """
        self.model.eval()

        total          = 0
        real_correct   = 0
        blank_correct  = 0
        neutral_correct = 0
        cf_correct     = 0
        flipped_count  = 0
        delta_p_sum    = 0.0

        pbar = tqdm(dataloader, desc="Mechanistic Audit")

        # Pre-move blank tensors to device once — avoids repetitive CPU→GPU transfers
        dummy_blank_ids  = self.blank_tokens["input_ids"].to(self.device)
        dummy_blank_mask = self.blank_tokens["attention_mask"].to(self.device)
        
        # Pre-move neutral tensors to device once
        dummy_neutral_ids  = self.neutral_tokens["input_ids"].to(self.device)
        dummy_neutral_mask = self.neutral_tokens["attention_mask"].to(self.device)

        for batch in pbar:
            imgs      = batch["image"].to(self.device, non_blocking=True)
            real_ids  = batch["input_ids"].to(self.device, non_blocking=True)
            real_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels    = batch["label"].to(self.device, non_blocking=True)

            B      = imgs.size(0)
            total += B

            # --- PROBE 1: REAL TEXT BASELINE ---
            with autocast(device_type=self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda"):
                real_logits, _, _ = self.model(imgs, real_ids, real_mask)
                real_probs        = torch.softmax(real_logits.float(), dim=1)
                _, real_preds     = torch.max(real_logits, 1)
                real_correct     += (real_preds == labels).sum().item()

            # --- PROBE 2: BLANK TEXT ABLATION ---
            # .contiguous() on expand() output to avoid potential .view() errors downstream
            b_ids  = dummy_blank_ids.expand(B, -1).contiguous()
            b_mask = dummy_blank_mask.expand(B, -1).contiguous()

            with autocast(device_type=self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda"):
                blank_logits, _, _ = self.model(imgs, b_ids, b_mask)
                _, blank_preds     = torch.max(blank_logits, 1)
                blank_correct     += (blank_preds == labels).sum().item()

            # --- PROBE 2.5: NEUTRAL TEXT PROBE ---
            n_ids  = dummy_neutral_ids.expand(B, -1).contiguous()
            n_mask = dummy_neutral_mask.expand(B, -1).contiguous()

            with autocast(device_type=self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda"):
                neutral_logits, _, _ = self.model(imgs, n_ids, n_mask)
                _, neutral_preds     = torch.max(neutral_logits, 1)
                neutral_correct     += (neutral_preds == labels).sum().item()

            # --- PROBE 3: COUNTERFACTUAL OVERRIDES ---
            cf_ids, cf_mask = self._get_override_tensors(labels)

            with autocast(device_type=self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda"):
                cf_logits, _, _ = self.model(imgs, cf_ids, cf_mask)
                cf_probs        = torch.softmax(cf_logits.float(), dim=1)
                _, cf_preds     = torch.max(cf_logits, 1)
                cf_correct     += (cf_preds == labels).sum().item()

            # CFR: proportion of samples where the prediction flipped
            flipped        = (real_preds != cf_preds)
            flipped_count += flipped.sum().item()

            # Delta P: mean absolute shift in confidence on the *original* predicted class
            # Advanced indexing extracts per-sample probability of the original prediction
            # Measuring shift on real_preds (not cf_preds) isolates original-decision confidence loss
            real_p  = real_probs[torch.arange(B), real_preds]
            cf_p    = cf_probs[torch.arange(B), real_preds]
            delta_p = torch.abs(real_p - cf_p)
            delta_p_sum += delta_p.sum().item()

        # Compile final metrics
        # All percentage values are in percentage points for consistent reporting
        results = {
            "Real_Accuracy":           100. * real_correct / total,
            "Blank_Accuracy":          100. * blank_correct / total,
            "Neutral_Accuracy":        100. * neutral_correct / total,
            "Counterfactual_Accuracy": 100. * cf_correct / total,
            "Blank_Accuracy_Drop":     100. * (real_correct - blank_correct) / total,  # structured for orchestrator
            "CFR":                     100. * flipped_count / total,
            "Mean_Delta_P":            100. * delta_p_sum / total      # percentage points of probability shift
        }

        return results

    def print_report(self, results: Dict[str, Any]):
        """Utility to elegantly print the audit metrics."""
        print(f"\n🔬 --- Mechanistic Audit Report ---")
        print(f"Transfer Accuracy:      {results['Real_Accuracy']:.2f}%")
        print(f"Blank-Text Accuracy:    {results['Blank_Accuracy']:.2f}%")
        print(f"  -> Accuracy Drop:     {results['Blank_Accuracy_Drop']:.2f}pp  (Higher = stronger text reliance)")
        print(f"\nCounterfactual Flip Rate (CFR):  {results['CFR']:.2f}%")
        print(f"Mean Probability Shift (ΔP):     {results['Mean_Delta_P']:.2f}pp")
        print("-" * 35)