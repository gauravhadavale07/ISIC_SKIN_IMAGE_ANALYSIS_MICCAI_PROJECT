"""Task 8: non-tautological cross-attention head patching.

The original Task 8 reconstructed the real-text forward pass and therefore
reported ~100% recovery by construction. This version uses a neutral-text
recipient state and a real-text donor state, then patches one attention head at a
time before the MultiheadAttention output projection.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier, CrossAttentionV2TClassifier


SEED = 1337
BATCH_SIZE = 16
EPS = 1e-8
MAX_SAMPLES = int(os.environ.get("TASK8_MAX_SAMPLES", "0"))


class RealNeutralDataset(MultimodalDermatologyDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        enc = self.tokenizer(
            cfg.audit.neutral_string,
            padding="max_length",
            truncation=True,
            max_length=cfg.data.max_text_len,
            return_tensors="pt",
        )
        item["neutral_input_ids"] = enc["input_ids"].squeeze(0)
        item["neutral_attention_mask"] = enc["attention_mask"].squeeze(0)
        return item


def load_model(model_cls, checkpoint_name, device):
    ckpt_path = os.path.join(
        cfg.paths.checkpoint_dir,
        f"{checkpoint_name}_seed_{SEED}",
        "best_model.pth",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = model_cls().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()
    return model


def split_qkv(mha, query_seq, key_seq, value_seq):
    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)
    q = F.linear(query_seq, w_q, b_q)
    k = F.linear(key_seq, w_k, b_k)
    v = F.linear(value_seq, w_v, b_v)
    return q, k, v


def per_head_attention(mha, query_seq, key_seq, value_seq, key_padding_mask=None):
    q, k, v = split_qkv(mha, query_seq, key_seq, value_seq)
    bsz, q_len, embed_dim = q.shape
    k_len = k.shape[1]
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads

    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
    if key_padding_mask is not None:
        mask = key_padding_mask[:, None, None, :].to(torch.bool)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def logits_from_heads(model, heads, pool_mode):
    bsz, num_heads, seq_len, head_dim = heads.shape
    combined = heads.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)
    attn_output = model.cross_attn.out_proj(combined)
    if pool_mode == "squeeze":
        fused = attn_output.squeeze(1)
    else:
        fused = attn_output.mean(dim=1)
    return model.classifier(fused)


def t2v_heads(model, images, input_ids, attention_mask):
    vision_seq = model.vision_encoder.forward_features(images)
    text_seq = model.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    return per_head_attention(model.cross_attn, text_seq, vision_seq, vision_seq)


def v2t_heads(model, images, input_ids, attention_mask):
    vision_feat = model.vision_encoder(images)
    vision_seq = vision_feat.unsqueeze(1)
    text_seq = model.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    key_padding_mask = attention_mask == 0
    return per_head_attention(
        model.cross_attn,
        vision_seq,
        text_seq,
        text_seq,
        key_padding_mask=key_padding_mask,
    )


def patch_head(base_heads, donor_heads, head_idx):
    patched = base_heads.clone()
    patched[:, head_idx, :, :] = donor_heads[:, head_idx, :, :]
    return patched


def compute_rows(arch_name, model, loader, device, head_fn, pool_mode):
    num_heads = model.cross_attn.num_heads
    rows = []
    aggregate = {
        h: {
            "recoveries": [],
            "patched_correct": 0,
            "patched_flips": 0,
            "valid": 0,
            "total": 0,
        }
        for h in list(range(num_heads)) + ["all"]
    }
    base_correct = 0
    donor_correct = 0
    total = 0
    diffs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Patching {arch_name}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            real_ids = batch["input_ids"].to(device)
            real_mask = batch["attention_mask"].to(device)
            neutral_ids = batch["neutral_input_ids"].to(device)
            neutral_mask = batch["neutral_attention_mask"].to(device)

            base_heads = head_fn(model, images, neutral_ids, neutral_mask)
            donor_heads = head_fn(model, images, real_ids, real_mask)

            base_logits = logits_from_heads(model, base_heads, pool_mode)
            donor_logits = logits_from_heads(model, donor_heads, pool_mode)
            base_pred = base_logits.argmax(dim=1)
            donor_pred = donor_logits.argmax(dim=1)

            idx = torch.arange(labels.shape[0], device=device)
            base_true = base_logits[idx, labels]
            donor_true = donor_logits[idx, labels]
            denom = donor_true - base_true
            valid = torch.abs(denom) > EPS

            base_correct += (base_pred == labels).sum().item()
            donor_correct += (donor_pred == labels).sum().item()
            total += labels.shape[0]
            diffs.extend(denom.detach().cpu().numpy().tolist())

            for head_idx in range(num_heads):
                patched_logits = logits_from_heads(
                    model, patch_head(base_heads, donor_heads, head_idx), pool_mode
                )
                patched_pred = patched_logits.argmax(dim=1)
                patched_true = patched_logits[idx, labels]
                recovery = torch.zeros_like(denom)
                recovery[valid] = (patched_true[valid] - base_true[valid]) / denom[valid]

                aggregate[head_idx]["recoveries"].extend(
                    recovery[valid].detach().cpu().numpy().tolist()
                )
                aggregate[head_idx]["valid"] += int(valid.sum().item())
                aggregate[head_idx]["total"] += labels.shape[0]
                aggregate[head_idx]["patched_correct"] += (patched_pred == labels).sum().item()
                aggregate[head_idx]["patched_flips"] += (patched_pred != base_pred).sum().item()

            all_logits = logits_from_heads(model, donor_heads, pool_mode)
            all_pred = all_logits.argmax(dim=1)
            all_recovery = torch.zeros_like(denom)
            all_recovery[valid] = (all_logits[idx, labels][valid] - base_true[valid]) / denom[valid]
            aggregate["all"]["recoveries"].extend(all_recovery[valid].detach().cpu().numpy().tolist())
            aggregate["all"]["valid"] += int(valid.sum().item())
            aggregate["all"]["total"] += labels.shape[0]
            aggregate["all"]["patched_correct"] += (all_pred == labels).sum().item()
            aggregate["all"]["patched_flips"] += (all_pred != base_pred).sum().item()

    for head_idx, stats in aggregate.items():
        rec = np.asarray(stats["recoveries"], dtype=float)
        rows.append(
            {
                "Architecture": arch_name,
                "Patch": f"Head {head_idx}" if head_idx != "all" else "All heads",
                "N": int(stats["total"]),
                "Valid_N": int(stats["valid"]),
                "Base_Neutral_Accuracy": base_correct / total,
                "Donor_Real_Accuracy": donor_correct / total,
                "Patched_Accuracy": stats["patched_correct"] / max(stats["total"], 1),
                "Mean_Real_minus_Neutral_True_Logit": float(np.mean(diffs)),
                "Mean_Recovery_Pct": float(np.mean(rec) * 100.0) if len(rec) else np.nan,
                "Median_Recovery_Pct": float(np.median(rec) * 100.0) if len(rec) else np.nan,
                "Patch_Flip_Rate": stats["patched_flips"] / max(stats["total"], 1),
            }
        )
    return rows


def main():
    print("=" * 70)
    print("TASK 8: NON-TAUTOLOGICAL CROSS-ATTENTION HEAD PATCHING")
    print("=" * 70)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = RealNeutralDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(),
    )
    if MAX_SAMPLES > 0:
        dataset.df = dataset.df.head(MAX_SAMPLES).reset_index(drop=True)
        print(f"Using first {len(dataset.df)} samples because TASK8_MAX_SAMPLES={MAX_SAMPLES}.")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    experiments = [
        (
            "Cross-Attention V->T",
            CrossAttentionV2TClassifier,
            "Cross-Attention_V\u2192T",
            v2t_heads,
            "squeeze",
        ),
        (
            "Cross-Attention T->V",
            CrossAttentionT2VClassifier,
            "Cross-Attention_T\u2192V",
            t2v_heads,
            "mean",
        ),
    ]

    rows = []
    for arch_name, model_cls, ckpt_name, head_fn, pool_mode in experiments:
        print(f"\nEvaluating {arch_name}...")
        model = load_model(model_cls, ckpt_name, device)
        rows.extend(compute_rows(arch_name, model, loader, device, head_fn, pool_mode))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.results_dir, "task8_activation_patching.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved corrected Task 8 results to {out_path}")


if __name__ == "__main__":
    main()
