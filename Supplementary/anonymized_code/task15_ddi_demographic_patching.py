import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from dataset import get_transforms
from models.cross_attention import CrossAttentionT2VClassifier
from task6_ddi_stratified_audit_rigorous import DDI_DISEASE_MAPPING


SEED = 1337
BATCH_SIZE = 16
LIGHT_TONE = 12
DARK_TONE = 56
NUM_HEADS = cfg.model.num_attention_heads


class DDIDemographicPairDataset(Dataset):
    def __init__(self, metadata_path, img_dir, tokenizer, transform=None):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform

        df = pd.read_csv(metadata_path)
        df = df.copy()
        df["miccai_class"] = df["disease"].map(DDI_DISEASE_MAPPING)
        df = df.dropna(subset=["miccai_class"]).reset_index(drop=True)
        df["label_idx"] = df["miccai_class"].map(cfg.data.LABEL_MAP)

        self.df = df
        self.pairs = self._build_pairs(df)
        print(
            f"Built {len(self.pairs)} exact-disease FST {LIGHT_TONE}->{DARK_TONE} "
            "pairs with same-tone and random-class controls."
        )

    def _pick_other(self, candidates, avoid_ddi_id, random_state):
        candidates = candidates[candidates["DDI_ID"] != avoid_ddi_id]
        if candidates.empty:
            return None
        return candidates.sample(1, random_state=random_state).iloc[0]

    def _build_pairs(self, df):
        pairs = []
        light = df[df["skin_tone"] == LIGHT_TONE]
        dark = df[df["skin_tone"] == DARK_TONE]

        rng = np.random.RandomState(SEED)
        for disease in sorted(df["disease"].dropna().unique()):
            light_d = light[light["disease"] == disease]
            dark_d = dark[dark["disease"] == disease]
            if light_d.empty or dark_d.empty:
                continue

            for _, dark_row in dark_d.iterrows():
                light_row = light_d.sample(1, random_state=int(dark_row["DDI_ID"]) + SEED).iloc[0]
                dark_ctrl = self._pick_other(
                    dark_d, dark_row["DDI_ID"], int(dark_row["DDI_ID"]) + 2 * SEED
                )

                random_light_pool = light[light["disease"] != disease]
                if random_light_pool.empty:
                    continue
                random_light = random_light_pool.sample(
                    1, random_state=int(rng.randint(0, 1_000_000))
                ).iloc[0]

                pairs.append(
                    {
                        "disease": disease,
                        "miccai_class": dark_row["miccai_class"],
                        "label_idx": int(dark_row["label_idx"]),
                        "light_file": light_row["DDI_file"],
                        "dark_file": dark_row["DDI_file"],
                        "dark_control_file": dark_ctrl["DDI_file"] if dark_ctrl is not None else None,
                        "random_light_file": random_light["DDI_file"],
                        "light_skin_tone": int(light_row["skin_tone"]),
                        "dark_skin_tone": int(dark_row["skin_tone"]),
                        "random_light_disease": random_light["disease"],
                    }
                )
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, filename):
        if filename is None:
            return torch.zeros(3, cfg.data.img_size, cfg.data.img_size)
        image = Image.open(os.path.join(self.img_dir, filename)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        text = "No patient metadata provided."
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=cfg.data.max_text_len,
            return_tensors="pt",
        )
        return {
            "light_img": self._load_image(pair["light_file"]),
            "dark_img": self._load_image(pair["dark_file"]),
            "dark_control_img": self._load_image(pair["dark_control_file"]),
            "random_light_img": self._load_image(pair["random_light_file"]),
            "has_dark_control": torch.tensor(pair["dark_control_file"] is not None),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label_idx": torch.tensor(pair["label_idx"], dtype=torch.long),
            "disease": pair["disease"],
            "miccai_class": pair["miccai_class"],
            "light_file": pair["light_file"],
            "dark_file": pair["dark_file"],
            "dark_control_file": pair["dark_control_file"] or "",
            "random_light_file": pair["random_light_file"],
            "random_light_disease": pair["random_light_disease"],
        }


def load_model(device):
    ckpt_path = os.path.join(
        cfg.paths.checkpoint_dir,
        f"Cross-Attention_T\u2192V_seed_{SEED}",
        "best_model.pth",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()
    return model


def cross_attention_heads(model, images, input_ids, attention_mask):
    """Return per-head T->V attention outputs before output projection."""
    vision_seq = model.vision_encoder.forward_features(images)
    text_seq = model.text_encoder(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state

    mha = model.cross_attn
    embed_dim = mha.embed_dim
    num_heads = mha.num_heads
    head_dim = embed_dim // num_heads

    w_q, w_k, w_v = mha.in_proj_weight.chunk(3, dim=0)
    b_q, b_k, b_v = mha.in_proj_bias.chunk(3, dim=0)

    q = F.linear(text_seq, w_q, b_q)
    k = F.linear(vision_seq, w_k, b_k)
    v = F.linear(vision_seq, w_v, b_v)

    bsz, q_len, _ = q.shape
    k_len = k.shape[1]

    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)

    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_probs, v)


def logits_from_heads(model, heads):
    bsz, num_heads, seq_len, head_dim = heads.shape
    combined = heads.transpose(1, 2).contiguous().view(bsz, seq_len, num_heads * head_dim)
    attn_output = model.cross_attn.out_proj(combined)
    fused = attn_output.mean(dim=1)
    return model.classifier(fused)


def patched_logits(model, base_heads, donor_heads, head_idx):
    patched = base_heads.clone()
    patched[:, head_idx, :, :] = donor_heads[:, head_idx, :, :]
    return logits_from_heads(model, patched)


def signed_recovery(patched_value, base_value, donor_value):
    denom = donor_value - base_value
    if abs(float(denom)) < 1e-8:
        return np.nan
    return float((patched_value - base_value) / denom * 100.0)


def main():
    print("=" * 70)
    print("TASK 15: DDI DEMOGRAPHIC BIAS HEAD PATCHING")
    print("=" * 70)
    print(
        "This patches individual T->V cross-attention heads from donor images; "
        "full fused-vector replacement is intentionally avoided because it is tautological."
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    metadata_path = os.path.join(
        cfg.paths.package_root, "data", "ddi", "ddidiversedermatologyimages", "ddi_metadata.csv"
    )
    img_dir = os.path.join(
        cfg.paths.package_root, "data", "ddi", "ddidiversedermatologyimages", "Images"
    )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"DDI metadata not found: {metadata_path}")

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = DDIDemographicPairDataset(
        metadata_path, img_dir, tokenizer, transform=get_transforms()
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = load_model(device)
    class_names = {idx: name for name, idx in cfg.data.LABEL_MAP.items()}

    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Head patching"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label_idx"].to(device)

            light_heads = cross_attention_heads(
                model, batch["light_img"].to(device), input_ids, attn_mask
            )
            dark_heads = cross_attention_heads(
                model, batch["dark_img"].to(device), input_ids, attn_mask
            )
            dark_control_heads = cross_attention_heads(
                model, batch["dark_control_img"].to(device), input_ids, attn_mask
            )
            random_light_heads = cross_attention_heads(
                model, batch["random_light_img"].to(device), input_ids, attn_mask
            )

            light_logits = logits_from_heads(model, light_heads)
            dark_logits = logits_from_heads(model, dark_heads)

            for head_idx in range(NUM_HEADS):
                main_logits = patched_logits(model, dark_heads, light_heads, head_idx)
                same_tone_logits = patched_logits(model, dark_heads, dark_control_heads, head_idx)
                random_class_logits = patched_logits(model, dark_heads, random_light_heads, head_idx)
                reverse_logits = patched_logits(model, light_heads, dark_heads, head_idx)

                for i in range(labels.shape[0]):
                    label_idx = int(labels[i].item())
                    label_name = class_names[label_idx]

                    base_dark = float(dark_logits[i, label_idx].item())
                    donor_light = float(light_logits[i, label_idx].item())
                    main_value = float(main_logits[i, label_idx].item())
                    same_value = float(same_tone_logits[i, label_idx].item())
                    random_value = float(random_class_logits[i, label_idx].item())
                    reverse_base = float(light_logits[i, label_idx].item())
                    reverse_donor = float(dark_logits[i, label_idx].item())
                    reverse_value = float(reverse_logits[i, label_idx].item())

                    rows.append(
                        {
                            "disease": batch["disease"][i],
                            "miccai_class": batch["miccai_class"][i],
                            "label_idx": label_idx,
                            "label_name": label_name,
                            "light_file": batch["light_file"][i],
                            "dark_file": batch["dark_file"][i],
                            "dark_control_file": batch["dark_control_file"][i],
                            "random_light_file": batch["random_light_file"][i],
                            "random_light_disease": batch["random_light_disease"][i],
                            "has_dark_control": bool(batch["has_dark_control"][i].item()),
                            "head": head_idx,
                            "dark_true_logit": base_dark,
                            "light_true_logit": donor_light,
                            "main_light_to_dark_true_logit": main_value,
                            "same_tone_dark_to_dark_true_logit": same_value,
                            "random_class_light_to_dark_true_logit": random_value,
                            "reverse_dark_to_light_true_logit": reverse_value,
                            "main_recovery_pct": signed_recovery(main_value, base_dark, donor_light),
                            "same_tone_recovery_pct": signed_recovery(same_value, base_dark, donor_light),
                            "random_class_recovery_pct": signed_recovery(random_value, base_dark, donor_light),
                            "reverse_recovery_pct": signed_recovery(reverse_value, reverse_base, reverse_donor),
                            "dark_pred": int(dark_logits[i].argmax().item()),
                            "light_pred": int(light_logits[i].argmax().item()),
                            "main_patched_pred": int(main_logits[i].argmax().item()),
                            "same_tone_patched_pred": int(same_tone_logits[i].argmax().item()),
                            "random_class_patched_pred": int(random_class_logits[i].argmax().item()),
                            "reverse_patched_pred": int(reverse_logits[i].argmax().item()),
                        }
                    )

    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.results_dir, "task15_ddi_demographic_patching.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    summary = (
        df.groupby("head")
        .agg(
            n=("main_recovery_pct", "count"),
            mean_main_recovery_pct=("main_recovery_pct", "mean"),
            mean_same_tone_recovery_pct=("same_tone_recovery_pct", "mean"),
            mean_random_class_recovery_pct=("random_class_recovery_pct", "mean"),
            mean_reverse_recovery_pct=("reverse_recovery_pct", "mean"),
            main_pred_flip_rate=("main_patched_pred", lambda s: np.mean(s.values != df.loc[s.index, "dark_pred"].values)),
        )
        .reset_index()
    )
    summary["specificity_gap_vs_same_tone"] = (
        summary["mean_main_recovery_pct"] - summary["mean_same_tone_recovery_pct"]
    )
    summary_path = os.path.join(
        cfg.paths.results_dir, "task15_ddi_demographic_patching_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\nHead-level summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved per-pair/head results to {out_path}")
    print(f"Saved summary results to {summary_path}")


if __name__ == "__main__":
    main()
