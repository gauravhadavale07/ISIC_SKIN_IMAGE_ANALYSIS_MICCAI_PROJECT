"""Task 20: center-crop sensitivity for SAE Feature 1449 in the correct space.

The SAE was trained on Cross-Attention T->V fused representations. This script
therefore uses the T->V model and extracts the same pre-classifier fused stream
before encoding Feature 1449.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE


FEATURE_ID = 1449
SEED = 1337
TOP_K_IMAGES = 50
BATCH_SIZE = 16


baseline_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.img_mean, std=cfg.data.img_std),
    ]
)

center_crop_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(200),
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.data.img_mean, std=cfg.data.img_std),
    ]
)


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


def load_sae(device):
    weights_path = os.path.join(cfg.paths.results_dir, "sae_weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"SAE weights not found: {weights_path}")
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    sae.load_state_dict(torch.load(weights_path, map_location=device))
    sae.eval()
    return sae


def encode_text(tokenizer, text):
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=cfg.data.max_text_len,
        return_tensors="pt",
    )


def t2v_fused(model, images, input_ids, attention_mask):
    vision_seq = model.vision_encoder.forward_features(images)
    text_seq = model.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    attn_output, _ = model.cross_attn(
        query=text_seq,
        key=vision_seq,
        value=vision_seq,
        need_weights=False,
    )
    return attn_output.mean(dim=1)


def batch_records(df, tokenizer, transform, start, end):
    images = []
    input_ids = []
    masks = []
    paths = []
    diagnostics = []
    for _, row in df.iloc[start:end].iterrows():
        path = str(row["filepath"])
        image = Image.open(path).convert("RGB")
        enc = encode_text(tokenizer, str(row["clinical_history"]))
        images.append(transform(image))
        input_ids.append(enc["input_ids"].squeeze(0))
        masks.append(enc["attention_mask"].squeeze(0))
        paths.append(path)
        diagnostics.append(str(row["diagnostic"]).strip().upper())
    return paths, diagnostics, torch.stack(images), torch.stack(input_ids), torch.stack(masks)


def feature_activations(model, sae, df, tokenizer, transform, device):
    rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Feature activations", leave=False):
            end = min(start + BATCH_SIZE, len(df))
            paths, diagnostics, images, input_ids, masks = batch_records(
                df, tokenizer, transform, start, end
            )
            fused = t2v_fused(
                model,
                images.to(device),
                input_ids.to(device),
                masks.to(device),
            )
            acts = sae.encode(fused)[:, FEATURE_ID].detach().cpu().numpy()
            for path, diag, act in zip(paths, diagnostics, acts):
                rows.append({"filepath": path, "diagnostic": diag, "feature_activation": float(act)})
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("TASK 20: T->V FEATURE 1449 CENTER-CROP SENSITIVITY")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    model = load_model(device)
    sae = load_sae(device)

    df = pd.read_csv(cfg.paths.pad_ufes_csv)
    df = df[df["diagnostic"].astype(str).str.upper() != "NAN"].reset_index(drop=True)

    base_df = feature_activations(model, sae, df, tokenizer, baseline_transform, device)
    top_df = base_df.sort_values("feature_activation", ascending=False).head(TOP_K_IMAGES).copy()

    crop_input = df[df["filepath"].isin(set(top_df["filepath"]))].copy()
    crop_df = feature_activations(model, sae, crop_input, tokenizer, center_crop_transform, device)
    merged = top_df.rename(columns={"feature_activation": "base_activation"}).merge(
        crop_df[["filepath", "feature_activation"]].rename(
            columns={"feature_activation": "cropped_activation"}
        ),
        on="filepath",
        how="inner",
    )
    merged["activation_drop"] = merged["base_activation"] - merged["cropped_activation"]
    merged["relative_drop_pct"] = np.where(
        merged["base_activation"].abs() > 1e-8,
        100.0 * merged["activation_drop"] / merged["base_activation"],
        np.nan,
    )

    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    per_sample_path = os.path.join(cfg.paths.results_dir, "task20_visual_biopsy_leak.csv")
    summary_path = os.path.join(cfg.paths.results_dir, "task20_visual_biopsy_leak_summary.csv")
    merged.to_csv(per_sample_path, index=False)

    summary = {
        "model": "Cross-Attention T->V",
        "seed": SEED,
        "feature_id": FEATURE_ID,
        "n": int(len(merged)),
        "mean_base_activation": float(merged["base_activation"].mean()),
        "mean_cropped_activation": float(merged["cropped_activation"].mean()),
        "mean_activation_drop": float(merged["activation_drop"].mean()),
        "mean_relative_drop_pct": float(merged["relative_drop_pct"].mean()),
        "median_relative_drop_pct": float(merged["relative_drop_pct"].median()),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print(pd.DataFrame([summary]).to_string(index=False))
    print(f"Saved per-sample results to {per_sample_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
