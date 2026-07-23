import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import stats
from transformers import AutoTokenizer

from config import cfg
from dataset import get_transforms
from models.cross_attention import CrossAttentionT2VClassifier


SEED = 456
N_PER_CLASS = 52
RESULTS_DIR = Path(cfg.paths.results_dir)
SAMPLE_CSV = RESULTS_DIR / "task21_matched_grounding_audit_samples.csv"
SUMMARY_JSON = RESULTS_DIR / "task21_matched_grounding_audit_summary.json"
SUMMARY_CSV = RESULTS_DIR / "task21_matched_grounding_audit_summary.csv"

TEXT_CONDITIONS = {
    "MEL_consistent": (
        "Patient history is highly suspicious for invasive melanoma: rapid evolution, "
        "irregular pigmentation, asymmetric growth, and recent change."
    ),
    "NEV_consistent": (
        "Patient history is consistent with a stable benign melanocytic nevus: "
        "long-standing, symmetric, unchanged, and without alarming symptoms."
    ),
}


def load_model(device):
    model = CrossAttentionT2VClassifier().to(device)
    ckpt_path = (
        Path(cfg.paths.checkpoint_dir)
        / f"Cross-Attention_T\u2192V_seed_{SEED}"
        / "best_model.pth"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def encode_text(tokenizer, text, device):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=cfg.data.max_text_len,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def load_image(path, transform, device):
    path = Path(path)
    if not path.is_absolute():
        path = Path(cfg.paths.package_root) / path
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def blank_image(transform, device):
    image = Image.new("RGB", (cfg.data.img_size, cfg.data.img_size), color="black")
    return transform(image).unsqueeze(0).to(device)


def cached_text_sequences(model, tokenizer, device):
    cache = {}
    with torch.no_grad():
        for text_condition, text in TEXT_CONDITIONS.items():
            input_ids, attention_mask = encode_text(tokenizer, text, device)
            text_outputs = model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cache[text_condition] = text_outputs.last_hidden_state.detach()
    return cache


def cached_vision_sequences(model, audit_df, transform, device, batch_size=16):
    all_sequences = []
    paths = audit_df["filepath"].astype(str).tolist()
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [
                load_image(path, transform, device).squeeze(0)
                for path in batch_paths
            ]
            image_batch = torch.stack(images, dim=0)
            all_sequences.append(model.vision_encoder.forward_features(image_batch).detach())
    return torch.cat(all_sequences, dim=0)


def logits_from_cached(model, text_seq, vision_seq):
    batch_size = vision_seq.shape[0]
    query = text_seq.expand(batch_size, -1, -1)
    attn_output, _ = model.cross_attn(
        query=query,
        key=vision_seq,
        value=vision_seq,
        need_weights=False,
    )
    fused_repr = attn_output.mean(dim=1)
    return model.classifier(fused_repr)


def mel_nev_scores(logits):
    mel_idx = cfg.data.LABEL_MAP["MEL"]
    nev_idx = cfg.data.LABEL_MAP["NEV"]
    mel = float(logits[mel_idx].item())
    nev = float(logits[nev_idx].item())
    return mel, nev, mel - nev


def true_margin(row):
    if row["true_class"] == "MEL":
        return row["mel_minus_nev_margin"]
    return -row["mel_minus_nev_margin"]


def safe_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.allclose(values, 0):
        return None
    try:
        return float(stats.wilcoxon(values).pvalue)
    except ValueError:
        return None


def summarize(samples):
    df = pd.DataFrame(samples)
    df["true_class_margin"] = df.apply(true_margin, axis=1)

    paired_rows = []
    for (filepath, image_condition, true_class), group in df.groupby(
        ["filepath", "image_condition", "true_class"]
    ):
        by_text = group.set_index("text_condition")
        aligned_key = f"{true_class}_consistent"
        contrad_key = "NEV_consistent" if true_class == "MEL" else "MEL_consistent"
        if aligned_key not in by_text.index or contrad_key not in by_text.index:
            continue
        aligned = by_text.loc[aligned_key]
        contrad = by_text.loc[contrad_key]
        paired_rows.append(
            {
                "filepath": filepath,
                "image_condition": image_condition,
                "true_class": true_class,
                "aligned_true_margin": float(aligned["true_class_margin"]),
                "contradictory_true_margin": float(contrad["true_class_margin"]),
                "aligned_minus_contradictory_margin": float(
                    aligned["true_class_margin"] - contrad["true_class_margin"]
                ),
                "aligned_pred": aligned["binary_pred"],
                "contradictory_pred": contrad["binary_pred"],
            }
        )

    paired = pd.DataFrame(paired_rows)
    summaries = []
    for (image_condition, true_class), group in paired.groupby(
        ["image_condition", "true_class"]
    ):
        shifts = group["aligned_minus_contradictory_margin"].to_numpy(dtype=float)
        aligned_correct = (group["aligned_pred"] == true_class).mean()
        contrad_correct = (group["contradictory_pred"] == true_class).mean()
        summaries.append(
            {
                "image_condition": image_condition,
                "true_class": true_class,
                "n": int(len(group)),
                "mean_aligned_minus_contradictory_margin": float(np.mean(shifts)),
                "median_aligned_minus_contradictory_margin": float(np.median(shifts)),
                "wilcoxon_p_vs_zero": safe_wilcoxon(shifts),
                "aligned_binary_accuracy": float(aligned_correct),
                "contradictory_binary_accuracy": float(contrad_correct),
                "contradiction_flip_rate": float(
                    (group["aligned_pred"] != group["contradictory_pred"]).mean()
                ),
            }
        )

    for image_condition, group in paired.groupby("image_condition"):
        shifts = group["aligned_minus_contradictory_margin"].to_numpy(dtype=float)
        summaries.append(
            {
                "image_condition": image_condition,
                "true_class": "ALL",
                "n": int(len(group)),
                "mean_aligned_minus_contradictory_margin": float(np.mean(shifts)),
                "median_aligned_minus_contradictory_margin": float(np.median(shifts)),
                "wilcoxon_p_vs_zero": safe_wilcoxon(shifts),
                "aligned_binary_accuracy": float(
                    (group["aligned_pred"] == group["true_class"]).mean()
                ),
                "contradictory_binary_accuracy": float(
                    (group["contradictory_pred"] == group["true_class"]).mean()
                ),
                "contradiction_flip_rate": float(
                    (group["aligned_pred"] != group["contradictory_pred"]).mean()
                ),
            }
        )

    return paired, summaries


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Task 21 matched grounding audit on {device}.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    transform = get_transforms()
    model = load_model(device)

    df = pd.read_csv(cfg.paths.pad_ufes_csv)
    df = df[df["diagnostic"].isin(["MEL", "NEV"])].copy()
    mel_df = df[df["diagnostic"] == "MEL"].head(N_PER_CLASS)
    nev_df = df[df["diagnostic"] == "NEV"].head(N_PER_CLASS)
    audit_df = pd.concat([mel_df, nev_df], ignore_index=True)

    print("Caching text and image encodings...")
    text_cache = cached_text_sequences(model, tokenizer, device)
    real_vision = cached_vision_sequences(model, audit_df, transform, device)
    blank_vision_single = model.vision_encoder.forward_features(blank_image(transform, device))
    blank_vision = blank_vision_single.expand(real_vision.shape[0], -1, -1).detach()

    samples = []

    with torch.no_grad():
        for row_idx, row in audit_df.iterrows():
            filepath = str(row["filepath"])
            true_class = str(row["diagnostic"]).strip().upper()
            image_conditions = {
                "real_image": real_vision[row_idx : row_idx + 1],
                "blank_image": blank_vision[row_idx : row_idx + 1],
            }

            for image_condition, vision_seq in image_conditions.items():
                for text_condition, text_seq in text_cache.items():
                    logits = logits_from_cached(model, text_seq, vision_seq)
                    logits = logits.squeeze(0).detach().cpu()
                    mel_logit, nev_logit, margin = mel_nev_scores(logits)
                    binary_pred = "MEL" if margin > 0 else "NEV"
                    full_pred_idx = int(torch.argmax(logits).item())
                    full_pred = {
                        idx: label for label, idx in cfg.data.LABEL_MAP.items()
                    }[full_pred_idx]

                    samples.append(
                        {
                            "filepath": filepath,
                            "true_class": true_class,
                            "image_condition": image_condition,
                            "text_condition": text_condition,
                            "mel_logit": mel_logit,
                            "nev_logit": nev_logit,
                            "mel_minus_nev_margin": margin,
                            "binary_pred": binary_pred,
                            "full_6way_pred": full_pred,
                        }
                    )

    paired, summaries = summarize(samples)

    with SAMPLE_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        writer.writeheader()
        writer.writerows(samples)

    paired.to_csv(RESULTS_DIR / "task21_matched_grounding_audit_pairs.csv", index=False)
    pd.DataFrame(summaries).to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summaries, indent=2))

    print("\nMatched grounding summary:")
    for row in summaries:
        print(
            f"{row['image_condition']:>11} | {row['true_class']:>3} | "
            f"N={row['n']:3d} | mean aligned-contradictory margin="
            f"{row['mean_aligned_minus_contradictory_margin']:+.4f} | "
            f"flip={row['contradiction_flip_rate']:.3f} | "
            f"p={row['wilcoxon_p_vs_zero']}"
        )
    print(f"\nSaved {SAMPLE_CSV}, {SUMMARY_CSV}, and {SUMMARY_JSON}.")


if __name__ == "__main__":
    main()
