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
RESULTS_DIR = Path("results")
SAMPLE_CSV = RESULTS_DIR / "task21b_generalized_grounding_audit_samples.csv"
SUMMARY_JSON = RESULTS_DIR / "task21b_generalized_grounding_audit_summary.json"
SUMMARY_CSV = RESULTS_DIR / "task21b_generalized_grounding_audit_summary.csv"

# Added class-specific descriptions for the extended audit
TEXT_CONDITIONS = {
    "MEL_consistent": (
        "Patient history is highly suspicious for invasive melanoma: rapid evolution, "
        "irregular pigmentation, asymmetric growth, and recent change."
    ),
    "NEV_consistent": (
        "Patient history is consistent with a stable benign melanocytic nevus: "
        "long-standing, symmetric, unchanged, and without alarming symptoms."
    ),
    "BCC_consistent": (
        "Patient history is consistent with a slow-growing basal cell carcinoma: "
        "pearly papule with telangiectasia, bleeding on minor trauma, and a rolled border."
    ),
    "SCC_consistent": (
        "Patient history is highly suspicious for squamous cell carcinoma: "
        "rapidly growing hyperkeratotic nodule with central ulceration and surrounding erythema."
    ),
    "ACK_consistent": (
        "Patient history is consistent with an actinic keratosis: "
        "rough, scaly, erythematous macule on chronically sun-damaged skin."
    ),
    "SEK_consistent": (
        "Patient history is consistent with a benign seborrheic keratosis: "
        "well-demarcated, waxy, stuck-on appearing plaque without malignant features."
    )
}

PAIRS = [("MEL", "NEV"), ("BCC", "SCC"), ("ACK", "SEK")]

def repo_root():
    return Path(__file__).resolve().parent

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
    image = Image.open(path).convert("RGB")
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

def class_margin(logits, class_a, class_b):
    idx_a = cfg.data.LABEL_MAP[class_a]
    idx_b = cfg.data.LABEL_MAP[class_b]
    score_a = float(logits[idx_a].item())
    score_b = float(logits[idx_b].item())
    return score_a, score_b, score_a - score_b

def safe_wilcoxon(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or np.allclose(values, 0):
        return None
    try:
        return float(stats.wilcoxon(values).pvalue)
    except ValueError:
        return None

def main():
    os.chdir(repo_root())
    RESULTS_DIR.mkdir(exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Task 21b generalized grounding audit on {device}.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    transform = get_transforms()
    model = load_model(device)

    df = pd.read_csv(cfg.paths.pad_ufes_csv)
    
    samples = []
    summaries = []

    print("Caching text encodings...")
    text_cache = cached_text_sequences(model, tokenizer, device)

    for pair in PAIRS:
        class_1, class_2 = pair
        print(f"\nProcessing pair: {class_1} vs {class_2}")
        
        pair_df = df[df["diagnostic"].isin([class_1, class_2])].copy()
        
        # Take N_PER_CLASS for each (if available, else min)
        min_available = min(len(pair_df[pair_df["diagnostic"] == class_1]), len(pair_df[pair_df["diagnostic"] == class_2]))
        n_to_sample = min(N_PER_CLASS, min_available)
        print(f"Sampling {n_to_sample} per class for {class_1}/{class_2}")
        
        c1_df = pair_df[pair_df["diagnostic"] == class_1].head(n_to_sample)
        c2_df = pair_df[pair_df["diagnostic"] == class_2].head(n_to_sample)
        audit_df = pd.concat([c1_df, c2_df], ignore_index=True)
        
        if len(audit_df) == 0:
            continue

        real_vision = cached_vision_sequences(model, audit_df, transform, device)

        with torch.no_grad():
            for row_idx, row in audit_df.iterrows():
                filepath = str(row["filepath"])
                true_class = str(row["diagnostic"]).strip().upper()
                
                vision_seq = real_vision[row_idx : row_idx + 1]
                
                text_c1 = text_cache[f"{class_1}_consistent"]
                text_c2 = text_cache[f"{class_2}_consistent"]
                
                logits_c1 = logits_from_cached(model, text_c1, vision_seq).squeeze(0).detach().cpu()
                logits_c2 = logits_from_cached(model, text_c2, vision_seq).squeeze(0).detach().cpu()
                
                _, _, margin_c1_text = class_margin(logits_c1, class_1, class_2)
                _, _, margin_c2_text = class_margin(logits_c2, class_1, class_2)
                
                pred_c1_text = class_1 if margin_c1_text > 0 else class_2
                pred_c2_text = class_1 if margin_c2_text > 0 else class_2
                
                # True class margin
                if true_class == class_1:
                    true_margin_aligned = margin_c1_text
                    true_margin_contrad = margin_c2_text
                    aligned_pred = pred_c1_text
                    contrad_pred = pred_c2_text
                else:
                    true_margin_aligned = -margin_c2_text
                    true_margin_contrad = -margin_c1_text
                    aligned_pred = pred_c2_text
                    contrad_pred = pred_c1_text
                
                samples.append({
                    "pair": f"{class_1}_{class_2}",
                    "filepath": filepath,
                    "true_class": true_class,
                    "aligned_true_margin": float(true_margin_aligned),
                    "contradictory_true_margin": float(true_margin_contrad),
                    "aligned_minus_contradictory_margin": float(true_margin_aligned - true_margin_contrad),
                    "aligned_pred": aligned_pred,
                    "contradictory_pred": contrad_pred,
                })

    df_samples = pd.DataFrame(samples)
    df_samples.to_csv(SAMPLE_CSV, index=False)

    for pair_name, group in df_samples.groupby("pair"):
        for true_class, sub_group in group.groupby("true_class"):
            shifts = sub_group["aligned_minus_contradictory_margin"].to_numpy(dtype=float)
            aligned_correct = (sub_group["aligned_pred"] == true_class).mean()
            contrad_correct = (sub_group["contradictory_pred"] == true_class).mean()
            
            summaries.append({
                "pair": pair_name,
                "true_class": true_class,
                "n": int(len(sub_group)),
                "mean_aligned_minus_contradictory_margin": float(np.mean(shifts)),
                "median_aligned_minus_contradictory_margin": float(np.median(shifts)),
                "wilcoxon_p_vs_zero": safe_wilcoxon(shifts),
                "aligned_binary_accuracy": float(aligned_correct),
                "contradictory_binary_accuracy": float(contrad_correct),
                "contradiction_flip_rate": float((sub_group["aligned_pred"] != sub_group["contradictory_pred"]).mean())
            })
            
        # Group summary
        shifts = group["aligned_minus_contradictory_margin"].to_numpy(dtype=float)
        summaries.append({
            "pair": pair_name,
            "true_class": "ALL",
            "n": int(len(group)),
            "mean_aligned_minus_contradictory_margin": float(np.mean(shifts)),
            "median_aligned_minus_contradictory_margin": float(np.median(shifts)),
            "wilcoxon_p_vs_zero": safe_wilcoxon(shifts),
            "aligned_binary_accuracy": float((group["aligned_pred"] == group["true_class"]).mean()),
            "contradictory_binary_accuracy": float((group["contradictory_pred"] == group["true_class"]).mean()),
            "contradiction_flip_rate": float((group["aligned_pred"] != group["contradictory_pred"]).mean())
        })

    pd.DataFrame(summaries).to_csv(SUMMARY_CSV, index=False)
    SUMMARY_JSON.write_text(json.dumps(summaries, indent=2))

    print("\nMatched grounding summary:")
    for row in summaries:
        if row["true_class"] == "ALL":
            print(
                f"{row['pair']:>7} | {row['true_class']:>3} | "
                f"N={row['n']:3d} | mean aligned-contradictory margin="
                f"{row['mean_aligned_minus_contradictory_margin']:+.4f} | "
                f"flip={row['contradiction_flip_rate']:.3f} | "
                f"p={row['wilcoxon_p_vs_zero']}"
            )
    print(f"\nSaved {SAMPLE_CSV}, {SUMMARY_CSV}, and {SUMMARY_JSON}.")

if __name__ == "__main__":
    main()
