"""
Task 27: Causal Mediation Analysis + Token-Level Grounding.

Improvements over baseline:
  - Correct Sobel test p-value using scipy.stats.norm directly (not sm.stats.zprob)
  - FDR (Benjamini-Hochberg) correction applied to mediation p-values
  - Span-level nulling in addition to token-level nulling
  - Reports only FDR-significant mediators
  - Saves comprehensive results with FDR column
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from scipy import stats as scipy_stats
import statsmodels.api as sm
from PIL import Image
from skimage import color

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE

# ─── ITA Computation ───────────────────────────────────────────────────────────

def compute_ita(image_tensor):
    """
    Compute Individual Typology Angle (ITA) from a normalised image tensor.
    image_tensor: (C, H, W) with ImageNet normalisation.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_unnorm = torch.clamp(image_tensor.cpu() * std + mean, 0, 1)
    img_np = img_unnorm.permute(1, 2, 0).numpy()

    h, w, _ = img_np.shape
    crop = img_np[h//4:3*h//4, w//4:3*w//4, :]

    lab = color.rgb2lab(crop)
    L   = lab[:, :, 0]
    b   = lab[:, :, 2]

    mask = (L > 5)
    L, b = L[mask], b[mask]
    if len(b) == 0:
        return 0.0

    ita = np.arctan((L - 50.0) / (b + 1e-8)) * (180.0 / np.pi)
    return float(np.mean(ita))


# ─── FDR (Benjamini-Hochberg) ─────────────────────────────────────────────────

def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Returns boolean mask: True = significant after BH FDR control."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_k = np.where(below)[0].max()
    result = np.zeros(n, dtype=bool)
    result[order[:max_k + 1]] = True
    return result


# ─── Token-Level & Span-Level Nulling ─────────────────────────────────────────

def token_level_nulling(model, tokenizer, batch, device):
    """
    Perform token-level and span-level nulling on clinical history tokens.

    For each sample, ablates:
      - age tokens (digits, 'years', 'old')
      - gender tokens ('male', 'female', ...)
      - location spans (the full noun phrase after 'on the')
    
    Measures per-category logit drop relative to baseline.
    """
    model.eval()
    imgs      = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels    = batch["label"].to(device)

    with torch.no_grad():
        base_logits, _, _ = model(imgs, input_ids, attn_mask)

    pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    results = []

    for i in range(len(imgs)):
        text      = batch["clinical_history"][i]
        label_idx = labels[i].item()
        base_logit = base_logits[i, label_idx].item()

        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=cfg.data.max_text_len,
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        # Categorize tokens
        age_tokens      = []
        gender_tokens   = []
        location_tokens = []
        in_location     = False

        for tok_idx, tok in enumerate(tokens):
            tl = tok.lower().replace("Ġ", "").replace("▁", "").strip()
            if tl in ("age", "years", "old") or tl.isdigit():
                age_tokens.append(tok_idx)
            if tl in ("male", "female", "man", "woman", "boy", "girl"):
                gender_tokens.append(tok_idx)
            # Location span: starts after 'on' + 'the', ends at period/comma
            if tl in ("on",):
                in_location = True
            elif in_location and tl in (".", ",", ";", ""):
                in_location = False
            elif in_location:
                location_tokens.append(tok_idx)

        def apply_nulling(target_indices, category):
            if not target_indices:
                return
            nulled_ids  = input_ids[i:i+1].clone()
            nulled_mask = attn_mask[i:i+1].clone()
            for idx in target_indices:
                if idx < nulled_ids.shape[1]:
                    nulled_ids[0, idx]  = pad_id
                    nulled_mask[0, idx] = 0
            with torch.no_grad():
                new_logits, _, _ = model(imgs[i:i+1], nulled_ids, nulled_mask)
            new_logit  = new_logits[0, label_idx].item()
            logit_drop = base_logit - new_logit
            results.append({
                "sample_idx": batch["idx"][i].item(),
                "category":   category,
                "n_tokens_nulled": len(target_indices),
                "base_logit":     base_logit,
                "nulled_logit":   new_logit,
                "logit_drop":     logit_drop,
                "abs_logit_drop": abs(logit_drop),
            })

        apply_nulling(age_tokens,      "age")
        apply_nulling(gender_tokens,   "gender")
        apply_nulling(location_tokens, "location_span")
        # Full text ablation (all non-special tokens) as upper bound
        all_content = [j for j, t in enumerate(tokens)
                       if t not in (tokenizer.cls_token, tokenizer.sep_token,
                                    "[CLS]", "[SEP]", "<s>", "</s>")
                          and j > 0]
        apply_nulling(all_content, "full_text")

    return results


# ─── Causal Mediation Analysis ────────────────────────────────────────────────

def run_causal_mediation(df_mediation: pd.DataFrame) -> pd.DataFrame:
    """
    Run causal mediation analysis using OLS + Sobel test.
    Path: ITA (X) → SAE Feature (M) → Output Logit (Y)

    Returns a DataFrame with FDR-corrected significance flags.
    """
    print("\nRunning Causal Mediation Analysis...")
    results = []

    df_mediation = df_mediation.copy()
    df_mediation["X"] = (
        (df_mediation["ITA"] - df_mediation["ITA"].mean()) /
        (df_mediation["ITA"].std() + 1e-8)
    )
    df_mediation["Y"] = (
        (df_mediation["Logit"] - df_mediation["Logit"].mean()) /
        (df_mediation["Logit"].std() + 1e-8)
    )
    df_mediation = df_mediation.replace([np.inf, -np.inf], np.nan).dropna(subset=["X", "Y"])

    for feature_id in tqdm(df_mediation["Feature_ID"].unique(), desc="Mediation analysis", leave=False):
        df_f = df_mediation[df_mediation["Feature_ID"] == feature_id].copy()

        if df_f["Activation"].std() < 1e-8:
            continue

        df_f["M"] = (df_f["Activation"] - df_f["Activation"].mean()) / (df_f["Activation"].std() + 1e-8)
        df_f = df_f.replace([np.inf, -np.inf], np.nan).dropna(subset=["X", "Y", "M"])

        if len(df_f) < 2:
            continue

        # Step 1: Mediator model M ~ X
        X_c = sm.add_constant(df_f["X"])
        med_model = sm.OLS(df_f["M"], X_c).fit()
        a, se_a = med_model.params["X"], med_model.bse["X"]

        # Step 2: Outcome model Y ~ X + M
        XM_c = sm.add_constant(df_f[["X", "M"]])
        out_model = sm.OLS(df_f["Y"], XM_c).fit()
        b      = out_model.params.get("M", 0.0)
        c_prime = out_model.params["X"]
        se_b   = out_model.bse.get("M", 0.0)

        # Step 3: Total effect Y ~ X
        tot_model = sm.OLS(df_f["Y"], X_c).fit()
        c = tot_model.params["X"]

        # Indirect effect and Sobel test
        ab = a * b
        sobel_se = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)
        if sobel_se > 0:
            sobel_z = ab / sobel_se
            # Correct p-value via scipy (two-tailed)
            p_val = float(2 * scipy_stats.norm.sf(abs(sobel_z)))
        else:
            sobel_z, p_val = np.nan, np.nan

        results.append({
            "Feature_ID":          int(feature_id),
            "Total_Effect_c":      float(c),
            "Direct_Effect_cP":    float(c_prime),
            "Indirect_Effect_ab":  float(ab),
            "Proportion_Mediated": float(ab / c) if c != 0 else np.nan,
            "Sobel_Z":             float(sobel_z) if not np.isnan(sobel_z) else np.nan,
            "p_value_raw":         p_val,
        })

    if not results:
        print("  No mediators found.")
        return pd.DataFrame()

    df_res = pd.DataFrame(results)

    # Apply FDR correction
    p_arr = df_res["p_value_raw"].fillna(1.0).values
    sig_mask = fdr_bh(p_arr, alpha=0.05)
    df_res["FDR_significant"] = sig_mask

    n_sig = sig_mask.sum()
    print(f"\nFDR correction (BH, alpha=0.05): {n_sig}/{len(df_res)} features are significant mediators.")

    df_res = df_res.sort_values("Proportion_Mediated", ascending=False, na_position="last")

    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.results_dir, "task27_mediation_results.csv")
    df_res.to_csv(out_path, index=False)
    print(f"Mediation results saved to {out_path}")
    print("\nTop 10 mediators:")
    print(df_res.head(10).to_string(index=False))

    # Print FDR-significant only
    fdr_df = df_res[df_res["FDR_significant"]]
    if not fdr_df.empty:
        print(f"\nFDR-significant mediators ({len(fdr_df)}):")
        print(fdr_df.to_string(index=False))
    else:
        print("\nNo features survive FDR correction. Results are exploratory only.")

    return df_res


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TASK 27: CAUSAL MEDIATION ANALYSIS + TOKEN-LEVEL GROUNDING")
    print("=" * 70)

    device    = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

    # Monkeypatch dataset to return clinical history and index
    original_getitem = MultimodalDermatologyDataset.__getitem__

    def new_getitem(self, idx):
        item = original_getitem(self, idx)
        item["clinical_history"] = str(self.df.iloc[idx].get("clinical_history", ""))
        item["idx"]              = torch.tensor(idx, dtype=torch.long)
        return item

    MultimodalDermatologyDataset.__getitem__ = new_getitem

    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load classifier model
    seed = 1337
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_{seed}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    model = CrossAttentionT2VClassifier().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    # Load SAE (optional)
    sae_path = os.path.join(cfg.paths.results_dir, "sae_weights.pth")
    sae = None
    if os.path.exists(sae_path):
        sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
        sae.load_state_dict(torch.load(sae_path, map_location=device))
        sae.eval()
        print("SAE weights loaded successfully.")
    else:
        print("Warning: SAE weights not found at results/sae_weights.pth")
        print("  Run task11_sae.py first to generate SAE weights.")
        print("  Proceeding with token-nulling analysis only.")

    nulling_results = []
    mediation_data  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            # 1. Token / span-level nulling
            batch_null_results = token_level_nulling(model, tokenizer, batch, device)
            nulling_results.extend(batch_null_results)

            # 2. Collect SAE mediation data
            if sae is not None:
                imgs      = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                labels    = batch["label"].to(device)

                itas = [compute_ita(imgs[i]) for i in range(len(imgs))]

                logits, _, _ = model(imgs, input_ids, attn_mask)

                vision_seq  = model.vision_encoder.forward_features(imgs)
                text_outs   = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
                text_seq    = text_outs.last_hidden_state

                attn_out, _ = model.cross_attn(
                    query=text_seq, key=vision_seq, value=vision_seq, need_weights=False
                )
                fused = attn_out.mean(dim=1)

                _, sparse_acts = sae(fused)
                sparse_acts_np = sparse_acts.cpu().numpy()

                for i in range(len(imgs)):
                    label_idx = labels[i].item()
                    logit     = logits[i, label_idx].item()
                    ita       = itas[i]
                    top_feats = np.argsort(sparse_acts_np[i])[::-1][:10]

                    for f in top_feats:
                        mediation_data.append({
                            "Sample_idx": batch["idx"][i].item(),
                            "Feature_ID": int(f),
                            "ITA":        ita,
                            "Logit":      logit,
                            "Activation": float(sparse_acts_np[i, f]),
                        })

    # ── Save token nulling results ─────────────────────────────────────────────
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    df_null = pd.DataFrame(nulling_results)
    null_path = os.path.join(cfg.paths.results_dir, "task27_token_nulling.csv")
    df_null.to_csv(null_path, index=False)

    print("\n" + "=" * 70)
    print("TOKEN NULLING SUMMARY")
    print("=" * 70)
    if not df_null.empty:
        summary = df_null.groupby("category").agg(
            n=("logit_drop", "count"),
            mean_logit_drop=("logit_drop", "mean"),
            std_logit_drop=("logit_drop", "std"),
            mean_abs_drop=("abs_logit_drop", "mean"),
        )
        print(summary.to_string())

        # Test: is each ablation's logit drop significantly different from zero?
        print("\nSignificance of logit drops (one-sample t-test vs 0):")
        for cat, grp in df_null.groupby("category"):
            drops = grp["logit_drop"].dropna().values
            if len(drops) > 1:
                t, p = scipy_stats.ttest_1samp(drops, popmean=0.0)
                print(f"  {cat:<20s}: mean={drops.mean():+.4f}  t={t:.3f}  p={p:.4e}  {'*' if p < 0.05 else ''}")

    # ── Run mediation analysis if SAE available ──────────────────────────────
    if sae is not None and mediation_data:
        df_med = pd.DataFrame(mediation_data)
        run_causal_mediation(df_med)
    else:
        print("\nSkipping mediation analysis (SAE not available or no data).")

    print("\nTask 27 complete.")


if __name__ == "__main__":
    main()
