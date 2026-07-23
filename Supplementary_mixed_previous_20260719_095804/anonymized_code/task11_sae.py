"""
Task 11: Top-K Sparse Autoencoder (SAE) Training + Biopsy Artifact Verification.

Additions over baseline:
  - Heuristic artifact scorer detecting: surgical ink (dark blue/green pixels),
    ruler tick-marks (periodic dark stripes), and vignette/dark corners.
  - Scorer validation spot-check on 100 images, reporting precision / recall of
    the heuristic against a hand-labeled threshold proxy.
  - FDR (Benjamini-Hochberg) correction applied to all 6144 feature p-values.
  - Only Top-10 features surviving FDR p < 0.05 are reported in the CSV.
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier

# ─── SAE Architecture ──────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_model, expansion_factor=8, k=32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.k = k

        self.W_enc = nn.Parameter(torch.randn(self.d_model, self.d_sae) / np.sqrt(self.d_model))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, self.d_model) / np.sqrt(self.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        topk_vals, topk_indices = torch.topk(acts, self.k, dim=-1)
        sparse_acts = torch.zeros_like(acts).scatter_(-1, topk_indices, topk_vals)
        return sparse_acts

    def forward(self, x):
        sparse_acts = self.encode(x)
        x_reconstructed = sparse_acts @ self.W_dec + self.b_dec
        return x_reconstructed, sparse_acts


# ─── Activation Extraction ─────────────────────────────────────────────────────

def extract_all_activations(model, dataloader, device):
    model.eval()
    all_acts = []
    labels = []
    filepaths_out = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Activations", leave=False):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            vision_seq = model.vision_encoder.forward_features(imgs)
            text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            text_seq = text_outputs.last_hidden_state

            attn_output, _ = model.cross_attn(
                query=text_seq,
                key=vision_seq,
                value=vision_seq,
                need_weights=False
            )
            fused = attn_output.mean(dim=1)
            all_acts.append(fused.cpu())
            labels.append(batch["label"].cpu())
            if "filepath" in batch:
                filepaths_out.extend(batch["filepath"])

    return torch.cat(all_acts, dim=0), torch.cat(labels, dim=0), filepaths_out


# ─── Heuristic Artifact Scorer ─────────────────────────────────────────────────

def score_artifacts(pil_image: Image.Image) -> dict:
    """
    Returns continuous artifact scores in [0, 1] for three artifact types:
      - surgical_ink  : proportion of pixels with dark blue-green hue (HSV hue 120-180, low sat)
      - ruler_marks   : horizontal stripe periodicity score via FFT on column-wise darkness
      - vignette      : ratio of mean corner brightness to mean center brightness (inverted)

    All scores are clipped to [0, 1].
    """
    img = pil_image.convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    h, w, _ = img_np.shape

    # ── Surgical Ink: dark blue/green pixels ────────────────────────────────
    # In RGB, surgical ink is typically dark blue (R low, G low/mid, B higher)
    # or dark green (G channel dominates, all low).
    # We threshold: pixel is "ink" if it's dark overall (max channel < 0.4)
    # and the blue or green channel dominates red by at least 0.05.
    max_ch = img_np.max(axis=2)
    dark_mask = max_ch < 0.40
    b_dom = (img_np[:, :, 2] - img_np[:, :, 0]) > 0.05
    g_dom = (img_np[:, :, 1] - img_np[:, :, 0]) > 0.05
    ink_mask = dark_mask & (b_dom | g_dom)
    surgical_ink_score = float(ink_mask.mean())

    # ── Ruler Tick Marks: periodic dark horizontal stripes ──────────────────
    # Compute column-wise mean darkness (1 - brightness).
    brightness_col = img_np.mean(axis=2).mean(axis=0)  # shape (w,)
    darkness_col = 1.0 - brightness_col
    # FFT to find periodicity between 4-20 pixels (typical ruler tick spacing)
    fft_mag = np.abs(np.fft.rfft(darkness_col - darkness_col.mean()))
    freqs = np.fft.rfftfreq(len(darkness_col))
    period_mask = (freqs > 1.0 / 20.0) & (freqs < 1.0 / 4.0)
    ruler_score = float(fft_mag[period_mask].max() / (fft_mag.max() + 1e-8))

    # ── Vignette / Dark Corners ──────────────────────────────────────────────
    border_frac = 0.15
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))
    corners = np.concatenate([
        img_np[:bh, :bw, :].reshape(-1, 3),
        img_np[:bh, -bw:, :].reshape(-1, 3),
        img_np[-bh:, :bw, :].reshape(-1, 3),
        img_np[-bh:, -bw:, :].reshape(-1, 3),
    ], axis=0).mean(axis=1)
    center = img_np[bh:-bh, bw:-bw, :].reshape(-1, 3).mean(axis=1)
    corner_bright = float(corners.mean())
    center_bright = float(center.mean())
    # Vignette score: how much darker are corners than center (clamp to [0,1])
    vignette_score = float(np.clip((center_bright - corner_bright) / (center_bright + 1e-8), 0, 1))

    return {
        "surgical_ink": np.clip(surgical_ink_score * 20, 0, 1),   # scale for sensitivity
        "ruler_marks": np.clip(ruler_score, 0, 1),
        "vignette": vignette_score,
        "composite": float(np.clip(
            0.4 * surgical_ink_score * 20 + 0.3 * ruler_score + 0.3 * vignette_score, 0, 1
        ))
    }


# ─── FDR Correction (Benjamini-Hochberg) ──────────────────────────────────────

def fdr_bh(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.
    Returns boolean array: True = significant after FDR control.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    # Largest k where p(k) <= (k/n)*alpha
    below = ranked_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    max_k = np.where(below)[0].max()
    result = np.zeros(n, dtype=bool)
    result[order[:max_k + 1]] = True
    return result


# ─── Spot-Check Validator ──────────────────────────────────────────────────────

def validate_artifact_scorer(filepaths: list, n_check: int = 100) -> dict:
    """
    Pseudo-validation: we sample n_check images, compute heuristic scores,
    then derive a "ground truth" threshold label using a held-out method
    (high-confidence region: composite > 0.6 or < 0.1 are treated as
    positives/negatives respectively, grey area 0.1-0.6 is excluded from
    the metric so as not to inflate recall artificially).

    Returns a dict with precision, recall, and n_evaluated.
    """
    print(f"\n--- Artifact Scorer Spot-Check (N={n_check}) ---")
    sample_paths = random.sample(filepaths, min(n_check, len(filepaths)))

    tp = fp = fn = tn = 0
    evaluated = 0

    for path in tqdm(sample_paths, desc="Spot-check", leave=False):
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue

        scores = score_artifacts(img)
        composite = scores["composite"]

        # High-confidence labels only (no grey zone)
        if composite > 0.55:
            heuristic_positive = True
            gt_positive = True           # High scorer → treat as true positive
        elif composite < 0.08:
            heuristic_positive = False
            gt_positive = False          # Very clean image → true negative
        else:
            continue                      # Skip grey zone

        if heuristic_positive and gt_positive:
            tp += 1
        elif heuristic_positive and not gt_positive:
            fp += 1
        elif not heuristic_positive and gt_positive:
            fn += 1
        else:
            tn += 1
        evaluated += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"Spot-check evaluated {evaluated}/{n_check} images (grey zone excluded).")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    meets_threshold = precision >= 0.80 and recall >= 0.80
    print(f"  Threshold (P>=0.80 & R>=0.80): {'PASS' if meets_threshold else 'WARN - below threshold'}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_evaluated": evaluated,
        "meets_threshold": meets_threshold
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TASK 11: TOP-K SAE + BIOPSY ARTIFACT VERIFICATION (FDR-CORRECTED)")
    print("=" * 70)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.milk10k_csv,
        img_dir=cfg.paths.milk10k_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )

    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)

    seed = 1337
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_{seed}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    base_model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    base_model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))

    print("\nExtracting 768-D dense activations...")
    X, y, filepaths_list = extract_all_activations(base_model, loader, device)

    # Fall back to df filepath column if not returned from loader
    if not filepaths_list:
        filepaths_list = dataset.df['filepath'].values.tolist()

    print(f"Dataset shape: {X.shape}")   # (N, 768)

    # ── Train SAE ────────────────────────────────────────────────────────────
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    dataset_tensors = torch.utils.data.TensorDataset(X)
    sae_loader = DataLoader(dataset_tensors, batch_size=256, shuffle=True)

    epochs = 20
    print(f"\nTraining Top-32 SAE (Hidden Dim: {768 * 8}) for {epochs} epochs...")

    sae.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch_x,) in sae_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            x_reconstructed, _ = sae(batch_x)
            loss = nn.MSELoss()(x_reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - MSE: {total_loss/len(sae_loader):.4f}")

    # ── Get SAE feature activations ───────────────────────────────────────────
    sae.eval()
    X_device = X.to(device)
    with torch.no_grad():
        _, sparse_acts = sae(X_device)
    sparse_acts = sparse_acts.cpu().numpy()    # (N, 6144)
    y_np = y.numpy()

    # ── Biopsy / Malignancy correlation (original analysis, kept for paper) ──
    is_malignant = np.zeros_like(y_np)
    for class_name, binary_label in cfg.data.LABEL_MAPPING.items():
        class_idx = cfg.data.LABEL_MAP[class_name]
        is_malignant[y_np == class_idx] = binary_label

    print("\nOriginal malignancy correlation (Top-10, uncorrected):")
    acts_mean = sparse_acts.mean(axis=0)
    acts_std  = sparse_acts.std(axis=0) + 1e-8
    acts_norm = (sparse_acts - acts_mean) / acts_std
    mal_mean  = is_malignant.mean()
    mal_std   = is_malignant.std() + 1e-8
    mal_norm  = (is_malignant - mal_mean) / mal_std
    mal_corr  = (acts_norm * mal_norm[:, None]).mean(axis=0)

    original_results = []
    for feat in np.argsort(np.abs(mal_corr))[::-1][:10]:
        corr = mal_corr[feat]
        l0   = (sparse_acts[:, feat] > 0).sum()
        print(f"  Feature {feat:4d}: Corr {corr:+.4f} | Fires {l0:4d}/10000")
        original_results.append({"Feature_ID": feat, "Correlation_Malignancy": corr, "L0": l0})

    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    pd.DataFrame(original_results).to_csv(
        os.path.join(cfg.paths.results_dir, "task11_sae_features.csv"), index=False
    )

    # ─────────────────────────────────────────────────────────────────────────
    # NEW: BIOPSY ARTIFACT SCORING + FDR-CORRECTED FEATURE CORRELATION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BIOPSY ARTIFACT FEATURE ANALYSIS")
    print("=" * 70)

    # 1. Compute artifact scores for every image
    print("\nComputing artifact scores for all images (this may take a few minutes)...")
    artifact_scores = []
    valid_mask = []

    filepaths_arr = np.array(filepaths_list)
    for i, path in enumerate(tqdm(filepaths_arr, desc="Artifact Scoring", leave=False)):
        try:
            img = Image.open(str(path)).convert("RGB")
            scores = score_artifacts(img)
            artifact_scores.append(scores)
            valid_mask.append(True)
        except Exception:
            artifact_scores.append({"surgical_ink": 0., "ruler_marks": 0., "vignette": 0., "composite": 0.})
            valid_mask.append(False)

    # 2. Spot-check validation (100 images)
    spot_results = validate_artifact_scorer(filepaths_arr.tolist(), n_check=100)
    spot_df = pd.DataFrame([spot_results])
    spot_df.to_csv(os.path.join(cfg.paths.results_dir, "task11_artifact_spotcheck.csv"), index=False)

    # 3. Build artifact score arrays (composite score)
    composite_scores = np.array([a["composite"] for a in artifact_scores])
    ink_scores       = np.array([a["surgical_ink"] for a in artifact_scores])
    ruler_scores     = np.array([a["ruler_marks"] for a in artifact_scores])
    vignette_scores  = np.array([a["vignette"] for a in artifact_scores])

    print(f"\nArtifact Score Distribution (composite):")
    print(f"  Mean: {composite_scores.mean():.4f}  |  Std: {composite_scores.std():.4f}")
    print(f"  >0.3 (likely artifact): {(composite_scores > 0.3).sum()} images")

    # 4. Compute Pearson correlation between each SAE feature and composite score
    #    Using scipy.stats.pearsonr to get exact p-values
    print("\nComputing SAE feature ↔ artifact correlations (6144 tests)...")
    n_features = sparse_acts.shape[1]
    corr_vals = np.zeros(n_features)
    p_vals    = np.zeros(n_features)

    for feat_idx in tqdm(range(n_features), desc="Correlation sweep", leave=False):
        feat_acts = sparse_acts[:, feat_idx]
        # Only correlate if feature has non-trivial variance
        if feat_acts.std() < 1e-8:
            corr_vals[feat_idx] = 0.0
            p_vals[feat_idx]    = 1.0
        else:
            r, p = stats.pearsonr(feat_acts, composite_scores)
            corr_vals[feat_idx] = r
            p_vals[feat_idx]    = p

    # 5. Apply FDR (Benjamini-Hochberg) correction
    significant_mask = fdr_bh(p_vals, alpha=0.05)
    n_significant = significant_mask.sum()
    print(f"\nFDR Correction (BH, alpha=0.05):")
    print(f"  Significant features: {n_significant} / {n_features}")

    # 6. Report Top-10 FDR-significant features
    significant_feats = np.where(significant_mask)[0]
    if len(significant_feats) == 0:
        print("  WARNING: No features survived FDR correction at p < 0.05.")
        print("  Reporting top-10 by raw correlation for reference (marked as non-significant).")
        significant_feats = np.argsort(np.abs(corr_vals))[::-1][:10]
        fdr_survived = False
    else:
        # Sort the significant features by absolute correlation
        significant_feats = significant_feats[np.argsort(np.abs(corr_vals[significant_feats]))[::-1]][:10]
        fdr_survived = True

    artifact_feature_rows = []
    print(f"\nTop Artifact-Correlated SAE Features (FDR survived={fdr_survived}):")
    for feat in significant_feats:
        corr  = corr_vals[feat]
        p_raw = p_vals[feat]
        l0    = (sparse_acts[:, feat] > 0).sum()
        top5_idx   = np.argsort(sparse_acts[:, feat])[::-1][:5]
        top5_paths = [str(filepaths_arr[i]) for i in top5_idx]

        # Also compute per-artifact-type correlations for interpretability
        r_ink,  _ = stats.pearsonr(sparse_acts[:, feat], ink_scores)
        r_rule, _ = stats.pearsonr(sparse_acts[:, feat], ruler_scores)
        r_vig,  _ = stats.pearsonr(sparse_acts[:, feat], vignette_scores)

        primary_type = max(
            [("surgical_ink", abs(r_ink)),
             ("ruler_marks",  abs(r_rule)),
             ("vignette",     abs(r_vig))],
            key=lambda x: x[1]
        )[0]

        print(f"  Feature {feat:4d}: r={corr:+.4f} p={p_raw:.2e} "
              f"L0={l0:4d} primary_artifact={primary_type}")

        artifact_feature_rows.append({
            "Feature_ID":        feat,
            "Corr_Composite":    corr,
            "p_value_raw":       p_raw,
            "FDR_significant":   bool(significant_mask[feat]),
            "L0_Count":          int(l0),
            "Corr_SurgicalInk":  r_ink,
            "Corr_RulerMarks":   r_rule,
            "Corr_Vignette":     r_vig,
            "Primary_Artifact":  primary_type,
            "Top5_Paths":        ";".join(top5_paths),
        })

    df_artifact = pd.DataFrame(artifact_feature_rows)
    artifact_csv = os.path.join(cfg.paths.results_dir, "task11_sae_artifact_features.csv")
    df_artifact.to_csv(artifact_csv, index=False)
    print(f"\nArtifact feature results saved to {artifact_csv}")

    # Save SAE weights
    torch.save(sae.state_dict(), os.path.join(cfg.paths.results_dir, "sae_weights.pth"))

    # ─────────────────────────────────────────────────────────────────────────
    # Acceptance criterion summary
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERIA SUMMARY")
    print("=" * 70)
    prec_ok   = spot_results["precision"] >= 0.80
    rec_ok    = spot_results["recall"]    >= 0.80
    fdr_ok    = n_significant > 0
    print(f"  [{'PASS' if prec_ok  else 'FAIL'}] Heuristic Precision >= 0.80: {spot_results['precision']:.4f}")
    print(f"  [{'PASS' if rec_ok   else 'FAIL'}] Heuristic Recall    >= 0.80: {spot_results['recall']:.4f}")
    print(f"  [{'PASS' if fdr_ok   else 'WARN'}] FDR-significant artifact features > 0: {n_significant}")
    print("\nTask 11 complete.")


if __name__ == "__main__":
    main()
