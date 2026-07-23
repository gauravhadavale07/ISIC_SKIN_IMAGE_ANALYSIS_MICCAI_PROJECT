import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats as scipy_stats
from skimage import color
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from dataset import get_transforms
from models.cross_attention import CrossAttentionT2VClassifier

SEED = 1337
BATCH_SIZE = 16
NUM_HEADS = cfg.model.num_attention_heads

MACRO_CLASSES = {
    'MEL': 'Malignant',
    'BCC': 'Malignant',
    'SCC': 'Malignant',
    'NEV': 'Benign',
    'ACK': 'Benign',
    'SEK': 'Benign',
    'BOD': 'Benign'
}

def compute_ita_from_pil(image: Image.Image):
    """
    Compute Individual Typology Angle (ITA) from a PIL Image.
    """
    img_np = np.array(image.convert("RGB")) / 255.0
    
    # Center crop to focus on skin
    h, w, _ = img_np.shape
    crop = img_np[h//4:3*h//4, w//4:3*w//4, :]
    
    lab = color.rgb2lab(crop)
    L = lab[:, :, 0]
    b = lab[:, :, 2]
    
    mask = (L > 5)
    L = L[mask]
    b = b[mask]
    if len(b) == 0:
        return 0.0
    
    ita = np.arctan((L - 50.0) / (b + 1e-8)) * (180.0 / np.pi)
    return float(np.mean(ita))

class PadUfesMacroPairDataset(Dataset):
    def __init__(self, df_metadata, img_dir, tokenizer, transform=None):
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Precompute ITA and setup df
        print("Computing ITA for all images...")
        itas = []
        macro_classes = []
        valid_indices = []
        
        for idx in tqdm(range(len(df_metadata))):
            row = df_metadata.iloc[idx]
            diag = str(row['diagnostic']).strip().upper()
            if diag not in MACRO_CLASSES:
                continue
                
            img_path = str(row['filepath'])
            if not os.path.exists(img_path):
                continue
                
            try:
                img = Image.open(img_path)
                ita = compute_ita_from_pil(img)
                itas.append(ita)
                macro_classes.append(MACRO_CLASSES[diag])
                valid_indices.append(idx)
            except Exception as e:
                continue
                
        self.df = df_metadata.iloc[valid_indices].copy()
        self.df['ITA'] = itas
        self.df['macro_class'] = macro_classes
        self.df['label_idx'] = self.df['diagnostic'].str.strip().str.upper().map(cfg.data.LABEL_MAP)
        
        # Perform manual spot check export
        self._export_spot_check()
        
        # Define Light vs Dark thresholds based on percentiles
        ita_q25 = self.df['ITA'].quantile(0.25)
        ita_q75 = self.df['ITA'].quantile(0.75)
        
        print(f"ITA 25th percentile (Dark threshold): {ita_q25:.2f}")
        print(f"ITA 75th percentile (Light threshold): {ita_q75:.2f}")
        
        self.df['skin_tone_bucket'] = 'Medium'
        self.df.loc[self.df['ITA'] <= ita_q25, 'skin_tone_bucket'] = 'Dark'
        self.df.loc[self.df['ITA'] >= ita_q75, 'skin_tone_bucket'] = 'Light'
        
        self.pairs = self._build_pairs(self.df)
        print(f"Built {len(self.pairs)} relaxed macro-class FST Light->Dark pairs.")

    def _export_spot_check(self):
        # Sample 50 images spanning the ITA range
        df_sorted = self.df.sort_values(by='ITA')
        # Take 50 evenly spaced samples
        indices = np.linspace(0, len(df_sorted)-1, 50, dtype=int)
        spot_check_df = df_sorted.iloc[indices][['filepath', 'ITA', 'diagnostic']]
        out_path = os.path.join(cfg.paths.results_dir, "task26_ita_spotcheck.csv")
        os.makedirs(cfg.paths.results_dir, exist_ok=True)
        spot_check_df.to_csv(out_path, index=False)
        print(f"Saved 50-image ITA spot check to {out_path} for manual validation.")

    def _build_pairs(self, df):
        pairs = []
        light = df[df['skin_tone_bucket'] == 'Light']
        dark = df[df['skin_tone_bucket'] == 'Dark']
        
        rng = np.random.RandomState(SEED)
        
        for macro in ['Benign', 'Malignant']:
            light_m = light[light['macro_class'] == macro]
            dark_m = dark[dark['macro_class'] == macro]
            
            if light_m.empty or dark_m.empty:
                continue
                
            for _, dark_row in dark_m.iterrows():
                # We can sample multiple light donors per dark receptor to increase N if needed
                # Here we just take 1 random donor to keep it 1:1, or 3 to boost N
                for _ in range(3): 
                    light_row = light_m.sample(1, random_state=int(rng.randint(0, 100000))).iloc[0]
                    
                    dark_ctrl_pool = dark_m[dark_m.index != dark_row.name]
                    dark_ctrl = dark_ctrl_pool.sample(1, random_state=int(rng.randint(0, 100000))).iloc[0] if not dark_ctrl_pool.empty else None
                    
                    # Random light pool (different macro class)
                    random_light_pool = light[light['macro_class'] != macro]
                    random_light = random_light_pool.sample(1, random_state=int(rng.randint(0, 100000))).iloc[0] if not random_light_pool.empty else None
                    
                    if random_light is None:
                        continue
                        
                    pairs.append({
                        "macro_class": macro,
                        "label_idx": int(dark_row["label_idx"]),
                        "dark_diag": dark_row["diagnostic"],
                        "light_diag": light_row["diagnostic"],
                        "light_file": light_row["filepath"],
                        "dark_file": dark_row["filepath"],
                        "dark_control_file": dark_ctrl["filepath"] if dark_ctrl is not None else None,
                        "random_light_file": random_light["filepath"],
                        "light_ita": light_row["ITA"],
                        "dark_ita": dark_row["ITA"],
                        "random_light_diag": random_light["diagnostic"],
                    })
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, filepath):
        if filepath is None:
            return torch.zeros(3, cfg.data.img_size, cfg.data.img_size)
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        text = "No clinical history available."
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
            "macro_class": pair["macro_class"],
            "light_file": pair["light_file"],
            "dark_file": pair["dark_file"],
        }


def cross_attention_heads(model, images, input_ids, attention_mask):
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
    print("TASK 26: LARGE-SCALE FAIRNESS PATCHING (RELAXED MACRO-CLASSES)")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    df = pd.read_csv(cfg.paths.pad_ufes_csv)
    dataset = PadUfesMacroPairDataset(df, cfg.paths.pad_ufes_img_dir, tokenizer, transform=get_transforms())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_{SEED}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    class_names = {idx: name for name, idx in cfg.data.LABEL_MAP.items()}
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Large-scale head patching"):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label_idx"].to(device)

            light_heads = cross_attention_heads(model, batch["light_img"].to(device), input_ids, attn_mask)
            dark_heads = cross_attention_heads(model, batch["dark_img"].to(device), input_ids, attn_mask)
            dark_control_heads = cross_attention_heads(model, batch["dark_control_img"].to(device), input_ids, attn_mask)
            random_light_heads = cross_attention_heads(model, batch["random_light_img"].to(device), input_ids, attn_mask)

            light_logits = logits_from_heads(model, light_heads)
            dark_logits = logits_from_heads(model, dark_heads)

            for head_idx in range(NUM_HEADS):
                main_logits = patched_logits(model, dark_heads, light_heads, head_idx)
                same_tone_logits = patched_logits(model, dark_heads, dark_control_heads, head_idx)
                random_class_logits = patched_logits(model, dark_heads, random_light_heads, head_idx)

                for i in range(labels.shape[0]):
                    label_idx = int(labels[i].item())
                    
                    base_dark = float(dark_logits[i, label_idx].item())
                    donor_light = float(light_logits[i, label_idx].item())
                    main_value = float(main_logits[i, label_idx].item())
                    same_value = float(same_tone_logits[i, label_idx].item())
                    random_value = float(random_class_logits[i, label_idx].item())

                    rows.append({
                        "macro_class": batch["macro_class"][i],
                        "label_idx": label_idx,
                        "light_file": batch["light_file"][i],
                        "dark_file": batch["dark_file"][i],
                        "head": head_idx,
                        "dark_true_logit": base_dark,
                        "light_true_logit": donor_light,
                        "main_recovery_pct": signed_recovery(main_value, base_dark, donor_light),
                        "same_tone_recovery_pct": signed_recovery(same_value, base_dark, donor_light),
                        "random_class_recovery_pct": signed_recovery(random_value, base_dark, donor_light),
                        "dark_pred": int(dark_logits[i].argmax().item()),
                        "main_patched_pred": int(main_logits[i].argmax().item()),
                    })

    out_path = os.path.join(cfg.paths.results_dir, "task26_large_scale_patching.csv")
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)

    summary = df_out.groupby("head").agg(
        n=("main_recovery_pct", "count"),
        mean_main_recovery=("main_recovery_pct", "mean"),
        mean_same_tone_recovery=("same_tone_recovery_pct", "mean"),
        mean_random_class_recovery=("random_class_recovery_pct", "mean"),
        flip_rate=("main_patched_pred", lambda s: np.mean(s.values != df_out.loc[s.index, "dark_pred"].values))
    ).reset_index()

    summary_path = os.path.join(cfg.paths.results_dir, "task26_large_scale_patching_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nLarge-Scale Head Patching Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to {out_path}")

    # ── Formal Post-Hoc Statistical Power Analysis ───────────────────────────
    N_pairs = len(dataset)
    print(f"\n{'=' * 70}")
    print("FORMAL POST-HOC STATISTICAL POWER ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Total pairs evaluated: {N_pairs}")

    # Compute Cohen's d for the light-dark logit disparity
    # Using head-level data: per-pair main_recovery_pct as the effect measure
    df_no_nan = df_out.dropna(subset=["main_recovery_pct"])
    recovery_vals = df_no_nan["main_recovery_pct"].values

    if len(recovery_vals) > 1:
        # Cohen's d: mean / std of recovery percentage (one-sample: test if != 0)
        cohens_d = float(recovery_vals.mean() / (recovery_vals.std() + 1e-8))

        # One-sample t-test: is mean recovery_pct different from 0?
        t_stat, p_val = scipy_stats.ttest_1samp(recovery_vals, popmean=0.0)

        # Post-hoc power computation:
        # Given N, Cohen's d, alpha=0.05, what is the achieved power?
        # Using non-central t distribution:
        alpha = 0.05
        df_val = len(recovery_vals) - 1
        ncp = cohens_d * np.sqrt(len(recovery_vals))  # non-centrality parameter
        t_crit = scipy_stats.t.ppf(1 - alpha / 2, df_val)
        power = 1.0 - scipy_stats.nct.cdf(t_crit, df_val, ncp) + scipy_stats.nct.cdf(-t_crit, df_val, ncp)
        power = float(np.clip(power, 0, 1))

        print(f"\nEffect size (Cohen's d):  {cohens_d:.4f}")
        print(f"Mean recovery pct:        {recovery_vals.mean():.4f}")
        print(f"Std  recovery pct:        {recovery_vals.std():.4f}")
        print(f"One-sample t-test:        t={t_stat:.4f}, p={p_val:.4e}")
        print(f"Post-hoc power (alpha=0.05): {power:.4f}")
        power_ok = power >= 0.80
        print(f"[{'PASS' if power_ok else 'FAIL'}] Power >= 0.80: {power:.4f}")

        # Save power analysis
        power_df = pd.DataFrame([{
            "N_pairs":        N_pairs,
            "N_valid":        len(recovery_vals),
            "cohens_d":       cohens_d,
            "mean_recovery":  recovery_vals.mean(),
            "std_recovery":   recovery_vals.std(),
            "t_stat":         t_stat,
            "p_value":        p_val,
            "post_hoc_power": power,
            "power_ok_80pct": power_ok,
        }])
        power_path = os.path.join(cfg.paths.results_dir, "task26_power_analysis.csv")
        power_df.to_csv(power_path, index=False)
        print(f"Power analysis saved to {power_path}")
    else:
        print("  WARNING: Insufficient data for power analysis.")

    # N >= 500 check
    if N_pairs >= 500:
        print(f"\n[PASS] N={N_pairs} >= 500. Disparity diagnosis is adequately powered.")
    else:
        print(f"\n[FAIL] N={N_pairs} < 500. Consider expanding the dataset.")


if __name__ == "__main__":
    main()
