import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE


FEATURE_ID = 1449
SEED = 1337
BATCH_SIZE = 32


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


def malignant_class_indices():
    return sorted(
        cfg.data.LABEL_MAP[class_name]
        for class_name, binary in cfg.data.LABEL_MAPPING.items()
        if binary == 1
    )


def class_name_lookup():
    return {idx: name for name, idx in cfg.data.LABEL_MAP.items()}


def choose_random_active_controls(sparse_acts, feature_id):
    """Choose one non-target active SAE feature per row as a local control."""
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    control_ids = []
    control_acts = []

    sparse_cpu = sparse_acts.detach().cpu()
    for row in sparse_cpu:
        active = torch.nonzero(row > 0, as_tuple=False).flatten()
        active = active[active != feature_id]
        if len(active) == 0:
            control_ids.append(-1)
            control_acts.append(0.0)
            continue
        pick = active[torch.randint(len(active), (1,), generator=generator).item()].item()
        control_ids.append(pick)
        control_acts.append(float(row[pick].item()))

    return (
        torch.tensor(control_ids, dtype=torch.long),
        torch.tensor(control_acts, dtype=sparse_acts.dtype),
    )


def main():
    print("=" * 70)
    print("TASK 14: CAUSAL FEATURE KNOCKOUT (SAE FEATURE 1449)")
    print("=" * 70)
    print(
        "Feature 1449 is treated as an artifact-associated SAE feature; "
        "we measure signed logit shifts rather than assuming malignancy direction."
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.milk10k_csv,
        img_dir=cfg.paths.milk10k_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = load_model(device)
    sae = load_sae(device)

    class_names = class_name_lookup()
    malignant_idx = malignant_class_indices()
    benign_idx = [i for i in range(cfg.model.num_classes) if i not in malignant_idx]

    sae_features_path = os.path.join(cfg.paths.results_dir, "task11_sae_features.csv")
    if os.path.exists(sae_features_path):
        df_sae = pd.read_csv(sae_features_path)
        row = df_sae[df_sae["Feature_ID"] == FEATURE_ID]
        if not row.empty:
            print(
                f"Feature {FEATURE_ID} prior: corr_with_malignancy="
                f"{row.iloc[0]['Correlation_with_Malignancy']:+.4f}, "
                f"L0={int(row.iloc[0]['L0_Count'])}, "
                f"max_act={row.iloc[0]['Max_Activation']:.4f}"
            )

    records = []
    row_offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Knockout passes"):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            batch_size = imgs.shape[0]

            vision_seq = model.vision_encoder.forward_features(imgs)
            text_seq = model.text_encoder(
                input_ids=input_ids, attention_mask=attn_mask
            ).last_hidden_state
            attn_output, _ = model.cross_attn(
                query=text_seq,
                key=vision_seq,
                value=vision_seq,
                need_weights=False,
            )

            fused = attn_output.mean(dim=1)
            baseline_logits = model.classifier(fused)
            sparse_acts = sae.encode(fused)

            fires_mask = sparse_acts[:, FEATURE_ID] > 0
            if not fires_mask.any():
                row_offset += batch_size
                continue

            fused_fires = fused[fires_mask]
            base_fires = baseline_logits[fires_mask]
            labels_fires = labels[fires_mask]
            acts_fires = sparse_acts[fires_mask]
            feature_act = acts_fires[:, FEATURE_ID]

            target_decoded = sae.W_dec[FEATURE_ID].unsqueeze(0)
            fused_target_knockout = fused_fires - feature_act.unsqueeze(1) * target_decoded
            target_logits = model.classifier(fused_target_knockout)

            control_ids_cpu, control_acts_cpu = choose_random_active_controls(
                acts_fires, FEATURE_ID
            )
            valid_ctrl = control_ids_cpu >= 0
            control_ids = control_ids_cpu.to(device)
            control_acts = control_acts_cpu.to(device)
            control_decoded = sae.W_dec[control_ids.clamp_min(0)]
            fused_control_knockout = fused_fires - control_acts.unsqueeze(1) * control_decoded
            control_logits = model.classifier(fused_control_knockout)

            batch_indices = torch.arange(batch_size, device=device)[fires_mask].cpu().numpy()
            filepaths = dataset.df.iloc[row_offset + batch_indices]["filepath"].tolist()
            diagnostics = dataset.df.iloc[row_offset + batch_indices]["diagnostic"].tolist()

            delta_target = target_logits - base_fires
            delta_control = control_logits - base_fires

            for i in range(len(filepaths)):
                rec = {
                    "filepath": filepaths[i],
                    "diagnostic": str(diagnostics[i]).strip().upper(),
                    "label_idx": int(labels_fires[i].item()),
                    "feature_id": FEATURE_ID,
                    "feature_activation": float(feature_act[i].item()),
                    "control_feature_id": int(control_ids_cpu[i].item()),
                    "control_feature_activation": float(control_acts_cpu[i].item()),
                    "control_valid": bool(valid_ctrl[i].item()),
                    "baseline_pred": int(base_fires[i].argmax().item()),
                    "feature_knockout_pred": int(target_logits[i].argmax().item()),
                    "control_knockout_pred": int(control_logits[i].argmax().item()),
                    "baseline_malignant_mean_logit": float(base_fires[i, malignant_idx].mean().item()),
                    "feature_knockout_malignant_mean_logit": float(target_logits[i, malignant_idx].mean().item()),
                    "control_knockout_malignant_mean_logit": float(control_logits[i, malignant_idx].mean().item()),
                    "feature_delta_malignant_minus_benign": float(
                        delta_target[i, malignant_idx].mean().item()
                        - delta_target[i, benign_idx].mean().item()
                    ),
                    "control_delta_malignant_minus_benign": float(
                        delta_control[i, malignant_idx].mean().item()
                        - delta_control[i, benign_idx].mean().item()
                    ),
                }
                for class_idx, class_name in class_names.items():
                    rec[f"baseline_logit_{class_name}"] = float(base_fires[i, class_idx].item())
                    rec[f"feature_delta_{class_name}"] = float(delta_target[i, class_idx].item())
                    rec[f"control_delta_{class_name}"] = float(delta_control[i, class_idx].item())
                records.append(rec)

            row_offset += batch_size

    if not records:
        print(f"Feature {FEATURE_ID} never fired; no CSV written.")
        return

    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.results_dir, "task14_knockout.csv")
    df = pd.DataFrame(records)
    df.to_csv(out_path, index=False)

    summary_rows = []
    for prefix, label in [("feature_delta", f"Feature {FEATURE_ID}"), ("control_delta", "Random active control")]:
        row = {"intervention": label, "n": len(df)}
        for class_idx, class_name in class_names.items():
            row[f"mean_delta_{class_name}"] = df[f"{prefix}_{class_name}"].mean()
        row["mean_delta_malignant_minus_benign"] = (
            df[[f"{prefix}_{class_names[i]}" for i in malignant_idx]].mean(axis=1)
            - df[[f"{prefix}_{class_names[i]}" for i in benign_idx]].mean(axis=1)
        ).mean()
        row["prediction_flip_rate"] = (
            df[f"{'feature' if prefix == 'feature_delta' else 'control'}_knockout_pred"]
            != df["baseline_pred"]
        ).mean()
        summary_rows.append(row)

    summary_path = os.path.join(cfg.paths.results_dir, "task14_knockout_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nFeature {FEATURE_ID} fired in {len(df)} images.")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved per-sample results to {out_path}")
    print(f"Saved summary results to {summary_path}")


if __name__ == "__main__":
    main()
