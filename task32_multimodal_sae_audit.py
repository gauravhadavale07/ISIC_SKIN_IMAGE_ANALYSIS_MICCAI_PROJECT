import os
import sys
import pandas as pd
import numpy as np
import torch
import argparse
import random
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

def build_2x2_dataframe(csv_path: str):
    """
    Constructs a strict 2x2 experimental DataFrame with four distinct groups:
    - Group A: Melanoma Image + Melanoma Text (Aligned)
    - Group B: Melanoma Image + Benign Text (Contradictory)
    - Group C: Benign Image + Benign Text (Aligned)
    - Group D: Benign Image + Melanoma Text (Contradictory)
    """
    print(f"Loading base dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out invalid diagnostics
    df = df[df['diagnostic'].astype(str).str.upper() != 'NAN']
    df = df.reset_index(drop=True)

    records = []
    
    malignant_classes = [k for k, v in cfg.data.LABEL_MAPPING.items() if v == 1]
    benign_classes = [k for k, v in cfg.data.LABEL_MAPPING.items() if v == 0]
    
    MALIGNANT_POOL = [
        "Patient: Male, 75 years old. Lesion located on the back.",
        "Patient: Female, 82 years old. Lesion located on the face.",
        "Patient: Male, 68 years old. Lesion located on the chest.",
        "Patient: Male, 79 years old. Lesion located on the scalp.",
        "Patient: Female, 71 years old. Lesion located on the lower leg.",
        "Patient: Male, 88 years old. Lesion located on the ear.",
        "Patient: Female, 65 years old. Lesion located on the neck."
    ]

    BENIGN_POOL = [
        "Patient: Female, 22 years old. Lesion located on the arm.",
        "Patient: Male, 28 years old. Lesion located on the thigh.",
        "Patient: Female, 19 years old. Lesion located on the abdomen.",
        "Patient: Male, 32 years old. Lesion located on the forearm.",
        "Patient: Female, 25 years old. Lesion located on the back.",
        "Patient: Male, 21 years old. Lesion located on the foot.",
        "Patient: Female, 30 years old. Lesion located on the hand."
    ]

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building 2x2 matrix"):
        diag = str(row['diagnostic']).strip().upper()
        if diag in malignant_classes:
            image_class = 'Malignant'
        elif diag in benign_classes:
            image_class = 'Benign'
        else:
            continue
            
        sample_id = f"sample_{idx}"
        filepath = row['filepath']
        label_idx = cfg.data.LABEL_MAP[diag]
        
        # Aligned pair
        aligned_text = random.choice(MALIGNANT_POOL) if image_class == 'Malignant' else random.choice(BENIGN_POOL)
        aligned_text_class = image_class
        group_aligned = 'A' if image_class == 'Malignant' else 'C'
        
        records.append({
            'sample_id': sample_id,
            'filepath': filepath,
            'diagnostic': diag,
            'true_label_idx': label_idx,
            'image_class': image_class,
            'text_class': aligned_text_class,
            'clinical_history': aligned_text,
            'group': group_aligned
        })
        
        # Contradictory pair
        contradictory_text = random.choice(BENIGN_POOL) if image_class == 'Malignant' else random.choice(MALIGNANT_POOL)
        contradictory_text_class = 'Benign' if image_class == 'Malignant' else 'Malignant'
        group_contradictory = 'B' if image_class == 'Malignant' else 'D'
        
        records.append({
            'sample_id': sample_id,
            'filepath': filepath,
            'diagnostic': diag,
            'true_label_idx': label_idx,
            'image_class': image_class,
            'text_class': contradictory_text_class,
            'clinical_history': contradictory_text,
            'group': group_contradictory
        })

    df_2x2 = pd.DataFrame(records)
    print(f"\nConstructed 2x2 DataFrame with {len(df_2x2)} rows.")
    print("Group distribution:")
    print(df_2x2['group'].value_counts().sort_index())
    
    return df_2x2


def load_model(device, seed=1337):
    from models.cross_attention import CrossAttentionT2VClassifier
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_{seed}", "best_model.pth")
    model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()
    return model


def load_sae(device):
    from task11_sae import TopKSAE
    weights_path = os.path.join(cfg.paths.results_dir, "sae_weights.pth")
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    sae.load_state_dict(torch.load(weights_path, map_location=device))
    sae.eval()
    return sae


def extract_sae_activations(df_2x2: pd.DataFrame):
    from dataset import MultimodalDermatologyDataset, get_transforms
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    device = cfg.train.device
    model = load_model(device)
    sae = load_sae(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    temp_csv = os.path.join(cfg.paths.results_dir, "temp_2x2.csv")
    df_2x2.to_csv(temp_csv, index=False)
    
    dataset = MultimodalDermatologyDataset(
        csv_file=temp_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(),
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_acts = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting SAE activations"):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
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
            sparse_acts = sae.encode(fused)
            
            all_acts.append(sparse_acts.cpu().numpy())
            
    all_acts = np.concatenate(all_acts, axis=0)
    
    feature_cols = [f"feat_{i}" for i in range(6144)]
    df_acts = pd.DataFrame(all_acts, columns=feature_cols)
    
    df_final = pd.concat([df_2x2.reset_index(drop=True), df_acts.reset_index(drop=True)], axis=1)
    
    if os.path.exists(temp_csv):
        os.remove(temp_csv)
        
    return df_final, feature_cols


def calculate_art_interaction(df: pd.DataFrame, feature_col: str):
    """
    Computes the Aligned Rank Transform (ART) p-value for the interaction 
    between image_class and text_class on a given SAE feature activation.
    """
    mu_image = df.groupby('image_class')[feature_col].transform('mean')
    mu_text = df.groupby('text_class')[feature_col].transform('mean')
    mu_total = df[feature_col].mean()
    
    aligned_val = df[feature_col] - mu_image - mu_text + mu_total
    
    df_temp = df[['image_class', 'text_class']].copy()
    df_temp['ranked_val'] = aligned_val.rank()
    
    model = ols('ranked_val ~ C(image_class) * C(text_class)', data=df_temp).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    interaction_p = anova_table.loc['C(image_class):C(text_class)', 'PR(>F)']
    interaction_F = anova_table.loc['C(image_class):C(text_class)', 'F']
    
    return interaction_F, interaction_p


def screen_features_art(df: pd.DataFrame, feature_cols: list):
    """
    Screens all SAE features using ART and applies FDR correction.
    """
    results = []
    for f in tqdm(feature_cols, desc="ART Screening"):
        if df[f].sum() == 0:
            results.append({'feature': f, 'F_stat': 0.0, 'p_value': 1.0})
            continue
            
        try:
            F, p = calculate_art_interaction(df, f)
            results.append({'feature': f, 'F_stat': F, 'p_value': p})
        except Exception as e:
            # Handle potential singular matrix errors for extremely sparse features
            results.append({'feature': f, 'F_stat': 0.0, 'p_value': 1.0})
        
    df_results = pd.DataFrame(results)
    
    # FDR Correction (Benjamini-Hochberg)
    passed, q_values = fdrcorrection(df_results['p_value'])
    df_results['q_value'] = q_values
    df_results['significant'] = passed
    
    # Calculate fusion score: F-statistic * -log10(FDR_q)
    df_results['fusion_score'] = df_results['F_stat'] * (-torch.tensor(df_results['q_value'].values).log10().numpy())
    df_results = df_results.sort_values(by='fusion_score', ascending=False).reset_index(drop=True)
    
    return df_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for clinical history text sampling")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"--- Running Audit with Text Sampling Seed: {args.seed} ---")
    
    csv_path = cfg.paths.pad_ufes_csv
    df_2x2 = build_2x2_dataframe(csv_path)
    
    df_final, feature_cols = extract_sae_activations(df_2x2)
    
    print("\nExtracting feature stats...")
    df_results = screen_features_art(df_final, feature_cols)
    
    surviving_features = df_results[df_results['significant'] == True]
    
    if len(surviving_features) == 0:
        print("\nNull Result: No statistically significant localized sparse multimodal interaction features were detected under the examined representation and methodology. This suggests that any image–text interaction may be weak, distributed, or encoded outside the analyzed representation, and is consistent with our behavioral findings of limited semantic text integration.")
        sys.exit(0)
    else:
        print(f"\nFound {len(surviving_features)} significant interaction features!")
        print("Top 10 candidates by Fusion Score:")
        print(df_results.head(10))
        
        # Save results
        out_path = os.path.join(cfg.paths.results_dir, f"task32_art_screening_results_seed{args.seed}.csv")
        df_results.to_csv(out_path, index=False)
        print(f"\nSaved screening results to {out_path}")
        
        # Save activations for the top 50 features for visualization
        top_50_features = df_results['feature'].head(50).tolist()
        df_top_acts = df_final[['sample_id', 'image_class', 'text_class', 'group'] + top_50_features]
        acts_out = os.path.join(cfg.paths.results_dir, f"task32_top50_activations_seed{args.seed}.csv")
        df_top_acts.to_csv(acts_out, index=False)
        print(f"Saved top 50 feature activations for visualization to {acts_out}")
        
if __name__ == "__main__":
    main()
