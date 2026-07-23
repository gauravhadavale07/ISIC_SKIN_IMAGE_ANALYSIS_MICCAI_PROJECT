import os
import sys
import pandas as pd
import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE

def build_2x2_dataframe(csv_path: str):
    """
    Constructs a strict 2x2 experimental DataFrame exactly as in task32.
    """
    df = pd.read_csv(csv_path)
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

    for idx, row in df.iterrows():
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
            'img_path': filepath, # adding for audit script
            'diagnostic': diag,
            'true_label_idx': label_idx,
            'image_class': image_class,
            'text_class': aligned_text_class,
            'clinical_history': aligned_text,
            'text': aligned_text, # adding for audit script
            'group': group_aligned
        })
        
        # Contradictory pair
        contradictory_text = random.choice(BENIGN_POOL) if image_class == 'Malignant' else random.choice(MALIGNANT_POOL)
        contradictory_text_class = 'Benign' if image_class == 'Malignant' else 'Malignant'
        group_contradictory = 'B' if image_class == 'Malignant' else 'D'
        
        records.append({
            'sample_id': sample_id,
            'filepath': filepath,
            'img_path': filepath, # adding for audit script
            'diagnostic': diag,
            'true_label_idx': label_idx,
            'image_class': image_class,
            'text_class': contradictory_text_class,
            'clinical_history': contradictory_text,
            'text': contradictory_text, # adding for audit script
            'group': group_contradictory
        })

    return pd.DataFrame(records)

def main():
    print("--- Step 1 & 10: Verification of Raw Extraction and E2E Pipeline ---")
    
    # EXACT REPRODUCTION OF ORIGINAL SEEDING
    seed = 1337
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 1. Rebuild inputs
    df_inputs = build_2x2_dataframe(cfg.paths.pad_ufes_csv)
    
    # 2. Load stored results
    acts_csv_path = os.path.join(cfg.paths.results_dir, "task32_top50_activations_seed1337.csv")
    df_stored = pd.read_csv(acts_csv_path)
    
    # Randomly select 20 rows that exist in the stored CSV
    np.random.seed(42)
    sample_indices = np.random.choice(len(df_stored), 20, replace=False)
    
    # Load Model and SAE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    model = CrossAttentionT2VClassifier().to(device)
    checkpoint = torch.load(os.path.join(cfg.paths.checkpoint_dir, "Cross-Attention_T→V_seed_1337", "best_model.pth"), map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
    model.eval()
    
    sae = TopKSAE(768, 8, 32).to(device)
    sae.load_state_dict(torch.load(os.path.join(cfg.paths.results_dir, "sae_weights.pth"), map_location=device))
    sae.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    fused_activations = []
    def get_fused_hook(module, input, output):
        fused_activations.append(output[0])
    hook = model.cross_attn.register_forward_hook(get_fused_hook)
    
    mismatches = 0
    checked = 0
    
    feat_cols = [c for c in df_stored.columns if c.startswith('feat_')]
    feat_indices = [int(c.split('_')[1]) for c in feat_cols]
    
    test_feats = np.random.choice(feat_indices, 5, replace=False)
    print(f"Testing manual indexing for features: {test_feats}")

    for idx in sample_indices:
        stored_row = df_stored.iloc[idx]
        sample_id = stored_row['sample_id']
        group = stored_row['group']
        
        input_row = df_inputs[(df_inputs['sample_id'] == sample_id) & (df_inputs['group'] == group)].iloc[0]
        
        img_path = input_row['filepath']
        if not img_path.startswith('./data'):
            full_img_path = os.path.join(cfg.paths.data_dir, "raw_pad_ufes", "images", os.path.basename(img_path))
        else:
            full_img_path = img_path
            
        text = input_row['clinical_history']
        
        fused_activations.clear()
        with torch.no_grad():
            image = Image.open(full_img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            _ = model(image_tensor, input_ids, attention_mask)
            fused_stream = fused_activations[0].mean(dim=1)
            
            _, f_x = sae(fused_stream)
        
        for feat_idx in feat_cols:
            actual_idx = int(feat_idx.split('_')[1])
            recomputed_val = f_x[0, actual_idx].item()
            stored_val = stored_row[feat_idx]
            
            if not np.isclose(recomputed_val, stored_val, atol=1e-5):
                print(f"Mismatch! {sample_id} Group {group} Feature {actual_idx}: Recomputed={recomputed_val:.6f}, Stored={stored_val:.6f}")
                mismatches += 1
                
        for tf in test_feats:
            manual_val = f_x[0, tf].item()
            csv_col_val = stored_row[f"feat_{tf}"]
            if not np.isclose(manual_val, csv_col_val, atol=1e-5):
                print(f"Indexing Mismatch! f_x[0, {tf}] = {manual_val:.6f} != df['feat_{tf}'] = {csv_col_val:.6f}")
                
        checked += 1
        print(f"Verified sample {checked}/20: {sample_id} Group {group}")

    hook.remove()
    print(f"\nStep 1, 3 & 10 Results: Checked 20 samples ({20 * len(feat_cols)} feature values). Mismatches: {mismatches}")

if __name__ == "__main__":
    main()
