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

# Try multiple seeds if one doesn't exist, but we will start with 1337
SEED = 1337

def load_model(device, seed=1337):
    ckpt_path = os.path.join(
        cfg.paths.checkpoint_dir,
        f"Cross-Attention_T\u2192V_seed_{seed}",
        "best_model.pth",
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()
    return model

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    try:
        model = load_model(device, SEED)
    except FileNotFoundError:
        # Fallback to the first available seed
        print(f"Seed {SEED} not found, falling back to first config seed {cfg.seeds[0]}")
        model = load_model(device, cfg.seeds[0])

    print("Loading PAD-UFES-20 test data...")
    df = pd.read_csv(cfg.paths.pad_ufes_csv)
    
    # Filter for MEL and NEV
    df_mel = df[df['diagnostic'].astype(str).str.upper() == 'MEL'].head(50)
    df_nev = df[df['diagnostic'].astype(str).str.upper() == 'NEV'].head(50)
    df_subset = pd.concat([df_mel, df_nev]).reset_index(drop=True)

    print(f"Isolated {len(df_mel)} MEL and {len(df_nev)} NEV images.")

    profile_A = "Male, age 85, presents with a lesion on the face."
    profile_B = "Female, age 18, presents with a lesion on the abdomen."

    # Profile A
    df_subset['clinical_history'] = profile_A
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    temp_csv_A = os.path.join(cfg.paths.results_dir, "temp_pad_ufes_A.csv")
    df_subset.to_csv(temp_csv_A, index=False)

    print("Creating dataset for Profile A (High Risk)...")
    dataset_A = MultimodalDermatologyDataset(
        csv_file=temp_csv_A,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(),
    )
    loader_A = DataLoader(dataset_A, batch_size=32, shuffle=False, num_workers=4)

    # Profile B
    df_subset['clinical_history'] = profile_B
    temp_csv_B = os.path.join(cfg.paths.results_dir, "temp_pad_ufes_B.csv")
    df_subset.to_csv(temp_csv_B, index=False)

    print("Creating dataset for Profile B (Low Risk)...")
    dataset_B = MultimodalDermatologyDataset(
        csv_file=temp_csv_B,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(),
    )
    loader_B = DataLoader(dataset_B, batch_size=32, shuffle=False, num_workers=4)

    mel_idx = cfg.data.LABEL_MAP['MEL']
    nev_idx = cfg.data.LABEL_MAP['NEV']

    def get_margins(loader, desc="Evaluating"):
        margins = []
        with torch.no_grad():
            # Support both architectures: models with and without dict output
            for batch in tqdm(loader, desc=desc):
                imgs = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                
                output = model(imgs, input_ids, attn_mask)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                    
                # Margin: Logit(MEL) - Logit(NEV)
                margin = logits[:, mel_idx] - logits[:, nev_idx]
                margins.extend(margin.cpu().tolist())
        return np.array(margins)

    margins_A = get_margins(loader_A, "Evaluating Profile A")
    margins_B = get_margins(loader_B, "Evaluating Profile B")

    margin_shift = margins_A - margins_B
    mean_shift = np.mean(margin_shift)

    print("\n" + "="*50)
    print("RESULTS: ADVERSARIAL DEMOGRAPHIC SWAP")
    print("="*50)
    print(f"Mean MEL-NEV logit margin | Profile A (High Risk): {np.mean(margins_A):.4f}")
    print(f"Mean MEL-NEV logit margin | Profile B (Low Risk):  {np.mean(margins_B):.4f}")
    print("-" * 50)
    print(f"Mean Margin Shift (A - B): {mean_shift:.4f}")
    print("="*50)

    if mean_shift > 0.5:
        print("CONCLUSION: The margin shifts heavily toward Melanoma for Profile A.")
        print("The model is using the demographic prior as a shortcut.")
    elif mean_shift < 0.1:
        print("CONCLUSION: The margin barely moves.")
        print("This proves the fusion model is visually dominant and structurally bottlenecked, practically ignoring the explicit demographic text despite its statistical relevance.")
    else:
        print("CONCLUSION: The model uses demographic text moderately. The margin shift is noticeable but not overwhelming.")
    
    # clean up temp files
    try:
        os.remove(temp_csv_A)
        os.remove(temp_csv_B)
    except:
        pass

if __name__ == '__main__':
    main()
