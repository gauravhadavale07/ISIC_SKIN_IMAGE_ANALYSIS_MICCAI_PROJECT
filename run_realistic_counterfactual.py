import os
import torch
import random
import re
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import cfg
from dataset import MultimodalDermatologyDataset
from models.cross_attention import CrossAttentionClassifier

def generate_realistic_counterfactual(text):
    """Perturbs the clinical metadata string realistically."""
    # 1. Swap Age
    text = re.sub(r'age \d+', f'age {random.randint(20, 85)}', text)
    
    # 2. Swap Sex
    # Use word boundaries to avoid turning 'Female' into 'FeFemale'
    if re.search(r'\bMale\b', text):
        text = re.sub(r'\bMale\b', 'Female', text)
    elif re.search(r'\bFemale\b', text):
        text = re.sub(r'\bFemale\b', 'Male', text)
        
    # 3. Swap Location
    locations = ['face', 'back', 'chest', 'abdomen', 'arm', 'leg', 'neck', 'shoulder', 'head', 'torso']
    for loc in locations:
        # Use word boundaries to avoid replacing 'head' inside 'forehead' -> 'foreabdomen'
        if re.search(rf'\b{loc}\b', text, re.IGNORECASE):
            new_loc = random.choice([l for l in locations if l != loc])
            text = re.sub(rf'\b{loc}\b', new_loc, text, flags=re.IGNORECASE)
            break
            
    # 4. Swap Morphological History (flip benign/malignant priors)
    if 'No prominent secondary morphological features' in text:
        text = re.sub(r'No prominent secondary morphological features.*?\.',
                      'Dermatoscopic evaluation indicates distinct erythema, irregular pigmentation, surface ulceration or crusting.', text)
    else:
        text = re.sub(r'Dermatoscopic evaluation indicates.*?\.', 
                      'No prominent secondary morphological features (erythema, ulceration, or abnormal vasculature) were confidently identified.', text)
                      
    return text

def run_realistic_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # Load Model
    model = CrossAttentionClassifier().to(device)
    
    # Load Checkpoint (seed 456)
    ckpt_path = 'checkpoints/Cross-Attention_V→T_seed_456/best_model.pth'
    if not os.path.exists(ckpt_path):
        # Fallback to look for .pt or other seeds
        print(f"Checkpoint not found at {ckpt_path}. Looking for others...")
        import glob
        ckpts = glob.glob('checkpoints/Cross-Attention_V→T_seed_*/best_model.pth')
        if not ckpts:
            raise FileNotFoundError("No Cross-Attention checkpoint found.")
        ckpt_path = ckpts[0]
        
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    # Neutral Tokens
    neutral_tokens = tokenizer(
        cfg.audit.neutral_string,
        padding="max_length",
        truncation=True,
        max_length=cfg.data.max_text_len,
        return_tensors="pt"
    )
    dummy_neutral_ids = neutral_tokens["input_ids"].to(device)
    dummy_neutral_mask = neutral_tokens["attention_mask"].to(device)
    
    # Dataset
    df = pd.read_csv('pad_ufes_20_test.csv')
    # Filter out empty images
    dataset = MultimodalDermatologyDataset(
        csv_file='pad_ufes_20_test.csv', 
        tokenizer=tokenizer,
        split='all'
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    total = 0
    real_correct = 0
    neutral_correct = 0
    cf_correct = 0
    
    flipped_count = 0
    delta_p_sum = 0.0
    
    print("Starting Audit...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            imgs = batch['image'].to(device)
            real_ids = batch['input_ids'].to(device)
            real_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            B = imgs.size(0)
            total += B
            
            # 1. Real Prediction
            real_logits, _, _ = model(imgs, real_ids, real_mask)
            real_probs = torch.softmax(real_logits, dim=1)
            _, real_preds = torch.max(real_logits, 1)
            real_correct += (real_preds == labels).sum().item()
            
            # 2. Neutral Prediction
            n_ids = dummy_neutral_ids.expand(B, -1).contiguous()
            n_mask = dummy_neutral_mask.expand(B, -1).contiguous()
            neutral_logits, _, _ = model(imgs, n_ids, n_mask)
            _, neutral_preds = torch.max(neutral_logits, 1)
            neutral_correct += (neutral_preds == labels).sum().item()
            
            # 3. Realistic Counterfactual Prediction
            # Get original text
            # Since the dataset doesn't return the raw text, we will get it from df directly based on batch indexing
            start_idx = i * 32
            end_idx = start_idx + B
            batch_df = dataset.df.iloc[start_idx:end_idx]
            
            cf_texts = []
            for text in batch_df['clinical_history']:
                cf_texts.append(generate_realistic_counterfactual(str(text)))
                
            cf_encodings = tokenizer(
                cf_texts,
                padding='max_length',
                truncation=True,
                max_length=cfg.data.max_text_len,
                return_tensors='pt'
            )
            cf_ids = cf_encodings['input_ids'].to(device)
            cf_mask = cf_encodings['attention_mask'].to(device)
            
            cf_logits, _, _ = model(imgs, cf_ids, cf_mask)
            cf_probs = torch.softmax(cf_logits, dim=1)
            _, cf_preds = torch.max(cf_logits, 1)
            cf_correct += (cf_preds == labels).sum().item()
            
            # Metrics
            flipped = (real_preds != cf_preds)
            flipped_count += flipped.sum().item()
            
            real_p = real_probs[torch.arange(B), real_preds]
            cf_p = cf_probs[torch.arange(B), real_preds]
            delta_p_sum += torch.abs(real_p - cf_p).sum().item()
            
            if total >= 1440:
                print("\n[CPU Subset mode: Stopping after 1440 samples]")
                break
            
    print("\n🔬 --- Realistic Counterfactual Audit Report ---")
    print(f"Original Accuracy:          {100. * real_correct / total:.2f}%")
    print(f"Neutral Text Accuracy:      {100. * neutral_correct / total:.2f}%")
    print(f"Realistic Perturb Accuracy: {100. * cf_correct / total:.2f}%")
    print(f"\nRealistic Counterfactual Flip Rate (CFR):  {100. * flipped_count / total:.2f}%")
    print(f"Mean Probability Shift (ΔP):               {100. * delta_p_sum / total:.2f}pp")
    print("-" * 45)

if __name__ == '__main__':
    run_realistic_audit()
