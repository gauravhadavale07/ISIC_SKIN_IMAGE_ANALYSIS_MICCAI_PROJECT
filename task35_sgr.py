import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE

def train_sgr():
    print("=" * 70)
    print("TASK 35: SEMANTIC GROUNDING REGULARIZATION (SGR) TRAINING")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    train_dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.milk10k_csv,
        img_dir=cfg.paths.milk10k_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )
    # Subset for faster demonstration
    train_size = int(len(train_dataset) * 0.1) # 10%
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, len(train_dataset)-train_size])
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=2)

    # 2. Load Model
    seed = 1337
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_{seed}", "best_model.pth")
    model = CrossAttentionT2VClassifier().to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        print(f"Loaded pretrained model from {ckpt_path}")
    else:
        print("Warning: Pretrained model not found, training from scratch (not recommended).")
        
    model.train()

    # 3. Load SAE
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    sae_path = os.path.join(cfg.paths.results_dir, "sae_weights.pth")
    if os.path.exists(sae_path):
        sae.load_state_dict(torch.load(sae_path, map_location=device))
        print("Loaded SAE.")
    sae.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # 4. Text pools
    BENIGN_POOL = ["Patient presents with a completely harmless, benign mole.", "Routine benign nevus."]
    MALIGNANT_POOL = ["Patient presents with a highly dangerous, malignant melanoma.", "Deadly malignant melanoma."]
    NEUTRAL_POOL = ["Patient is here for a routine skin check.", "Clinical history is not available."]
    BLANK_TEXT = [""]
    
    # Extract SAE hook
    activations = {}
    def hook_fn(module, input, output):
        activations['h'] = output
    hook_handle = model.multimodal_fusion.register_forward_hook(hook_fn)

    lambdas = {'ground': 0.5, 'neutral': 0.2, 'counter': 0.1, 'shortcut': 1.0}
    delta_ground = 1.0
    
    # Lists for tracking metrics
    epoch_losses = []
    
    print("\nStarting SGR Fine-Tuning for 1 Epoch (Demonstration)...")
    progress_bar = tqdm(train_loader, desc="Training SGR")
    
    for batch in progress_bar:
        imgs = batch['image'].to(device)
        labels = batch['label'].to(device) # 0 for benign, 1 for malignant
        
        batch_size = imgs.size(0)
        
        # Original Classification
        texts_correct = [MALIGNANT_POOL[0] if l == 1 else BENIGN_POOL[0] for l in labels]
        texts_contradict = [BENIGN_POOL[0] if l == 1 else MALIGNANT_POOL[0] for l in labels]
        texts_neutral = [NEUTRAL_POOL[0]] * batch_size
        texts_blank = [BLANK_TEXT[0]] * batch_size
        
        def run_forward(texts):
            inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            return model(imgs, inputs['input_ids'], inputs['attention_mask'])
            
        # Standard
        logits_correct = run_forward(texts_correct)
        acts_correct = activations['h'].clone()
        
        # Contradictory
        logits_contradict = run_forward(texts_contradict)
        
        # Neutral & Blank
        logits_neutral = run_forward(texts_neutral)
        logits_blank = run_forward(texts_blank)
        
        def get_margin(logits, y):
            m = logits[:, 1] - logits[:, 0]
            m[y == 0] = -m[y == 0]
            return m
            
        m_c = get_margin(logits_correct, labels)
        m_x = get_margin(logits_contradict, labels)
        
        # Loss 0: Standard Classification (Cross Entropy)
        L_cls = F.cross_entropy(logits_correct, labels)
        
        # Loss 1: Contradictory Text Margin
        L_ground = torch.clamp(delta_ground - (m_c - m_x), min=0).mean()
        
        # Loss 2: Neutral / Blank Consistency
        L_neutral = F.mse_loss(logits_neutral, logits_blank)
        
        # Loss 3: Counterfactual Sensitivity
        m_neutral = get_margin(logits_neutral, labels)
        m_counter = get_margin(logits_contradict, labels)
        
        W_amb = torch.exp(-m_neutral.detach().clamp(min=0))
        L_counter = (W_amb * torch.clamp(m_counter - m_neutral + 0.5, min=0)).mean()
        
        # Loss 4: Shortcut Suppression (Feature 1449)
        with torch.no_grad():
            pass # SAE weights are fixed
        _, sparse_acts = sae(acts_correct)
        L_shortcut = sparse_acts[:, 1449].mean()
        
        # Total Loss
        L_total = L_cls + lambdas['ground'] * L_ground + lambdas['neutral'] * L_neutral + lambdas['counter'] * L_counter + lambdas['shortcut'] * L_shortcut
        
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        
        epoch_losses.append({
            'L_cls': L_cls.item(),
            'L_ground': L_ground.item(),
            'L_neutral': L_neutral.item(),
            'L_counter': L_counter.item(),
            'L_shortcut': L_shortcut.item(),
            'L_total': L_total.item()
        })
        
        progress_bar.set_postfix({'Total': f"{L_total.item():.3f}", 'Ground': f"{L_ground.item():.3f}", 'Shrt': f"{L_shortcut.item():.3f}"})
        
    hook_handle.remove()
    
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.paths.results_dir, "sgr_tuned_model.pth")
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved SGR tuned model to {out_path}")
    
    df_losses = pd.DataFrame(epoch_losses)
    print("\nFinal Loss Component Averages:")
    print(df_losses.mean())
    df_losses.to_csv(os.path.join(cfg.paths.results_dir, "task35_sgr_training_losses.csv"), index=False)
    
if __name__ == "__main__":
    train_sgr()
