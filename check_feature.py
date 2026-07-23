import os
import torch
import numpy as np
import pandas as pd
from dataset import MultimodalDermatologyDataset, get_transforms
from transformers import AutoTokenizer
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE, extract_all_activations
from config import cfg

def check_feature():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.milk10k_csv,
        img_dir=cfg.paths.milk10k_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    base_model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(os.path.join(cfg.paths.checkpoint_dir, "Cross-Attention_T→V_seed_1337", "best_model.pth"), map_location=device)
    base_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    
    print("Extracting X...")
    X, _, filepaths = extract_all_activations(base_model, loader, device)
    
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    sae.load_state_dict(torch.load(os.path.join(cfg.paths.results_dir, "sae_weights.pth"), map_location=device))
    sae.eval()

    print("Extracting SAE...")
    with torch.no_grad():
        _, sparse_acts = sae(X.to(device))
    
    feat1449 = sparse_acts[:, 1449].cpu().numpy()
    non_zero = np.sum(feat1449 > 0)
    print(f"Total images: {len(feat1449)}")
    print(f"Images with non-zero activation for Feature 1449: {non_zero}")

if __name__ == "__main__":
    check_feature()
