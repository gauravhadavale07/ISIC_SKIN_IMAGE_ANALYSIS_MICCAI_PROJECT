import os
import torch
import numpy as np
import pandas as pd
from dataset import MultimodalDermatologyDataset, get_transforms
from transformers import AutoTokenizer
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE, extract_all_activations, score_artifact
from config import cfg

def test_medians():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.milk10k_csv,
        img_dir=cfg.paths.milk10k_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )
    
    # Load model and SAE
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    sae_path = os.path.join(cfg.paths.results_dir, "sae_weights.pth")
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    # I can just load X from a cached file or extract it
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    base_model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_1337", "best_model.pth"), map_location=device)
    base_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    
    print("Extracting X...")
    X, _, _ = extract_all_activations(base_model, loader, device)
    
    print("Extracting SAE...")
    with torch.no_grad():
        _, sparse_acts = sae(X.to(device))
    feat1449 = sparse_acts[:, 1449].cpu().numpy()
    
    print("Scoring heuristics...")
    heuristic_scores = []
    for filepath in dataset.df['filepath']:
        score = score_artifact(filepath)
        heuristic_scores.append(1 if score > 0.5 else 0)
        
    df = pd.DataFrame({'f1449': feat1449, 'heuristic': heuristic_scores})
    
    acts_present = df[df['heuristic'] == 1]['f1449'].values
    acts_absent = df[df['heuristic'] == 0]['f1449'].values
    
    print("Median heuristic=1:", np.median(acts_present))
    print("Median heuristic=0:", np.median(acts_absent))
    
if __name__ == "__main__":
    test_medians()
