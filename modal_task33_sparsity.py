import modal
import subprocess

app = modal.App("miccai-task33-sparsity")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio",
    "pandas", "numpy", "scipy", "scikit-learn",
    "transformers", "timm",
    "huggingface_hub==0.23.2",
    "accelerate", "pillow", "statsmodels", "tqdm"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT",
    remote_path="/root/project",
    ignore=["data", "results", "logs", "checkpoints", ".git", "__pycache__"]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data",
    remote_path="/root/project/data"
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/checkpoints",
    remote_path="/root/project/checkpoints"
)

@app.function(
    gpu="T4",
    image=image,
    timeout=3600,
    volumes={"/root/project/results": vol_results},
)
def run_task():
    script = """
import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE
from task33_audit_extraction import build_2x2_dataframe

def main():
    print("Running Sparsity Extraction on GPU...")
    device = torch.device('cuda')
    
    # 1. Prepare data
    df_2x2 = build_2x2_dataframe(cfg.paths.pad_ufes_csv)
    temp_csv = "results/temp_sparsity.csv"
    df_2x2.to_csv(temp_csv, index=False)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = MultimodalDermatologyDataset(
        csv_file=temp_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 2. Load model
    model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(os.path.join(cfg.paths.checkpoint_dir, "Cross-Attention_T→V_seed_1337", "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()
    
    sae = TopKSAE(768, 8, 32).to(device)
    sae.load_state_dict(torch.load(os.path.join(cfg.paths.results_dir, "sae_weights.pth"), map_location=device))
    sae.eval()

    all_acts = []
    mse_losses = []
    variances = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            vision_seq = model.vision_encoder.forward_features(imgs)
            text_seq = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
            
            attn_output, _ = model.cross_attn(query=text_seq, key=vision_seq, value=vision_seq, need_weights=False)
            fused = attn_output.mean(dim=1)
            
            # SAE Forward
            x_hat, f_x = sae(fused)
            all_acts.append(f_x.cpu().numpy())
            
            # Metrics
            mse = torch.nn.functional.mse_loss(x_hat, fused, reduction='none').mean(dim=1).cpu().numpy()
            var = fused.var(dim=1).cpu().numpy()
            mse_losses.append(mse)
            variances.append(var)
            
    all_acts = np.concatenate(all_acts, axis=0) # Shape: [N, 6144]
    mse_losses = np.concatenate(mse_losses, axis=0)
    variances = np.concatenate(variances, axis=0)
    
    # Global Metrics
    avg_mse = float(mse_losses.mean())
    avg_var = float(variances.mean())
    fvu = avg_mse / avg_var if avg_var > 0 else 0
    
    # Sparsity Metrics
    non_zero_counts = (all_acts > 0).sum(axis=0) # Shape: [6144]
    dead_features = int((non_zero_counts == 0).sum())
    
    # Count of activations per feature
    activation_counts = {int(k): int(v) for k, v in enumerate(non_zero_counts)}
    
    stats = {
        "reconstruction_mse": avg_mse,
        "fvu": fvu,
        "dead_features_count": dead_features,
        "dead_features_fraction": dead_features / 6144,
        "total_samples": int(all_acts.shape[0]),
        "feature_activation_counts": activation_counts
    }
    
    with open("results/task33_global_sparsity_stats.json", "w") as f:
        json.dump(stats, f)
        
    print("Saved stats successfully.")
    
if __name__ == '__main__':
    main()
"""
    with open("run_sparsity.py", "w") as f:
        f.write(script)
        
    subprocess.run(["python3", "run_sparsity.py"], check=True)
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_task.remote()
