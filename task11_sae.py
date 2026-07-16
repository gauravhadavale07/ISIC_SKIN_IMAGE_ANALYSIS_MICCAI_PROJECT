import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier

class TopKSAE(nn.Module):
    def __init__(self, d_model, expansion_factor=8, k=32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.k = k
        
        self.W_enc = nn.Parameter(torch.randn(self.d_model, self.d_sae) / np.sqrt(self.d_model))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        
        # Tie weights or use separate? Usually separate for TopK
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, self.d_model) / np.sqrt(self.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))
        
    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        acts = torch.relu(pre_acts)
        
        # Top-K routing
        topk_vals, topk_indices = torch.topk(acts, self.k, dim=-1)
        sparse_acts = torch.zeros_like(acts).scatter_(-1, topk_indices, topk_vals)
        return sparse_acts
        
    def forward(self, x):
        sparse_acts = self.encode(x)
        x_reconstructed = sparse_acts @ self.W_dec + self.b_dec
        return x_reconstructed, sparse_acts

def extract_all_activations(model, dataloader, device):
    model.eval()
    all_acts = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Activations", leave=False):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            vision_seq = model.vision_encoder.forward_features(imgs)
            text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            text_seq = text_outputs.last_hidden_state
            
            attn_output, _ = model.cross_attn(
                query=text_seq,
                key=vision_seq,
                value=vision_seq,
                need_weights=False
            )
            # The residual stream we want to model is the fused representation
            fused = attn_output.mean(dim=1)
            all_acts.append(fused.cpu())
            labels.append(batch["label"].cpu())
            
    return torch.cat(all_acts, dim=0), torch.cat(labels, dim=0)

def main():
    print("=" * 70)
    print("TASK 11: TOP-K SPARSE AUTOENCODER (SAE) TRAINING")
    print("=" * 70)

    device = cfg.train.device
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    # Use training set to train the SAE
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.milk10k_csv,
        img_dir=cfg.paths.milk10k_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )
    
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)
    
    seed = 1337
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_{seed}", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
        
    base_model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    base_model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    
    print("\\nExtracting 768-D dense activations...")
    X, y = extract_all_activations(base_model, loader, device)
    
    print(f"Dataset shape: {X.shape}") # Should be (10000, 768)
    
    # Train SAE
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    
    dataset_tensors = torch.utils.data.TensorDataset(X)
    sae_loader = DataLoader(dataset_tensors, batch_size=256, shuffle=True)
    
    epochs = 20
    print(f"\\nTraining Top-32 SAE (Hidden Dim: {768 * 8}) for {epochs} epochs...")
    
    sae.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch_x,) in sae_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            
            x_reconstructed, _ = sae(batch_x)
            
            # MSE loss (No L1 loss needed for Top-K SAE!)
            loss = nn.MSELoss()(x_reconstructed, batch_x)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - MSE: {total_loss/len(sae_loader):.4f}")
            
    # Evaluation: Do specific features correlate with Malignant/Benign?
    sae.eval()
    X_device = X.to(device)
    with torch.no_grad():
        _, sparse_acts = sae(X_device)
        
    sparse_acts = sparse_acts.cpu().numpy()
    y = y.numpy()
    
    # Binary mapping (0=Benign, 1=Malignant based on cfg.data.LABEL_MAPPING)
    # Actually y is the 6-class index. Let's map to binary.
    is_malignant = np.zeros_like(y)
    for class_name, binary_label in cfg.data.LABEL_MAPPING.items():
        class_idx = cfg.data.LABEL_MAP[class_name]
        is_malignant[y == class_idx] = binary_label
        
    print("\\nAnalyzing SAE Features for Clinical Concept Alignment (Malignant vs Benign)...")
    # For each feature (6144), check correlation with malignancy
    # Using point-biserial correlation (Pearson correlation between binary and continuous)
    # Fast vectorized correlation
    acts_mean = sparse_acts.mean(axis=0)
    acts_std = sparse_acts.std(axis=0) + 1e-8
    acts_norm = (sparse_acts - acts_mean) / acts_std
    
    mal_mean = is_malignant.mean()
    mal_std = is_malignant.std() + 1e-8
    mal_norm = (is_malignant - mal_mean) / mal_std
    
    correlations = (acts_norm * mal_norm[:, None]).mean(axis=0)
    
    top_features = np.argsort(np.abs(correlations))[::-1][:10]
    
    results = []
    # To find top 5 images, we need the filepaths from the dataset
    filepaths = dataset.df['filepath'].values
    
    for feat in top_features:
        corr = correlations[feat]
        max_act = sparse_acts[:, feat].max()
        l0 = (sparse_acts[:, feat] > 0).sum()
        
        # Get top 5 indices for this feature
        top5_idx = np.argsort(sparse_acts[:, feat])[::-1][:5]
        top5_paths = filepaths[top5_idx].tolist()
        
        print(f"Feature {feat:4d}: Corr {corr:+.4f} | Fires in {l0:4d}/10000 images | Max Act: {max_act:.2f}")
        results.append({
            "Feature_ID": feat,
            "Correlation_with_Malignancy": corr,
            "L0_Count": l0,
            "Max_Activation": max_act,
            "Top5_Paths": ";".join(top5_paths)
        })
        
    df = pd.DataFrame(results)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    df.to_csv(os.path.join(cfg.paths.results_dir, "task11_sae_features.csv"), index=False)
    
    # Save SAE weights
    torch.save(sae.state_dict(), os.path.join(cfg.paths.results_dir, "sae_weights.pth"))
    print("\\nResults saved to results/task11_sae_features.csv and sae_weights.pth")
    
if __name__ == "__main__":
    main()
