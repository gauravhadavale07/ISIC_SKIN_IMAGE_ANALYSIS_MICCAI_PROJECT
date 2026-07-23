import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
import re

from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier

# --- 1. Top-K SAE Definition ---
class TopKSAE(nn.Module):
    def __init__(self, d_model: int, expansion_factor: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_model * expansion_factor
        self.k = k
        self.W_enc = nn.Parameter(torch.randn(self.d_model, self.d_sae) / np.sqrt(self.d_model))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.randn(self.d_sae, self.d_model) / np.sqrt(self.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))
        
    def encode(self, x):
        acts = x @ self.W_enc + self.b_enc
        acts = torch.relu(acts)
        topk_vals, topk_indices = torch.topk(acts, self.k, dim=-1)
        sparse_acts = torch.zeros_like(acts).scatter_(-1, topk_indices, topk_vals)
        return sparse_acts

    def forward(self, x):
        sparse_acts = self.encode(x)
        x_reconstructed = sparse_acts @ self.W_dec + self.b_dec
        return x_reconstructed, sparse_acts

def extract_text_activations(model, dataloader, device):
    model.eval()
    all_acts = []
    labels = []
    metadata = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Text Acts", leave=False):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            text_seq = text_outputs.last_hidden_state  # (B, L, 768)
            
            # Extract valid tokens
            seq_mask = attn_mask.bool()
            valid_tokens = text_seq[seq_mask]
            all_acts.append(valid_tokens.cpu())
            
            labels.append(batch["label"].cpu())
            
            # Batch decoding
            decoded_texts = dataloader.dataset.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            for i in range(len(batch["label"])):
                metadata.append({"text": decoded_texts[i], "label": batch["label"][i].item()})
                
    return torch.cat(all_acts, dim=0), torch.cat(labels, dim=0), metadata

def main():
    torch.manual_seed(1337)
    np.random.seed(1337)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(is_training=False)
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    malignant_idx = [cfg.data.LABEL_MAP[lbl] for lbl, val in cfg.data.LABEL_MAPPING.items() if val == 1]
    benign_idx = [cfg.data.LABEL_MAP[lbl] for lbl, val in cfg.data.LABEL_MAPPING.items() if val == 0]

    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_1337", "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Error: missing model {ckpt_path}")
        return
    model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    print("Extracting Text Activations from PAD-UFES-20...")
    X_tokens, y, metadata = extract_text_activations(model, loader, device)
    
    print(f"Extracted {X_tokens.shape[0]} valid tokens.")
    
    print("Training Demographic SAE on All Text Tokens...")
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)
    dataset_tensors = TensorDataset(X_tokens)
    sae_loader = DataLoader(dataset_tensors, batch_size=256, shuffle=True)
    
    epochs = 10 # 10 epochs on dense tokens is plenty
    sae.train()
    for ep in range(epochs):
        total_loss = 0
        for batch in tqdm(sae_loader, desc=f"Epoch {ep}", leave=False):
            bx = batch[0].to(device)
            optimizer.zero_grad()
            x_rec, _ = sae(bx)
            loss = nn.MSELoss()(x_rec, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {ep} Loss: {total_loss/len(sae_loader):.4f}")

    sae.eval()
    
    # Extract Sequence-Level Activations to find correlations
    print("Evaluating Sequence-Level Feature Correlations...")
    seq_activations = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding sequences"):
            imgs = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            
            text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            text_seq = text_outputs.last_hidden_state  # (B, L, 768)
            
            sparse_acts_seq = sae.encode(text_seq) # (B, L, d_sae)
            
            # Mask out padding tokens
            seq_mask = attn_mask.unsqueeze(-1).expand_as(sparse_acts_seq).bool()
            sparse_acts_seq = sparse_acts_seq * seq_mask
            
            # Max pool over sequence length to get if feature fired in this sentence
            sparse_acts_pooled = sparse_acts_seq.max(dim=1).values # (B, d_sae)
            seq_activations.append(sparse_acts_pooled.cpu())
            
    seq_activations = torch.cat(seq_activations, dim=0) # (N, d_sae)
    
    ages = np.array([
        int(m.group(1)) if (m := re.search(r'\bage[:\s]+(\d+)\b', t['text'], re.IGNORECASE)) else 0 
        for t in metadata
    ])
    is_old = (ages >= 60).astype(float)
    is_male = np.array([1.0 if re.search(r'\bMale\b', t['text']) else 0.0 for t in metadata])
    is_malignant = np.array([lbl in malignant_idx for lbl in y.numpy()])
    
    # Calculate correlations
    age_corrs = []
    mal_corrs = []
    for f in range(seq_activations.shape[1]):
        f_vals = seq_activations[:, f].numpy()
        if (f_vals > 0).sum() < 20: 
            age_corrs.append(0)
            mal_corrs.append(0)
            continue
        age_corr = np.corrcoef(f_vals, is_old)[0, 1]
        mal_corr = np.corrcoef(f_vals, is_malignant)[0, 1]
        age_corrs.append(age_corr if not np.isnan(age_corr) else 0)
        mal_corrs.append(mal_corr if not np.isnan(mal_corr) else 0)
        
    age_corrs = np.array(age_corrs)
    top_age_features = np.argsort(np.abs(age_corrs))[-5:][::-1]
    
    print("\nTop 5 Age >= 60 Features:")
    for f in top_age_features:
        print(f"Feature {f}: Age Corr = {age_corrs[f]:.4f}, Malignancy Corr = {mal_corrs[f]:.4f}")

    def ablate_features(features_to_ablate):
        original_logits = []
        ablated_logits = []
        
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                
                vision_seq = model.vision_encoder.forward_features(imgs)
                text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
                text_seq = text_outputs.last_hidden_state
                
                attn_out_base, _ = model.cross_attn(query=text_seq, key=vision_seq, value=vision_seq, need_weights=False)
                fused_base = attn_out_base.mean(dim=1)
                logits_base = model.classifier(fused_base).cpu()
                
                sparse = sae.encode(text_seq) # (B, L, d_sae)
                shift = torch.zeros_like(text_seq)
                for f in features_to_ablate:
                    act = sparse[:, :, f].unsqueeze(-1)
                    dir_vec = sae.W_dec[f].unsqueeze(0).unsqueeze(0).to(device)
                    shift += act * dir_vec
                    
                text_seq_ablated = text_seq - shift
                
                if len(original_logits) == 0:
                    print(f"Text norm : {text_seq.norm().item():.4f}")
                    print(f"Shift norm: {shift.norm().item():.4f}")
                    print(f"Ratio: {shift.norm().item()/text_seq.norm().item():.6f}")
                    cos = torch.nn.functional.cosine_similarity(
                        text_seq.reshape(-1,768), text_seq_ablated.reshape(-1,768), dim=1
                    ).mean()
                    print(f"Cosine: {cos.item():.6f}")

                attn_out_abl, _ = model.cross_attn(query=text_seq_ablated, key=vision_seq, value=vision_seq, need_weights=False)
                
                if len(original_logits) == 0:
                    print(f"Attn out diff norm: {(attn_out_base - attn_out_abl).norm().item():.4f}")
                    print(f"Attn out diff mean abs: {torch.mean(torch.abs(attn_out_base - attn_out_abl)).item():.6f}")

                fused_abl = attn_out_abl.mean(dim=1)
                logits_abl = model.classifier(fused_abl).cpu()
                
                original_logits.append(logits_base)
                ablated_logits.append(logits_abl)
                
        original_logits = torch.cat(original_logits, dim=0)
        ablated_logits = torch.cat(ablated_logits, dim=0)
        
        def get_margin(logits):
            mal_probs = torch.softmax(logits, dim=-1)[:, malignant_idx].sum(dim=-1)
            ben_probs = torch.softmax(logits, dim=-1)[:, benign_idx].sum(dim=-1)
            return (mal_probs - ben_probs).mean().item()
            
        base_margin = get_margin(original_logits)
        abl_margin = get_margin(ablated_logits)
        
        base_preds = original_logits.argmax(dim=-1)
        abl_preds = ablated_logits.argmax(dim=-1)
        flip_rate = (base_preds != abl_preds).float().mean().item()
        
        return base_margin, abl_margin, abl_margin - base_margin, flip_rate
        
    print("\n--- Causal Intervention: Ablating Top N Age Features (Full Sequence) ---")
    os.makedirs("results", exist_ok=True)
    with open("results/demographic_sae_results.txt", "w") as f:
        for n in [1, 3, 5]:
            feats = top_age_features[:n]
            b_m, a_m, diff, flip = ablate_features(feats)
            print(f"\nAblating Top {n} Features: {feats.tolist()}")
            print(f"Baseline Margin: {b_m:.4f} | Ablated Margin: {a_m:.4f} | Shift: {diff:.4f} | Flips: {flip*100:.2f}%")
            
            f.write(f"Top {n} Features: {feats.tolist()}\n")
            f.write(f"Shift: {diff:.4f}\n")
            f.write(f"Flip Rate: {flip*100:.2f}%\n\n")

if __name__ == "__main__":
    main()
