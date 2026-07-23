import torch
from task37_demographic_sae import TopKSAE

# Load a single batch and model
import os
from config import cfg
from transformers import AutoTokenizer
from dataset import MultimodalDermatologyDataset, get_transforms
from torch.utils.data import DataLoader
from models.cross_attention import CrossAttentionT2VClassifier

def main():
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms(is_training=False)
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_1337", "best_model.pth")
    model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    # Create dummy SAE to check shift norm logic
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    # We will just see the ratio of decoded W to text_seq
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        text_outputs = model.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
        text_seq = text_outputs.last_hidden_state
        
        sparse = sae.encode(text_seq)
        
        # Pick feature 0
        f = 0
        act = sparse[:, :, f].unsqueeze(-1)
        dir_vec = sae.W_dec[f].unsqueeze(0).unsqueeze(0).to(device)
        shift = act * dir_vec
        
        print(f"Shift norm: {shift.norm().item():.4f}")
        print(f"Text norm: {text_seq.norm().item():.4f}")
        
        # also print what the sae normally does
        x_rec, _ = sae(text_seq)
        print(f"X_rec norm: {x_rec.norm().item():.4f}")
        break

if __name__ == "__main__":
    main()
