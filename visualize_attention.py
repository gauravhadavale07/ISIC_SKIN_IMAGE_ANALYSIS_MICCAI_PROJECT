import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier

def main():
    print("Visualizing Attention Maps...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model (Cross-Attention T->V, Seed 456)
    model = CrossAttentionT2VClassifier().to(device)
    ckpt_path = f"{cfg.paths.checkpoint_dir}/Cross-Attention_T2V_seed_456/best_model.pth"
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    # 2. Load Dataset & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    
    # Use the OOD test set for a real challenge case
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()  # Includes normalization, we need inverse later
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    # Inverse transform for normalization
    inv_mean = [-m/s for m, s in zip(cfg.data.img_mean, cfg.data.img_std)]
    inv_std = [1/s for s in cfg.data.img_std]
    
    def denormalize(tensor):
        t = tensor.clone().detach().squeeze(0).cpu()
        for c, m, s in zip(t, inv_mean, inv_std):
            c.sub_(m).div_(s)
        t = t.numpy().transpose(1, 2, 0)
        t = np.clip(t, 0, 1)
        return t

    # 3. Find a sample to visualize
    sample_found = False
    for i, batch in enumerate(loader):
        if i < 5: continue # Skip a few to get a random one
        
        img = batch['image'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(img, ids, mask)
        
        # Get attention weights (B=1, seq_len, 197)
        # Note: 197 = 1 CLS token + 14x14 spatial patches
        attn = model.last_attention_weights # (1, 128, 197)
        
        # We average attention across all non-padding text tokens to see what the whole text pathway is looking at
        seq_len = mask[0].sum().item()
        # Shape: (seq_len, 197)
        valid_attn = attn[0, :seq_len, :]
        
        # Average across text tokens -> shape (197,)
        mean_attn = valid_attn.mean(dim=0)
        
        # Discard CLS token (index 0) and reshape spatial patches (1:197) to 14x14
        spatial_attn = mean_attn[1:].reshape(14, 14).cpu().numpy()
        
        # Normalize to [0, 1] for visualization
        spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-8)
        
        # Resize to image size (224, 224)
        attn_map = cv2.resize(spatial_attn, (cfg.data.img_size, cfg.data.img_size))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Get original image
        orig_img = denormalize(img)
        orig_img_uint8 = np.uint8(255 * orig_img)
        
        # Overlay
        overlay = cv2.addWeighted(orig_img_uint8, 0.5, heatmap, 0.5, 0)
        
        # Decode text
        text_str = tokenizer.decode(ids[0][:seq_len], skip_special_tokens=True)
        class_name = [k for k, v in cfg.data.LABEL_MAP.items() if v == label.item()][0]
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Original Image\nClass: {class_name}")
        axes[0].axis('off')
        
        axes[1].imshow(attn_map, cmap='jet')
        axes[1].set_title("Raw Attention Map (14x14->224x224)")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Text Attention Overlay")
        axes[2].axis('off')
        
        plt.suptitle(f"Text Input: '{text_str}'\nThis map shows where the Text pathway is looking in the Image.", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"results/attention_visualization_sample.png", dpi=150)
        plt.close()
        
        print(f"Saved visualization to results/attention_visualization_sample.png")
        sample_found = True
        break
        
if __name__ == "__main__":
    main()
