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
    print("Visualizing Attention Maps (Multiple Samples)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CrossAttentionT2VClassifier().to(device)
    ckpt_path = f"{cfg.paths.checkpoint_dir}/Cross-Attention_T2V_seed_456/best_model.pth"
    if not os.path.exists(ckpt_path): return
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv, img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer, transform=get_transforms()
    )
    
    # Shuffle False to get consistent interesting samples, just pick specific indices
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    inv_mean = [-m/s for m, s in zip(cfg.data.img_mean, cfg.data.img_std)]
    inv_std = [1/s for s in cfg.data.img_std]
    def denormalize(tensor):
        t = tensor.clone().detach().squeeze(0).cpu()
        for c, m, s in zip(t, inv_mean, inv_std): c.sub_(m).div_(s)
        t = t.numpy().transpose(1, 2, 0)
        return np.clip(t, 0, 1)

    samples_to_grab = [5, 12, 42, 88] # somewhat arbitrary, just to get different classes
    saved_count = 1
    
    for i, batch in enumerate(loader):
        if i not in samples_to_grab: continue
        
        img = batch['image'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        label = batch['label'].to(device)
        
        with torch.no_grad():
            _ = model(img, ids, mask)
        
        attn = model.last_attention_weights
        seq_len = mask[0].sum().item()
        valid_attn = attn[0, :seq_len, :]
        mean_attn = valid_attn.mean(dim=0)
        
        spatial_attn = mean_attn[1:].reshape(14, 14).cpu().numpy()
        spatial_attn = (spatial_attn - spatial_attn.min()) / (spatial_attn.max() - spatial_attn.min() + 1e-8)
        attn_map = cv2.resize(spatial_attn, (cfg.data.img_size, cfg.data.img_size))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        orig_img = denormalize(img)
        orig_img_uint8 = np.uint8(255 * orig_img)
        overlay = cv2.addWeighted(orig_img_uint8, 0.5, heatmap, 0.5, 0)
        
        text_str = tokenizer.decode(ids[0][:seq_len], skip_special_tokens=True)
        class_name = [k for k, v in cfg.data.LABEL_MAP.items() if v == label.item()][0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Original Image\n{class_name}")
        axes[0].axis('off')
        axes[1].imshow(attn_map, cmap='jet')
        axes[1].set_title("Raw Attention Map")
        axes[1].axis('off')
        axes[2].imshow(overlay)
        axes[2].set_title(f"Attention Overlay")
        axes[2].axis('off')
        
        plt.suptitle(f"Text: '{text_str}'", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"results/attention_sample_{saved_count}.png", dpi=150)
        plt.close()
        saved_count += 1
        
if __name__ == "__main__":
    main()
