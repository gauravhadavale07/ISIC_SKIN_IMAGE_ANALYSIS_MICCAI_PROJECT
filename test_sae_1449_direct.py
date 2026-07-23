import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import extract_all_activations, TopKSAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

dataset = MultimodalDermatologyDataset(
    csv_file=cfg.paths.milk10k_csv,
    img_dir=cfg.paths.milk10k_img_dir,
    tokenizer=tokenizer,
    transform=get_transforms()
)
# Just take a subset for quick test
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

base_model = CrossAttentionT2VClassifier().to(device)
ckpt_path = os.path.join(cfg.paths.checkpoint_dir, "Cross-Attention_T→V_seed_1337", "best_model.pth")
ckpt = torch.load(ckpt_path, map_location=device)
base_model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))

sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
sae_path = os.path.join(cfg.paths.results_dir, "sae_weights.pth")
sae.load_state_dict(torch.load(sae_path, map_location=device))
sae.eval()

# We just want to find ANY batch where Feature 1449 fires.
# Task 14 used sae.encode(fused).
fire_count_direct = 0
fire_count_batch = 0

with torch.no_grad():
    for i, batch in enumerate(loader):
        if i > 200: break # don't check everything
        imgs = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        
        vision_seq = base_model.vision_encoder.forward_features(imgs)
        text_seq = base_model.text_encoder(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
        attn_output, _ = base_model.cross_attn(query=text_seq, key=vision_seq, value=vision_seq, need_weights=False)
        fused = attn_output.mean(dim=1)
        
        # Test 1: directly like Task 14
        sparse_acts = sae.encode(fused)
        fire_count_direct += (sparse_acts[:, 1449] > 0).sum().item()
        
        # Test 2: like Task 31
        X = fused.cpu()
        X_device = X.to(device)
        _, sparse_acts2 = sae(X_device)
        fire_count_batch += (sparse_acts2[:, 1449] > 0).sum().item()

print(f"Direct fires in first 200 batches: {fire_count_direct}")
print(f"Batch fires in first 200 batches: {fire_count_batch}")
