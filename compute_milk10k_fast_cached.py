import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import timm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from dataset import MultimodalDermatologyDataset, get_transforms
from config import cfg

from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionClassifier, CrossAttentionT2VClassifier
from models.image_only import ImageOnlyClassifier
from models.text_only import TextOnlyClassifier

device = cfg.train.device

print("Loading Backbones...")
vision_encoder = timm.create_model(cfg.model.vision_backbone, pretrained=True, num_classes=0).to(device)
vision_encoder.eval()
text_encoder = AutoModel.from_pretrained(cfg.model.text_backbone).to(device)
text_encoder.eval()

tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
dataset = MultimodalDermatologyDataset("./milk10k_clean_val_temp.csv", "./data/raw_milk10k/", tokenizer, get_transforms())
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

SEEDS = cfg.seeds

# Initialize dictionaries to hold predictions for all models
# predictions[arch_name][seed] = {"y_true": [], "y_prob": [], "y_pred": []}
MODELS = {
    "Image-Only": (ImageOnlyClassifier, "ImageOnly"),
    "Text-Only": (TextOnlyClassifier, "TextOnly"),
    "Late Fusion": (LateFusionClassifier, "Late_Fusion"),
    "GMU Baseline": (GMUClassifier, "GMU_Baseline"),
    "Cross-Attention (V->T)": (CrossAttentionClassifier, "Cross-Attention"),
    "Cross-Attention T->V": (CrossAttentionT2VClassifier, "Cross-Attention_T2V"),
}

predictions = {arch: {s: {"y_true": [], "y_prob": [], "y_pred": []} for s in SEEDS} for arch in MODELS}

# Load all model heads into memory
loaded_heads = {}
for arch, (cls, prefix) in MODELS.items():
    loaded_heads[arch] = {}
    for s in SEEDS:
        ckpt_path = f"./checkpoints/{prefix}_seed_{s}/best_model.pth"
        if os.path.exists(ckpt_path):
            model = cls().to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device).get("model_state_dict", torch.load(ckpt_path, map_location=device)))
            if hasattr(model, 'vision_encoder'):
                del model.vision_encoder
            if hasattr(model, 'text_encoder'):
                del model.text_encoder
            torch.cuda.empty_cache()
            model.eval()
            loaded_heads[arch][s] = model

print("Extracting embeddings and running fusion heads...")
with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        img = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].cpu().numpy()
        
        # 1. Forward Pass Backbones
        vision_seq = vision_encoder.forward_features(img)  # (B, 197, 768)
        vision_features = vision_seq[:, 0, :]  # CLS (B, 768)
        
        text_outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = text_outputs.last_hidden_state  # (B, seq_len, 768)
        text_features = text_seq[:, 0, :]  # CLS (B, 768)
        
        key_padding_mask = (attention_mask == 0)
        
        # 2. Forward Pass Fusion Heads
        for arch, seeds_dict in loaded_heads.items():
            for s, head in seeds_dict.items():
                
                if arch == "Image-Only":
                    logits = head.classifier(vision_features)
                elif arch == "Text-Only":
                    logits = head.classifier(text_features)
                elif arch == "Late Fusion":
                    combined = torch.cat([vision_features, text_features], dim=1)
                    logits = head.fusion(combined)
                elif arch == "GMU Baseline":
                    vision_proj = head.vision_proj(vision_features)
                    text_proj = head.text_proj(text_features)
                    gate_input = torch.cat([vision_features, text_features], dim=1)
                    gate = head.gate(gate_input)
                    h_fused = gate * vision_proj + (1 - gate) * text_proj
                    logits = head.classifier(h_fused)
                elif arch == "Cross-Attention (V->T)":
                    v_seq_unsqueeze = vision_features.unsqueeze(1)
                    attn_out, _ = head.cross_attn(query=v_seq_unsqueeze, key=text_seq, value=text_seq, key_padding_mask=key_padding_mask, need_weights=False)
                    logits = head.classifier(attn_out.squeeze(1))
                elif arch == "Cross-Attention T->V":
                    attn_out, _ = head.cross_attn(query=text_seq, key=vision_seq, value=vision_seq, need_weights=False)
                    fused = attn_out.mean(dim=1)
                    logits = head.classifier(fused)
                
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                
                predictions[arch][s]["y_true"].extend(labels)
                predictions[arch][s]["y_prob"].extend(probs)
                predictions[arch][s]["y_pred"].extend(preds)
        
        if batch_idx % 5 == 0:
            print(f"Processed batch {batch_idx+1}/{len(loader)}")

print("\nComputing Final Metrics...")
results = {}
for arch in MODELS:
    accs, f1s, aurocs = [], [], []
    for s in SEEDS:
        if s not in loaded_heads[arch]: continue
        y_t = np.array(predictions[arch][s]["y_true"])
        y_p = np.array(predictions[arch][s]["y_pred"])
        y_prob = np.array(predictions[arch][s]["y_prob"])
        
        accs.append(accuracy_score(y_t, y_p))
        f1s.append(f1_score(y_t, y_p, average='macro'))
        
        try:
            uniq = np.unique(y_t)
            if len(uniq) == 6:
                aurocs.append(roc_auc_score(y_t, y_prob, multi_class='ovr', average='macro'))
            else:
                p_filt = y_prob[:, uniq]
                p_filt = p_filt / p_filt.sum(axis=1, keepdims=True)
                aurocs.append(roc_auc_score(y_t, p_filt, multi_class='ovr', average='macro', labels=uniq))
        except:
            pass
    if accs:
        results[arch] = (np.mean(accs)*100, np.mean(f1s)*100, np.mean(aurocs))
        print(f"{arch:25s}: Acc={results[arch][0]:.2f}%, F1={results[arch][1]:.2f}%, AUROC={results[arch][2]:.4f}")

# Update LaTeX file automatically
with open("paper/main_template.tex", "r") as f:
    content = f.read()

for arch, var_prefix in [
    ("Image-Only", "IMG"), ("Text-Only", "TXT"), ("Late Fusion", "LF"), 
    ("GMU Baseline", "GMU"), ("Cross-Attention (V->T)", "VT"), ("Cross-Attention T->V", "TV")
]:
    if arch in results:
        acc, f1, auc = results[arch]
        content = content.replace(f"$MILK10K_{var_prefix}_ACC", f"{acc:.2f}\\%")
        content = content.replace(f"$MILK10K_{var_prefix}_F1", f"{f1:.2f}\\%")
        content = content.replace(f"$MILK10K_{var_prefix}_AUC", f"{auc:.3f}")

with open("paper/main.tex", "w") as f:
    f.write(content)

print("Updated main.tex with real numbers!")
