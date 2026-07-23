import torch
from torch.utils.data import DataLoader
import pandas as pd
from models.cross_attention import CrossAttentionT2VClassifier
from dataset import MultimodalDermatologyDataset, get_transforms
from transformers import AutoTokenizer
from config import cfg

def run_scaffold_ablation():
    print("--- Running Task 18: Cross-Attention Scaffold Ablation ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = CrossAttentionT2VClassifier().to(device)
    try:
        ckpt = torch.load('checkpoints/Cross-Attention_T→V_seed_456/best_model.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Loaded CrossAttentionT2VClassifier checkpoint.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
        
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    val_transforms = get_transforms()
    test_ds = MultimodalDermatologyDataset('pad_ufes_20_test.csv', img_dir=None, tokenizer=tokenizer, transform=val_transforms, max_length=cfg.data.max_text_len)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    correct_baseline = 0
    correct_ablated = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 1. Baseline Forward
            logits_base, _, _ = model(images, input_ids, attention_mask)
            preds_base = torch.argmax(logits_base, dim=1)
            correct_baseline += (preds_base == labels).sum().item()
            
            # 2. Ablated Forward (Zero out [CLS]=101 and [SEP]=102 in attention mask)
            ablated_mask = attention_mask.clone()
            
            # Find CLS and SEP tokens
            cls_positions = (input_ids == 101)
            sep_positions = (input_ids == 102)
            
            ablated_mask[cls_positions] = 0
            ablated_mask[sep_positions] = 0
            
            logits_ablated, _, _ = model(images, input_ids, ablated_mask)
            preds_ablated = torch.argmax(logits_ablated, dim=1)
            correct_ablated += (preds_ablated == labels).sum().item()
            
            total += labels.size(0)
            
    acc_base = correct_baseline / total * 100
    acc_ablated = correct_ablated / total * 100
    
    print(f"Baseline Accuracy (Full Scaffold): {acc_base:.2f}%")
    print(f"Ablated Accuracy (No CLS/SEP): {acc_ablated:.2f}%")
    
    with open('results/task18_scaffold_ablation.csv', 'w') as f:
        f.write("Condition,Accuracy\n")
        f.write(f"Baseline,{acc_base:.2f}\n")
        f.write(f"ScaffoldAblated,{acc_ablated:.2f}\n")
    print("Results saved to results/task18_scaffold_ablation.csv")

if __name__ == "__main__":
    run_scaffold_ablation()
