import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from torch.utils.data import DataLoader
from models.cross_attention import CrossAttentionT2VClassifier
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer

# Set seed for reproducibility
torch.manual_seed(1337)
np.random.seed(1337)

def main():
    print("======================================================================")
    print("TASK 12: CONTRASTIVE ACTIVATION STEERING (ACT-ADD)")
    print("======================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
        transform=get_transforms()
    )
    test_loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)
    
    ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"Cross-Attention_T→V_seed_1337", "best_model.pth")
    model = CrossAttentionT2VClassifier().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model.eval()

    # 3. Compute Mean Activations
    print("\nPhase 1: Computing Mean Activations...")
    real_fused_reprs = []
    neutral_fused_reprs = []
    
    # We will hook the classifier to just extract the fused representation
    extracted_features = []
    def extract_hook(module, args):
        extracted_features.append(args[0].detach().cpu())
        return args

    hook_handle = model.classifier.register_forward_pre_hook(extract_hook)
    
    # Real Text Pass
    print("Extracting Real Text Representations...")
    baseline_preds = []
    neutral_preds = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            extracted_features.clear()
            logits, _, _ = model(images, input_ids, attn_mask)
            real_fused_reprs.append(extracted_features[0])
            baseline_preds.append(logits.argmax(dim=1).cpu())
            labels_list.append(labels.cpu())
            
    real_fused = torch.cat(real_fused_reprs, dim=0)
    baseline_preds = torch.cat(baseline_preds, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)
    
    mean_real = real_fused.mean(dim=0)
    
    # Neutral Text Pass
    print("Extracting Neutral Text Representations...")
    neutral_text = "No clinical history available"
    neutral_encoded = tokenizer(
        neutral_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['image'].to(device)
            b_size = images.size(0)
            n_ids = neutral_encoded['input_ids'].expand(b_size, -1).to(device)
            n_mask = neutral_encoded['attention_mask'].expand(b_size, -1).to(device)
            
            extracted_features.clear()
            logits, _, _ = model(images, n_ids, n_mask)
            neutral_fused_reprs.append(extracted_features[0])
            neutral_preds.append(logits.argmax(dim=1).cpu())
            
    neutral_fused = torch.cat(neutral_fused_reprs, dim=0)
    neutral_preds = torch.cat(neutral_preds, dim=0)
    mean_neutral = neutral_fused.mean(dim=0)
    
    hook_handle.remove()
    
    # 4. Compute Steering Vector
    v_steer = (mean_neutral - mean_real).to(device)
    print(f"\nSteering Vector Computed. L2 Norm: {torch.norm(v_steer).item():.4f}")
    
    # 5. Inference with Steering
    alphas = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    
    print("\nPhase 2: Steering Evaluation")
    
    # We must use a list to store alpha to be accessible in the hook
    hook_state = {'alpha': 0.0}
    
    def steering_hook(module, args):
        fused = args[0]
        steered_fused = fused + hook_state['alpha'] * v_steer.unsqueeze(0)
        return (steered_fused,)
        
    steer_handle = model.classifier.register_forward_pre_hook(steering_hook)
    
    for alpha in alphas:
        hook_state['alpha'] = alpha
        steered_preds = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                
                logits, _, _ = model(images, input_ids, attn_mask)
                steered_preds.append(logits.argmax(dim=1).cpu())
                
        steered_preds = torch.cat(steered_preds, dim=0)
        
        acc = accuracy_score(labels_tensor.numpy(), steered_preds.numpy())
        cfr = (steered_preds != baseline_preds).float().mean().item()
        shortcut_removal = (steered_preds == neutral_preds).float().mean().item()
        
        print(f"Alpha: {alpha:4.1f} | Accuracy: {acc:.4f} | CFR: {cfr:.4f} | Match Neutral: {shortcut_removal:.4f}")
        
        results.append({
            'alpha': alpha,
            'accuracy': acc,
            'cfr': cfr,
            'match_neutral': shortcut_removal
        })
        
    steer_handle.remove()
    
    # Save results
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/task12_contrastive_steering.csv", index=False)
    print("\nSaved results to results/task12_contrastive_steering.csv")

if __name__ == "__main__":
    main()
