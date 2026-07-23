import os
import sys
import torch
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from models.cross_attention import CrossAttentionT2VClassifier
from task11_sae import TopKSAE
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model and SAE
    print("Loading model and SAE...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    model = CrossAttentionT2VClassifier().to(device)
    
    model_path = os.path.join(cfg.paths.checkpoint_dir, "Cross-Attention_T→V_seed_1337", "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Residual stream dim comes from cross-attention output (v_dim)
    # v_dim is 768 for vit-base
    d_model = 768 
    expansion_factor = 8
    k = 32
    sae = TopKSAE(d_model, expansion_factor, k).to(device)
    sae_path = os.path.join(cfg.paths.results_dir, "sae_weights.pth")
    sae.load_state_dict(torch.load(sae_path, map_location=device))
    sae.eval()

    # 2. Get the outlier image
    df = pd.read_csv(cfg.paths.pad_ufes_csv)
    target_row = df.iloc[1587]
    img_path = target_row['filepath']
    print(f"Target Image: {img_path}")
    
    image = Image.open(img_path).convert('RGB')
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 3. Define 10 semantic variations of "Malignant" + 1 control
    test_texts = [
        "Biopsy confirmed melanoma.",
        "Fast-growing asymmetric dark spot.",
        "Patient history of malignant neoplasms.",
        "Suspicious lesion with irregular borders, high risk of malignancy.",
        "Pathology report indicates malignant melanoma.",
        "Rapidly evolving pigmented macule, deeply invasive.",
        "Malignant growth excised from the shoulder.",
        "Dermatoscopy shows atypical network and blue-white veil characteristic of melanoma.",
        "Confirmed case of skin cancer, malignant type.",
        "Aggressive spreading of a cancerous skin lesion.",
        "Benign nevus, no sign of malignancy." # Negative control
    ]
    
    feat_idx = 819
    results = []
    
    print(f"\n--- Testing feat_{feat_idx} Activation ---")
    
    fused_activations = []
    def get_fused_hook(module, input, output):
        fused_activations.append(output[0])
        
    hook = model.cross_attn.register_forward_hook(get_fused_hook)
    
    with torch.no_grad():
        for text in test_texts:
            fused_activations.clear()
            encoded = tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Forward pass to get cross-attention output
            _ = model(image_tensor, input_ids, attention_mask)
            # Fused residual stream is the mean across sequence length
            fused_stream = fused_activations[0].mean(dim=1) 
            
            # SAE extraction
            _, f_x = sae(fused_stream)
            act_val = f_x[0, feat_idx].item()
            results.append((text, act_val))
            
            print(f"Activation: {act_val:.6f} | Text: '{text}'")
            
    hook.remove()

if __name__ == "__main__":
    main()
