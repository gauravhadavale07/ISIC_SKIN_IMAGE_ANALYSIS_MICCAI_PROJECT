import sys
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from config import cfg

from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionClassifier, CrossAttentionT2VClassifier
from models.image_only import ImageOnlyClassifier
from models.text_only import TextOnlyClassifier

def main():
    print("="*70)
    print("CAPACITY & FLOP ACCOUNTING")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy inputs matching our shapes
    # Image: (B, C, H, W)
    dummy_img = torch.randn(1, 3, cfg.data.img_size, cfg.data.img_size).to(device)
    # Text input_ids and attention_mask
    dummy_ids = torch.ones(1, cfg.data.max_text_len, dtype=torch.long).to(device)
    dummy_mask = torch.ones(1, cfg.data.max_text_len, dtype=torch.long).to(device)
    
    inputs = (dummy_img, dummy_ids, dummy_mask)
    
    models_to_test = {
        "Image-Only": (ImageOnlyClassifier(), inputs),
        "Text-Only": (TextOnlyClassifier(), inputs),
        "Late Fusion": (LateFusionClassifier(), inputs),
        "GMU Baseline": (GMUClassifier(), inputs),
        "Cross-Attention (V->T)": (CrossAttentionClassifier(), inputs),
        "Cross-Attention T->V": (CrossAttentionT2VClassifier(), inputs),
    }
    
    print(f"{'Architecture':<25} | {'Total Params':<15} | {'Trainable Params':<18} | {'GFLOPs':<10}")
    print("-" * 75)
    
    for name, (model, inputs) in models_to_test.items():
        model = model.to(device)
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate FLOPs using fvcore
        # Note: Since the text encoder (ClinicalBERT) is frozen and its embeddings are pre-computed 
        # for our dataset, the inputs we feed to the models are (Image, Text_Embeddings).
        # We need to explicitly count the ViT FLOPs since it's inside the model, but NOT the 
        # ClinicalBERT FLOPs (since they are generated offline in the dataset prep).
        flops = FlopCountAnalysis(model, inputs)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total_flops = flops.total()
        
        gflops = total_flops / 1e9
        
        print(f"{name:<25} | {total_params:<15,} | {trainable_params:<18,} | {gflops:<10.3f}")

if __name__ == "__main__":
    main()
