import torch
from models import MoEFusionClassifier

def test_moe():
    print("Initializing MoEFusionClassifier...")
    model = MoEFusionClassifier()
    model.eval()
    
    # Dummy inputs
    B = 2
    # image: 3 channels, 224x224
    image = torch.randn(B, 3, 224, 224)
    # text: input_ids (sequence length 64)
    input_ids = torch.randint(0, 1000, (B, 64))
    # attention_mask
    attention_mask = torch.ones(B, 64)
    
    print("Running forward pass...")
    with torch.no_grad():
        logits, fused_repr, vision_features = model(image, input_ids, attention_mask)
        
    print(f"Output logits shape: {logits.shape}")
    print(f"Fused repr shape: {fused_repr.shape}")
    print(f"Vision features shape: {vision_features.shape}")
    
    assert logits.shape == (B, 6), f"Expected logits shape (B, 6), got {logits.shape}"
    assert fused_repr.shape == (B, 768), f"Expected fused_repr shape (B, 768), got {fused_repr.shape}"
    assert vision_features.shape == (B, 768), f"Expected vision_features shape (B, 768), got {vision_features.shape}"
    
    print("Forward pass successful! All shapes match expected output.")

if __name__ == "__main__":
    test_moe()
