"""Image-only classifier (vision baseline)."""

import torch
import torch.nn as nn
import timm
from config import cfg


class ImageOnlyClassifier(nn.Module):
    """
    Image-Only: Uses only the frozen ViT-B/16 backbone, no text input.
    Serves as the vision baseline for collapse testing.
    """
    def __init__(self):
        super().__init__()
        
        # Frozen Vision Encoder
        self.vision_encoder = timm.create_model(
            cfg.model.vision_backbone,
            pretrained=True,
            num_classes=0
        )
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Classification head (trainable)
        self.classifier = nn.Linear(cfg.model.vision_dim, cfg.model.num_classes)
    
    def forward(self, image, input_ids, attention_mask):
        # Vision pathway only
        vision_features = self.vision_encoder(image)  # (B, 768)
        
        # Classify
        logits = self.classifier(vision_features)  # (B, 6)
        
        # For CKA compatibility, return vision features as both fused and visual
        return logits, vision_features, vision_features
