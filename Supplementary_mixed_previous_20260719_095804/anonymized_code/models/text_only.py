"""Text-only classifier (text baseline)."""

import torch
import torch.nn as nn
from transformers import AutoModel
from config import cfg


class TextOnlyClassifier(nn.Module):
    """
    Text-Only: Uses only the frozen Bio_ClinicalBERT backbone, no image input.
    Serves as the text baseline for collapse testing.
    """
    def __init__(self):
        super().__init__()
        
        # Frozen Text Encoder
        self.text_encoder = AutoModel.from_pretrained(cfg.model.text_backbone)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Classification head (trainable)
        self.classifier = nn.Linear(cfg.model.text_dim, cfg.model.num_classes)
        
        # Project to vision dimension for CKA compatibility
        self.to_vision_dim = nn.Linear(cfg.model.text_dim, cfg.model.vision_dim)
    
    def forward(self, image, input_ids, attention_mask):
        # Text pathway only
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token (B, 768)
        
        # Classify
        logits = self.classifier(text_features)  # (B, 6)
        
        # Project to vision dimension for CKA compatibility
        text_for_cka = self.to_vision_dim(text_features)  # (B, 768)
        
        # For CKA compatibility, return projected text as both fused and a dummy visual
        return logits, text_for_cka, text_for_cka
