"""Late Fusion multimodal classifier with capacity-matched hidden layer."""

import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from config import cfg


class LateFusionClassifier(nn.Module):
    """
    Late Fusion: Concatenate frozen ViT-B/16 and Bio_ClinicalBERT CLS embeddings,
    pass through capacity-matched hidden layer (512-d), then classify.
    """
    def __init__(self):
        super().__init__()
        
        # Frozen Vision Encoder (ViT-B/16 from timm)
        self.vision_encoder = timm.create_model(
            cfg.model.vision_backbone,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        # Frozen Text Encoder (Bio_ClinicalBERT)
        self.text_encoder = AutoModel.from_pretrained(cfg.model.text_backbone)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # Capacity-matched fusion layer (512-d per paper Eq. 2)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.model.vision_dim + cfg.model.text_dim, cfg.model.capacity_matched_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.model.capacity_matched_hidden_dim, cfg.model.num_classes)
        )
        
        # Project fused representation back to 768-d for CKA compatibility
        self.fused_proj = nn.Linear(cfg.model.capacity_matched_hidden_dim, cfg.model.vision_dim)
    
    def forward(self, image, input_ids, attention_mask):
        # Vision pathway
        vision_features = self.vision_encoder(image)  # (B, 768)
        
        # Text pathway
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token (B, 768)
        
        # Concatenate and fuse
        combined = torch.cat([vision_features, text_features], dim=1)  # (B, 1536)
        logits = self.fusion(combined)  # (B, 6)
        
        # Project fused representation for CKA audit
        fused_repr = self.fused_proj(combined[:, :cfg.model.capacity_matched_hidden_dim])  # (B, 768)
        
        return logits, fused_repr, vision_features
