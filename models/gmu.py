"""Gated Multimodal Unit (GMU) fusion classifier."""

import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from config import cfg


class GMUClassifier(nn.Module):
    """
    GMU: Learn a gating mechanism to weight vision vs. text contributions.
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
        
        # Frozen Text Encoder
        self.text_encoder = AutoModel.from_pretrained(cfg.model.text_backbone)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # GMU components
        self.vision_proj = nn.Linear(cfg.model.vision_dim, cfg.model.capacity_matched_hidden_dim)
        self.text_proj = nn.Linear(cfg.model.text_dim, cfg.model.capacity_matched_hidden_dim)
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(cfg.model.vision_dim + cfg.model.text_dim, cfg.model.capacity_matched_hidden_dim),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Linear(cfg.model.capacity_matched_hidden_dim, cfg.model.num_classes)
        
        # Project fused representation for CKA compatibility
        self.fused_proj = nn.Linear(cfg.model.capacity_matched_hidden_dim, cfg.model.vision_dim)
    
    def forward(self, image, input_ids, attention_mask):
        # Vision pathway
        vision_features = self.vision_encoder(image)  # (B, 768)
        
        # Text pathway
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token (B, 768)
        
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # (B, 512)
        text_proj = self.text_proj(text_features)  # (B, 512)
        
        # Compute gate
        gate_input = torch.cat([vision_features, text_features], dim=1)  # (B, 1536)
        gate = self.gate(gate_input)  # (B, 512)
        
        # Gated fusion
        h_fused = gate * vision_proj + (1 - gate) * text_proj  # (B, 512)
        
        # Classify
        logits = self.classifier(h_fused)  # (B, 6)
        
        # Project fused representation for CKA audit
        fused_repr = self.fused_proj(h_fused)  # (B, 768)
        
        return logits, fused_repr, vision_features
