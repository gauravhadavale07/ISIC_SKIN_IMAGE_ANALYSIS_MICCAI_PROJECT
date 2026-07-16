"""Mixture of Experts (MoE) Adaptive Fusion classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel
from config import cfg


class MoEFusionClassifier(nn.Module):
    """
    MoE Fusion: Dynamically routes samples to modality-specific experts using a learned gating network.
    We implement a three-expert multimodal architecture. Expert specialization emerges through 
    end-to-end optimization rather than being manually prescribed.
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
            
        # Feature projections
        self.vision_proj = nn.Sequential(
            nn.Linear(cfg.model.vision_dim, cfg.model.capacity_matched_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.model.text_dim, cfg.model.capacity_matched_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # =======================
        # MoE Components
        # =======================
        
        # Router (Gating Network)
        # Looks at both modalities and decides which expert(s) to trust
        self.router = nn.Sequential(
            nn.Linear(cfg.model.vision_dim + cfg.model.text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3) # 3 Experts
        )
        
        # 3 Generic Experts
        # Specialization emerges via end-to-end routing optimization
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.model.capacity_matched_hidden_dim * 2, cfg.model.capacity_matched_hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.model.capacity_matched_hidden_dim, cfg.model.num_classes)
            ) for _ in range(3)
        ])
        
        # Project fused representation for CKA audit (to maintain compatibility with codebase)
        # We simulate a fused hidden state by weighting the hidden states.
        self.fused_proj = nn.Linear(cfg.model.capacity_matched_hidden_dim * 2, cfg.model.vision_dim)

    def forward(self, image, input_ids, attention_mask):
        B = image.size(0)
        
        # 1. Vision pathway
        vision_features = self.vision_encoder(image)  # (B, 768)
        
        # 2. Text pathway
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # CLS token (B, 768)
        
        # 3. Project to common dimension
        v_h = self.vision_proj(vision_features)  # (B, 512)
        t_h = self.text_proj(text_features)      # (B, 512)
        
        # 4. Routing
        router_input = torch.cat([vision_features, text_features], dim=1)
        routing_logits = self.router(router_input)      # (B, 3)
        routing_weights = F.softmax(routing_logits, dim=1) # (B, 3)
        
        # 5. Expert Execution
        fused_input = torch.cat([v_h, t_h], dim=1)
        expert_outputs = [expert(fused_input) for expert in self.experts]
        
        # Stack expert outputs: (B, 3, 6)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # 6. Weight by routing probabilities
        # routing_weights is (B, 3) -> unsqueeze to (B, 3, 1) to broadcast multiply
        # Sum over the 3 experts to get final (B, 6) logits
        logits = (expert_outputs * routing_weights.unsqueeze(-1)).sum(dim=1)
        
        # Create a fused representation for CKA compatibility
        h_fused_concat = torch.cat([v_h, t_h], dim=1)
        fused_repr = self.fused_proj(h_fused_concat)
        
        # Return standard tuple expected by trainer/auditor
        return logits, fused_repr, vision_features
