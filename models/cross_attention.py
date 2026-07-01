"""Cross-Attention multimodal classifier with bidirectional variants."""

import torch
import torch.nn as nn
import timm
from transformers import AutoModel
from config import cfg


class CrossAttentionClassifier(nn.Module):
    """
    Cross-Attention: Vision attends to Text (V→T direction).
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
        
        # Cross-Attention layer (Vision attends to Text)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.model.vision_dim,
            num_heads=cfg.model.num_attention_heads,
            dropout=cfg.model.attention_dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Linear(cfg.model.vision_dim, cfg.model.num_classes)
    
    def forward(self, image, input_ids, attention_mask):
        # Vision pathway
        vision_features = self.vision_encoder(image)  # (B, 768) -> reshape to (B, 1, 768)
        vision_seq = vision_features.unsqueeze(1)  # (B, 1, 768)
        
        # Text pathway
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = text_outputs.last_hidden_state  # (B, seq_len, 768)
        
        # Create key padding mask for attention (True = padding position)
        key_padding_mask = (attention_mask == 0)  # (B, seq_len)
        
        # Cross-attention: Vision (query) attends to Text (key/value)
        attn_output, attn_weights = self.cross_attn(
            query=vision_seq,
            key=text_seq,
            value=text_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Extract CLS-like representation
        fused_cls = attn_output.squeeze(1)  # (B, 768)
        
        # Classify
        logits = self.classifier(fused_cls)  # (B, 6)
        
        return logits, fused_cls, vision_features


class CrossAttentionV2TClassifier(nn.Module):
    """
    Cross-Attention V→T: Vision attends to Text (explicit directional variant).
    Same as CrossAttentionClassifier but with explicit naming for clarity.
    """
    def __init__(self):
        super().__init__()
        self.vision_encoder = timm.create_model(
            cfg.model.vision_backbone,
            pretrained=True,
            num_classes=0
        )
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        self.text_encoder = AutoModel.from_pretrained(cfg.model.text_backbone)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.model.vision_dim,
            num_heads=cfg.model.num_attention_heads,
            dropout=cfg.model.attention_dropout,
            batch_first=True
        )
        
        self.classifier = nn.Linear(cfg.model.vision_dim, cfg.model.num_classes)
    
    def forward(self, image, input_ids, attention_mask):
        vision_features = self.vision_encoder(image)
        vision_seq = vision_features.unsqueeze(1)
        
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = text_outputs.last_hidden_state
        
        key_padding_mask = (attention_mask == 0)
        
        attn_output, _ = self.cross_attn(
            query=vision_seq,
            key=text_seq,
            value=text_seq,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        fused_cls = attn_output.squeeze(1)
        logits = self.classifier(fused_cls)
        
        return logits, fused_cls, vision_features


class CrossAttentionT2VClassifier(nn.Module):
    """
    Cross-Attention T→V: Text attends to Vision (reverse direction).
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
        
        # Cross-Attention layer (Text attends to Vision)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.model.text_dim,
            num_heads=cfg.model.num_attention_heads,
            dropout=cfg.model.attention_dropout,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Linear(cfg.model.text_dim, cfg.model.num_classes)
        
        # Project to vision dimension for CKA compatibility
        self.to_vision_dim = nn.Linear(cfg.model.text_dim, cfg.model.vision_dim)
    
    def forward(self, image, input_ids, attention_mask):
        # Vision pathway
        vision_features = self.vision_encoder(image)  # (B, 768) -> reshape to (B, 1, 768)
        vision_seq = vision_features.unsqueeze(1)  # (B, 1, 768)
        
        # Text pathway
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = text_outputs.last_hidden_state  # (B, seq_len, 768)
        
        # Cross-attention: Text (query) attends to Vision (key/value)
        attn_output, attn_weights = self.cross_attn(
            query=text_seq,
            key=vision_seq,
            value=vision_seq,
            need_weights=False
        )
        
        # Pool over sequence (mean pooling)
        fused_repr = attn_output.mean(dim=1)  # (B, 768)
        
        # Classify
        logits = self.classifier(fused_repr)  # (B, 6)
        
        # Project to vision dimension for CKA compatibility
        fused_for_cka = self.to_vision_dim(fused_repr)  # (B, 768)
        
        return logits, fused_for_cka, vision_features
