import torch
from models.cross_attention import CrossAttentionClassifier
model = CrossAttentionClassifier()
print("V->T Total:", sum(p.numel() for p in model.parameters()))
