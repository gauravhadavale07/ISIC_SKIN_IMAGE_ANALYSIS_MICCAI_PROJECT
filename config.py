import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class PathConfig:
    """Centralized path management for datasets and artifact logging."""
    # Dataset Paths - FIXED to local directory
    milk10k_csv: str = "./milk10k_train.csv"
    milk10k_img_dir: str = "./data/raw_milk10k"
    pad_ufes_csv: str = "./pad_ufes_20_test.csv"
    pad_ufes_img_dir: str = "./data/raw_pad_ufes"

    # Output Paths
    checkpoint_dir: str = "./checkpoints"
    results_dir: str = "./results"
    log_dir: str = "./logs"

    def __post_init__(self):
        """Ensure output directories exist before runtime."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

@dataclass
class DataConfig:
    """Configuration for data preprocessing and label harmonization."""
    img_size: int = 224
    max_text_len: int = 128

    # Normalization statistics strictly locked to ImageNet priors
    img_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    img_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # FIX: Canonical class-name -> integer-index mapping (6-class). This is now
    # the SINGLE SOURCE OF TRUTH for label indices. It used to be duplicated
    # inline inside dataset.py (a separate hardcoded dict), which is exactly
    # the kind of drift that let counterfactual.py silently assume a binary
    # (0/1) problem instead of this real 6-class one. dataset.py now reads
    # this directly, and counterfactual.py uses it (together with
    # LABEL_MAPPING below) to build a correct benign/malignant routing table.
    LABEL_MAP: Dict[str, int] = field(default_factory=lambda: {
        'MEL': 0,
        'BCC': 1,
        'SCC': 2,
        'ACK': 3,
        'NEV': 4,
        'SEK': 5
    })

    # 6-Class to Binary Mapping definition
    # 0 = Benign, 1 = Malignant
    # FIX: this dict previously existed but was never referenced anywhere in
    # the pipeline ("dead config"). It is now actively used by
    # CounterfactualAuditor to decide which override text (benign vs.
    # malignant) to inject for EACH of the 6 diagnostic classes — fixing the
    # bug where only label==0 / label==1 were handled and labels 2-5
    # (SCC/ACK/NEV/SEK, 61% of PAD-UFES-20) silently received an all-zero
    # "counterfactual" that was really just a second blank-text probe.
    LABEL_MAPPING: Dict[str, int] = field(default_factory=lambda: {
        "ACK": 0, "NEV": 0, "SEK": 0,  # Benign Cohort
        "MEL": 1, "BCC": 1, "SCC": 1   # Malignant Cohort
    })

    def __post_init__(self):
        """
        Defensive check: LABEL_MAP and LABEL_MAPPING must describe the exact
        same set of classes. If they ever drift apart again (e.g. someone
        adds a 7th class to one but not the other), this fails loudly at
        startup instead of silently dropping samples deep inside the
        counterfactual auditor the way the original bug did.
        """
        assert set(self.LABEL_MAP.keys()) == set(self.LABEL_MAPPING.keys()), (
            f"DataConfig.LABEL_MAP keys {set(self.LABEL_MAP.keys())} must match "
            f"DataConfig.LABEL_MAPPING keys {set(self.LABEL_MAPPING.keys())}."
        )

@dataclass
class TrainConfig:
    """Identical training hyperparameters for all architectures to ensure fair comparison."""
    # Core loop
    batch_size: int = 16
    epochs: int = 5
    patience: int = 2  # For Early Stopping

    # Optimizer (AdamW)
    learning_rate: float = 2e-5
    weight_decay: float = 0.01

    # Scheduler
    warmup_ratio: float = 0.1

    # Hardware & Precision
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp: bool = True  # Automatic Mixed Precision for VRAM efficiency
    max_grad_norm: float = 1.0  # Crucial for deep attention networks

@dataclass
class ModelConfig:
    """Architectural dimensions and pre-trained backbone pointers."""
    # NOTE: this is now the ONE vision backbone string used by ALL THREE
    # architectures (LateFusion, GMU, CrossAttention) via timm.create_model().
    # Previously LateFusionClassifier/GMUClassifier hardcoded torchvision's
    # vit_b_16(IMAGENET1K_V1) and ignored this field entirely, while only
    # CrossAttentionClassifier actually read it — meaning two DIFFERENT
    # pretrained ViT-B/16 checkpoints were being compared, not one held
    # constant. Confirmed empirically: a real run showed Vis_Feat_Norm=18.50
    # for the torchvision backbone vs. 34.78 for the timm backbone on
    # identical input images.
    vision_backbone: str = "vit_base_patch16_224"  # timm backbone, shared by all architectures
    text_backbone: str = "emilyalsentzer/Bio_ClinicalBERT"

    # Latent dimensions
    vision_dim: int = 768
    text_dim: int = 768
    num_classes: int = 6

    # Cross Attention Specifics
    num_attention_heads: int = 8
    attention_dropout: float = 0.1

    # Capacity Matching Specifics
    # FIX: was 2048. The paper (Eq. 2) specifies W1 ∈ R^(512x1536), i.e. a
    # hidden dim of 512. At 2048, Late Fusion's trainable-parameter count
    # (~4.34M) was nearly DOUBLE Cross-Attention's (~2.37M) — the opposite of
    # "capacity matched". At 512, Late Fusion comes to ~1.97M trainable
    # params, much closer to Cross-Attention's ~2.37M, and matches the paper.
    capacity_matched_hidden_dim: int = 512

@dataclass
class AuditConfig:
    """Strings and settings required for the mechanistic behavioral audit."""
    blank_string: str = ""
    benign_override: str = "[CLINICAL OVERRIDE: stable benign melanocytic nevus]"
    malignant_override: str = "[CLINICAL OVERRIDE: highly suspicious for invasive melanoma]"

@dataclass
class ExperimentConfig:
    """Master configuration object wrapping all sub-configs."""
    protocol_version: str = "v3_nobiopsy_6seeds"
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999, 1337])

    # Instantiate sub-configs
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

# Global singleton to be imported across the pipeline
cfg = ExperimentConfig()