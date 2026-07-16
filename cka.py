import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.amp import autocast
from tqdm import tqdm
from typing import Dict, Any
from config import cfg


class CKAAuditor:
    """
    Centered Kernel Alignment (CKA) Auditor.
    Quantifies the geometric divergence between the pre-fusion visual latent space
    and the post-fusion multimodal latent space.

    A CKA score near 1.0 means the fusion layer has not meaningfully changed the
    representational geometry — strong evidence of Modality Collapse.
    A CKA score well below 1.0 means text has actively warped the latent space.

    Reference: Kornblith et al., "Similarity of Neural Network Representations
               Revisited" (ICML 2019).

    CKA AUDIT CONTRACT (enforced by assertion in run_audit):
        Models must return (logits, fused_feat, vis_feat) where both
        fused_feat and vis_feat are (B, 768). Guaranteed by:
            - LateFusionClassifier:      fused_proj   (B, 768), v_feat     (B, 768)
            - CrossAttentionClassifier:  fused_cls    (B, 768), vis_cls    (B, 768)
            - GMUClassifier:             h_fused      (B, 768), v_feat     (B, 768)
    """
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def _center_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Centers columns of X by subtracting the column mean.
        Required for unbiased HSIC estimation underlying linear CKA.
        X: (N, D) -> returns (N, D)
        """
        return X - torch.mean(X, dim=0, keepdim=True)

    def _compute_linear_cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """
        Computes Linear CKA in feature space:
            CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

        This feature-space formulation is mathematically equivalent to the
        Gram-matrix CKA but runs in O(N * D^2) instead of O(N^2 * D),
        preventing RAM exhaustion on large datasets (e.g. full PAD-UFES-20).

        Args:
            X: (N, D) — pre-fusion visual features
            Y: (N, D) — post-fusion multimodal features
        Returns:
            CKA score in [0, 1]
        """
        X = self._center_features(X)    # (N, D)
        Y = self._center_features(Y)    # (N, D)

        # Feature-space cross/self dot products: (D, N) @ (N, D) -> (D, D)
        dot_XX = torch.matmul(X.t(), X)
        dot_YY = torch.matmul(Y.t(), Y)
        dot_XY = torch.matmul(X.t(), Y)

        # Frobenius norms
        norm_XY = torch.linalg.matrix_norm(dot_XY, ord='fro') ** 2
        norm_XX = torch.linalg.matrix_norm(dot_XX, ord='fro')
        norm_YY = torch.linalg.matrix_norm(dot_YY, ord='fro')

        cka_score = norm_XY / (norm_XX * norm_YY)
        return cka_score.item()

    @torch.no_grad()
    def run_audit(self, dataloader) -> Dict[str, Any]:
        """
        Executes the CKA Audit over the provided dataloader.

        Accumulates (vis_feat, fused_feat) pairs across all batches on CPU
        to avoid GPU OOM, then computes exact linear CKA on the full (N, 768)
        matrices in one shot.

        Returns:
            Dict with keys:
                Linear_CKA      — float in [0, 1]
                N_samples       — int, total samples processed
                Vis_Feat_Norm   — float, mean L2 norm of visual features (sanity check)
                Fused_Feat_Norm — float, mean L2 norm of fused features (sanity check)
        """
        self.model.eval()

        all_vis_feats   = []
        all_fused_feats = []

        pbar = tqdm(dataloader, desc="CKA Geometric Audit")

        for batch in pbar:
            imgs      = batch["image"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda"):
                # CKA Contract: models return (logits, fused_feat, vis_feat)
                _, fused_feat, vis_feat = self.model(imgs, input_ids, attn_mask)

            # Move to CPU immediately after each batch to prevent GPU OOM
            # .contiguous() ensures no non-contiguous memory layout from expand() ops
            all_vis_feats.append(vis_feat.float().contiguous().cpu())
            all_fused_feats.append(fused_feat.float().contiguous().cpu())

        # Concatenate all batches: (N, 768)
        X_vis   = torch.cat(all_vis_feats,   dim=0)    # (N, 768)
        Y_fused = torch.cat(all_fused_feats, dim=0)    # (N, 768)

        # Enforce CKA audit contract — catches backbone mismatches early
        assert X_vis.shape == Y_fused.shape, (
            f"CKA dimension mismatch: vis_feat {X_vis.shape} vs fused_feat {Y_fused.shape}. "
            f"Check that fused_proj in LateFusionClassifier projects to cfg.model.vision_dim."
        )

        cka_score = self._compute_linear_cka(X_vis, Y_fused)

        # Sanity-check norms: if either is near-zero, features have collapsed to a degenerate space
        vis_norm   = torch.linalg.norm(X_vis,   dim=1).mean().item()
        fused_norm = torch.linalg.norm(Y_fused, dim=1).mean().item()

        return {
            "Linear_CKA":      cka_score,
            "N_samples":       X_vis.shape[0],
            "Vis_Feat_Norm":   vis_norm,
            "Fused_Feat_Norm": fused_norm,
        }

    def print_report(self, results: Dict[str, Any], model_name: str = ""):
        """
        Prints a structured CKA audit report.
        Includes norm sanity checks alongside the CKA interpretation.
        """
        cka        = results["Linear_CKA"]
        n          = results["N_samples"]
        vis_norm   = results["Vis_Feat_Norm"]
        fused_norm = results["Fused_Feat_Norm"]

        header = f"{model_name} " if model_name else ""
        print(f"\n📐 --- {header}Latent Space Geometric Audit ---")
        print(f"Samples evaluated:               {n}")
        print(f"Mean Visual Feature L2 Norm:     {vis_norm:.4f}")
        print(f"Mean Fused Feature L2 Norm:      {fused_norm:.4f}")

        # Norm collapse warning — catches silent feature degeneration
        if vis_norm < 0.1 or fused_norm < 0.1:
            print("  -> 🚨 WARNING: Near-zero feature norms detected. "
                  "Possible feature collapse — CKA score unreliable.")

        print(f"\nLinear CKA (Visual vs. Fused):   {cka:.4f}")

        if cka > 0.95:
            print("  -> 🚨 MODALITY COLLAPSE: CKA ≈ 1.0.")
            print("  -> Fused space is geometrically identical to the pure visual space.")
            print("  -> Text modality is being ignored by this architecture.")
        elif cka > 0.85:
            print("  -> ⚠️  Moderate geometric perturbation detected.")
            print("  -> Text has some influence but visual features dominate the geometry.")
        else:
            print("  -> ✅ Healthy multimodal fusion.")
            print("  -> Text has actively warped the latent space away from pure visual geometry.")

        print("-" * 48)