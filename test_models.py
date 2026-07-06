import torch
from config import cfg
from models.late_fusion import LateFusionClassifier
from models.cross_attention import CrossAttentionClassifier
from models.gmu import GMUClassifier  # FIX: GMU is trained/audited in run_experiment.py but was never covered here

def test_architectures():
    print("🚀 Initializing Architecture Audit...\n")

    # 1. Create dummy tensors that mimic our dataset outputs
    B = 2  # Batch size of 2
    dummy_img = torch.randn(B, 3, cfg.data.img_size, cfg.data.img_size)
    dummy_ids = torch.randint(0, 28000, (B, cfg.data.max_text_len))

    # Simulate an attention mask (e.g., first 10 tokens valid, rest are padding)
    dummy_mask = torch.zeros(B, cfg.data.max_text_len, dtype=torch.long)
    dummy_mask[:, :10] = 1

    print("✅ Dummy Tensors Generated.")
    print(f"   Image: {dummy_img.shape}")
    print(f"   Text:  {dummy_ids.shape}\n")

    fused_shapes = {}  # collects (fused, v_feat) shapes per architecture for the CKA contract check below

    # 2. Test Late Fusion (Capacity Matched)
    print("🏗️ Building Late Fusion (Capacity Matched)...")
    lf_model = LateFusionClassifier()
    lf_logits, lf_fused, lf_vfeat = lf_model(dummy_img, dummy_ids, dummy_mask)

    print("   ✅ Forward Pass Successful.")
    # FIX: was hardcoded "(Expected: B, 2)" — stale from an earlier binary
    # version of the project. num_classes is 6 (cfg.model.num_classes).
    print(f"   Logits: {lf_logits.shape} (Expected: {B}, {cfg.model.num_classes})")
    # FIX: was hardcoded "(Expected: B, 1536)". fused_proj is explicitly
    # projected DOWN to cfg.model.vision_dim (768) for CKA compatibility —
    # the 1536-D representation only exists internally, before fused_proj.
    print(f"   Fused Representation: {lf_fused.shape} (Expected: {B}, {cfg.model.vision_dim})")
    print(f"   Pre-fusion Visual: {lf_vfeat.shape} (Expected: {B}, {cfg.model.vision_dim})\n")

    # FIX: turned print-only "expected" comments into real assertions, so a
    # future regression actually fails loudly instead of silently printing a
    # wrong number that nobody reads closely (which is how the stale
    # comments above went unnoticed for so long in the first place).
    assert lf_logits.shape == (B, cfg.model.num_classes), f"Late Fusion logits shape mismatch: {lf_logits.shape}"
    assert lf_fused.shape == (B, cfg.model.vision_dim), f"Late Fusion fused_proj shape mismatch: {lf_fused.shape}"
    assert lf_vfeat.shape == (B, cfg.model.vision_dim), f"Late Fusion v_feat shape mismatch: {lf_vfeat.shape}"
    fused_shapes['Late Fusion'] = (lf_fused.shape, lf_vfeat.shape)

    # 3. Test Cross-Attention
    print("🏗️ Building Masked Cross-Attention...")
    ca_model = CrossAttentionClassifier()
    ca_logits, ca_fused, ca_vfeat = ca_model(dummy_img, dummy_ids, dummy_mask)

    print("   ✅ Forward Pass Successful.")
    print(f"   Logits: {ca_logits.shape} (Expected: {B}, {cfg.model.num_classes})")
    print(f"   Fused Representation: {ca_fused.shape} (Expected: {B}, {cfg.model.vision_dim})")
    print(f"   Pre-fusion Visual: {ca_vfeat.shape} (Expected: {B}, {cfg.model.vision_dim})\n")

    assert ca_logits.shape == (B, cfg.model.num_classes), f"Cross-Attention logits shape mismatch: {ca_logits.shape}"
    assert ca_fused.shape == (B, cfg.model.vision_dim), f"Cross-Attention fused_cls shape mismatch: {ca_fused.shape}"
    assert ca_vfeat.shape == (B, cfg.model.vision_dim), f"Cross-Attention vis_cls shape mismatch: {ca_vfeat.shape}"
    fused_shapes['Cross-Attention'] = (ca_fused.shape, ca_vfeat.shape)

    # 4. Test GMU Baseline
    # FIX: previously untested here despite being trained and audited
    # alongside the other two architectures in run_experiment.py.
    print("🏗️ Building GMU Baseline...")
    gmu_model = GMUClassifier()
    gmu_logits, gmu_fused, gmu_vfeat = gmu_model(dummy_img, dummy_ids, dummy_mask)

    print("   ✅ Forward Pass Successful.")
    print(f"   Logits: {gmu_logits.shape} (Expected: {B}, {cfg.model.num_classes})")
    print(f"   Fused Representation: {gmu_fused.shape} (Expected: {B}, {cfg.model.vision_dim})")
    print(f"   Pre-fusion Visual: {gmu_vfeat.shape} (Expected: {B}, {cfg.model.vision_dim})\n")

    assert gmu_logits.shape == (B, cfg.model.num_classes), f"GMU logits shape mismatch: {gmu_logits.shape}"
    assert gmu_fused.shape == (B, cfg.model.vision_dim), f"GMU h_fused shape mismatch: {gmu_fused.shape}"
    assert gmu_vfeat.shape == (B, cfg.model.vision_dim), f"GMU v_feat shape mismatch: {gmu_vfeat.shape}"
    fused_shapes['GMU Baseline'] = (gmu_fused.shape, gmu_vfeat.shape)

    # FIX: explicitly assert the CKA audit contract here (every architecture
    # must emit identically-shaped (fused_feat, vis_feat) pairs), instead of
    # only discovering a contract violation deep inside CKAAuditor.run_audit()
    # after a full training run has already completed.
    unique_shape_pairs = set(fused_shapes.values())
    assert len(unique_shape_pairs) == 1, f"CKA contract violated — shapes differ across architectures: {fused_shapes}"

    print("🏁 All Architectures passed tensor dimension audits (Late Fusion, Cross-Attention, GMU)!")

if __name__ == "__main__":
    test_architectures()