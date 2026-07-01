import torch
from config import cfg
from models.late_fusion import LateFusionClassifier
from models.cross_attention import CrossAttentionV2TClassifier, CrossAttentionT2VClassifier
from models.gmu import GMUClassifier
from models.image_only import ImageOnlyClassifier
from models.text_only import TextOnlyClassifier

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
    print(f"   Logits: {lf_logits.shape} (Expected: {B}, {cfg.model.num_classes})")
    print(f"   Fused Representation: {lf_fused.shape} (Expected: {B}, {cfg.model.vision_dim})")
    print(f"   Pre-fusion Visual: {lf_vfeat.shape} (Expected: {B}, {cfg.model.vision_dim})\n")

    assert lf_logits.shape == (B, cfg.model.num_classes), f"Late Fusion logits shape mismatch: {lf_logits.shape}"
    assert lf_fused.shape == (B, cfg.model.vision_dim), f"Late Fusion fused_proj shape mismatch: {lf_fused.shape}"
    assert lf_vfeat.shape == (B, cfg.model.vision_dim), f"Late Fusion v_feat shape mismatch: {lf_vfeat.shape}"
    fused_shapes['Late Fusion'] = (lf_fused.shape, lf_vfeat.shape)

    # 3. Test GMU Baseline
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

    # 4. Test Cross-Attention V→T
    print("🏗️ Building Cross-Attention V→T...")
    ca_v2t_model = CrossAttentionV2TClassifier()
    ca_v2t_logits, ca_v2t_fused, ca_v2t_vfeat = ca_v2t_model(dummy_img, dummy_ids, dummy_mask)

    print("   ✅ Forward Pass Successful.")
    print(f"   Logits: {ca_v2t_logits.shape} (Expected: {B}, {cfg.model.num_classes})")
    print(f"   Fused Representation: {ca_v2t_fused.shape} (Expected: {B}, {cfg.model.vision_dim})")
    print(f"   Pre-fusion Visual: {ca_v2t_vfeat.shape} (Expected: {B}, {cfg.model.vision_dim})\n")

    assert ca_v2t_logits.shape == (B, cfg.model.num_classes), f"Cross-Attn V→T logits shape mismatch: {ca_v2t_logits.shape}"
    assert ca_v2t_fused.shape == (B, cfg.model.vision_dim), f"Cross-Attn V→T fused_cls shape mismatch: {ca_v2t_fused.shape}"
    assert ca_v2t_vfeat.shape == (B, cfg.model.vision_dim), f"Cross-Attn V→T vis_cls shape mismatch: {ca_v2t_vfeat.shape}"
    fused_shapes['Cross-Attn V→T'] = (ca_v2t_fused.shape, ca_v2t_vfeat.shape)

    # 5. Test Cross-Attention T→V
    print("🏗️ Building Cross-Attention T→V...")
    ca_t2v_model = CrossAttentionT2VClassifier()
    ca_t2v_logits, ca_t2v_fused, ca_t2v_vfeat = ca_t2v_model(dummy_img, dummy_ids, dummy_mask)

    print("   ✅ Forward Pass Successful.")
    print(f"   Logits: {ca_t2v_logits.shape} (Expected: {B}, {cfg.model.num_classes})")
    print(f"   Fused Representation: {ca_t2v_fused.shape} (Expected: {B}, {cfg.model.vision_dim})")
    print(f"   Pre-fusion Visual: {ca_t2v_vfeat.shape} (Expected: {B}, {cfg.model.vision_dim})\n")

    assert ca_t2v_logits.shape == (B, cfg.model.num_classes), f"Cross-Attn T→V logits shape mismatch: {ca_t2v_logits.shape}"
    assert ca_t2v_fused.shape == (B, cfg.model.vision_dim), f"Cross-Attn T→V fused_cls shape mismatch: {ca_t2v_fused.shape}"
    assert ca_t2v_vfeat.shape == (B, cfg.model.vision_dim), f"Cross-Attn T→V vis_cls shape mismatch: {ca_t2v_vfeat.shape}"
    fused_shapes['Cross-Attn T→V'] = (ca_t2v_fused.shape, ca_t2v_vfeat.shape)

    # 6. Test Image-Only
    print("🏗️ Building Image-Only...")
    img_model = ImageOnlyClassifier()
    img_logits, img_fused, img_vfeat = img_model(dummy_img, dummy_ids, dummy_mask)

    print("   ✅ Forward Pass Successful.")
    print(f"   Logits: {img_logits.shape} (Expected: {B}, {cfg.model.num_classes})")
    print(f"   Fused Representation: {img_fused.shape} (Expected: {B}, {cfg.model.vision_dim})")
    print(f"   Pre-fusion Visual: {img_vfeat.shape} (Expected: {B}, {cfg.model.vision_dim})\n")

    assert img_logits.shape == (B, cfg.model.num_classes), f"Image-Only logits shape mismatch: {img_logits.shape}"
    assert img_fused.shape == (B, cfg.model.vision_dim), f"Image-Only fused shape mismatch: {img_fused.shape}"
    assert img_vfeat.shape == (B, cfg.model.vision_dim), f"Image-Only v_feat shape mismatch: {img_vfeat.shape}"
    fused_shapes['Image-Only'] = (img_fused.shape, img_vfeat.shape)

    # 7. Test Text-Only
    print("🏗️ Building Text-Only...")
    txt_model = TextOnlyClassifier()
    txt_logits, txt_fused, txt_vfeat = txt_model(dummy_img, dummy_ids, dummy_mask)

    print("   ✅ Forward Pass Successful.")
    print(f"   Logits: {txt_logits.shape} (Expected: {B}, {cfg.model.num_classes})")
    print(f"   Fused Representation: {txt_fused.shape} (Expected: {B}, {cfg.model.vision_dim})")
    print(f"   Pre-fusion Visual: {txt_vfeat.shape} (Expected: {B}, {cfg.model.vision_dim})\n")

    assert txt_logits.shape == (B, cfg.model.num_classes), f"Text-Only logits shape mismatch: {txt_logits.shape}"
    assert txt_fused.shape == (B, cfg.model.vision_dim), f"Text-Only fused shape mismatch: {txt_fused.shape}"
    assert txt_vfeat.shape == (B, cfg.model.vision_dim), f"Text-Only v_feat shape mismatch: {txt_vfeat.shape}"
    fused_shapes['Text-Only'] = (txt_fused.shape, txt_vfeat.shape)

    # Assert CKA audit contract (every architecture must emit identically-shaped pairs)
    unique_shape_pairs = set(fused_shapes.values())
    assert len(unique_shape_pairs) == 1, f"CKA contract violated — shapes differ across architectures: {fused_shapes}"

    print("🏁 All 6 Architectures passed tensor dimension audits!")

if __name__ == "__main__":
    test_architectures()