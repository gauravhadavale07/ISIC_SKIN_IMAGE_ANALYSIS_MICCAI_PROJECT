"""
full_analysis.py — Three pending items, run sequentially:

  1. Train Image-Only and Text-Only baselines (seeds 456/789/1337)
     so significance tests against them are meaningful.

  2. Silhouette / class-separability on T→V fused features,
     compared against V→T and Late Fusion.

  3. Full pairwise significance tests:
       - All 4 multimodal archs vs. Image-Only
       - All 4 multimodal archs vs. Text-Only
       - Cross-Attention (V→T) vs. Cross-Attention T→V
       - GMU vs. Late Fusion
     Metrics: Accuracy, AUROC, Macro F1, CFR, Mean_Delta_P, Linear_CKA

Protocol: v4_verified_3seeds  Seeds: 456, 789, 1337
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from scipy import stats
from sklearn.metrics import silhouette_score, silhouette_samples

from config import cfg
from dataset import MultimodalDermatologyDataset
from transforms import get_train_transforms, get_eval_transforms
from trainer import MultimodalTrainer, set_seed
from evaluate import Evaluator
from counterfactual import CounterfactualAuditor
from cka import CKAAuditor
from statistical_analyzer import StatisticalAnalyzer
from models.image_only import ImageOnlyClassifier
from models.text_only import TextOnlyClassifier
from models.late_fusion import LateFusionClassifier
from models.cross_attention import CrossAttentionV2TClassifier, CrossAttentionT2VClassifier
from models.gmu import GMUClassifier

DEVICE = cfg.train.device
SEEDS  = cfg.seeds  # [456, 789, 1337]
MAJORITY_BASELINE = 845 / 2298 * 100  # 36.77%

def main_logic():
    print(f"\n{'='*70}")
    print(f"FULL ANALYSIS — Baselines + Silhouette + Significance Tests")
    print(f"Protocol: {cfg.protocol_version}   Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")
    print(f"{'='*70}\n")
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────────
    
    def load_checkpoint(model, path, device):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt
    
    def checkpoint_is_compatible(model, path, device):
        if not os.path.exists(path):
            return False
        try:
            load_checkpoint(model, path, device)
            return True
        except Exception as e:
            print(f"  ⚠️  Incompatible checkpoint {path}: {e}")
            return False
    
    def build_test_loader(tokenizer):
        test_ds = MultimodalDermatologyDataset(
            csv_file=cfg.paths.pad_ufes_csv, img_dir="",
            tokenizer=tokenizer, transform=get_eval_transforms()
        )
        return DataLoader(test_ds, batch_size=cfg.train.batch_size,
                          shuffle=False, num_workers=0, pin_memory=True)
    
    def build_train_val_loaders(tokenizer, seed):
        train_ds = MultimodalDermatologyDataset(
            csv_file=cfg.paths.milk10k_csv, img_dir="",
            tokenizer=tokenizer, transform=get_train_transforms(),
            split="train"
        )
        val_ds = MultimodalDermatologyDataset(
            csv_file=cfg.paths.milk10k_csv, img_dir="",
            tokenizer=tokenizer, transform=get_eval_transforms(),
            split="val"
        )
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=cfg.train.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)
        return train_loader, val_loader
    
    def run_full_audit(model, test_loader, tokenizer, arch_name, seed):
        evaluator  = Evaluator(model, DEVICE)
        cf_auditor = CounterfactualAuditor(model, tokenizer, DEVICE)
        cka_auditor = CKAAuditor(model, DEVICE)
        std_res  = evaluator.evaluate(test_loader)
        cf_res   = cf_auditor.run_audit(test_loader)
        cka_res  = cka_auditor.run_audit(test_loader)
        evaluator.print_report(std_res, prefix=f"{arch_name} (Seed {seed})")
        cf_auditor.print_report(cf_res)
        cka_auditor.print_report(cka_res, model_name=arch_name)
        return {**std_res, **cf_res, **cka_res}
    
    def train_baseline(arch_name, model_cls, run_id, seed, train_loader, val_loader):
        model = model_cls()
        best_path = os.path.join(cfg.paths.checkpoint_dir, run_id, "best_model.pth")
    
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable parameters: {sum(p.numel() for p in trainable):,}")
    
        optimizer = torch.optim.AdamW(trainable, lr=cfg.train.learning_rate,
                                      weight_decay=cfg.train.weight_decay)
        total_steps  = len(train_loader) * cfg.train.epochs
        warmup_steps = int(cfg.train.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        criterion = torch.nn.CrossEntropyLoss()
    
        trainer = MultimodalTrainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler, criterion=criterion,
            device=DEVICE, run_name=run_id
        )
    
        if checkpoint_is_compatible(model, best_path, DEVICE):
            print(f"  📂 Compatible checkpoint found — skipping training.")
            load_checkpoint(model, best_path, DEVICE)
        else:
            if os.path.exists(best_path):
                os.remove(best_path)
            trainer.fit()
            load_checkpoint(model, best_path, DEVICE)
    
        return model
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PART 1: Train Image-Only and Text-Only baselines
    # ─────────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'='*70}")
    print("PART 1: BASELINE TRAINING (Image-Only & Text-Only)")
    print(f"{'='*70}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    test_loader = build_test_loader(tokenizer)
    
    stats_analyzer = StatisticalAnalyzer()
    
    BASELINES = {
        "Text-Only":  TextOnlyClassifier,
    }
    
    for arch_name, model_cls in BASELINES.items():
        print(f"\n{'─'*60}")
        print(f"Architecture: {arch_name}")
        print(f"{'─'*60}")
    
        for seed in SEEDS:
            run_key = f"{seed}:{arch_name}"
            if stats_analyzer.is_complete(run_key):
                print(f"  ⏭️  {arch_name} seed={seed} already complete — skipping.")
                continue
    
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"\n  🌱 Seed: {seed} [Attempt {attempt}/{max_retries}]")
                    set_seed(seed)
            
                    run_id = f"{arch_name.replace('-', '').replace(' ', '_')}_seed_{seed}"
                    train_loader, val_loader = build_train_val_loaders(tokenizer, seed)
            
                    model = train_baseline(arch_name, model_cls, run_id, seed,
                                           train_loader, val_loader)
                    model.to(DEVICE).eval()
            
                    combined = run_full_audit(model, test_loader, tokenizer, arch_name, seed)
                    stats_analyzer.add_run(arch_name, combined, run_key)
                    print(f"  ✅ {arch_name} seed={seed} complete.")
                    break
                except Exception as e:
                    import traceback
                    print(f"  ❌ ERROR: {e}")
                    traceback.print_exc()
                    if attempt == max_retries:
                        print(f"  ⚠️ PERMANENT FAILURE for seed {seed}. Skipping.")
                    else:
                        import time
                        print("  ⏳ Waiting 30s before retry...")
                        time.sleep(30)
    
    print("\n\n✅ PART 1 COMPLETE — Baselines trained and audited.\n")
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PART 2: Silhouette / class-separability on fused features
    #         Architectures: Late Fusion, Cross-Attention (V→T), Cross-Attention T→V
    # ─────────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'='*70}")
    print("PART 2: SILHOUETTE / CLASS SEPARABILITY ON FUSED FEATURES")
    print("Architectures: Late Fusion | Cross-Attention (V→T) | Cross-Attention T→V")
    print(f"{'='*70}\n")
    
    SILHOUETTE_ARCHS = {
        "Late Fusion":          (LateFusionClassifier,      "Late_Fusion"),
        "Cross-Attention (V→T)":(CrossAttentionV2TClassifier,"Cross-Attention"),
        "Cross-Attention T→V":  (CrossAttentionT2VClassifier,"Cross-Attention_T2V"),
    }
    
    sil_results = {}
    
    for arch_label, (model_cls, ckpt_prefix) in SILHOUETTE_ARCHS.items():
        print(f"\n── {arch_label} ──────────────────────────────────────────────────")
        seed_sils = []
        seed_per_class = []
    
        for seed in SEEDS:
            ckpt_path = f"./checkpoints/{ckpt_prefix}_seed_{seed}/best_model.pth"
            if not os.path.exists(ckpt_path):
                print(f"  ⚠️  Checkpoint not found: {ckpt_path} — skipping seed {seed}")
                continue
    
            model = model_cls().to(DEVICE).eval()
            load_checkpoint(model, ckpt_path, DEVICE)
    
            all_fused = []
            all_labels = []
            with torch.no_grad():
                for batch in test_loader:
                    imgs   = batch["image"].to(DEVICE)
                    ids    = batch["input_ids"].to(DEVICE)
                    mask   = batch["attention_mask"].to(DEVICE)
                    labels = batch["label"]
                    _, fused, _ = model(imgs, ids, mask)
                    all_fused.append(fused.cpu().float().numpy())
                    all_labels.append(labels.numpy())
    
            fused_np  = np.concatenate(all_fused,  axis=0)   # (N, 768)
            labels_np = np.concatenate(all_labels, axis=0)   # (N,)
    
            # Use a random subsample for speed (silhouette is O(N^2))
            rng = np.random.RandomState(seed)
            max_samples = 1000
            if len(fused_np) > max_samples:
                idx = rng.choice(len(fused_np), max_samples, replace=False)
                fused_s  = fused_np[idx]
                labels_s = labels_np[idx]
            else:
                fused_s, labels_s = fused_np, labels_np
    
            sil_global = silhouette_score(fused_s, labels_s, metric="cosine")
            sil_per    = silhouette_samples(fused_s, labels_s, metric="cosine")
    
            CLASS_NAMES = ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']
            per_class = {}
            for ci, cn in enumerate(CLASS_NAMES):
                mask_c = (labels_s == ci)
                per_class[cn] = float(sil_per[mask_c].mean()) if mask_c.sum() > 0 else float("nan")
    
            print(f"  Seed {seed}: global silhouette (cosine) = {sil_global:.4f}")
            for cn, s in per_class.items():
                print(f"    {cn}: {s:.4f}")
    
            seed_sils.append(sil_global)
            seed_per_class.append(per_class)
    
        if seed_sils:
            mean_sil = np.mean(seed_sils)
            std_sil  = np.std(seed_sils)
            print(f"\n  ➡  {arch_label}: mean silhouette = {mean_sil:.4f} ± {std_sil:.4f}")
            sil_results[arch_label] = {
                "mean": mean_sil, "std": std_sil,
                "per_seed": seed_sils,
                "per_class_mean": {
                    cn: float(np.nanmean([s[cn] for s in seed_per_class]))
                    for cn in CLASS_NAMES
                }
            }
    
    print(f"\n\n── Silhouette Summary ───────────────────────────────────────────────")
    print(f"  {'Architecture':<30} {'Mean Sil':>10} {'Std':>8}")
    print(f"  {'─'*50}")
    for arch, res in sil_results.items():
        print(f"  {arch:<30} {res['mean']:>10.4f} {res['std']:>8.4f}")
    
    print(f"\n  Per-class silhouette means:")
    print(f"  {'Class':<8}", end="")
    for arch in sil_results:
        print(f"  {arch[:22]:>22}", end="")
    print()
    for cn in ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']:
        print(f"  {cn:<8}", end="")
        for res in sil_results.values():
            v = res['per_class_mean'].get(cn, float('nan'))
            print(f"  {v:>22.4f}", end="")
        print()
    
    print("\n\n✅ PART 2 COMPLETE\n")
    
    
    # ─────────────────────────────────────────────────────────────────────────────
    # PART 3: Pairwise significance tests
    # ─────────────────────────────────────────────────────────────────────────────
    
    print(f"\n{'='*70}")
    print("PART 3: PAIRWISE SIGNIFICANCE TESTS (paired t-test, α=0.05)")
    print(f"{'='*70}\n")
    
    # Reload fresh from disk (includes newly trained baselines)
    stats_analyzer2 = StatisticalAnalyzer()
    results_db = stats_analyzer2.results
    
    def get_vals(arch, metric):
        vals = list(results_db.get(arch, {}).get(metric, []))
        return vals if len(vals) == 3 else None
    
    def ttest_pair(name_a, name_b, metric):
        a = get_vals(name_a, metric)
        b = get_vals(name_b, metric)
        if a is None or b is None:
            return f"{'N/A':>12} — missing data (a={a}, b={b})"
        t, p = stats.ttest_rel(a, b)
        delta = np.mean(b) - np.mean(a)   # positive = b > a
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        return f"t={t:+6.3f}  p={p:.4f} {sig}  Δ={delta:+.4f}  a={np.mean(a):.4f}  b={np.mean(b):.4f}"
    
    SIG_METRICS = ["Accuracy", "AUROC", "F1 (Macro)", "CFR", "Mean_Delta_P", "Linear_CKA",
                   "Blank_Accuracy_Drop"]
    
    COMPARISONS = [
        # (label, baseline, proposed)
        ("Late Fusion    vs. Image-Only",         "Image-Only",  "Late Fusion"),
        ("GMU Baseline   vs. Image-Only",         "Image-Only",  "GMU Baseline"),
        ("V→T            vs. Image-Only",         "Image-Only",  "Cross-Attention"),
        ("T→V            vs. Image-Only",         "Image-Only",  "Cross-Attention T→V"),
        ("Late Fusion    vs. Text-Only",          "Text-Only",   "Late Fusion"),
        ("GMU Baseline   vs. Text-Only",          "Text-Only",   "GMU Baseline"),
        ("V→T            vs. Text-Only",          "Text-Only",   "Cross-Attention"),
        ("T→V            vs. Text-Only",          "Text-Only",   "Cross-Attention T→V"),
        ("V→T  vs. T→V  (direction ablation)",   "Cross-Attention", "Cross-Attention T→V"),
        ("GMU  vs. Late Fusion",                  "Late Fusion", "GMU Baseline"),
        ("V→T  vs. Late Fusion",                  "Late Fusion", "Cross-Attention"),
        ("T→V  vs. Late Fusion",                  "Late Fusion", "Cross-Attention T→V"),
        ("V→T  vs. GMU",                          "GMU Baseline","Cross-Attention"),
        ("T→V  vs. GMU",                          "GMU Baseline","Cross-Attention T→V"),
    ]
    
    for label, arch_a, arch_b in COMPARISONS:
        print(f"\n  {label}")
        print(f"  {'─'*66}")
        for metric in SIG_METRICS:
            result = ttest_pair(arch_a, arch_b, metric)
            print(f"    {metric:<22} {result}")

print(f"\n\n{'='*70}")
import modal

app = modal.App("miccai-multimodal-pipeline")

vol_checkpoints = modal.Volume.from_name("miccai-checkpoints", create_if_missing=True)
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers", "pandas", "scikit-learn", "timm", "scipy", "pillow", "tqdm", "networkx"
).workdir("/root/project").add_local_dir(
    "/tmp/miccai_code_fresh", 
    remote_path="/root/project"
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data", 
    remote_path="/root/project/data"
).add_local_file(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/milk10k_train.csv",
    remote_path="/root/project/milk10k_train.csv"
).add_local_file(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/pad_ufes_20_test.csv",
    remote_path="/root/project/pad_ufes_20_test.csv"
)

@app.function(
    gpu="H200",
    image=image,
    volumes={
        "/root/project/checkpoints": vol_checkpoints,
        "/root/project/results": vol_results
    },
    timeout=86400
)
def run_heavy_workload():
    main_logic()

@app.local_entrypoint()
def main():
    run_heavy_workload.remote()

if __name__ == "__main__":
    modal.runner.deploy_stub = lambda *args, **kwargs: None

