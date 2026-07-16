"""
train_t2v.py — Trains CrossAttentionT2VClassifier at seeds 456, 789, 1337.

Operates identically to run_experiment.py's training loop but:
  - Only runs Cross-Attention T→V (skips Late Fusion / GMU / V→T)
  - Respects the existing `completed_runs` skip logic so a partially-completed
    run can be safely resumed
  - Saves results under key "Cross-Attention T→V" in experiment_progress.json,
    which does NOT collide with the existing "Cross-Attention" (V→T) entries
  - Tags every checkpoint's saved state_dict with protocol_version

Protocol: v4_verified_3seeds (see config.py for full fix list)
Seeds:     456, 789, 1337 only
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import cfg
from dataset import MultimodalDermatologyDataset
from transforms import get_train_transforms, get_eval_transforms
from trainer import MultimodalTrainer, set_seed
from evaluate import Evaluator
from counterfactual import CounterfactualAuditor
from cka import CKAAuditor
from statistical_analyzer import StatisticalAnalyzer
from models.cross_attention import CrossAttentionT2VClassifier

ARCH_NAME = "Cross-Attention T→V"
ARCH_KEY_PREFIX = "Cross-Attention T→V"   # used in run_key = f"{seed}:{ARCH_NAME}"

def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def checkpoint_is_compatible(model, checkpoint_path, device) -> bool:
    if not os.path.exists(checkpoint_path):
        return False
    try:
        load_checkpoint(model, checkpoint_path, device)
        return True
    except Exception as e:
        print(f"⚠️  Incompatible checkpoint at {checkpoint_path}: {e}")
        return False

def main():
    device = cfg.train.device
    print(f"\n{'='*65}")
    print(f"CROSS-ATTENTION T→V TRAINING")
    print(f"Protocol: {cfg.protocol_version}")
    print(f"Seeds:    {cfg.seeds}")
    print(f"Device:   {device}")
    print(f"{'='*65}\n")

    # ── Shared resources ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    stats_analyzer = StatisticalAnalyzer()

    # Build OOD test loader once
    test_ds = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir="",
        tokenizer=tokenizer,
        transform=get_eval_transforms()
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # ── Seed loop ──────────────────────────────────────────────────────────────
    for seed in cfg.seeds:
        print(f"\n{'='*65}")
        print(f"🌱 SEED: {seed}  |  Architecture: {ARCH_NAME}")
        print(f"{'='*65}")

        run_key = f"{seed}:{ARCH_NAME}"
        if stats_analyzer.is_complete(run_key):
            print(f"⏭️  Already complete — skipping.")
            continue

        set_seed(seed)

        # ── Build train/val split ─────────────────────────────────────────────
        milk10k_csv = cfg.paths.milk10k_csv
        if not os.path.exists(milk10k_csv):
            raise FileNotFoundError(f"Training CSV not found: {milk10k_csv}")

        train_ds = MultimodalDermatologyDataset(
            csv_file=milk10k_csv, img_dir="",
            tokenizer=tokenizer, transform=get_train_transforms()
        )
        val_ds = MultimodalDermatologyDataset(
            csv_file=milk10k_csv, img_dir="",
            tokenizer=tokenizer, transform=get_eval_transforms()
        )

        train_size = int(0.85 * len(train_ds))
        val_size   = len(train_ds) - train_size
        gen = torch.Generator().manual_seed(seed)
        train_subset, _ = random_split(train_ds, [train_size, val_size], generator=gen)
        gen = torch.Generator().manual_seed(seed)
        _, val_subset   = random_split(val_ds,   [train_size, val_size], generator=gen)

        train_loader = DataLoader(train_subset, batch_size=cfg.train.batch_size,
                                  shuffle=True,  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_subset,   batch_size=cfg.train.batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

        # ── Model + optimizer ─────────────────────────────────────────────────
        run_identifier = f"Cross-Attention_T2V_seed_{seed}"
        best_model_path = os.path.join(cfg.paths.checkpoint_dir, run_identifier, "best_model.pth")

        model = CrossAttentionT2VClassifier()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay
        )

        total_steps   = len(train_loader) * cfg.train.epochs
        warmup_steps  = int(cfg.train.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        criterion = torch.nn.CrossEntropyLoss()

        # ── PHASE 1: TRAINING ─────────────────────────────────────────────────
        trainer = MultimodalTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            run_name=run_identifier
        )

        if checkpoint_is_compatible(model, best_model_path, device):
            print(f"📂 Compatible checkpoint found — skipping training.")
        else:
            if os.path.exists(best_model_path):
                print("♻️  Removing stale checkpoint before retraining.")
                os.remove(best_model_path)
            trainer.fit()

        # ── PHASE 2: OOD EVALUATION + FULL AUDIT ─────────────────────────────
        load_checkpoint(model, best_model_path, device)
        model.to(device)
        model.eval()

        print(f"\n🩺 OOD Audit on PAD-UFES-20 ({ARCH_NAME}, seed={seed})...")

        # A. Standard metrics
        evaluator   = Evaluator(model, device)
        std_results = evaluator.evaluate(test_loader)

        # B. Full counterfactual + neutral audit
        cf_auditor  = CounterfactualAuditor(model, tokenizer, device)
        cf_results  = cf_auditor.run_audit(test_loader)

        # C. CKA geometric audit
        cka_auditor  = CKAAuditor(model, device)
        cka_results  = cka_auditor.run_audit(test_loader)

        combined = {**std_results, **cf_results, **cka_results}

        stats_analyzer.add_run(ARCH_NAME, combined, run_key)
        evaluator.print_report(std_results, prefix=f"{ARCH_NAME} (Seed {seed})")
        cf_auditor.print_report(cf_results)
        cka_auditor.print_report(cka_results, model_name=ARCH_NAME)

        print(f"\n  ✅ Seed {seed} complete. Results saved.")

    # ── Final aggregate report ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"🏁 CROSS-ATTENTION T→V — FINAL MULTI-SEED REPORT")
    print(f"{'='*65}")
    stats_analyzer.print_report()

if __name__ == "__main__":
    main()
