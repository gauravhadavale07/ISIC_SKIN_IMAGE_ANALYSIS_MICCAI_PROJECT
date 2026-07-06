"""
run_inference_audit.py — Inference-only re-audit for existing checkpoints.

Runs the full CounterfactualAuditor (Real / Blank-empty / Neutral / CF) against
every existing checkpoint found in ./checkpoints/, updates experiment_progress.json
with the new fields (Counterfactual_Accuracy, Neutral_Accuracy), and prints a
per-seed comparison table.

NO TRAINING — reads existing best_model.pth files only.
"""

import os
import json
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
from dataset import MultimodalDermatologyDataset
from counterfactual import CounterfactualAuditor

# Architecture registry — only the ones with existing checkpoints
from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionClassifier  # original V→T (what was trained)

ARCH_MAP = {
    "Late Fusion":      LateFusionClassifier,
    "GMU Baseline":     GMUClassifier,
    "Cross-Attention":  CrossAttentionClassifier,   # original checkpoints were saved under this class
}

MAJORITY_BASELINE = 845 / 2298 * 100  # BCC class, 36.77%

# ──────────────────────────────────────────────────────────────────────────────
def discover_checkpoints():
    """Return dict {arch_name: {seed: ckpt_path}} from ./checkpoints/ directory."""
    ckpt_root = cfg.paths.checkpoint_dir
    mapping = {}
    for entry in sorted(os.listdir(ckpt_root)):
        full = os.path.join(ckpt_root, entry)
        if not os.path.isdir(full):
            continue
        # Expected format: <ArchName>_seed_<N>  (spaces replaced with underscores)
        # e.g. Cross-Attention_seed_456, Late_Fusion_seed_789
        best_ckpt = os.path.join(full, "best_model.pth")
        if not os.path.exists(best_ckpt):
            print(f"  [SKIP] {entry}: no best_model.pth")
            continue

        # Normalise arch name back: e.g. "Late_Fusion" -> "Late Fusion"
        parts = entry.rsplit("_seed_", 1)
        if len(parts) != 2:
            print(f"  [SKIP] {entry}: can't parse arch/seed")
            continue
        arch_raw, seed_str = parts
        arch_name = arch_raw.replace("_", " ").replace("- ", "-")
        # Fix: "Cross- Attention" -> "Cross-Attention"
        arch_name = arch_name.replace("Cross- Attention", "Cross-Attention")
        try:
            seed = int(seed_str)
        except ValueError:
            print(f"  [SKIP] {entry}: non-integer seed '{seed_str}'")
            continue

        if arch_name not in ARCH_MAP:
            print(f"  [SKIP] {entry}: unknown arch '{arch_name}'")
            continue

        mapping.setdefault(arch_name, {})[seed] = best_ckpt

    return mapping


def load_model(arch_name: str, ckpt_path: str, device: torch.device):
    model_cls = ARCH_MAP[arch_name]
    model = model_cls()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def build_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    dataset = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir=cfg.paths.pad_ufes_img_dir,
        tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return loader, tokenizer


def print_result_row(arch, seed, r):
    real  = r["Real_Accuracy"]
    blank = r["Blank_Accuracy"]
    neut  = r["Neutral_Accuracy"]
    cf    = r["Counterfactual_Accuracy"]
    drop  = r["Blank_Accuracy_Drop"]
    cfr   = r["CFR"]
    dp    = r["Mean_Delta_P"]
    below = "⚠️  BELOW BASELINE" if blank < MAJORITY_BASELINE else ""
    below_n = "⚠️  BELOW BASELINE" if neut < MAJORITY_BASELINE else ""
    print(f"  {arch} | seed={seed:6d} | Real={real:6.2f}% | Blank={blank:6.2f}%{below} | "
          f"Neutral={neut:6.2f}%{below_n} | CF={cf:6.2f}% | Drop={drop:+6.2f}pp | "
          f"CFR={cfr:5.2f}% | ΔP={dp:5.2f}pp")


def main():
    device = cfg.train.device
    print(f"\n{'='*70}")
    print("INFERENCE-ONLY COUNTERFACTUAL + NEUTRAL RE-AUDIT")
    print(f"Device: {device}  |  Majority baseline: {MAJORITY_BASELINE:.2f}%")
    print(f"Neutral string: \"{cfg.audit.neutral_string}\"")
    print(f"{'='*70}\n")

    # ── discover checkpoints ───────────────────────────────────────────────────
    ckpt_map = discover_checkpoints()
    if not ckpt_map:
        print("ERROR: no checkpoints found in", cfg.paths.checkpoint_dir)
        sys.exit(1)

    print("Found checkpoints:")
    for arch, seeds in sorted(ckpt_map.items()):
        for seed, path in sorted(seeds.items()):
            print(f"  {arch}  seed={seed}  → {path}")
    print()

    # ── build dataloader once ─────────────────────────────────────────────────
    print("Building PAD-UFES-20 test dataloader…")
    loader, tokenizer = build_dataloader()
    print(f"Dataset: {len(loader.dataset)} samples, {len(loader)} batches\n")

    # ── load results JSON ──────────────────────────────────────────────────────
    results_path = os.path.join(cfg.paths.results_dir, "experiment_progress.json")
    with open(results_path) as f:
        progress = json.load(f)

    # ── run audits ─────────────────────────────────────────────────────────────
    all_new_results = {}   # arch -> seed -> result_dict

    for arch_name, seed_map in sorted(ckpt_map.items()):
        print(f"\n{'─'*60}")
        print(f"Architecture: {arch_name}")
        print(f"{'─'*60}")

        for seed, ckpt_path in sorted(seed_map.items()):
            print(f"\n  Loading seed={seed} from {ckpt_path}")
            model = load_model(arch_name, ckpt_path, device)

            auditor = CounterfactualAuditor(model=model, tokenizer=tokenizer, device=device)
            results = auditor.run_audit(loader)

            print_result_row(arch_name, seed, results)
            all_new_results.setdefault(arch_name, {})[seed] = results

            # Immediately write to JSON in case of crash
            if arch_name not in progress["results"]:
                progress["results"][arch_name] = {}
            for key, val in results.items():
                if key not in progress["results"][arch_name]:
                    progress["results"][arch_name][key] = []
                # Find the seed index and update/append
                # We'll rebuild lists from scratch at the end to stay consistent

            with open(results_path, "w") as f:
                json.dump(progress, f, indent=2)

    # ── merge new results into progress JSON ──────────────────────────────────
    print(f"\n\n{'='*70}")
    print("SUMMARY — FOUR-PROBE COMPARISON TABLE")
    print(f"Majority-class baseline (BCC): {MAJORITY_BASELINE:.2f}%")
    print(f"{'='*70}")
    print(f"{'Architecture':<22} {'Seed':>6} | {'Real':>7} | {'Blank-∅':>9} | {'Neutral':>9} | {'CF':>7} | {'Drop':>7} | {'CFR':>6} | {'ΔP':>6}")
    print("─" * 110)

    for arch_name in sorted(all_new_results.keys()):
        for seed in sorted(all_new_results[arch_name].keys()):
            r = all_new_results[arch_name][seed]
            below    = "⚠️" if r["Blank_Accuracy"]   < MAJORITY_BASELINE else "  "
            below_n  = "⚠️" if r["Neutral_Accuracy"] < MAJORITY_BASELINE else "  "
            print(f"  {arch_name:<20} {seed:>6} | {r['Real_Accuracy']:>6.2f}% | "
                  f"{r['Blank_Accuracy']:>7.2f}%{below} | {r['Neutral_Accuracy']:>7.2f}%{below_n} | "
                  f"{r['Counterfactual_Accuracy']:>6.2f}% | {r['Blank_Accuracy_Drop']:>+6.2f}pp | "
                  f"{r['CFR']:>5.2f}% | {r['Mean_Delta_P']:>5.2f}pp")

    # ── write final updated progress JSON with all new fields ─────────────────
    print(f"\nUpdating {results_path} with new audit fields…")

    for arch_name, seed_results in all_new_results.items():
        # For each new metric key from the audit, rebuild per-metric lists
        # sorted by seed order matching whatever is already in the JSON
        sorted_seeds = sorted(seed_results.keys())
        new_keys = list(next(iter(seed_results.values())).keys())

        for key in new_keys:
            progress["results"][arch_name][key] = [
                seed_results[s][key] for s in sorted_seeds
            ]

    with open(results_path, "w") as f:
        json.dump(progress, f, indent=2)
    print("Done. Results saved.")


if __name__ == "__main__":
    main()
