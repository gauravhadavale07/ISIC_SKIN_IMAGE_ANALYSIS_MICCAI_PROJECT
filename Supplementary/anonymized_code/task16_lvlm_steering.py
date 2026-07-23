"""
Task 16: LVLM Causal Activation Steering – Robustness Expansion.

New features over baseline:
  - Multi-model sweep: llava-med-v1.5-mistral-7b, llava-1.5-7b, Qwen-VL
  - Two steering methods: ActAdd (original) and Contrastive Activation Addition (CAA)
  - Subspace Entanglement Diagnostic: cosine similarity between each steering
    vector and the estimated MEL–NEV decision direction, explaining why a
    method fails (orthogonal ≈ decoupled from the decision boundary).
  - Varied prompt templates: zero-shot open-ended, multiple-choice, and
    original clinical format.
  - --dry-run flag: loads model, extracts V_mel/V_nev, prints cosine diagnostics
    without running the full sweep.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg

# ─── Prompt Templates ──────────────────────────────────────────────────────────

PROMPT_TEMPLATES = {
    "clinical": lambda hist: (
        f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: "
        f"[{hist}]. What is the diagnosis? Is the diagnosis MEL or NEV? Diagnosis:"
    ),
    "open_ended": lambda hist: (
        f"[INST] <image>\nYou are a dermatologist. Examine this skin lesion image. "
        f"Patient history: {hist}. Provide your diagnosis (MEL for melanoma or NEV for nevus):"
    ),
    "multiple_choice": lambda hist: (
        f"[INST] <image>\nChoose one:\n"
        f"A) MEL (melanoma)\nB) NEV (nevus)\n"
        f"Patient history: {hist}\n"
        f"The correct answer is:"
    ),
}

# Benign/malignant prompt sets for steering vector extraction
BENIGN_TEXTS = [
    "The clinical history strongly indicates a benign nevus.",
    "Patient presents with a completely harmless, benign mole.",
    "Dermatoscopic examination confirms a classic benign melanocytic nevus.",
    "The lesion is a benign nevus with no signs of malignancy.",
    "This is a routine benign nevus, typical and unremarkable.",
    "Clinical assessment points directly to a benign nevus.",
    "Findings are consistent with a benign, non-cancerous nevus.",
    "The spot is a benign nevus, entirely safe.",
    "Pathology would show a benign nevus.",
    "A perfectly benign nevus on the patient's skin.",
]
MALIGNANT_TEXTS = [
    "The clinical history strongly indicates a malignant melanoma.",
    "Patient presents with a highly dangerous, malignant melanoma.",
    "Dermatoscopic examination confirms a classic malignant melanoma.",
    "The lesion is a malignant melanoma with severe signs of malignancy.",
    "This is a deadly malignant melanoma, atypical and concerning.",
    "Clinical assessment points directly to a malignant melanoma.",
    "Findings are consistent with a malignant, cancerous melanoma.",
    "The spot is a malignant melanoma, highly unsafe.",
    "Pathology would show a malignant melanoma.",
    "A definitively malignant melanoma on the patient's skin.",
]

# ─── Model Registry ────────────────────────────────────────────────────────────

MODEL_CONFIGS = [
    {
        "name":       "llava-med-v1.5-mistral-7b",
        "model_id":   "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        "type":       "llava",
        "layer_idx":  15,
    },
    {
        "name":       "llava-1.5-7b",
        "model_id":   "llava-hf/llava-1.5-7b-hf",
        "type":       "llava",
        "layer_idx":  15,
    },
    # Qwen-VL is added as best-effort; fails gracefully if not available
    {
        "name":       "Qwen-VL",
        "model_id":   "Qwen/Qwen-VL-Chat",
        "type":       "qwenvl",
        "layer_idx":  15,
    },
]


# ─── Model Loading ─────────────────────────────────────────────────────────────

def load_model(cfg: dict, device: torch.device):
    """Load model and processor for the given model config."""
    model_id = cfg["model_id"]
    model_type = cfg["type"]

    if model_type == "llava":
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
        model.eval()
        return model, processor

    elif model_type == "qwenvl":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)
        model.eval()
        return model, tokenizer

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_target_layer(model, cfg: dict):
    """Navigate to the target transformer layer for hook registration."""
    layer_idx = cfg["layer_idx"]
    # LLaVA family
    for attr in ["language_model.model.layers", "language_model.layers", "model.layers"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj[layer_idx]
        except (AttributeError, IndexError, TypeError):
            continue
    raise RuntimeError(f"Cannot find layer {layer_idx} in model {cfg['name']}")


def get_dummy_image():
    return Image.new("RGB", (336, 336), color="black")


# ─── Hidden State Capture ──────────────────────────────────────────────────────

def capture_last_token_hook(captured: dict):
    """Returns a hook that stores the last-token hidden state."""
    def _hook(module, args, kwargs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["h"] = hs[:, -1, :].detach().cpu()
        return output
    return _hook


# ─── Steering Vector Extraction ────────────────────────────────────────────────

def extract_actadd_vector(model, processor, device, layer_idx: int, model_cfg: dict):
    """
    ActAdd (original): mean(benign hiddens) − mean(malignant hiddens).
    Returns (V_nev, V_mel, V_dir_normed, text_norm).
    V_dir = V_NEV − V_MEL  (push towards benign when added)
    """
    print(f"  Extracting ActAdd steering vector at layer {layer_idx}...")
    captured = {}

    try:
        target_layer = get_target_layer(model, model_cfg)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return None, None, None, None

    hook_handle = target_layer.register_forward_hook(
        capture_last_token_hook(captured), with_kwargs=True
    )

    dummy_img = get_dummy_image()
    benign_h, mal_h = [], []

    for text in BENIGN_TEXTS:
        prompt = PROMPT_TEMPLATES["clinical"](text)
        try:
            inputs = processor(text=prompt, images=dummy_img, return_tensors="pt").to(device, torch.float16)
        except Exception:
            inputs = processor(text=prompt, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            model(**inputs)
        benign_h.append(captured["h"])

    for text in MALIGNANT_TEXTS:
        prompt = PROMPT_TEMPLATES["clinical"](text)
        try:
            inputs = processor(text=prompt, images=dummy_img, return_tensors="pt").to(device, torch.float16)
        except Exception:
            inputs = processor(text=prompt, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            model(**inputs)
        mal_h.append(captured["h"])

    hook_handle.remove()

    V_nev = torch.cat(benign_h, dim=0).mean(dim=0)
    V_mel = torch.cat(mal_h,    dim=0).mean(dim=0)
    V_dir = V_nev - V_mel        # Push: benign – malignant (flip MEL → NEV)
    V_dir_normed = F.normalize(V_dir, dim=0).to(device)
    text_norm = torch.cat(benign_h, dim=0).norm(dim=-1).mean().item()

    print(f"    V_dir norm: {V_dir.norm().item():.4f}  text_norm: {text_norm:.4f}")
    return V_nev.to(device), V_mel.to(device), V_dir_normed, text_norm


def extract_caa_vector(model, processor, device, layer_idx: int, model_cfg: dict):
    """
    Contrastive Activation Addition (CAA):
    For each paired (positive, negative) prompt, compute the difference at the
    target layer, then average. This is more robust than ActAdd's mean difference
    because it pairs each prompt with its direct contrast.

    Returns (V_caa_normed, text_norm).
    """
    print(f"  Extracting CAA steering vector at layer {layer_idx}...")
    captured = {}
    try:
        target_layer = get_target_layer(model, model_cfg)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return None, None

    hook_handle = target_layer.register_forward_hook(
        capture_last_token_hook(captured), with_kwargs=True
    )

    dummy_img = get_dummy_image()
    pair_diffs = []

    for b_text, m_text in zip(BENIGN_TEXTS, MALIGNANT_TEXTS):
        b_prompt = PROMPT_TEMPLATES["clinical"](b_text)
        m_prompt = PROMPT_TEMPLATES["clinical"](m_text)

        try:
            b_inputs = processor(text=b_prompt, images=dummy_img, return_tensors="pt").to(device, torch.float16)
        except Exception:
            b_inputs = processor(text=b_prompt, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            model(**b_inputs)
        h_b = captured["h"].clone()

        try:
            m_inputs = processor(text=m_prompt, images=dummy_img, return_tensors="pt").to(device, torch.float16)
        except Exception:
            m_inputs = processor(text=m_prompt, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            model(**m_inputs)
        h_m = captured["h"].clone()

        pair_diffs.append(h_b - h_m)   # Positive class minus negative class

    hook_handle.remove()

    V_caa = torch.stack(pair_diffs, dim=0).mean(dim=0).squeeze(0)
    V_caa_normed = F.normalize(V_caa, dim=0).to(device)
    text_norm = torch.stack(pair_diffs, dim=0).norm(dim=-1).mean().item()

    print(f"    V_caa norm: {V_caa.norm().item():.4f}")
    return V_caa_normed, text_norm


# ─── Subspace Entanglement Diagnostic ─────────────────────────────────────────

def diagnose_subspace_entanglement(
    V_actadd: torch.Tensor,
    V_caa: torch.Tensor,
    V_mel: torch.Tensor,
    V_nev: torch.Tensor,
    model_name: str,
):
    """
    Compute cosine similarities between each steering vector and the estimated
    MEL–NEV decision direction at the hook layer.

    V_decision = V_NEV - V_MEL  (normalised): direction along the decision boundary.
    High cosine sim (close to ±1) means the steering vector aligns with the
    decision direction → steering should work.
    Low cosine sim (close to 0)  means the vectors are orthogonal → the steering
    is operating in a subspace that the model's decision-making ignores.
    """
    V_decision = F.normalize(V_nev - V_mel, dim=0)
    results = {}

    if V_actadd is not None:
        sim_actadd = F.cosine_similarity(V_actadd.unsqueeze(0), V_decision.unsqueeze(0)).item()
        results["ActAdd_cos_sim_to_decision"] = sim_actadd
        entangled = abs(sim_actadd) > 0.3
        print(f"    [ActAdd] cos(V_actadd, V_decision) = {sim_actadd:+.4f}  "
              f"→ {'ALIGNED (entangled)' if entangled else 'ORTHOGONAL (decoupled)'}")

    if V_caa is not None:
        sim_caa = F.cosine_similarity(V_caa.unsqueeze(0), V_decision.unsqueeze(0)).item()
        results["CAA_cos_sim_to_decision"] = sim_caa
        entangled_caa = abs(sim_caa) > 0.3
        print(f"    [CAA]    cos(V_caa,    V_decision) = {sim_caa:+.4f}  "
              f"→ {'ALIGNED (entangled)' if entangled_caa else 'ORTHOGONAL (decoupled)'}")

    if V_actadd is not None and V_caa is not None:
        sim_ab = F.cosine_similarity(V_actadd.unsqueeze(0), V_caa.unsqueeze(0)).item()
        results["ActAdd_CAA_inter_cos_sim"] = sim_ab
        print(f"    [Inter]  cos(V_actadd, V_caa)      = {sim_ab:+.4f}")

    results["model"] = model_name
    return results


# ─── Token Extraction for LLaVA ───────────────────────────────────────────────

def extract_mel_nev_ids(processor):
    mel_words = ["MEL", " MEL", "Melanoma", " Melanoma", "Malignant", " Malignant",
                 "Cancer", " Cancer", "Positive", " Positive", "M", " M"]
    nev_words = ["NEV", " NEV", "Nevus", " Nevus", "Benign", " Benign",
                 "Normal", " Normal", "Negative", " Negative", "N", " N"]

    mel_ids = list(set(sum(
        [processor.tokenizer.encode(w, add_special_tokens=False) for w in mel_words], []
    )))
    nev_ids = list(set(sum(
        [processor.tokenizer.encode(w, add_special_tokens=False) for w in nev_words], []
    )))
    return mel_ids, nev_ids


# ─── Steering Sweep ────────────────────────────────────────────────────────────

def run_steering_sweep(
    model, processor, device,
    mel_samples, nev_samples,
    V_actadd, V_caa, text_norm_actadd, text_norm_caa,
    model_cfg, alphas=(0.0, 0.5, 1.0, 2.0),
    prompt_template="clinical",
):
    """Run actadd and CAA steering sweeps across alphas and collect results."""
    layer_idx = model_cfg["layer_idx"]
    try:
        target_layer = get_target_layer(model, model_cfg)
    except RuntimeError as e:
        print(f"  ERROR getting layer: {e}")
        return []

    mel_ids, nev_ids = extract_mel_nev_ids(processor)

    current_alpha = 0.0
    current_vec   = V_actadd if V_actadd is not None else V_caa
    current_norm  = text_norm_actadd if V_actadd is not None else text_norm_caa

    def steering_hook(module, args, kwargs, output):
        is_tuple = isinstance(output, tuple)
        hs = output[0] if is_tuple else output
        seq_len = hs.shape[1]
        v = current_vec.to(hs.dtype).to(hs.device)
        inject = 40
        if seq_len <= inject:
            hs = hs + (current_alpha * current_norm) * v
        else:
            hs = hs.clone()
            hs[:, -inject:, :] += (current_alpha * current_norm) * v
        return (hs,) + output[1:] if is_tuple else hs

    hook_handle = target_layer.register_forward_hook(steering_hook, with_kwargs=True)

    def score_sample(sample, vec, alpha, t_norm):
        nonlocal current_alpha, current_vec, current_norm
        current_alpha  = alpha
        current_vec    = vec
        current_norm   = t_norm
        prompt = PROMPT_TEMPLATES[prompt_template](sample["hist"])
        inputs = processor(text=prompt, images=sample["img"], return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        ms = logits[mel_ids].max().item()
        ns = logits[nev_ids].max().item()
        return ms, ns, "MEL" if ms > ns else "NEV"

    sweep_rows = []
    for alpha in alphas:
        for method_name, vec, tnorm in [
            ("ActAdd", V_actadd, text_norm_actadd),
            ("CAA",    V_caa,    text_norm_caa),
        ]:
            if vec is None:
                continue

            # Primary: MEL → NEV using +vec (vec is NEV-MEL)
            mel_flips, mel_mel_scores, mel_nev_scores = [], [], []
            for s in tqdm(mel_samples, desc=f"{method_name} alpha={alpha}", leave=False):
                ms, ns, pred = score_sample(s, vec, alpha, tnorm)
                mel_flips.append(pred == "NEV")
                mel_mel_scores.append(ms)
                mel_nev_scores.append(ns)

            # Reverse: NEV → MEL using −vec
            nev_flips = []
            for s in tqdm(nev_samples, desc=f"{method_name} rev alpha={alpha}", leave=False):
                ms, ns, pred = score_sample(s, -vec, alpha, tnorm)
                nev_flips.append(pred == "MEL")

            # Random control
            torch.manual_seed(42)
            V_rand = F.normalize(torch.randn_like(vec), dim=0)
            rand_flips = []
            for s in tqdm(mel_samples, desc=f"Rand alpha={alpha}", leave=False):
                ms, ns, pred = score_sample(s, V_rand, alpha, tnorm)
                rand_flips.append(pred == "NEV")

            if mel_flips:
                sweep_rows.append({
                    "model":       model_cfg["name"],
                    "method":      method_name,
                    "prompt_tmpl": prompt_template,
                    "alpha":       alpha,
                    "condition":   "Primary MEL→NEV",
                    "n":           len(mel_flips),
                    "flip_rate":   float(np.mean(mel_flips)),
                    "mean_mel_score": float(np.mean(mel_mel_scores)),
                    "mean_nev_score": float(np.mean(mel_nev_scores)),
                    "mean_margin":    float(np.mean([m - n for m, n in zip(mel_mel_scores, mel_nev_scores)])),
                })
                sweep_rows.append({
                    "model":       model_cfg["name"],
                    "method":      method_name,
                    "prompt_tmpl": prompt_template,
                    "alpha":       alpha,
                    "condition":   "Reverse NEV→MEL",
                    "n":           len(nev_flips),
                    "flip_rate":   float(np.mean(nev_flips)),
                    "mean_mel_score": np.nan,
                    "mean_nev_score": np.nan,
                    "mean_margin":    np.nan,
                })
                sweep_rows.append({
                    "model":       model_cfg["name"],
                    "method":      method_name,
                    "prompt_tmpl": prompt_template,
                    "alpha":       alpha,
                    "condition":   "Random Control",
                    "n":           len(rand_flips),
                    "flip_rate":   float(np.mean(rand_flips)),
                    "mean_mel_score": np.nan,
                    "mean_nev_score": np.nan,
                    "mean_margin":    np.nan,
                })

    hook_handle.remove()
    return sweep_rows


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False):
    print("=" * 70)
    print("TASK 16: LVLM ROBUSTNESS – MULTI-MODEL STEERING + ENTANGLEMENT DIAGNOSTIC")
    print("=" * 70)
    if dry_run:
        print(">>> DRY RUN MODE: will load first model, extract vectors, print diagnostics only <<<")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_sweep_rows   = []
    all_entanglement = []

    models_to_run = [MODEL_CONFIGS[0]] if dry_run else MODEL_CONFIGS

    for model_cfg in models_to_run:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_cfg['name']}")
        print(f"{'=' * 70}")

        try:
            model, processor = load_model(model_cfg, device)
        except Exception as e:
            print(f"  Failed to load {model_cfg['name']}: {e}")
            print(f"  Skipping model. (May require HF token or gated access.)")
            all_entanglement.append({
                "model": model_cfg["name"],
                "status": "LOAD_FAILED",
                "error": str(e),
            })
            continue

        layer_idx = model_cfg["layer_idx"]

        # Extract steering vectors
        V_nev, V_mel, V_actadd, text_norm_actadd = extract_actadd_vector(
            model, processor, device, layer_idx, model_cfg
        )
        V_caa, text_norm_caa = extract_caa_vector(
            model, processor, device, layer_idx, model_cfg
        )

        # Subspace entanglement diagnostic
        print(f"\n--- Subspace Entanglement Diagnostic [{model_cfg['name']}] ---")
        ent_row = diagnose_subspace_entanglement(
            V_actadd, V_caa, V_mel, V_nev, model_cfg["name"]
        )
        all_entanglement.append(ent_row)

        if dry_run:
            print("\n[DRY RUN] Stopping after entanglement diagnostic.")
            continue

        # Load data
        try:
            df = pd.read_csv(cfg.paths.pad_ufes_csv)
        except FileNotFoundError:
            print(f"  ERROR: PAD-UFES split not found at {cfg.paths.pad_ufes_csv}. Skipping sweep.")
            continue

        df = df[df["diagnostic"].isin(["MEL", "NEV"])].reset_index(drop=True)
        mel_rows = df[df["diagnostic"] == "MEL"]
        nev_rows = df[df["diagnostic"] == "NEV"]
        print(f"  Found {len(mel_rows)} MEL and {len(nev_rows)} NEV images.")

        mel_ids_tok, nev_ids_tok = extract_mel_nev_ids(processor)

        # Baseline filtering
        neutral_hist = "No clinical history available."
        mel_samples, nev_samples = [], []

        print("  Baseline filtering...")
        for subset, container in [(mel_rows, mel_samples), (nev_rows, nev_samples)]:
            for _, row in tqdm(subset.iterrows(), total=len(subset), leave=False):
                img_path = Path(str(row["filepath"]))
                if not img_path.is_absolute():
                    img_path = Path(cfg.paths.package_root) / img_path
                if not os.path.exists(img_path):
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue

                prompt = PROMPT_TEMPLATES["clinical"](neutral_hist)
                inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                ms = logits[mel_ids_tok].max().item()
                ns = logits[nev_ids_tok].max().item()
                pred = "MEL" if ms > ns else "NEV"
                true_diag = str(row["diagnostic"]).strip().upper()

                if pred == true_diag:
                    container.append({"filepath": str(img_path), "true_diag": true_diag, "img": img, "hist": neutral_hist})
                    if len(container) >= 50:
                        break

        print(f"  After baseline filter: {len(mel_samples)} MEL, {len(nev_samples)} NEV")

        # Steering sweep across prompt templates
        for tmpl in PROMPT_TEMPLATES.keys():
            print(f"\n  -- Steering sweep with prompt_template='{tmpl}' --")
            rows = run_steering_sweep(
                model, processor, device,
                mel_samples, nev_samples,
                V_actadd, V_caa, text_norm_actadd, text_norm_caa,
                model_cfg, alphas=[0.0, 0.5, 1.0, 2.0],
                prompt_template=tmpl,
            )
            all_sweep_rows.extend(rows)

        # Offload model to free VRAM
        del model
        torch.cuda.empty_cache()

    # ── Save Results ──────────────────────────────────────────────────────────
    os.makedirs(cfg.paths.results_dir, exist_ok=True)

    if all_sweep_rows:
        out_df = pd.DataFrame(all_sweep_rows)
        out_df.to_csv(os.path.join(cfg.paths.results_dir, "task16_results.csv"), index=False)
        print("\n\nSweep results:")
        print(out_df.to_string(index=False))

    if all_entanglement:
        ent_df = pd.DataFrame(all_entanglement)
        ent_df.to_csv(os.path.join(cfg.paths.results_dir, "task16_entanglement_diagnostics.csv"), index=False)
        print("\nEntanglement diagnostics:")
        print(ent_df.to_string(index=False))

    print("\nTask 16 complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LVLM Robustness Steering – Task 16")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load first model, extract vectors, print diagnostics only.")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
