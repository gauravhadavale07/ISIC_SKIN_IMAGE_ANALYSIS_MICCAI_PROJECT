import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import torch.nn.functional as F

def get_dummy_image():
    # LLaVA-Med expects 336x336 images for its CLIP encoder
    return Image.new('RGB', (336, 336), color='black')

def extract_steering_vector(model, processor, device, layer_idx=15):
    print("\n--- Extracting Paraphrased Steering Vector ---")
    benign_prompts = [
        "The clinical history strongly indicates a benign nevus.",
        "Patient presents with a completely harmless, benign mole.",
        "Dermatoscopic examination confirms a classic benign melanocytic nevus.",
        "The lesion is a benign nevus with no signs of malignancy.",
        "This is a routine benign nevus, typical and unremarkable.",
        "Clinical assessment points directly to a benign nevus.",
        "Findings are consistent with a benign, non-cancerous nevus.",
        "The spot is a benign nevus, entirely safe.",
        "Pathology would show a benign nevus.",
        "A perfectly benign nevus on the patient's skin."
    ]
    
    malignant_prompts = [
        "The clinical history strongly indicates a malignant melanoma.",
        "Patient presents with a highly dangerous, malignant melanoma.",
        "Dermatoscopic examination confirms a classic malignant melanoma.",
        "The lesion is a malignant melanoma with severe signs of malignancy.",
        "This is a deadly malignant melanoma, atypical and concerning.",
        "Clinical assessment points directly to a malignant melanoma.",
        "Findings are consistent with a malignant, cancerous melanoma.",
        "The spot is a malignant melanoma, highly unsafe.",
        "Pathology would show a malignant melanoma.",
        "A definitively malignant melanoma on the patient's skin."
    ]

    dummy_img = get_dummy_image()
    
    benign_hiddens = []
    malignant_hiddens = []
    
    # We register a hook to capture the hidden states of the final token at layer 15
    captured_hiddens = {}
    def capture_hook(module, args, kwargs, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        # hidden_states is safely (batch, seq_len, hidden_dim)
        captured_hiddens['h'] = hidden_states[:, -1, :].detach().cpu()
        
        # Return exactly what came in
        return output

    try:
        target_layer = model.language_model.model.layers[layer_idx]
    except AttributeError:
        target_layer = model.language_model.layers[layer_idx]
        
    hook_handle = target_layer.register_forward_hook(capture_hook, with_kwargs=True)

    print("Extracting Benign activations...")
    for text in benign_prompts:
        prompt = f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: [{text}]. What is the diagnosis? Is the diagnosis MEL or NEV? Diagnosis:"
        inputs = processor(text=prompt, images=dummy_img, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            model(**inputs)
        benign_hiddens.append(captured_hiddens['h'])
        
    print("Extracting Malignant activations...")
    for text in malignant_prompts:
        prompt = f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: [{text}]. What is the diagnosis? Is the diagnosis MEL or NEV? Diagnosis:"
        inputs = processor(text=prompt, images=dummy_img, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            model(**inputs)
        malignant_hiddens.append(captured_hiddens['h'])
        
    hook_handle.remove()

    benign_hiddens = torch.cat(benign_hiddens, dim=0) # (10, hidden_dim)
    malignant_hiddens = torch.cat(malignant_hiddens, dim=0) # (10, hidden_dim)
    
    # Hold-out check (using first 7 for mean, last 3 for holdout check)
    V_train_benign = benign_hiddens[:7].mean(dim=0)
    V_train_malignant = malignant_hiddens[:7].mean(dim=0)
    V_mean = V_train_benign - V_train_malignant
    V_mean_normed = F.normalize(V_mean, dim=0)
    
    print("Hold-out cosine similarities to V_mean:")
    for i in range(7, 10):
        v_holdout = benign_hiddens[i] - malignant_hiddens[i]
        sim = F.cosine_similarity(v_holdout.unsqueeze(0), V_mean.unsqueeze(0)).item()
        print(f"  Pair {i}: {sim:.4f}")

    # Final vector uses all 10 pairs
    V_final = benign_hiddens.mean(dim=0) - malignant_hiddens.mean(dim=0)
    V_dir = F.normalize(V_final, dim=0).to(device)
    
    # Calculate mean norm of text tokens (approximated by the last token norm)
    text_norm = benign_hiddens.norm(dim=-1).mean().item()
    print(f"Mean Text Token Norm at Layer {layer_idx}: {text_norm:.4f}")
    
    return V_dir, text_norm

def run_inference():
    print("======================================================================")
    print("TASK 16: LVLM CAUSAL ACTIVATION STEERING (THE HALLUCINATION HIJACK)")
    print("======================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    
    print(f"Loading Model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    layer_idx = 15
    V_dir, text_norm = extract_steering_vector(model, processor, device, layer_idx)

    # Prepare forced choice tokens
    # Symmetrically expand the target vocabularies to catch shifts to related synonyms
    mel_words = ["MEL", " MEL", "Melanoma", " Melanoma", "Malignant", " Malignant", "Cancer", " Cancer", "Positive", " Positive", "M", " M"]
    nev_words = ["NEV", " NEV", "Nevus", " Nevus", "Benign", " Benign", "Normal", " Normal", "Negative", " Negative", "N", " N"]
    
    mel_ids = []
    for w in mel_words:
        mel_ids.extend(processor.tokenizer.encode(w, add_special_tokens=False))
    mel_ids = list(set(mel_ids))
    
    nev_ids = []
    for w in nev_words:
        nev_ids.extend(processor.tokenizer.encode(w, add_special_tokens=False))
    nev_ids = list(set(nev_ids))
    # Remove duplicates
    mel_ids = list(set(mel_ids))
    nev_ids = list(set(nev_ids))
    print(f"Target Token IDs -> MEL: {mel_ids}, NEV: {nev_ids}")

    df = pd.read_csv('./pad_ufes_20_test.csv')
    df = df[df['diagnostic'].isin(['MEL', 'NEV'])]
    print(f"Found {len(df[df['diagnostic']=='MEL'])} MEL and {len(df[df['diagnostic']=='NEV'])} NEV images in test set.")

    # We need to filter for baseline correctness
    neutral_history = "No clinical history available."
    
    clean_samples = []
    
    print("\n--- Baseline Filtering (alpha=0) ---")
    for idx in tqdm(range(len(df))):
        row = df.iloc[idx]
        img_path = str(row['filepath'])
        if not os.path.exists(img_path): continue
            
        img = Image.open(img_path).convert('RGB')
        true_diag = row['diagnostic'].strip().upper()
        
        prompt = f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: [{neutral_history}]. What is the diagnosis? Is the diagnosis MEL or NEV? Diagnosis:"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get logits of the last token (the prediction for the masked diagnosis)
        next_token_logits = outputs.logits[0, -1, :]
        
        mel_prob = torch.softmax(next_token_logits[mel_ids], dim=-1).sum().item()
        nev_prob = torch.softmax(next_token_logits[nev_ids], dim=-1).sum().item()
        
        # We just compare raw logits for the forced choice
        mel_score = next_token_logits[mel_ids].max().item()
        nev_score = next_token_logits[nev_ids].max().item()
        
        pred_diag = "MEL" if mel_score > nev_score else "NEV"
        if pred_diag == true_diag:
            clean_samples.append({
                'filepath': img_path,
                'true_diag': true_diag,
                'img': img,
                'prompt': prompt,
                'base_mel_score': mel_score,
                'base_nev_score': nev_score
            })
            
    # Subsample to keep it balanced and fast (N=50 per class)
    mel_samples = [s for s in clean_samples if s['true_diag'] == 'MEL'][:50]
    nev_samples = [s for s in clean_samples if s['true_diag'] == 'NEV'][:50]
    print(f"Baseline filtered: {len(mel_samples)} MEL and {len(nev_samples)} NEV images correctly classified.")

    # Sweep parameters
    alphas = [0.0, 0.5, 1.0, 2.0]
    
    # Random direction control
    torch.manual_seed(42)
    V_random = F.normalize(torch.randn_like(V_dir), dim=0).to(device)

    # We will hook the layer to add alpha * norm * V_dir
    current_alpha = 0.0
    current_vec = V_dir

    def steering_hook(module, args, kwargs, output):
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output
        
        seq_len = hidden_states.shape[1]
        v = current_vec.to(hidden_states.dtype).to(hidden_states.device)
        
        inject_length = 40
        if seq_len == 1:
            # Incremental decode step (though we aren't using .generate() currently)
            hidden_states = hidden_states + (current_alpha * text_norm) * v
        elif seq_len > inject_length:
            # Inject only at the final text tokens to avoid the modality gap
            # We don't use clone() here as modifying hidden_states in-place or returning a new one is fine 
            # for decoder layers, but cloning is safer for gradients. Since we are in no_grad(), direct slice addition is okay.
            hidden_states = hidden_states.clone()
            hidden_states[:, -inject_length:, :] += (current_alpha * text_norm) * v
        else:
            hidden_states = hidden_states + (current_alpha * text_norm) * v
            
        return (hidden_states,) + output[1:] if is_tuple else hidden_states

    try:
        target_layer = model.language_model.model.layers[layer_idx]
    except AttributeError:
        target_layer = model.language_model.layers[layer_idx]
        
    hook_handle = target_layer.register_forward_hook(steering_hook, with_kwargs=True)

    def score_sample(sample, vec, alpha):
        nonlocal current_alpha, current_vec
        current_alpha = alpha
        current_vec = vec
        inputs = processor(
            text=sample['prompt'],
            images=sample['img'],
            return_tensors="pt",
        ).to(device, torch.float16)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        mel_score = logits[mel_ids].max().item()
        nev_score = logits[nev_ids].max().item()
        pred = "MEL" if mel_score > nev_score else "NEV"
        return mel_score, nev_score, pred, logits

    def summarize_condition(condition, samples, vec, alpha, target_class):
        mel_scores, nev_scores, margins, preds = [], [], [], []
        for sample in tqdm(samples, desc=f"{condition} alpha={alpha}", leave=False):
            mel_score, nev_score, pred, _ = score_sample(sample, vec, alpha)
            mel_scores.append(mel_score)
            nev_scores.append(nev_score)
            margins.append(mel_score - nev_score)
            preds.append(pred)

        if len(samples) == 0:
            return {
                "condition": condition,
                "alpha": alpha,
                "n": 0,
                "target_class": target_class,
                "flip_rate": np.nan,
                "mean_mel_score": np.nan,
                "mean_nev_score": np.nan,
                "mean_mel_minus_nev_margin": np.nan,
            }

        flips = [pred == target_class for pred in preds]
        return {
            "condition": condition,
            "alpha": alpha,
            "n": len(samples),
            "target_class": target_class,
            "flip_rate": float(np.mean(flips)),
            "mean_mel_score": float(np.mean(mel_scores)),
            "mean_nev_score": float(np.mean(nev_scores)),
            "mean_mel_minus_nev_margin": float(np.mean(margins)),
        }

    print("\n--- Real Steering Sweep ---")
    sweep_rows = []
    for alpha in alphas:
        sweep_rows.append(
            summarize_condition(
                "Primary (MEL->NEV)",
                mel_samples,
                V_dir,
                alpha,
                target_class="NEV",
            )
        )
        sweep_rows.append(
            summarize_condition(
                "Random Control",
                mel_samples,
                V_random,
                alpha,
                target_class="NEV",
            )
        )
        sweep_rows.append(
            summarize_condition(
                "Reverse (NEV->MEL)",
                nev_samples,
                -V_dir,
                alpha,
                target_class="MEL",
            )
        )

    out_df = pd.DataFrame(sweep_rows)
    os.makedirs("results", exist_ok=True)
    out_df.to_csv("results/task16_results.csv", index=False)
    out_df.to_csv("task16_results.csv", index=False)
    print("\nSweep results:")
    print(out_df.to_string(index=False))

    # --- Single Sample Deep Diagnostic ---
    print("\n--- Single Sample Deep Diagnostic ---")
    if len(mel_samples) == 0:
        print("No MEL samples found for diagnostic.")
        hook_handle.remove()
        return

    sample = mel_samples[0]
    print(f"Sample True Diag: {sample['true_diag']}")
    diagnostic_rows = []

    for alpha in [0.0, 2.0, 50.0]:
        mel_score, nev_score, pred, logits = score_sample(sample, V_dir, alpha)

        top_token_id = logits.argmax().item()
        top_token_str = processor.tokenizer.decode([top_token_id])
        mel_top_id = mel_ids[logits[mel_ids].argmax().item()]
        mel_top_str = processor.tokenizer.decode([mel_top_id])
        nev_top_id = nev_ids[logits[nev_ids].argmax().item()]
        nev_top_str = processor.tokenizer.decode([nev_top_id])

        diagnostic_rows.append({
            "alpha": alpha,
            "pred": pred,
            "mel_score": mel_score,
            "nev_score": nev_score,
            "mel_minus_nev_margin": mel_score - nev_score,
            "top_token": top_token_str,
            "mel_bucket_token": mel_top_str,
            "nev_bucket_token": nev_top_str,
        })

        print(f"\n[Alpha = {alpha}]")
        print(f"  Top-1 Token (Absolute): '{top_token_str}' (ID {top_token_id}) | Logit: {logits[top_token_id]:.2f}")
        print(f"  MEL Bucket Max: '{mel_top_str}' (ID {mel_top_id}) | Logit: {mel_score:.2f}")
        print(f"  NEV Bucket Max: '{nev_top_str}' (ID {nev_top_id}) | Logit: {nev_score:.2f}")

    pd.DataFrame(diagnostic_rows).to_csv("results/task16_single_sample_diagnostic.csv", index=False)
    hook_handle.remove()
    print("\nDiagnostic completed.")

if __name__ == "__main__":
    run_inference()
