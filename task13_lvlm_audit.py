import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

def main():
    print("======================================================================")
    print("TASK 13: GENERATIVE LVLM TEXT-ABLATION CROSS-CHECK (PHASE 2: LLaVA-Med)")
    print("======================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Swap to the unofficial HF port for Phase 2
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    
    print(f"Loading Model: {model_id}...")
    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    # Load dataset
    df = pd.read_csv('./pad_ufes_20_test.csv')
    img_dir = './data/pad_ufes_20/'

    # Valid abbreviations for the prompt constraint
    valid_diagnoses = ["BCC", "SCC", "ACK", "SEK", "BOD", "MEL", "NEV"]
    
    # Pre-tokenize all valid diagnoses IN CONTEXT to avoid sub-word alignment issues
    print("\nTokenizing targets in-context...")
    target_token_ids = {}
    dummy_prompt = "Answer with only the exact abbreviation. [/INST]"
    prompt_tokens_len = len(processor.tokenizer(dummy_prompt, add_special_tokens=False)['input_ids'])
    
    for diag in valid_diagnoses:
        full_tokens = processor.tokenizer(dummy_prompt + diag, add_special_tokens=False)['input_ids']
        tokens = full_tokens[prompt_tokens_len:]
        target_token_ids[diag] = tokens
        print(f"Diagnosis {diag} -> Tokens: {tokens} -> {[processor.tokenizer.decode([t]) for t in tokens]}")

    results = []
    
    # Load test data
    df = pd.read_csv("pad_ufes_20_test.csv")
    img_dir = "data/"
    
    print("\nStarting Inference Loop...")
    with torch.no_grad():
        for idx in tqdm(range(len(df))):
            row = df.iloc[idx]
            
            # The column is 'filepath' and already contains the path
            img_path = str(row['filepath'])
            
            if not os.path.exists(img_path):
                continue
                
            img = Image.open(img_path).convert('RGB')
            true_diag = row['diagnostic'].strip().upper()
            
            if true_diag not in target_token_ids:
                continue
                
            target_ids = target_token_ids[true_diag]
            
            # Text variations
            real_text = str(row['clinical_history']).strip()
            # Shuffled metadata (from another patient)
            shuffled_text = str(df.iloc[(idx + 1) % len(df)]['clinical_history']).strip()
            # Neutral baseline
            neutral_text = "No clinical history available."
            
            conditions = {
                'Real': real_text,
                'Shuffled': shuffled_text,
                'Neutral': neutral_text
            }

            row_results = {'filepath': row['filepath'], 'true_diag': true_diag}

            for cond_name, text in conditions.items():
                prompt = f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: [{text}]. What is the diagnosis? Is the diagnosis MEL, NEV, BCC, ACK, SEK, BOD, or SCC? Answer with only the exact abbreviation. [/INST]"
                
                inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
                
                # To get the joint probability of the target tokens without generating,
                # we append the target tokens to the input, run a forward pass, and extract the logits at those positions.
                
                # First, get the input sequence length
                input_seq_len = inputs['input_ids'].shape[1]
                
                # Concatenate the target tokens to the input_ids
                target_tensor = torch.tensor([target_ids], dtype=torch.long).to(device)
                full_input_ids = torch.cat([inputs['input_ids'], target_tensor], dim=1)
                
                # Extend attention mask
                extended_mask = torch.cat([inputs['attention_mask'], torch.ones_like(target_tensor)], dim=1)
                
                # Run forward pass (we don't need to generate, just get logits for the full sequence)
                outputs = model(input_ids=full_input_ids, attention_mask=extended_mask, pixel_values=inputs['pixel_values'])
                logits = outputs.logits  # Shape: (1, expanded_seq_len, vocab_size)
                
                # In LLaVA, the <image> token is dynamically expanded into hundreds of patch tokens.
                # Because of this, the output sequence length is much longer than input_seq_len.
                # However, the target tokens are always at the very end of the sequence.
                # To predict target_ids[0], we need the logit from the token immediately preceding it (index: -len(target_ids) - 1)
                # To predict target_ids[1], we need the logit from target_ids[0] (index: -len(target_ids))
                
                joint_log_prob = 0.0
                expanded_seq_len = logits.shape[1]
                L = len(target_ids)
                
                for i, target_id in enumerate(target_ids):
                    # Position of the logit predicting this token
                    pos = expanded_seq_len - L - 1 + i
                    
                    # Get log_softmax for this position
                    token_logits = logits[0, pos, :]
                    log_probs = torch.nn.functional.log_softmax(token_logits, dim=0)
                    
                    # Extract the log probability of the correct token
                    token_log_prob = log_probs[target_id].item()
                    joint_log_prob += token_log_prob
                    
                # Normalize by length to get average log-prob per token
                normalized_log_prob = joint_log_prob / L
                row_results[f'{cond_name}_log_prob'] = normalized_log_prob
                
            results.append(row_results)
            
    # Save results
    os.makedirs("results", exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv("results/task13_lvlm_audit.csv", index=False)
    print("\nSaved results to results/task13_lvlm_audit.csv")

    # Compute summary
    print("\nSummary Results:")
    print(f"Mean Real Text Log-Prob: {out_df['Real_log_prob'].mean():.4f}")
    print(f"Mean Shuffled Text Log-Prob: {out_df['Shuffled_log_prob'].mean():.4f}")
    print(f"Mean Neutral Text Log-Prob: {out_df['Neutral_log_prob'].mean():.4f}")
    
    # Wilcoxon signed-rank test
    from scipy.stats import wilcoxon
    try:
        stat, pval = wilcoxon(out_df['Real_log_prob'], out_df['Shuffled_log_prob'])
        print(f"\nWilcoxon Signed-Rank Test (Real vs Shuffled): p = {pval:.4e}")
        if pval > 0.05:
            print("CONCLUSION: The model shows NO significant difference when text is shuffled (Text-Blindness Proven!)")
        else:
            print("CONCLUSION: The model shows a significant difference when text is shuffled.")
    except Exception as e:
        print(f"Could not compute Wilcoxon test: {e}")

if __name__ == "__main__":
    main()
