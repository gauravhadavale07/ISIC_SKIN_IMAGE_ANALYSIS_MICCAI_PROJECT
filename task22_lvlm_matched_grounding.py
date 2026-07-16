import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

def main():
    print("======================================================================")
    print("TASK 22: LVLM 2x2 MATCHED GROUNDING AUDIT (LLaVA-Med-v1.5-Mistral-7B)")
    print("======================================================================")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    print(f"Loading Model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    df = pd.read_csv('./pad_ufes_20_test.csv')
    img_dir = './data/pad_ufes_20/'

    # We need 52 MEL and 52 NEV
    df_mel = df[df['diagnostic'].str.upper().str.strip() == 'MEL'].head(52)
    df_nev = df[df['diagnostic'].str.upper().str.strip() == 'NEV'].head(52)
    audit_df = pd.concat([df_mel, df_nev])

    # Text conditions
    TEXT_CONDITIONS = {
        "MEL_consistent": "Patient history is highly suspicious for invasive melanoma: rapid evolution, irregular pigmentation, asymmetric growth, and recent change.",
        "NEV_consistent": "Patient history is consistent with a stable benign melanocytic nevus: long-standing, symmetric, unchanged, and without alarming symptoms."
    }

    results = []

    # Target tokens
    dummy_prompt = "Answer with only the exact abbreviation. [/INST]"
    prompt_tokens_len = len(processor.tokenizer(dummy_prompt, add_special_tokens=False)['input_ids'])
    
    target_token_ids = {}
    for diag in ["MEL", "NEV"]:
        full_tokens = processor.tokenizer(dummy_prompt + diag, add_special_tokens=False)['input_ids']
        target_token_ids[diag] = full_tokens[prompt_tokens_len:]

    print("\nStarting Inference Loop...")
    with torch.no_grad():
        for idx in tqdm(range(len(audit_df))):
            row = audit_df.iloc[idx]
            img_path = str(row['filepath'])
            true_diag = row['diagnostic'].strip().upper()
            
            # Real image and Blank image
            if not os.path.exists(img_path):
                print(f"Missing {img_path}")
                continue
            
            real_img = Image.open(img_path).convert('RGB')
            blank_img = Image.new("RGB", real_img.size, color="black")

            for img_name, img in [("real_image", real_img), ("blank_image", blank_img)]:
                for text_cond, text_val in TEXT_CONDITIONS.items():
                    # Randomize answer order
                    for ans_order in ["MEL, NEV", "NEV, MEL"]:
                        prompt = f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: [{text_val}]. What is the diagnosis? Is the diagnosis {ans_order}? Answer with only the exact abbreviation. [/INST]"
                        
                        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
                        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], pixel_values=inputs['pixel_values'])
                        logits = outputs.logits
                        
                        # Compute joint log prob for MEL and NEV
                        diag_log_probs = {}
                        expanded_seq_len = logits.shape[1]
                        
                        for diag_target in ["MEL", "NEV"]:
                            t_ids = target_token_ids[diag_target]
                            L = len(t_ids)
                            joint_log_prob = 0.0
                            for i, t_id in enumerate(t_ids):
                                pos = expanded_seq_len - L - 1 + i
                                token_logits = logits[0, pos, :]
                                log_probs = torch.nn.functional.log_softmax(token_logits, dim=0)
                                joint_log_prob += log_probs[t_id].item()
                            diag_log_probs[diag_target] = joint_log_prob / L
                        
                        pred_diag = "MEL" if diag_log_probs["MEL"] > diag_log_probs["NEV"] else "NEV"
                        
                        res = {
                            "filepath": img_path,
                            "true_class": true_diag,
                            "image_condition": img_name,
                            "text_condition": text_cond,
                            "answer_order": ans_order,
                            "MEL_logprob": diag_log_probs["MEL"],
                            "NEV_logprob": diag_log_probs["NEV"],
                            "pred_class": pred_diag,
                            "correct": (pred_diag == true_diag),
                            "aligned_text": (true_diag in text_cond),
                            "text_override": (pred_diag != true_diag and pred_diag in text_cond)
                        }
                        results.append(res)

    os.makedirs("results", exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv("results/task22_lvlm_matched_grounding_samples.csv", index=False)
    
    # Compute summary
    summary = []
    for img_cond in ["real_image", "blank_image"]:
        sub_df = out_df[out_df["image_condition"] == img_cond]
        mel_first = sub_df[sub_df["answer_order"] == "MEL, NEV"]
        nev_first = sub_df[sub_df["answer_order"] == "NEV, MEL"]
        
        # Aligned vs contradictory
        aligned = sub_df[sub_df["aligned_text"] == True]
        contradictory = sub_df[sub_df["aligned_text"] == False]
        
        aligned_acc = aligned["correct"].mean() if len(aligned) > 0 else 0
        contradictory_acc = contradictory["correct"].mean() if len(contradictory) > 0 else 0
        flip_rate = contradictory["text_override"].mean() if len(contradictory) > 0 else 0
        
        # Answer order bias
        mel_first_mel_preds = (mel_first["pred_class"] == "MEL").mean() if len(mel_first) > 0 else 0
        nev_first_nev_preds = (nev_first["pred_class"] == "NEV").mean() if len(nev_first) > 0 else 0
        
        summary.append({
            "image_condition": img_cond,
            "aligned_accuracy": aligned_acc,
            "contradictory_accuracy": contradictory_acc,
            "flip_rate": flip_rate,
            "mel_first_mel_preds_rate": mel_first_mel_preds,
            "nev_first_nev_preds_rate": nev_first_nev_preds
        })

    sum_df = pd.DataFrame(summary)
    sum_df.to_csv("results/task22_lvlm_matched_grounding_summary.csv", index=False)
    sum_df.to_json("results/task22_lvlm_matched_grounding_summary.json", orient="records", indent=2)
    print("Done! Saved to results/")

if __name__ == "__main__":
    main()
