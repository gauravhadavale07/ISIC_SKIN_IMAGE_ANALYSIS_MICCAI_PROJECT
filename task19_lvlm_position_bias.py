import modal
import os

image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate", "pandas", "pillow", "sentencepiece", "protobuf").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=[".git", "__pycache__"]
)
app = modal.App("task19-lvlm-position-bias")

@app.function(
    image=image, 
    gpu="a100", 
    timeout=1800
)
def run_position_bias():
    import os
    os.chdir("/root/project")
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from PIL import Image

    import pandas as pd
    from PIL import Image

    print("--- Running Task 19: LLaVA-Med Position Bias ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load LLaVA-Med
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    # Load 52 clear-cut MEL images
    df = pd.read_csv('pad_ufes_20_test.csv')
    mel_df = df[df['diagnostic'] == 'MEL'].head(52)
    img_paths = mel_df['filepath'].tolist()
    
    prompt_orig = "[INST] <image>\nAnalyze this clinical photograph. What is the diagnosis? Is the diagnosis MEL or NEV? Diagnosis:"
    prompt_rev = "[INST] <image>\nAnalyze this clinical photograph. What is the diagnosis? Is the diagnosis NEV or MEL? Diagnosis:"

    m_id = processor.tokenizer.encode("M", add_special_tokens=False)[-1]
    n_id = processor.tokenizer.encode("N", add_special_tokens=False)[-1]

    orig_m_margins = []
    rev_m_margins = []

    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')
        
        # Original Order
        inputs_orig = processor(text=prompt_orig, images=img, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            outputs_orig = model(**inputs_orig)
        logits_orig = outputs_orig.logits[0, -1, :]
        margin_orig = logits_orig[m_id].item() - logits_orig[n_id].item()
        orig_m_margins.append(margin_orig)
        
        # Reversed Order
        inputs_rev = processor(text=prompt_rev, images=img, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            outputs_rev = model(**inputs_rev)
        logits_rev = outputs_rev.logits[0, -1, :]
        margin_rev = logits_rev[m_id].item() - logits_rev[n_id].item()
        rev_m_margins.append(margin_rev)

    avg_margin_orig = sum(orig_m_margins) / len(orig_m_margins)
    avg_margin_rev = sum(rev_m_margins) / len(rev_m_margins)
    
    print(f"\n--- Original Order: 'MEL or NEV?' ---")
    print(f"Average MEL Margin (+ favors MEL, - favors NEV): {avg_margin_orig:.2f}")

    print(f"\n--- Reversed Order: 'NEV or MEL?' ---")
    print(f"Average MEL Margin (+ favors MEL, - favors NEV): {avg_margin_rev:.2f}")
    
    print("\nTask 19 Completed.")

@app.local_entrypoint()
def main():
    run_position_bias.remote()
