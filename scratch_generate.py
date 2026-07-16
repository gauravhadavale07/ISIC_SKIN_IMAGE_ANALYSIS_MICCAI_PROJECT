import pandas as pd
import modal
import os
import subprocess

app = modal.App("lvlm-generate-sanity-check")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio", "tqdm", "pandas", "numpy",
    "scikit-learn", "transformers", "timm", "huggingface_hub==0.23.2", "accelerate", "pillow", "scipy"
).run_commands(
    "huggingface-cli download chaoyinshe/llava-med-v1.5-mistral-7b-hf"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=["data", "results", "logs", ".git", "__pycache__"]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data", 
    remote_path="/root/project/data"
)

@app.function(image=image, gpu="H200")
def generate_samples():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image

    print("Loading model...")
    device = "cuda"
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    df = pd.read_csv("pad_ufes_20_test.csv")
    
    # We just do 5 samples
    for i in range(5):
        row = df.iloc[i]
        img_path = str(row['filepath'])
        real_text = str(row['clinical_history']).strip()
        shuffled_text = str(df.iloc[(i + 1) % len(df)]['clinical_history']).strip()
        neutral_text = "No clinical history available."
        
        img = Image.open(img_path).convert('RGB')
        
        conditions = {
            'Real': real_text,
            'Shuffled': shuffled_text,
            'Neutral': neutral_text
        }
        
        print(f"\n--- IMAGE {i+1} (True: {row['diagnostic']}) ---")
        for cond, text in conditions.items():
            prompt = f"USER: <image>\nAnalyze this clinical photograph of a patient with this history: [{text}]. What is the diagnosis? Is the diagnosis MEL, NEV, BCC, ACK, SEK, BOD, or SCC? Answer with only the exact abbreviation.\nASSISTANT: "
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10)
            
            generated_ids = out[0][inputs.input_ids.shape[1]:]
            gen_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            print(f"{cond} Output: '{gen_text}' (Raw IDs: {generated_ids.tolist()})")

@app.local_entrypoint()
def main():
    generate_samples.remote()
