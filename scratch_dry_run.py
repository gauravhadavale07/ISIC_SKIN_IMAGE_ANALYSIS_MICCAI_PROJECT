import pandas as pd
import modal
import os

app = modal.App("lvlm-dry-run")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

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
def dry_run():
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image

    print("Loading processor/model...")
    device = "cuda"
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    df = pd.read_csv("pad_ufes_20_test.csv")
    
    print("\n--- 5-IMAGE PREFILLED PROMPT DRY RUN ---")
    for i in range(5):
        row = df.iloc[i]
        img_path = str(row['filepath'])
        real_text = str(row['clinical_history']).strip()
        img = Image.open(img_path).convert('RGB')
        
        prompt = f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: [{real_text}]. What is the diagnosis? Is the diagnosis MEL, NEV, BCC, ACK, SEK, BOD, or SCC? Answer with only the exact abbreviation. [/INST]The diagnosis is "
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3)
        
        generated_ids = out[0][inputs.input_ids.shape[1]:]
        gen_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"Image {i+1} (True: {row['diagnostic']}): '{gen_text}' (Raw IDs: {generated_ids.tolist()})")

@app.local_entrypoint()
def main():
    dry_run.remote()
