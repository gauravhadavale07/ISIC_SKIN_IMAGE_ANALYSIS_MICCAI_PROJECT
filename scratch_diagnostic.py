import pandas as pd
import modal
import os
import json
from datetime import datetime

app = modal.App("lvlm-diagnostic")
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

@app.function(
    image=image,
    gpu="H200",
    timeout=3600,
    volumes={"/root/project/results": vol_results},
)
def run_diagnostic():
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    import torch
    import torch.nn.functional as F
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image

    result_txt = "/root/project/results/lvlm_diagnostic_output.txt"
    result_json = "/root/project/results/lvlm_diagnostic_output.json"
    records = {"started_at": datetime.utcnow().isoformat() + "Z", "samples": []}

    def log(msg):
        line = f"[{datetime.utcnow().isoformat()}Z] {msg}"
        print(line, flush=True)
        with open(result_txt, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    # Clear previous diagnostic output inside the mounted results volume.
    for path in (result_txt, result_json):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    log("Loading processor/model...")
    device = "cuda"
    model_id = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    log("Processor loaded.")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    log("Model loaded and moved to CUDA.")

    if hasattr(processor.tokenizer, "chat_template") and processor.tokenizer.chat_template:
        log("Chat template exists on tokenizer.")
        log(processor.tokenizer.chat_template)
    else:
        log("No chat template found on tokenizer.")

    df = pd.read_csv("pad_ufes_20_test.csv")
    row = df.iloc[0]
    img = Image.open(str(row['filepath'])).convert('RGB')
    real_text = str(row['clinical_history']).strip()

    log("=" * 50)
    log("TEST 1: OLD PROMPT (USER/ASSISTANT)")
    log("=" * 50)
    old_prompt = f"USER: <image>\nAnalyze this clinical photograph of a patient with this history: [{real_text}]. What is the diagnosis? Is the diagnosis MEL, NEV, BCC, ACK, SEK, BOD, or SCC? Answer with only the exact abbreviation.\nASSISTANT: "
    inputs_old = processor(text=old_prompt, images=img, return_tensors="pt").to(device, torch.float16)
    
    with torch.no_grad():
        out_old = model(**inputs_old)
        logits_old = out_old.logits[0, -1, :] # Last position
        probs_old = F.softmax(logits_old, dim=-1)
        top5_probs_old, top5_ids_old = torch.topk(probs_old, 5)
        eos_prob_old = probs_old[processor.tokenizer.eos_token_id].item()
        
    log(f"EOS token: {processor.tokenizer.eos_token!r} (ID: {processor.tokenizer.eos_token_id})")
    log(f"Old prompt EOS probability: {eos_prob_old:.6%}")
    log("Top 5 tokens for old prompt:")
    old_top5 = []
    for i in range(5):
        token_str = processor.tokenizer.decode([top5_ids_old[i].item()])
        prob = top5_probs_old[i].item()
        old_top5.append({"token": token_str, "id": top5_ids_old[i].item(), "prob": prob})
        log(f"  {token_str!r} (ID: {top5_ids_old[i].item()}): {prob:.4%}")


    log("=" * 50)
    log("TEST 2: NEW MISTRAL PROMPT ([INST])")
    log("=" * 50)
    # If no chat template, we construct it manually using the standard Mistral LLaVA template
    new_prompt = f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: [{real_text}]. What is the diagnosis? Is the diagnosis MEL, NEV, BCC, ACK, SEK, BOD, or SCC? Answer with only the exact abbreviation. [/INST]"
    inputs_new = processor(text=new_prompt, images=img, return_tensors="pt").to(device, torch.float16)
    
    with torch.no_grad():
        out_new = model(**inputs_new)
        logits_new = out_new.logits[0, -1, :] # Last position
        probs_new = F.softmax(logits_new, dim=-1)
        top5_probs_new, top5_ids_new = torch.topk(probs_new, 5)
        eos_prob_new = probs_new[processor.tokenizer.eos_token_id].item()
        
    log(f"New prompt EOS probability: {eos_prob_new:.6%}")
    log("Top 5 tokens for NEW prompt:")
    new_top5 = []
    for i in range(5):
        token_str = processor.tokenizer.decode([top5_ids_new[i].item()])
        prob = top5_probs_new[i].item()
        new_top5.append({"token": token_str, "id": top5_ids_new[i].item(), "prob": prob})
        log(f"  {token_str!r} (ID: {top5_ids_new[i].item()}): {prob:.4%}")

    records["old_prompt"] = {"eos_prob": eos_prob_old, "top5": old_top5}
    records["new_prompt"] = {"eos_prob": eos_prob_new, "top5": new_top5}

    log("=" * 50)
    log("TEST 3: GENERATING 20 SAMPLES WITH NEW PROMPT")
    log("=" * 50)
    for i in range(20):
        row = df.iloc[i]
        img_path = str(row['filepath'])
        real_text = str(row['clinical_history']).strip()
        img = Image.open(img_path).convert('RGB')
        
        prompt = f"[INST] <image>\nAnalyze this clinical photograph of a patient with this history: [{real_text}]. What is the diagnosis? Is the diagnosis MEL, NEV, BCC, ACK, SEK, BOD, or SCC? Answer with only the exact abbreviation. [/INST]"
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
        
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=10)
        
        generated_ids = out[0][inputs.input_ids.shape[1]:]
        gen_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
        sample = {
            "index": i + 1,
            "true": str(row["diagnostic"]),
            "generated": gen_text,
            "raw_ids": generated_ids.tolist(),
        }
        records["samples"].append(sample)
        log(f"Image {i+1} (True: {row['diagnostic']}): {gen_text!r} (Raw IDs: {generated_ids.tolist()})")

    records["finished_at"] = datetime.utcnow().isoformat() + "Z"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    vol_results.commit()
    log(f"Saved diagnostic outputs to {result_txt} and {result_json}")
    return records

@app.local_entrypoint()
def main():
    run_diagnostic.remote()
