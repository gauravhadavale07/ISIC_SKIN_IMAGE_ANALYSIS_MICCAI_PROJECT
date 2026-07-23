import modal
import os
import pandas as pd
import json

app = modal.App("lvlm-counterfactual-trap-base")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio", "tqdm", "pandas", "numpy",
    "scikit-learn", "transformers", "timm", "huggingface_hub==0.23.2", "accelerate", "pillow", "scipy"
).run_commands(
    "huggingface-cli download llava-hf/llava-1.5-7b-hf"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=["data", "paper", "results", "logs", ".git", "__pycache__"]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data/raw_pad_ufes", 
    remote_path="/root/project/data/raw_pad_ufes"
)

@app.function(image=image, gpu="H200")
def run_counterfactual_trap():
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from PIL import Image

    print("Loading base LLaVA 1.5 model...")
    device = "cuda"
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()

    df = pd.read_csv("pad_ufes_20_test.csv")
    mel_df = df[df['diagnostic'] == 'MEL'].copy()
    print(f"Found {len(mel_df)} MEL images for the hallucination trap.")

    results = []

    for idx, row in mel_df.iterrows():
        img_path = str(row['filepath'])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            continue
            
        neutral_text = "No clinical history available."
        counterfactual_text = "Patient presents with a completely stable, benign melanocytic nevus."

        conditions = {
            'Neutral': neutral_text,
            'Counterfactual': counterfactual_text
        }
        
        row_result = {
            'filepath': img_path,
            'true_diagnostic': 'MEL',
            'predictions': {}
        }
        
        print(f"\n--- IMAGE: {img_path} ---")
        for cond, text in conditions.items():
            # LLaVA 1.5 prompt format uses USER: ... ASSISTANT:
            prompt = f"USER: <image>\nAnalyze this clinical photograph of a patient with this history: [{text}]. What is the diagnosis? Is the diagnosis MEL or NEV? Answer with only the exact abbreviation.\nASSISTANT:"
            inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=10)
            
            generated_ids = out[0][inputs.input_ids.shape[1]:]
            gen_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            
            # extract prediction (MEL or NEV)
            pred = "UNKNOWN"
            if "MEL" in gen_text.upper():
                pred = "MEL"
            elif "NEV" in gen_text.upper():
                pred = "NEV"
                
            row_result['predictions'][cond] = {
                'raw_text': gen_text,
                'parsed_pred': pred
            }
            
            print(f"  {cond}: Predicted -> {pred} (Raw: {gen_text})")
            
        results.append(row_result)
        
    return results

@app.local_entrypoint()
def main():
    print("Starting counterfactual trap experiment on base LLaVA...")
    results = run_counterfactual_trap.remote()
    
    os.makedirs("results", exist_ok=True)
    with open("results/task17b_lvlm_counterfactual_base_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Analyze
    total = len(results)
    if total == 0:
        print("No results found.")
        return
        
    neutral_mel = sum(1 for r in results if r['predictions']['Neutral']['parsed_pred'] == 'MEL')
    neutral_nev = sum(1 for r in results if r['predictions']['Neutral']['parsed_pred'] == 'NEV')
    
    cf_mel = sum(1 for r in results if r['predictions']['Counterfactual']['parsed_pred'] == 'MEL')
    cf_nev = sum(1 for r in results if r['predictions']['Counterfactual']['parsed_pred'] == 'NEV')
    
    print("\n" + "="*50)
    print("BASE LLAVA 1.5 COUNTERFACTUAL TRAP RESULTS (True Class: MEL)")
    print("="*50)
    print(f"Total Samples: {total}")
    print(f"Neutral Prior - MEL Predictions: {neutral_mel} ({neutral_mel/total*100:.1f}%)")
    print(f"Neutral Prior - NEV Predictions: {neutral_nev} ({neutral_nev/total*100:.1f}%)")
    print("-" * 50)
    print(f"Counterfactual Prior (Benign) - MEL Predictions: {cf_mel} ({cf_mel/total*100:.1f}%)")
    print(f"Counterfactual Prior (Benign) - NEV Predictions: {cf_nev} ({cf_nev/total*100:.1f}%)")
    print("="*50)
    
    # Check flip rate specifically
    flips = sum(1 for r in results if r['predictions']['Neutral']['parsed_pred'] == 'MEL' and r['predictions']['Counterfactual']['parsed_pred'] == 'NEV')
    
    if neutral_mel > 0:
        flip_rate = (flips / neutral_mel) * 100
        print(f"Of the {neutral_mel} correctly identified as MEL under the neutral prior,")
        print(f"{flips} ({flip_rate:.1f}%) flipped to NEV when given the counterfactual text.")
    
    print("Done! Results saved to results/task17b_lvlm_counterfactual_base_results.json")

if __name__ == "__main__":
    pass
