import modal
import subprocess

app = modal.App("miccai-task13-lvlm-audit")

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

vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

@app.function(
    gpu="H200",
    image=image,
    timeout=3600,
    volumes={
        "/root/project/results": vol_results,
    }
)
def run_task13():
    print("\n==================================================")
    print("Running task13_lvlm_audit.py...")
    print("==================================================\n")
    subprocess.run(["python3", "task13_lvlm_audit.py"], check=True)
    
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_task13.remote()
