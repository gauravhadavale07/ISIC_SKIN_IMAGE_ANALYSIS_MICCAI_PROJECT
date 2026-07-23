import modal
import subprocess

app = modal.App("miccai-task31-feature1449-validation")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio", "tqdm", "pandas", "numpy",
    "scikit-learn", "transformers", "timm", "huggingface_hub==0.23.2", "accelerate", "pillow", "scipy",
    "matplotlib", "seaborn", "scikit-posthocs"
).run_commands(
    "huggingface-cli download llava-hf/llava-1.5-7b-hf"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=["data", "results", "logs", ".git", "__pycache__"]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data", 
    remote_path="/root/project/data"
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/checkpoints", 
    remote_path="/root/project/checkpoints"
)

vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

@app.function(
    gpu="H100",
    image=image,
    timeout=3600,
    volumes={
        "/root/project/results": vol_results,
    }
)
def run_task31():
    print("\n==================================================")
    print("Running task31_sae_feature_1449.py...")
    print("==================================================\n")
    subprocess.run(["python3", "task31_sae_feature_1449.py"], check=True)
    
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_task31.remote()
# Force modal rebuild
