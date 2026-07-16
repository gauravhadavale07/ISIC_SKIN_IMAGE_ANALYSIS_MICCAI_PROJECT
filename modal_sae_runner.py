import modal
import subprocess
import os

app = modal.App("mi4medfm-sae-runner")

vol_checkpoints = modal.Volume.from_name("miccai-checkpoints", create_if_missing=True)
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers", "pandas", "scikit-learn", "timm", "scipy", "pillow", "tqdm"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=["data", "results", "logs", ".git", "__pycache__"]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data", 
    remote_path="/root/project/data"
)

@app.function(
    gpu="H200",
    image=image,
    timeout=86400,
    volumes={
        "/root/project/results": vol_results,
    }
)
def run_sae():
    print(f"\\n{'='*50}\\nRunning task11_sae.py...\\n{'='*50}\\n")
    try:
        subprocess.run(["python3", "task11_sae.py"], check=True, cwd="/root/project")
    except subprocess.CalledProcessError as e:
        print(f"Error running SAE: {e}")
        
    print("Committing SAE results to volume...")
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_sae.remote()
