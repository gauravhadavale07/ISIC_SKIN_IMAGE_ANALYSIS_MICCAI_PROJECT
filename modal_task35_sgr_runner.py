import modal
import subprocess

app = modal.App("miccai-task35-sgr")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)
vol_hf = modal.Volume.from_name("miccai-hf-cache", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio", "tqdm", "pandas", "numpy",
    "scikit-learn", "transformers", "accelerate", "pillow", "scipy", "hf_transfer",
    "matplotlib", "timm"
).env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=[
        "results", "logs", ".git", "__pycache__",
        "modal_*_out.txt", "modal_task*_stop_duplicate.txt",
    ]
)

@app.function(
    gpu="H100",
    image=image,
    timeout=3600,
    volumes={
        "/root/project/results": vol_results,
        "/root/.cache/huggingface": vol_hf,
    }
)
def run_task35():
    print("\n==================================================")
    print("Running task35_sgr.py...")
    print("==================================================\n")
    subprocess.run(["python3", "-u", "task35_sgr.py"], check=True)
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_task35.remote()
