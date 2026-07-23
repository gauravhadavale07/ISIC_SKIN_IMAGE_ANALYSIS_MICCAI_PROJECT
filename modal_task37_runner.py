import modal
import subprocess

app = modal.App("miccai-task37-demographic-sae")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio", "tqdm", "pandas", "numpy",
    "scikit-learn", "transformers", "timm", "huggingface_hub==0.23.2", "accelerate", "pillow", "scipy", "statsmodels"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=[
        "data", "results", "logs", "checkpoints", ".git", "__pycache__",
        "modal_*_out.txt", "modal_task*_stop_duplicate.txt",
    ]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data/raw_pad_ufes", 
    remote_path="/root/project/data/raw_pad_ufes"
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/checkpoints/Cross-Attention_T→V_seed_1337",
    remote_path="/root/project/checkpoints/Cross-Attention_T→V_seed_1337",
)

@app.function(
    gpu="H100",
    image=image,
    timeout=3600,
    volumes={
        "/root/project/results": vol_results,
    }
)
def run_task37():
    print("\n==================================================")
    print("Running task37_demographic_sae.py...")
    print("==================================================\n")
    subprocess.run(["python3", "-u", "task37_demographic_sae.py"], check=True)
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_task37.remote()
