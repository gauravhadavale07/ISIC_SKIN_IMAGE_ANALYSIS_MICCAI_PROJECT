import modal
import subprocess

app = modal.App("miccai-task16-test-hook")
vol_hf = modal.Volume.from_name("miccai-hf-cache", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio", "tqdm", "pandas", "numpy",
    "scikit-learn", "transformers", "accelerate", "pillow", "scipy", "hf_transfer"
).env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=[
        "results", "logs", "checkpoints", ".git", "__pycache__",
        "modal_*_out.txt", "modal_task*_stop_duplicate.txt",
    ]
)

@app.function(
    gpu="H100", 
    image=image,
    timeout=600,
    volumes={
        "/root/.cache/huggingface": vol_hf,
    }
)
def run_test():
    subprocess.run(["python3", "test_hook.py"], check=True)

@app.local_entrypoint()
def main():
    run_test.remote()
