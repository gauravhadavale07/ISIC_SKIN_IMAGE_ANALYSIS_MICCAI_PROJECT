import modal
import subprocess

app = modal.App("miccai-task14-knockout")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio", "tqdm", "pandas", "numpy",
    "scikit-learn", "transformers", "timm", "huggingface_hub==0.23.2", "accelerate", "pillow", "scipy"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=[
        "data", "results", "logs", "checkpoints", ".git", "__pycache__",
        "modal_*_out.txt", "modal_task*_stop_duplicate.txt",
    ]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data/raw_milk10k", 
    remote_path="/root/project/data/raw_milk10k"
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/checkpoints/Cross-Attention_T→V_seed_1337",
    remote_path="/root/project/checkpoints/Cross-Attention_T→V_seed_1337",
)

@app.function(
    gpu="A10G", # A10G is fast enough for CrossAttentionT2VClassifier and we don't need H200 here
    image=image,
    timeout=3600,
    volumes={
        "/root/project/results": vol_results,
    }
)
def run_task14():
    print("\n==================================================")
    print("Running task14_feature_knockout.py...")
    print("==================================================\n")
    subprocess.run(["python3", "task14_feature_knockout.py"], check=True)
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_task14.remote()
