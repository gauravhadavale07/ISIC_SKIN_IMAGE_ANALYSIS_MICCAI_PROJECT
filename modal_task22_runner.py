import modal
import subprocess

app = modal.App("miccai-task22-lvlm-grounding")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "pandas",
    "numpy",
    "tqdm",
    "transformers",
    "pillow",
    "accelerate"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT",
    remote_path="/root/project",
    ignore=["data", "results", "logs", "checkpoints", ".git", "__pycache__"]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data/raw_pad_ufes",
    remote_path="/root/project/data/raw_pad_ufes",
)

@app.function(
    gpu="H100",
    image=image,
    timeout=7200,
    volumes={"/root/project/results": vol_results},
)
def run_task22():
    subprocess.run(["python3", "-u", "task22_lvlm_matched_grounding.py"], check=True)
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_task22.remote()
