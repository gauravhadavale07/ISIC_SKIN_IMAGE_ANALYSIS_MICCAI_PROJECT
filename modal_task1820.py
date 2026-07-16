import modal
import subprocess
import os

app = modal.App("miccai-task1820-final")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "torchaudio", "pandas", "numpy",
    "transformers", "timm", "pillow", "scipy"
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=[".git", "__pycache__"]
)

@app.function(image=image, gpu="a100", timeout=1800)
def run_evals():
    os.chdir("/root/project")
    print("--- Running Task 18 on A100 ---")
    subprocess.run(["python", "task18_scaffold_ablation.py"], check=True)
    print("\n--- Running Task 19 on A100 ---")
    subprocess.run(["python", "task19_lvlm_position_bias.py"], check=True)
    print("\n--- Running Task 20 on A100 ---")
    subprocess.run(["python", "task20_visual_biopsy_leak.py"], check=True)

@app.local_entrypoint()
def main():
    run_evals.remote()
