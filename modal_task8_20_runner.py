import os
import shutil
import subprocess
from pathlib import Path

import modal


app = modal.App("miccai-task8-20-corrected")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

PROJECT = "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT"

image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "torchaudio",
    "tqdm",
    "pandas",
    "numpy",
    "scikit-learn",
    "transformers",
    "timm",
    "huggingface_hub==0.23.2",
    "accelerate",
    "pillow",
    "scipy",
).workdir("/root/project").add_local_dir(
    PROJECT,
    remote_path="/root/project",
    ignore=[
        "data",
        "results",
        "logs",
        "checkpoints",
        ".git",
        "__pycache__",
        "modal_*_out.txt",
        "modal_task*_stop_duplicate.txt",
    ],
).add_local_dir(
    f"{PROJECT}/data/raw_pad_ufes",
    remote_path="/root/project/data/raw_pad_ufes",
).add_local_dir(
    f"{PROJECT}/checkpoints/Cross-Attention_T→V_seed_1337",
    remote_path="/root/project/checkpoints/Cross-Attention_T→V_seed_1337",
).add_local_dir(
    f"{PROJECT}/checkpoints/Cross-Attention_V→T_seed_1337",
    remote_path="/root/project/checkpoints/Cross-Attention_V→T_seed_1337",
).add_local_file(
    f"{PROJECT}/results/sae_weights.pth",
    remote_path="/root/project/bootstrap_sae_weights.pth",
)


@app.function(
    gpu="H200",
    image=image,
    timeout=7200,
    volumes={"/root/project/results": vol_results},
)
def run_corrected_task8_20():
    Path("results").mkdir(exist_ok=True)
    if not Path("results/sae_weights.pth").exists() and Path("bootstrap_sae_weights.pth").exists():
        shutil.copyfile("bootstrap_sae_weights.pth", "results/sae_weights.pth")

    subprocess.run(["python3", "-u", "task8_activation_patching.py"], check=True)
    subprocess.run(["python3", "-u", "task20_visual_biopsy_leak.py"], check=True)
    vol_results.commit()

    output_files = [
        "results/task8_activation_patching.csv",
        "results/task20_visual_biopsy_leak.csv",
        "results/task20_visual_biopsy_leak_summary.csv",
    ]
    return {path: Path(path).read_text() for path in output_files}


@app.local_entrypoint()
def main():
    outputs = run_corrected_task8_20.remote()
    for path, text in outputs.items():
        local_path = Path(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(text)
        print(f"Wrote {local_path}")
