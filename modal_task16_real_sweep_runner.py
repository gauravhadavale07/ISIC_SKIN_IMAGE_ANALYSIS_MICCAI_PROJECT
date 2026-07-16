import subprocess
from pathlib import Path

import modal


app = modal.App("miccai-task16-real-sweep")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)
vol_hf = modal.Volume.from_name("miccai-hf-cache", create_if_missing=True)

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
    "accelerate",
    "pillow",
    "scipy",
    "hf_transfer",
).env({
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
}).workdir("/root/project").add_local_dir(
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
)


@app.function(
    gpu="H200",
    image=image,
    timeout=7200,
    volumes={
        "/root/project/results": vol_results,
        "/root/.cache/huggingface": vol_hf,
    },
)
def run_task16_real_sweep():
    subprocess.run(["python3", "-u", "task16_lvlm_steering.py"], check=True)
    vol_results.commit()
    output_files = [
        "results/task16_results.csv",
        "results/task16_single_sample_diagnostic.csv",
    ]
    out = {}
    for path in output_files:
        p = Path(path)
        if p.exists():
            out[path] = p.read_text()
    if Path("task16_results.csv").exists():
        out["task16_results.csv"] = Path("task16_results.csv").read_text()
    return out


@app.local_entrypoint()
def main():
    outputs = run_task16_real_sweep.remote()
    for path, text in outputs.items():
        local_path = Path(path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(text)
        print(f"Wrote {local_path}")
