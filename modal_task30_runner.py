import subprocess

import modal


app = modal.App("miccai-task30-textonly-random-lesion-cv")
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "torchaudio",
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "transformers",
    "huggingface_hub==0.23.2",
    "accelerate",
    "tqdm",
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT",
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
).add_local_file(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/milk10k_train.csv",
    remote_path="/root/project/milk10k_train.csv",
)


@app.function(
    gpu="H100",
    image=image,
    timeout=14400,
    volumes={"/root/project/results": vol_results},
)
def run_task30():
    print("\n==================================================")
    print("Running task30_textonly_random_lesion_cv.py on Modal H100...")
    print("==================================================\n")
    subprocess.run(
        [
            "python3",
            "-u",
            "task30_textonly_random_lesion_cv.py",
            "--epochs",
            "5",
            "--seeds",
            "101",
            "202",
            "303",
            "404",
            "505",
        ],
        check=True,
    )
    vol_results.commit()


@app.local_entrypoint()
def main():
    run_task30.remote()
