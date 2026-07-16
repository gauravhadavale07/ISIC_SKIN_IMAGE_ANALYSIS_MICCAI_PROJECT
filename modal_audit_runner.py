import modal
import subprocess
import os

app = modal.App("audit-runner")

vol_checkpoints = modal.Volume.from_name("miccai-checkpoints", create_if_missing=True)
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers", "pandas", "scikit-learn", "timm", "scipy", "pillow", "tqdm", "networkx", "matplotlib", "seaborn", "statsmodels"
).workdir("/root/project").add_local_dir(
    "/tmp/miccai_code_fresh", 
    remote_path="/root/project"
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data", 
    remote_path="/root/project/data"
).add_local_file(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/milk10k_train.csv",
    remote_path="/root/project/milk10k_train.csv"
).add_local_file(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/pad_ufes_20_test.csv",
    remote_path="/root/project/pad_ufes_20_test.csv"
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/checkpoints",
    remote_path="/root/project/checkpoints"
)

@app.function(
    gpu="H200",
    image=image,
    timeout=86400,
    volumes={
        "/root/project/results": vol_results,
    }
)
def run_all_audits():
    scripts = [
        "task1_metadata_shuffle_significance.py",
        "task2_gmu_gate_analysis.py"
    ]
    for script in scripts:
        print(f"\\n{'='*50}\\nRunning {script}...\\n{'='*50}\\n")
        # Ensure outputs flush to logs immediately
        subprocess.run(["python3", script], check=True, cwd="/root/project")
        
    print("\\n🎉 All audit scripts completed successfully on H200! 🎉")
    
    # We must explicitly commit the volume since we write to ./results inside the scripts using standard python IO
    print("Committing results to volume...")
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_all_audits.remote()
