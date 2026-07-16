import modal
import subprocess
import os

app = modal.App("mi4medfm-runner")

vol_checkpoints = modal.Volume.from_name("miccai-checkpoints", create_if_missing=True)
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers", "pandas", "scikit-learn", "timm", "scipy", "pillow", "tqdm"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project",
    ignore=["data", "results", "logs", ".git", "__pycache__"]
).add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data", 
    remote_path="/root/project/data"
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
        "task2_gmu_gate_analysis.py",
        "task8_activation_patching.py",
        "task9_attention_steering.py",
        "task10_linear_probing.py"
    ]
    for script in scripts:
        print(f"\\n{'='*50}\\nRunning {script}...\\n{'='*50}\\n")
        try:
            subprocess.run(["python3", script], check=True, cwd="/root/project")
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
        
    print("\\n🎉 All MI4MedFM scripts completed! 🎉")
    
    print("Committing results to volume...")
    vol_results.commit()

@app.local_entrypoint()
def main():
    run_all_audits.remote()
