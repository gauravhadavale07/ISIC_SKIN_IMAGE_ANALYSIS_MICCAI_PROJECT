import modal
import subprocess
import os
import glob

app = modal.App("run-all-tasks")

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers", "pandas", "scikit-learn", "timm", "scipy", "pillow", "tqdm", "networkx", "matplotlib", "seaborn", "statsmodels"
).workdir("/root/project").add_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", 
    remote_path="/root/project"
)

@app.function(
    gpu="H200",
    image=image,
    timeout=86400,
)
def run_all():
    print("Running task1...")
    with open("task1_output.txt", "w") as f:
        subprocess.run(["python3", "task1_metadata_shuffle_significance.py"], cwd="/root/project", stdout=f, stderr=subprocess.STDOUT)
    
    print("Running task6...")
    subprocess.run(["python3", "task6_ddi_stratified_audit_rigorous.py"], cwd="/root/project")
    
    print("Running figures...")
    subprocess.run(["python3", "figures/run_all_figures.py"], cwd="/root/project")
    
    with open("task1_output.txt", "r") as f:
        t1_out = f.read()
    
    with open("results/task6_ddi_stratified_audit_rigorous.csv", "r") as f:
        t6_out = f.read()
        
    figures_dict = {}
    for filepath in glob.glob("figures/output/*.*"):
        filename = os.path.basename(filepath)
        with open(filepath, "rb") as f:
            figures_dict[filename] = f.read()
            
    return t1_out, t6_out, figures_dict

@app.local_entrypoint()
def main():
    t1_out, t6_out, figures_dict = run_all.remote()
    with open("task1_output.txt", "w") as f:
        f.write(t1_out)
    
    with open("results/task6_ddi_stratified_audit_rigorous.csv", "w") as f:
        f.write(t6_out)
        
    os.makedirs("figures/output", exist_ok=True)
    for filename, content in figures_dict.items():
        with open(os.path.join("figures/output", filename), "wb") as f:
            f.write(content)
            
    # Also copy them directly to paper/figures for the latex compile!
    os.makedirs("paper/figures", exist_ok=True)
    for filename, content in figures_dict.items():
        with open(os.path.join("paper/figures", filename), "wb") as f:
            f.write(content)
        
    print(f"Saved task1_output.txt, task6_ddi_stratified_audit_rigorous.csv, and {len(figures_dict)} figures locally!")
