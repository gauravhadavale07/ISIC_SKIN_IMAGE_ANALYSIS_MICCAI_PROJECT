import modal
import os

app = modal.App("miccai-multimodal-pipeline")

vol_checkpoints = modal.Volume.from_name("miccai-checkpoints", create_if_missing=True)
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

data_mount = modal.Mount.from_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT/data", 
    remote_path="/root/project/data"
)

# We mount the code manually to exclude data/ and checkpoints/ which are massive
code_mount = modal.Mount.from_local_dir(
    "/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT",
    remote_path="/root/project",
    condition=lambda path: all(exclude not in path for exclude in ["/data", "/checkpoints"])
)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers", "pandas", "scikit-learn", "timm", "scipy", "pillow", "tqdm", "networkx"
).workdir("/root/project")

@app.function(
    gpu="H200",
    image=image,
    mounts=[data_mount, code_mount],
    volumes={
        "/root/project/checkpoints": vol_checkpoints,
        "/root/project/results": vol_results
    },
    timeout=86400
)
def execute_experiment():
    import run_experiment
    run_experiment.main_logic()

@app.function(
    gpu="H200",
    image=image,
    mounts=[data_mount, code_mount],
    volumes={
        "/root/project/checkpoints": vol_checkpoints,
        "/root/project/results": vol_results
    },
    timeout=86400
)
def execute_analysis():
    # Because we need checkpoints, let's copy the local checkpoints into the volume if they don't exist
    import os
    import shutil
    import full_analysis
    full_analysis.main_logic()

@app.local_entrypoint()
def run_exp():
    print("Executing experiment on Modal...")
    execute_experiment.remote()

@app.local_entrypoint()
def run_ana():
    print("Executing full analysis on Modal...")
    execute_analysis.remote()
