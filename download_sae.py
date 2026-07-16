import modal
import os

app = modal.App("download-sae")
vol_results = modal.Volume.from_name("miccai-results")

@app.local_entrypoint()
def main():
    # Read the file contents from the volume
    contents = b""
    for chunk in vol_results.read_file("/sae_weights.pth"):
        contents += chunk
    
    with open("results/sae_weights.pth", "wb") as f:
        f.write(contents)
    print("Downloaded sae_weights.pth from volume!")
