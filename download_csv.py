import modal
import os

app = modal.App("download-results")
vol_results = modal.Volume.from_name("miccai-results")

@app.local_entrypoint()
def main():
    # Read the file contents from the volume
    contents = b""
    for chunk in vol_results.read_file("/task13_lvlm_audit.csv"):
        contents += chunk
    
    with open("results/task13_lvlm_audit.csv", "wb") as f:
        f.write(contents)
    print("Downloaded task13_lvlm_audit.csv from volume!")
