import modal
import os

app = modal.App("download-results")
vol_results = modal.Volume.from_name("miccai-results")

@app.local_entrypoint()
def main():
    files_to_download = [
        "task14_knockout.csv",
        "task14_knockout_summary.csv",
        "task15_ddi_demographic_patching.csv",
        "task15_ddi_demographic_patching_summary.csv",
        "task22_lvlm_matched_grounding_samples.csv",
        "task22_lvlm_matched_grounding_summary.csv",
        "task22_lvlm_matched_grounding_summary.json"
    ]
    
    os.makedirs("results", exist_ok=True)
    
    for filename in files_to_download:
        try:
            contents = b""
            for chunk in vol_results.read_file(f"/{filename}"):
                contents += chunk
            
            with open(f"results/{filename}", "wb") as f:
                f.write(contents)
            print(f"Downloaded {filename} from volume!")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
