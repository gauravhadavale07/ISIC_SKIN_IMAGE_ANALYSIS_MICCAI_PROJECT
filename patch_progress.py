import json
import modal

app = modal.App("progress-patcher")
volume = modal.Volume.from_name("miccai-results")

@app.function(volumes={"/results": volume})
def patch():
    filepath = "/results/experiment_progress.json"
    
    # Load current progress
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # 1. Remove the target run from completed_runs
    target_run = "1337:Cross-Attention V\u2192T"
    if target_run in data["completed_runs"]:
        data["completed_runs"].remove(target_run)
        print(f"Removed '{target_run}' from completed_runs.")
    
    # 2. Trim the metrics arrays back to the 4 valid baseline/previous runs
    model_key = "Cross-Attention V\u2192T"
    if model_key in data["results"]:
        metrics = data["results"][model_key]
        for metric_name, values_list in metrics.items():
            if isinstance(values_list, list) and len(values_list) == 5:
                # Pop the 5th (corrupted/incomplete) entry off the list
                popped_val = values_list.pop()
        print(f"Successfully trimmed '{model_key}' metrics back to 4 entries.")
    
    # Save the cleaned progress back to the Modal volume
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    # Commit the volume changes changes instantly
    modal.Volume.from_name("miccai-results").commit()
    print("Volume changes committed successfully.")

if __name__ == "__main__":
    with app.run():
        patch.remote()
