import json
import shutil
import os

print("Purging poisoned model from experiment_progress.json...")
try:
    with open('results/experiment_progress.json', 'r') as f:
        data = json.load(f)
    
    # Remove from completed_runs
    target = "1337:Cross-Attention V→T"
    if target in data["completed_runs"]:
        data["completed_runs"].remove(target)
        
    # We also need to remove the appended metric arrays for index 2 (Seed 1337 is the 3rd seed, wait, Seed 1337 is actually index 2, but wait, Seed 1337 was evaluated 3rd)
    # Actually, it's safer to just delete the entire "Cross-Attention V→T" key, but it has Seed 456 and Seed 789!
    # Wait, the arrays are just appended to. If we delete the last element of every array in "Cross-Attention V→T", we remove Seed 1337!
    arch_data = data["results"]["Cross-Attention V→T"]
    for metric in arch_data:
        if isinstance(arch_data[metric], list) and len(arch_data[metric]) == 3:
            arch_data[metric].pop(-1)
            
    with open('results/experiment_progress.json', 'w') as f:
        json.dump(data, f, indent=2)
    print("Purged from JSON.")
except Exception as e:
    print(f"Error: {e}")

print("Removing corrupted checkpoint directory...")
shutil.rmtree('./checkpoints/Cross-Attention_V→T_seed_1337', ignore_errors=True)
print("Done.")
