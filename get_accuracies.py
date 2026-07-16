import re

with open('experiment_run.log', 'r') as f:
    text = f.read()

# Look for Acc= in the validation phase
print("Finding all Acc= in the log...")
matches = re.findall(r'Acc=(\d+\.\d+)%', text)
if matches:
    print(f"Found {len(matches)} accuracy entries. Last 10: {matches[-10:]}")
else:
    print("No Acc= matches found")
