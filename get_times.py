import re
with open("experiment_run.log") as f:
    text = f.read()

# Look for time taken per epoch or total run
# E.g., "Epoch 5/5 [Val]: 100%|... [00:15<00:00]"
matches = re.findall(r'Epoch 5/5 \[Val\]: 100%\|.*?\[(\d+:\d+)', text)
print("Val times:", matches)

matches = re.findall(r'Epoch 5/5 \[Train\]: 100%\|.*?\[(\d+:\d+)', text)
print("Train times:", matches)

