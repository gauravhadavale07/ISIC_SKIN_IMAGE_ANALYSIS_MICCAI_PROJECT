import re
import numpy as np

with open('experiment_run.log', 'r') as f:
    text = f.read()

# Extract training times
# Usually looks like: [00:12<00:24, 11.62it/s, Loss=1.2714] or similar per epoch
# Let's search for "Total runtime" or "Total training time" if logged
times = re.findall(r'Total training time: ([\d\.]+)s', text)
if not times:
    # If not explicitly logged, let's look for "Epoch 5/5" ending time
    pass

# We can also find the Image-Only accuracy and T->V accuracy
# e.g., "Image-Only: 61.42%"
img_only = re.findall(r'Image-Only:\s+([\d\.]+)%', text)
t2v = re.findall(r'Cross-Attention \(T->V\):\s+([\d\.]+)%', text)
print("Image-Only Accs:", img_only)
print("T->V Accs:", t2v)

