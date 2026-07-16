import pandas as pd
import torch
from torch.utils.data import random_split
import re

csv_path = "./milk10k_train.csv"
df = pd.read_csv(csv_path)

# Extract lesion ID from the filepath (e.g., IL_0000652)
# Pattern matches IL_ followed by digits
df['lesion_id'] = df['filepath'].apply(lambda x: re.search(r'(IL_\d+)', x).group(1) if re.search(r'(IL_\d+)', x) else None)

total_images = len(df)
total_lesions = df['lesion_id'].nunique()
print(f"Total images: {total_images}")
print(f"Total unique lesions: {total_lesions}")
print(f"Average images per lesion: {total_images / total_lesions:.2f}")

n = len(df)
train_size = int(0.85 * n)
val_size = n - train_size

seeds = [456, 789, 1337]

for seed in seeds:
    print(f"\n--- Checking Seed {seed} ---")
    gen = torch.Generator().manual_seed(seed)
    
    # We just need the indices to simulate the split
    indices = list(range(n))
    
    # Simulate random_split
    import torch.utils.data as data
    class DummyDataset(data.Dataset):
        def __len__(self): return n
        def __getitem__(self, idx): return idx
    
    dummy_ds = DummyDataset()
    train_sub, val_sub = random_split(dummy_ds, [train_size, val_size], generator=gen)
    
    train_indices = train_sub.indices
    val_indices = val_sub.indices
    
    train_lesions = set(df.iloc[train_indices]['lesion_id'].dropna())
    val_lesions = set(df.iloc[val_indices]['lesion_id'].dropna())
    
    leaking_lesions = train_lesions.intersection(val_lesions)
    num_leaking = len(leaking_lesions)
    
    # Count how many validation *samples* belong to leaking lesions
    val_df = df.iloc[val_indices]
    leaking_val_samples = val_df[val_df['lesion_id'].isin(leaking_lesions)]
    num_leaking_samples = len(leaking_val_samples)
    
    print(f"Unique lesions in train: {len(train_lesions)}")
    print(f"Unique lesions in val: {len(val_lesions)}")
    print(f"Leaking lesions (in both): {num_leaking}")
    print(f"Validation samples associated with leaking lesions: {num_leaking_samples} out of {val_size}")
    print(f"Leakage rate (val samples): {num_leaking_samples / val_size * 100:.2f}%")

