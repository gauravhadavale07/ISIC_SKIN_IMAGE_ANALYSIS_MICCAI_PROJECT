import torch
import os

architectures = {
    "Late Fusion": "Late_Fusion",
    "GMU": "GMU_Baseline",
    "V->T": "Cross-Attention",
    "T->V": "Cross-Attention_T2V"
}

seeds = [456, 789, 1337]

print("Architecture | Seed | Best Epoch")
print("-" * 35)

for arch_name, prefix in architectures.items():
    for seed in seeds:
        ckpt_dir = f"./checkpoints/{prefix}_seed_{seed}"
        ckpt_file = os.path.join(ckpt_dir, "best_model.pth")
        
        if os.path.exists(ckpt_file):
            try:
                # Load only on CPU to avoid taking up GPU memory or crashing
                ckpt = torch.load(ckpt_file, map_location='cpu')
                # Check if epoch is saved in dict
                if isinstance(ckpt, dict) and 'epoch' in ckpt:
                    epoch = ckpt['epoch']
                else:
                    epoch = "Unknown (not in dict)"
                print(f"{arch_name:<12} | {seed:<4} | {epoch}")
            except Exception as e:
                print(f"{arch_name:<12} | {seed:<4} | Error loading")
        else:
            print(f"{arch_name:<12} | {seed:<4} | Missing file")
