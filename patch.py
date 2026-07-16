import re

with open("run_milk10k_clean_audit.py", "r") as f:
    content = f.read()

# Fix the corrupted SEEDS line
content = re.sub(r'SEEDS = \[456, 789, 1# ── Build lesion-disjoint clean held-out set ───────────────────────────────────',
                 r'SEEDS = [456, 789, 1337]\n\n# ── Build lesion-disjoint clean held-out set ───────────────────────────────────',
                 content)

# Fix the end corruption
content = content.replace("s(),\n)\nclean_val_loader = DataLoader(\n    clean_val_dataset,", "")

with open("run_milk10k_clean_audit.py", "w") as f:
    f.write(content)
