import os
import glob
import re

files_to_patch = [
    "compute_metrics_from_npz.py",
    "compute_milk10k_f1_auroc.py",
    "compute_milk10k_fast_cached.py",
    "compute_table12_std.py",
    "corrected_significance.py",
    "milk10k_significance.py",
    "run_lexical_control_audit.py",
    "run_milk10k_audit.py",
    "run_milk10k_clean_audit.py"
]

for f in files_to_patch:
    path = os.path.join("/home/ec2-user/ISIC_SKIN_IMAGE_ANALYSIS_MICCAI_PROJECT", f)
    if not os.path.exists(path): continue
    with open(path, "r") as fp:
        content = fp.read()
    
    # Check if 'from config import cfg' is already there
    has_import = "from config import cfg" in content
    
    # We replace `SEEDS = [456, 789, 1337]` with `SEEDS = cfg.seeds`
    new_content = re.sub(r'([ \t]*)SEEDS\s*=\s*\[456,\s*789,\s*1337\]', r'\1SEEDS = cfg.seeds', content)
    
    if new_content != content:
        if not has_import:
            # Add import at the top after standard imports, or just at the top
            new_content = "import sys\nimport os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nfrom config import cfg\n" + new_content
        with open(path, "w") as fp:
            fp.write(new_content)
        print(f"Patched {f}")

