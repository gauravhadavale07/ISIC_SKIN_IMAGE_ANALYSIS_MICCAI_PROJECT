import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
import os
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

MODELS = {
    "Image-Only": "ImageOnly",
    "Text-Only": "TextOnly",
    "Late Fusion": "Late_Fusion",
    "GMU Baseline": "GMU_Baseline",
    "Cross-Attention (V->T)": "Cross-Attention",
    "Cross-Attention T->V": "Cross-Attention_T2V",
}

SEEDS = cfg.seeds

print("=== TABLE 1 (MILK10k) ===")
for arch_name, prefix in MODELS.items():
    accs, f1s, aurocs = [], [], []
    for seed in SEEDS:
        path = f"results/milk10k_raw_preds_{prefix}_seed_{seed}.npz"
        if not os.path.exists(path):
            continue
        data = np.load(path)
        y_true, y_pred, y_prob = data['y_true'], data['y_pred'], data['y_prob']
        accs.append(accuracy_score(y_true, y_pred) * 100)
        f1s.append(f1_score(y_true, y_pred, average='macro') * 100)
        try:
            aurocs.append(roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'))
        except:
            aurocs.append(0.0)
    if accs:
        print(f"{arch_name}: Acc={np.mean(accs):.2f}\\pm{np.std(accs):.2f} | F1={np.mean(f1s):.2f}\\pm{np.std(f1s):.2f} | AUC={np.mean(aurocs):.3f}\\pm{np.std(aurocs):.3f}")

print("\n=== TABLE 2 (PAD-UFES-20) ===")
# read experiment_progress.json
try:
    with open("results/experiment_progress.json", "r") as f:
        prog = json.load(f)["results"]
except Exception as e:
    prog = {}
    print("Could not load experiment_progress.json:", e)

for arch_name in MODELS.keys():
    # architecture names in experiment_progress.json:
    json_key = arch_name
    if json_key == "Cross-Attention (V->T)": json_key = "Cross-Attention V\u2192T"
    if json_key == "Cross-Attention T->V": json_key = "Cross-Attention T\u2192V"
    if json_key in prog and len(prog[json_key].get("Real_Accuracy", [])) >= len(SEEDS):
        m = prog[json_key]
        n_seeds = len(SEEDS)
        # metrics
        accs = m["Real_Accuracy"][:n_seeds]
        aurocs = m["AUROC"][:n_seeds]
        f1s = [f * 100 for f in m["F1 (Macro)"][:n_seeds]]
        cfrs = m["CFR"][:n_seeds]
        deltas = m["Mean_Delta_P"][:n_seeds]
        ckas = m["Linear_CKA"][:n_seeds]
        print(f"{arch_name}:")
        print(f"  Acc:  {np.mean(accs):.2f}\\pm{np.std(accs):.2f}")
        print(f"  AUC:  {np.mean(aurocs):.3f}\\pm{np.std(aurocs):.3f}")
        print(f"  F1:   {np.mean(f1s):.2f}\\pm{np.std(f1s):.2f}")
        print(f"  CFR:  {np.mean(cfrs):.2f}\\pm{np.std(cfrs):.2f}")
        print(f"  DelP: {np.mean(deltas):.2f}\\pm{np.std(deltas):.2f}")
        print(f"  CKA:  {np.mean(ckas):.3f}\\pm{np.std(ckas):.3f}")
