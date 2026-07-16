import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
import os
import numpy as np
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

for arch_name, prefix in MODELS.items():
    accs, f1s, aurocs = [], [], []
    for seed in SEEDS:
        path = f"results/milk10k_raw_preds_{prefix}_seed_{seed}.npz"
        if not os.path.exists(path):
            continue
            
        data = np.load(path)
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_prob = data['y_prob']
        
        acc = accuracy_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred, average='macro') * 100
        
        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except:
            auroc = 0.0
            
        accs.append(acc)
        f1s.append(f1)
        aurocs.append(auroc)
        
    if accs:
        print(f"{arch_name:<25}: Acc={np.mean(accs):.2f}% | F1={np.mean(f1s):.2f}% | AUROC={np.mean(aurocs):.4f}")
