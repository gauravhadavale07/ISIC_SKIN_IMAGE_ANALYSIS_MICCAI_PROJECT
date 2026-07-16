import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import cfg
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy import stats

from corrected_significance import hedges_g_paired, raw_ttest, holm_bonferroni, d_label

def main():
    if not os.path.exists("./milk10k_clean_audit_results.json"):
        print("MILK10k results JSON not found. Run run_milk10k_clean_audit.py first.")
        return
        
    with open("./milk10k_clean_audit_results.json") as f:
        data = json.load(f)
    results_db = data["results"]

    ALPHA = 0.05
    SEEDS = cfg.seeds

    print(f"\n{'='*72}")
    print("MILK10K IN-DOMAIN CORRECTED SIGNIFICANCE ANALYSIS")
    print(f"Holm-Bonferroni correction, α={ALPHA}")
    print(f"{'='*72}")

    def get_vals(arch, metric):
        vals = results_db.get(arch, {}).get(metric, [])
        if len(vals) < len(SEEDS):
            return None
        return list(vals[:len(SEEDS)])

    def custom_raw_ttest(arch_a, arch_b, metric):
        a = get_vals(arch_a, metric)
        b = get_vals(arch_b, metric)
        if a is None or b is None:
            return None
        t, p = stats.ttest_rel(a, b)
        g = hedges_g_paired(a, b)
        return float(t), float(p), float(np.mean(b) - np.mean(a)), g

    BASELINE = "Image-Only"
    MODELS = [
        "Late Fusion",
        "GMU Baseline",
        "Cross-Attention (V->T)",
        "Cross-Attention T->V"
    ]
    METRICS = ["Accuracy", "F1 (Macro)"]

    all_raw = []
    for model in MODELS:
        if model not in results_db:
            continue
        for metric in METRICS:
            comp_label = f"{model} vs. {BASELINE}"
            res = custom_raw_ttest(BASELINE, model, metric)
            if res is not None:
                t, p, delta, d = res
                all_raw.append((comp_label, metric, BASELINE, model, t, p, delta, d))

    m_total = len(all_raw)
    print(f"Test Matrix (m = {m_total} tests, α={ALPHA})")
    
    if m_total == 0:
        print("No valid tests found.")
        return

    test_tuples = [(f"{r[0]} | {r[1]}", r[4], r[5], r[6], r[7]) for r in all_raw]
    corrected = holm_bonferroni(test_tuples)

    print("\nResults:")
    print(f"{'Comparison':<45} | {'Metric':<12} | {'p-value':>8} | {'adj-α':>8} | {'Sig?':>4} | {'Hedges g':>8} (Size)")
    print("-" * 105)

    n_surv = 0
    for i, (comp_label, metric, arch_a, arch_b, t, p, delta, d) in enumerate(all_raw):
        _, _, _, _, _, adj_alpha, reject = corrected[i]
        if reject:
            n_surv += 1
        
        sig_str = "YES*" if reject else "no"
        d_str = f"{d:+.2f}" if d != float('inf') else "+INF"
        
        print(f"{comp_label:<45} | {metric:<12} | {p:>8.4f} | {adj_alpha:>8.4f} | {sig_str:>4} | {d_str:>8} ({d_label(d)})")
        
    print(f"\nSurvivors after Holm-Bonferroni: {n_surv} / {m_total}")

if __name__ == "__main__":
    main()
