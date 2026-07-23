import json
import numpy as np
import scipy.stats as stats
import pandas as pd

def compute_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def main():
    with open("./results/experiment_progress.json", "r") as f:
        progress = json.load(f)
        
    with open("./figures/data/calibration.json", "r") as f:
        calib_db = json.load(f)

    results_db = progress["results"]
    architectures = [
        "Late Fusion",
        "GMU Baseline",
        "Cross-Attention (V->T)",
        "Cross-Attention T→V",
        "Image-Only",
        "Text-Only"
    ]

    metrics_to_ci = ["Accuracy", "AUROC", "F1 (Macro)"]
    
    records = []
    
    print("==========================================================")
    print("TASK 28: STATISTICAL RIGOR (95% CIs & Calibration)")
    print("==========================================================\n")

    for arch in architectures:
        if arch not in results_db:
            continue
            
        record = {"Architecture": arch}
        print(f"--- {arch} ---")
        for metric in metrics_to_ci:
            if metric in results_db[arch]:
                vals = results_db[arch][metric]
                if len(vals) > 1:
                    m, low, high = compute_ci(vals)
                    record[f"{metric}_mean"] = m
                    record[f"{metric}_95CI_low"] = low
                    record[f"{metric}_95CI_high"] = high
                    
                    if metric == "Accuracy":
                        print(f"  {metric}: {m*100:.2f}% (95% CI: [{low*100:.2f}%, {high*100:.2f}%])")
                    else:
                        print(f"  {metric}: {m:.4f} (95% CI: [{low:.4f}, {high:.4f}])")
                        
        ece = calib_db.get(arch, {}).get("ece", None)
        if ece is not None:
            record["ECE"] = ece
            print(f"  Expected Calibration Error (ECE): {ece:.4f}")
            
        records.append(record)
        print()
        
    df = pd.DataFrame(records)
    df.to_csv("results/task28_statistical_rigor_cis.csv", index=False)
    print("Saved 95% Confidence Intervals to results/task28_statistical_rigor_cis.csv")

if __name__ == "__main__":
    main()
