import json
import os
import numpy as np
from scipy import stats
from typing import Dict, List, Any
from collections import defaultdict

class StatisticalAnalyzer:
    """
    Aggregates multi-seed results and performs statistical significance testing.
    Crucial for defending claims against the 'lucky seed' critique in medical AI.
    """
    def __init__(self, progress_path: str = "./results/experiment_progress.json"):
        self.progress_path = progress_path
        # Structure: self.results[model_name][metric_name] = [seed1_val, seed2_val, seed3_val]
        self.results = defaultdict(lambda: defaultdict(list))
        self.completed_runs = set()
        self._load_progress()

    def _load_progress(self):
        if not os.path.exists(self.progress_path):
            return
        with open(self.progress_path, "r") as f:
            data = json.load(f)
        self.completed_runs = set(data.get("completed_runs", []))
        for model_name, metric_data in data.get("results", {}).items():
            for metric_name, values in metric_data.items():
                self.results[model_name][metric_name] = values

    def save_progress(self):
        os.makedirs(os.path.dirname(self.progress_path), exist_ok=True)
        payload = {
            "completed_runs": sorted(self.completed_runs),
            "results": {
                model: dict(metrics)
                for model, metrics in self.results.items()
            },
        }
        with open(self.progress_path, "w") as f:
            json.dump(payload, f, indent=2)

    def is_complete(self, run_key: str) -> bool:
        return run_key in self.completed_runs

    def add_run(self, model_name: str, metrics: Dict[str, Any], run_key: str):
        """Appends a single run's metrics to the aggregator."""
        for metric_name, value in metrics.items():
            # We skip complex types like Confusion Matrices for simple scalar statistics
            if isinstance(value, (int, float)):
                self.results[model_name][metric_name].append(value)
        self.completed_runs.add(run_key)
        self.save_progress()

    def compute_aggregation(self) -> Dict[str, Dict[str, str]]:
        """Computes Mean ± Std Dev for all scalar metrics across all models."""
        summary = defaultdict(dict)
        for model, metric_data in self.results.items():
            for metric, values in metric_data.items():
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                
                # Format smartly depending on if it's a percentage (0-100) or decimal (0-1)
                if mean_val > 1.0:
                    summary[model][metric] = f"{mean_val:.2f} ± {std_val:.2f}"
                else:
                    summary[model][metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                    
        return summary

    def paired_ttest(self, baseline_model: str, proposed_model: str, metric: str) -> str:
        """
        Performs a paired t-test to prove statistical significance.
        Null Hypothesis (H0): The models perform identically.
        """
        baseline_vals = self.results[baseline_model].get(metric)
        proposed_vals = self.results[proposed_model].get(metric)

        if not baseline_vals or not proposed_vals:
            return "Insufficient data for t-test."

        if len(baseline_vals) < 2 or len(proposed_vals) < 2:
            return "Need at least 2 seeds for a t-test."

        # Scipy paired t-test (NaN-aware for metrics like AUROC on small splits)
        t_stat, p_val = stats.ttest_rel(baseline_vals, proposed_vals, nan_policy='omit')
        
        if np.isnan(t_stat) or np.isnan(p_val):
            return "t=nan, p=nan (Not enough valid paired samples)"
        
        # Determine significance (alpha = 0.05)
        if p_val < 0.001:
            significance = "p < 0.001 (Highly Sig.)"
        elif p_val < 0.05:
            significance = f"p={p_val:.4f} (Sig.)"
        else:
            significance = f"p={p_val:.4f} (Not Sig.)"
            
        return f"t={t_stat:.3f}, {significance}"

    def print_report(self):
        """Prints a publication-ready statistical summary for the paper."""
        summary = self.compute_aggregation()
        print("\n" + "="*65)
        print("🏆 MICCAI FINAL MULTI-SEED STATISTICAL REPORT 🏆")
        print("="*65)
        
        for model, metrics in summary.items():
            print(f"\n🚀 Architecture: {model}")
            print("-" * 40)
            for metric, value in metrics.items():
                print(f"  {metric:<20}: {value}")

        # Explicit pairwise comparisons for the key narrative metrics
        print("\n⚖️ Statistical Significance (Cross-Attention vs Late Fusion)")
        print("-" * 65)
        
        key_metrics = ["AUROC", "F1 (Macro)", "Linear_CKA", "CFR", "Mean_Delta_P"]
        for metric in key_metrics:
            p_test = self.paired_ttest("Late Fusion", "Cross-Attention", metric)
            print(f"  {metric:<20} -> {p_test}")
            
        print("="*65)