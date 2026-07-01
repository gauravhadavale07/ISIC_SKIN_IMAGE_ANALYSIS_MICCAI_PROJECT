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
    def __init__(self, progress_path: str = "./results/experiment_progress_v3.json"):
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

        # Full 12-pair comparison family for collapse testing
        print("\n⚖️ Statistical Significance Tests (12-pair family)")
        print("-" * 65)
        
        # Define all 12 comparison pairs
        comparison_pairs = [
            # Original 6 pairs
            ("Late Fusion", "Cross-Attn V→T"),
            ("Late Fusion", "Cross-Attn T→V"),
            ("GMU Baseline", "Cross-Attn V→T"),
            ("Cross-Attn V→T", "Cross-Attn T→V"),
            ("Image-Only", "Late Fusion"),
            ("Text-Only", "Late Fusion"),
            # New 6 pairs for collapse testing
            ("GMU Baseline", "Image-Only"),
            ("GMU Baseline", "Text-Only"),
            ("Cross-Attn V→T", "Image-Only"),
            ("Cross-Attn V→T", "Text-Only"),
            ("Cross-Attn T→V", "Image-Only"),
            ("Cross-Attn T→V", "Text-Only"),
        ]
        
        key_metrics = ["AUROC", "F1 (Macro)", "Linear_CKA", "CFR", "Mean_Delta_P"]
        
        # Collect all p-values for Holm-Bonferroni correction
        all_p_values = []
        test_results = []
        
        for model_a, model_b in comparison_pairs:
            for metric in key_metrics:
                baseline_vals = self.results[model_a].get(metric)
                proposed_vals = self.results[model_b].get(metric)
                if baseline_vals and proposed_vals and len(baseline_vals) >= 2 and len(proposed_vals) >= 2:
                    t_stat, p_val = stats.ttest_rel(baseline_vals, proposed_vals, nan_policy='omit')
                    if not np.isnan(t_stat) and not np.isnan(p_val):
                        all_p_values.append(p_val)
                        test_results.append((model_a, model_b, metric, t_stat, p_val))
        
        # Apply Holm-Bonferroni correction
        if all_p_values:
            from statsmodels.stats.multitest import multipletests
            rejected, p_corrected, _, _ = multipletests(all_p_values, method='holm')
            
            # Print corrected results
            idx = 0
            for model_a, model_b, metric, t_stat, p_val in test_results:
                is_sig = rejected[idx]
                sig_marker = "***" if is_sig else "ns"
                print(f"  {model_a} vs {model_b} ({metric}): t={t_stat:.3f}, p={p_val:.4f} -> {p_corrected[idx]:.4f} {sig_marker}")
                idx += 1
        
        # Generate collapse-test summary table
        self._generate_collapse_summary(comparison_pairs, key_metrics)
        
        print("="*65)
    
    def _generate_collapse_summary(self, comparison_pairs, key_metrics):
        """Generate collapse-test summary table for paper Results section."""
        collapse_summary = {}
        
        architectures = ["Late Fusion", "GMU Baseline", "Cross-Attn V→T", "Cross-Attn T→V"]
        
        for arch in architectures:
            collapse_summary[arch] = {
                "vs_Image_Only": {"distinguishable": False, "p": None, "p_corrected": None},
                "vs_Text_Only": {"distinguishable": False, "p": None, "p_corrected": None}
            }
        
        # Collect p-values for correction
        all_p_values = []
        test_metadata = []
        
        for arch in architectures:
            for baseline in ["Image-Only", "Text-Only"]:
                for metric in key_metrics:
                    arch_vals = self.results[arch].get(metric)
                    baseline_vals = self.results[baseline].get(metric)
                    if arch_vals and baseline_vals and len(arch_vals) >= 2 and len(baseline_vals) >= 2:
                        t_stat, p_val = stats.ttest_rel(arch_vals, baseline_vals, nan_policy='omit')
                        if not np.isnan(t_stat) and not np.isnan(p_val):
                            all_p_values.append(p_val)
                            test_metadata.append((arch, baseline, metric, t_stat, p_val))
        
        # Apply Holm-Bonferroni correction
        if all_p_values:
            from statsmodels.stats.multitest import multipletests
            rejected, p_corrected, _, _ = multipletests(all_p_values, method='holm')
            
            idx = 0
            for arch, baseline, metric, t_stat, p_val in test_metadata:
                is_sig = rejected[idx]
                baseline_key = f"vs_{baseline.replace('-', '_')}"
                collapse_summary[arch][baseline_key]["p"] = float(p_val)
                collapse_summary[arch][baseline_key]["p_corrected"] = float(p_corrected[idx])
                collapse_summary[arch][baseline_key]["distinguishable"] = bool(is_sig)
                idx += 1
        
        # Save to JSON
        import json
        import os
        os.makedirs(os.path.dirname(self.progress_path), exist_ok=True)
        summary_path = self.progress_path.replace("experiment_progress_v3.json", "collapse_test_summary.json")
        with open(summary_path, "w") as f:
            json.dump(collapse_summary, f, indent=2)
        
        print(f"\n📊 Collapse-test summary saved to {summary_path}")
        
        # Print summary table
        print("\n📊 Collapse-Test Summary Table")
        print("-" * 65)
        print(f"{'Architecture':<20} {'vs Image-Only':<25} {'vs Text-Only':<25}")
        print("-" * 65)
        for arch in architectures:
            img_status = "Y" if collapse_summary[arch]["vs_Image_Only"]["distinguishable"] else "N"
            txt_status = "Y" if collapse_summary[arch]["vs_Text_Only"]["distinguishable"] else "N"
            img_p = collapse_summary[arch]["vs_Image_Only"]["p_corrected"]
            txt_p = collapse_summary[arch]["vs_Text_Only"]["p_corrected"]
            img_str = f"{img_status} (p={img_p:.4f})" if img_p else "N (insufficient data)"
            txt_str = f"{txt_status} (p={txt_p:.4f})" if txt_p else "N (insufficient data)"
            print(f"{arch:<20} {img_str:<25} {txt_str:<25}")
        print("-" * 65)