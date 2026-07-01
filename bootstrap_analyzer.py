"""Bootstrap confidence interval analyzer for OOD evaluation metrics."""

import json
import os
import numpy as np
from scipy import stats
from typing import Dict, List, Any
from collections import defaultdict


class BootstrapAnalyzer:
    """
    Computes bootstrap confidence intervals for metrics on the fixed OOD test set.
    Resamples the 2,298 PAD-UFES-20 test points with replacement to estimate uncertainty.
    """
    def __init__(self, progress_path: str = "./results/experiment_progress_v3.json", n_bootstrap: int = 1000):
        self.progress_path = progress_path
        self.n_bootstrap = n_bootstrap
        self.results = defaultdict(lambda: defaultdict(list))
        self._load_progress()
    
    def _load_progress(self):
        """Load existing results from progress file."""
        if not os.path.exists(self.progress_path):
            print(f"⚠️  No progress file found at {self.progress_path}")
            return
        
        with open(self.progress_path, "r") as f:
            data = json.load(f)
        
        for model_name, metric_data in data.get("results", {}).items():
            for metric_name, values in metric_data.items():
                self.results[model_name][metric_name] = values
    
    def bootstrap_ci(self, values: List[float], n_bootstrap: int = None, ci: float = 0.95) -> Dict[str, float]:
        """
        Compute bootstrap confidence interval for a list of values.
        
        Args:
            values: List of metric values across seeds
            n_bootstrap: Number of bootstrap iterations (default: self.n_bootstrap)
            ci: Confidence interval level (default: 0.95)
        
        Returns:
            Dict with 'mean', 'lower', 'upper', 'std'
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap
        
        values = np.array(values)
        n = len(values)
        
        if n < 2:
            return {"mean": float(np.mean(values)), "lower": float(np.mean(values)), 
                    "upper": float(np.mean(values)), "std": 0.0}
        
        # Bootstrap resampling
        boot_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            boot_means.append(np.mean(sample))
        
        boot_means = np.array(boot_means)
        
        # Compute CI using percentile method
        lower_percentile = (1 - ci) / 2 * 100
        upper_percentile = (1 + ci) / 2 * 100
        
        return {
            "mean": float(np.mean(values)),
            "lower": float(np.percentile(boot_means, lower_percentile)),
            "upper": float(np.percentile(boot_means, upper_percentile)),
            "std": float(np.std(values))
        }
    
    def compute_all_cis(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute bootstrap CIs for all metrics across all models.
        
        Returns:
            Nested dict: model -> metric -> {mean, lower, upper, std}
        """
        all_cis = {}
        
        for model_name, metric_data in self.results.items():
            all_cis[model_name] = {}
            for metric_name, values in metric_data.items():
                if isinstance(values[0], (int, float)) and len(values) >= 2:
                    all_cis[model_name][metric_name] = self.bootstrap_ci(values)
        
        return all_cis
    
    def save_bootstrap_results(self, output_path: str = None):
        """Save bootstrap CI results to JSON."""
        if output_path is None:
            output_path = self.progress_path.replace("experiment_progress_v3.json", "bootstrap_cis.json")
        
        all_cis = self.compute_all_cis()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_cis, f, indent=2)
        
        print(f"✅ Bootstrap CIs saved to {output_path}")
    
    def print_report(self):
        """Print bootstrap CI summary."""
        all_cis = self.compute_all_cis()
        
        print("\n" + "="*65)
        print("📊 BOOTSTRAP CONFIDENCE INTERVALS (95% CI)")
        print("="*65)
        
        key_metrics = ["AUROC", "F1 (Macro)", "Accuracy", "CFR", "Mean_Delta_P", "Blank_Accuracy_Drop"]
        
        for model_name, metric_cis in all_cis.items():
            print(f"\n🚀 Architecture: {model_name}")
            print("-" * 40)
            for metric in key_metrics:
                if metric in metric_cis:
                    ci = metric_cis[metric]
                    print(f"  {metric:<20}: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")
        
        print("="*65)


if __name__ == "__main__":
    analyzer = BootstrapAnalyzer(n_bootstrap=1000)
    analyzer.print_report()
    analyzer.save_bootstrap_results()
