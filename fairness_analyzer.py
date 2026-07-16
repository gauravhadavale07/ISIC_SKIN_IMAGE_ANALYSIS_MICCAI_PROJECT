"""Fairness analysis: skin tone stratification for model evaluation."""

import json
import os
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


class FairnessAnalyzer:
    """
    Computes fairness metrics stratified by skin tone groups.
    
    Expected metadata:
    - MILK10k: skin_tone_class field (if available)
    - PAD-UFES-20: ITA-based Fitzpatrick pseudo-labels (if available)
    
    Outputs:
    - AUROC by skin tone group
    - Performance disparity metrics
    - Supplementary table for paper
    """
    def __init__(self, progress_path: str = "./results/experiment_progress_v3.json"):
        self.progress_path = progress_path
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
    
    def compute_fairness_metrics(self):
        """
        Compute fairness metrics stratified by skin tone.
        
        Note: This requires per-sample skin tone labels which may not be
        available in current checkpoint structure. This is a framework for
        future integration when skin tone metadata is added to the dataset.
        
        Returns:
            Dict with fairness metrics per model
        """
        fairness_summary = {}
        
        for model_name in self.results.keys():
            fairness_summary[model_name] = {
                "note": "Skin tone stratification requires per-sample metadata",
                "status": "framework_ready_awaiting_data",
                "recommended_action": "Add skin_tone field to dataset CSVs and re-export predictions with labels"
            }
        
        return fairness_summary
    
    def save_fairness_report(self, output_path: str = None):
        """Save fairness analysis report to JSON."""
        if output_path is None:
            output_path = self.progress_path.replace("experiment_progress_v3.json", "fairness_summary.json")
        
        fairness_metrics = self.compute_fairness_metrics()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(fairness_metrics, f, indent=2)
        
        print(f"✅ Fairness summary saved to {output_path}")
    
    def print_report(self):
        """Print fairness analysis summary."""
        fairness_metrics = self.compute_fairness_metrics()
        
        print("\n" + "="*65)
        print("📊 FAIRNESS ANALYSIS: SKIN TONE STRATIFICATION")
        print("="*65)
        
        for model_name, metrics in fairness_metrics.items():
            print(f"\n🚀 Architecture: {model_name}")
            print("-" * 40)
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*65)
        print("💡 RECOMMENDATIONS FOR FAIRNESS EVALUATION:")
        print("="*65)
        print("1. Add skin_tone field to MILK10k and PAD-UFES-20 CSVs")
        print("2. For PAD-UFES-20: compute ITA from RGB images → Fitzpatrick groups")
        print("3. Modify export_figure_data.py to save predictions with skin tone labels")
        print("4. Compute AUROC per skin tone group (Fitzpatrick I-VI)")
        print("5. Report max AUROC disparity as fairness metric")
        print("6. Include supplementary table in paper submission")
        print("="*65)


if __name__ == "__main__":
    analyzer = FairnessAnalyzer()
    analyzer.print_report()
    analyzer.save_fairness_report()
