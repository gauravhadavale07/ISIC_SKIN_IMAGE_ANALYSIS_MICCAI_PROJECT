#!/usr/bin/env python3
"""
Generate all publication figures.

Usage (from project root):
    python figures/run_all_figures.py
    python figures/run_all_figures.py --skip-export   # use cached data only
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURES_DIR.parent

FIGURE_SCRIPTS = [
    "fig01_pipeline_diagram.py",
    "fig02_architecture_comparison.py",
    "fig03_training_curves.py",
    "fig04_overall_performance.py",
    "fig05_blank_text_ablation.py",
    "fig06_counterfactual_flip_rate.py",
    "fig07_mean_delta_p.py",
    "fig08_roc_curves.py",
    "fig09_precision_recall_curves.py",
    "fig10_confusion_matrices.py",
    "fig11_per_class_metrics.py",
    "fig12_cka_visualization.py",
    "fig13_tsne_embeddings.py",
    "fig14_umap_embeddings.py",
    "fig15_counterfactual_case_studies.py",
    "fig16_cross_attention_visualization.py",
    "fig17_calibration_plot.py",
    "fig18_statistical_summary.py",
    "fig19_feature_norm_comparison.py",
    "fig20_publication_summary.py",
]


def run_script(script_name: str):
    path = FIGURES_DIR / script_name
    print(f"\n{'='*60}\nGenerating: {script_name}\n{'='*60}")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(FIGURES_DIR))
    spec.loader.exec_module(mod)
    if hasattr(mod, "main"):
        mod.main()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    if not args.skip_export:
        print("Step 1: Exporting figure data artifacts...")
        subprocess.run(
            [sys.executable, str(FIGURES_DIR / "export_figure_data.py")],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    print("\nStep 2: Generating all figures...")
    for script in FIGURE_SCRIPTS:
        try:
            run_script(script)
        except Exception as e:
            print(f"  ERROR in {script}: {e}")

    print(f"\nDone. Figures saved to {FIGURES_DIR / 'output'}/")


if __name__ == "__main__":
    main()
