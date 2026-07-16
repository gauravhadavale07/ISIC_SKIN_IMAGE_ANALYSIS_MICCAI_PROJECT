#!/bin/bash
set -e
echo "Starting run_experiment.py (Multimodal)..."
python run_experiment.py > run_experiment.log 2>&1
echo "Starting full_analysis.py (Baselines & Stats Audit)..."
python full_analysis.py > full_analysis.log 2>&1
echo "Starting run_milk10k_clean_audit.py (Table 1)..."
python run_milk10k_clean_audit.py > run_milk10k_clean_audit.log 2>&1
echo "All done!"
