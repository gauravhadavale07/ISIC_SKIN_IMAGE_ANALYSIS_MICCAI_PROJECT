#!/bin/bash
echo "Starting MILK10k Audit..."
python run_milk10k_clean_audit.py > run_milk10k_final.log 2>&1
echo "MILK10k Audit Complete."

echo "Starting MILK10k Significance Testing..."
python milk10k_significance.py > milk10k_sig_final.log 2>&1
echo "MILK10k Significance Testing Complete."

echo "Starting DDI Stratified Audit..."
python task6_ddi_stratified_audit_rigorous.py > run_ddi_final.log 2>&1
echo "DDI Stratified Audit Complete."

echo "All tasks finished successfully!"
