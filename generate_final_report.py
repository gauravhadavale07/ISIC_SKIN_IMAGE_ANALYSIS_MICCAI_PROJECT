import json
import numpy as np
from typing import Dict, List, Any

# Load all data
with open('./results/experiment_progress.json', 'r') as f:
    progress_data = json.load(f)

with open('./figures/data/per_class_metrics.json', 'r') as f:
    per_class_data = json.load(f)

with open('./figures/data/confusion_matrices.json', 'r') as f:
    confusion_data = json.load(f)

with open('./figures/data/calibration.json', 'r') as f:
    calibration_data = json.load(f)

# Class names from config
CLASS_NAMES = ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']

def compute_mean_std(values: List[float]) -> tuple:
    """Compute mean and std for a list of values."""
    mean_val = np.mean(values)
    std_val = np.std(values)
    return mean_val, std_val

def format_metric(value: float, is_percentage: bool = False) -> str:
    """Format metric appropriately."""
    if is_percentage:
        return f"{value:.2f}"
    elif value < 1.0:
        return f"{value:.4f}"
    else:
        return f"{value:.2f}"

def print_model_report(model_name: str):
    """Print comprehensive report for a single model."""
    print("=" * 70)
    print(f"{model_name.upper()}")
    print("=" * 70)
    
    # Get aggregated metrics
    metrics = progress_data['results'][model_name]
    
    # 1. Overall Performance
    print("\n1. Overall Performance")
    print("-" * 40)
    
    acc_mean, acc_std = compute_mean_std(metrics['Accuracy'])
    auroc_mean, auroc_std = compute_mean_std(metrics['AUROC'])
    f1_mean, f1_std = compute_mean_std(metrics['F1 (Macro)'])
    prec_mean, prec_std = compute_mean_std(metrics['Precision (Macro)'])
    rec_mean, rec_std = compute_mean_std(metrics['Recall (Macro)'])
    
    cal_key = model_name.replace('Cross-Attention V->T', 'Cross-Attention (V->T)').replace('Cross-Attention V→T', 'Cross-Attention (V->T)').replace('Cross-Attention T→V', 'Cross-Attention T->V')
    ece = calibration_data.get(cal_key, {}).get('ece', 0.0)
    
    print(f"- Accuracy: {format_metric(acc_mean * 100)} ± {format_metric(acc_std * 100)}%")
    print(f"- AUROC (Macro One-vs-Rest): {format_metric(auroc_mean)} ± {format_metric(auroc_std)}")
    print(f"- Macro F1 Score: {format_metric(f1_mean)} ± {format_metric(f1_std)}")
    print(f"- Macro Precision: {format_metric(prec_mean)} ± {format_metric(prec_std)}")
    print(f"- Macro Recall: {format_metric(rec_mean)} ± {format_metric(rec_std)}")
    print(f"- Expected Calibration Error (ECE): {format_metric(ece)}")
    
    # 2. Confusion Matrix
    print("\n2. Confusion Matrix")
    print("-" * 40)
    cm = confusion_data.get(cal_key, {})
    print("- Confusion Matrix:")
    for row in cm:
        print(f"  {row}")
    print(f"- Class ordering used: {CLASS_NAMES}")
    
    # 3. Per-Class Metrics
    print("\n3. Per-Class Metrics")
    print("-" * 40)
    
    pc_metrics = per_class_data.get(cal_key, {})
    
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}:")
        print(f"    Precision: {format_metric(pc_metrics['precision'][i])}")
        print(f"    Recall: {format_metric(pc_metrics['recall'][i])}")
        print(f"    F1: {format_metric(pc_metrics['f1'][i])}")
        print(f"    Accuracy: {format_metric(pc_metrics['per_class_accuracy'][i])}")
        print(f"    n: {pc_metrics['support'][i]}")
        print()
    
    # 4. Mechanistic Audit
    print("4. Mechanistic Audit")
    print("-" * 40)
    
    real_acc_mean, real_acc_std = compute_mean_std(metrics['Real_Accuracy'])
    blank_acc_mean, blank_acc_std = compute_mean_std(metrics['Blank_Accuracy'])
    blank_drop_mean, blank_drop_std = compute_mean_std(metrics['Blank_Accuracy_Drop'])
    cfr_mean, cfr_std = compute_mean_std(metrics['CFR'])
    delta_p_mean, delta_p_std = compute_mean_std(metrics['Mean_Delta_P'])

    # Counterfactual accuracy — read real stored value, NOT algebraic derivation.
    # (Derivation bug: real_acc - blank_drop == blank_acc algebraically, so it
    #  was just a copy of blank_acc with zero information content.)
    if 'Counterfactual_Accuracy' in metrics and metrics['Counterfactual_Accuracy']:
        cf_acc_mean, cf_acc_std = compute_mean_std(metrics['Counterfactual_Accuracy'])
        cf_acc_source = "real forward pass"
    else:
        # Fallback for checkpoints not yet re-audited — flag clearly
        cf_acc_mean = real_acc_mean - blank_drop_mean  # == blank_acc_mean (stale)
        cf_acc_std = blank_drop_std
        cf_acc_source = "STALE FALLBACK (== blank_acc; re-run counterfactual audit)"

    # Neutral-text accuracy (Bug 5 probe)
    if 'Neutral_Accuracy' in metrics and metrics['Neutral_Accuracy']:
        neutral_acc_mean, neutral_acc_std = compute_mean_std(metrics['Neutral_Accuracy'])
        neutral_acc_str = f"{format_metric(neutral_acc_mean)} ± {format_metric(neutral_acc_std)}%"
    else:
        neutral_acc_str = "NOT YET RUN"

    # Text gain (real - blank)
    text_gain_mean = real_acc_mean - blank_acc_mean
    text_gain_std = np.std(np.array(metrics['Real_Accuracy']) - np.array(metrics['Blank_Accuracy']))

    print("Report:")
    print(f"- Real Accuracy:              {format_metric(real_acc_mean)} ± {format_metric(real_acc_std)}%")
    print(f"- Blank-Text Accuracy:        {format_metric(blank_acc_mean)} ± {format_metric(blank_acc_std)}%")
    print(f"- Neutral-Text Accuracy:      {neutral_acc_str}")
    print(f"- Counterfactual Accuracy:    {format_metric(cf_acc_mean)} ± {format_metric(cf_acc_std)}%  [{cf_acc_source}]")
    print(f"- Blank Accuracy Drop:        {format_metric(blank_drop_mean)} ± {format_metric(blank_drop_std)}%")
    print(f"- Text Gain (real - blank):   {format_metric(text_gain_mean)} ± {format_metric(text_gain_std)}%")

    print("\nLexical Counterfactual Audit")
    print(f"- CFR: {format_metric(cfr_mean)} ± {format_metric(cfr_std)}")
    print(f"- Malignant → Benign flips: N/A")
    print(f"- Benign → Malignant flips: N/A")
    print(f"- Mean ΔP: {format_metric(delta_p_mean)} ± {format_metric(delta_p_std)}")

    print("\nLexical Shortcut Diagnostic")
    print(f"- Diagnostic CFR: {format_metric(cfr_mean)} ± {format_metric(cfr_std)}")
    print(f"- Diagnostic Mean ΔP: {format_metric(delta_p_mean)} ± {format_metric(delta_p_std)}")
    print(f"- Real / Blank / Neutral / CF Accuracy: {format_metric(real_acc_mean)}% / {format_metric(blank_acc_mean)}% / {neutral_acc_str} / {format_metric(cf_acc_mean)}%")
    
    # 5. Latent Space Geometric Audit
    print("\n5. Latent Space Geometric Audit")
    print("-" * 40)
    
    print("Report:")
    
    n_samples = metrics['N_samples'][0]
    vis_norm_mean, vis_norm_std = compute_mean_std(metrics['Vis_Feat_Norm'])
    fused_norm_mean, fused_norm_std = compute_mean_std(metrics['Fused_Feat_Norm'])
    cka_mean, cka_std = compute_mean_std(metrics['Linear_CKA'])
    
    print(f"- Samples Evaluated: {n_samples}")
    print(f"- Mean Visual Feature L2 Norm: {format_metric(vis_norm_mean)} ± {format_metric(vis_norm_std)}")
    print(f"- Mean Fused Feature L2 Norm: {format_metric(fused_norm_mean)} ± {format_metric(fused_norm_std)}")
    print(f"- Linear CKA (Visual vs Fused): {format_metric(cka_mean)} ± {format_metric(cka_std)}")
    
    # Interpretation
    print("\nInterpretation:")
    if cka_mean >= 0.95:
        interpretation = "Modality collapse"
        explanation = "Extremely high CKA suggests the fused representation is nearly identical to visual features, indicating minimal text contribution."
    elif cka_mean > 0.85 and cka_mean < 0.95:
        interpretation = "Moderate geometric perturbation detected"
        explanation = "CKA indicates moderate divergence between visual and fused representations, suggesting some text influence but not full integration."
    else:
        interpretation = "Healthy multimodal fusion"
        explanation = "CKA indicates significant divergence between visual and fused representations, suggesting strong text influence and genuine multimodal integration."
    
    print(f"- {interpretation}")
    print(f"  {explanation}")

def print_comparison_table(sort_by: str = 'Accuracy'):
    """Print comparison table sorted by specified metric."""
    print("\n" + "=" * 70)
    print(f"MULTI-MODEL COMPARISON (Sorted by {sort_by})")
    print("=" * 70)
    
    # Collect all model data
    model_data = []
    for model_name in progress_data['results'].keys():
        metrics = progress_data['results'][model_name]
        
        acc_mean, _ = compute_mean_std(metrics['Accuracy'])
        auroc_mean, _ = compute_mean_std(metrics['AUROC'])
        f1_mean, _ = compute_mean_std(metrics['F1 (Macro)'])
        prec_mean, _ = compute_mean_std(metrics['Precision (Macro)'])
        rec_mean, _ = compute_mean_std(metrics['Recall (Macro)'])
        ece = calibration_data[model_name]['ece']
        
        real_acc_mean, _ = compute_mean_std(metrics['Real_Accuracy'])
        blank_acc_mean, _ = compute_mean_std(metrics['Blank_Accuracy'])
        blank_drop_mean, _ = compute_mean_std(metrics['Blank_Accuracy_Drop'])
        cfr_mean, _ = compute_mean_std(metrics['CFR'])
        delta_p_mean, _ = compute_mean_std(metrics['Mean_Delta_P'])
        cka_mean, _ = compute_mean_std(metrics['Linear_CKA'])
        
        # Counterfactual accuracy - use actual field if available, otherwise derive
        if 'Counterfactual_Accuracy' in metrics and metrics['Counterfactual_Accuracy']:
            cf_acc_mean, _ = compute_mean_std(metrics['Counterfactual_Accuracy'])
        else:
            # Fallback for old data without Counterfactual_Accuracy field
            cf_acc_mean = real_acc_mean - blank_drop_mean
        
        # Text gain
        text_gain_mean = real_acc_mean - blank_acc_mean
        
        model_data.append({
            'Model': model_name,
            'Accuracy': acc_mean * 100,
            'AUROC': auroc_mean,
            'Macro F1': f1_mean,
            'Macro Precision': prec_mean,
            'Macro Recall': rec_mean,
            'ECE': ece,
            'Real Accuracy': real_acc_mean,
            'Blank Accuracy': blank_acc_mean,
            'Counterfactual Accuracy': cf_acc_mean,
            'Blank Accuracy Drop': blank_drop_mean,
            'Text Gain': text_gain_mean,
            'CFR': cfr_mean,
            'Diagnostic CFR': cfr_mean,
            'Mean ΔP': delta_p_mean,
            'Linear CKA': cka_mean
        })
    
    # Sort
    reverse_sort = sort_by in ['Accuracy', 'AUROC', 'Macro F1', 'Macro Precision', 'Macro Recall', 'Real Accuracy', 'Blank Accuracy', 'Counterfactual Accuracy', 'Text Gain', 'Linear CKA']
    model_data.sort(key=lambda x: x[sort_by], reverse=reverse_sort)
    
    # Print table header
    headers = ['Model', 'Accuracy', 'AUROC', 'Macro F1', 'Macro Precision', 'Macro Recall', 'ECE', 
               'Real Accuracy', 'Blank Accuracy', 'Counterfactual Accuracy', 'Blank Accuracy Drop', 'Text Gain',
               'CFR', 'Diagnostic CFR', 'Mean ΔP', 'Linear CKA']
    
    # Calculate column widths
    col_widths = [max(len(h), 15) for h in headers]
    
    # Print header
    header_line = "|".join(h.center(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for data in model_data:
        row = []
        for h, w in zip(headers, col_widths):
            val = data[h]
            if h in ['Accuracy', 'Real Accuracy', 'Blank Accuracy', 'Counterfactual Accuracy', 'Blank Accuracy Drop', 'Text Gain']:
                row.append(f"{val:.2f}".center(w))
            elif h in ['AUROC', 'Macro F1', 'Macro Precision', 'Macro Recall', 'ECE', 'CFR', 'Diagnostic CFR', 'Mean ΔP', 'Linear CKA']:
                row.append(f"{val:.4f}".center(w))
            else:
                row.append(str(val).center(w))
        print("|".join(row))

def print_best_model_summary():
    """Print summary of best performing models."""
    print("\n" + "=" * 70)
    print("BEST MODEL SUMMARY")
    print("=" * 70)
    
    # Collect all model data
    model_data = {}
    for model_name in progress_data['results'].keys():
        metrics = progress_data['results'][model_name]
        
        acc_mean, _ = compute_mean_std(metrics['Accuracy'])
        auroc_mean, _ = compute_mean_std(metrics['AUROC'])
        f1_mean, _ = compute_mean_std(metrics['F1 (Macro)'])
        ece = calibration_data[model_name]['ece']
        cka_mean, _ = compute_mean_std(metrics['Linear_CKA'])
        cfr_mean, _ = compute_mean_std(metrics['CFR'])
        delta_p_mean, _ = compute_mean_std(metrics['Mean_Delta_P'])
        blank_drop_mean, _ = compute_mean_std(metrics['Blank_Accuracy_Drop'])
        
        blank_acc_mean, _ = compute_mean_std(metrics['Blank_Accuracy'])
        neutral_acc_mean = np.mean(metrics['Neutral_Accuracy']) if 'Neutral_Accuracy' in metrics and metrics['Neutral_Accuracy'] else None

        model_data[model_name] = {
            'Accuracy': acc_mean * 100,
            'AUROC': auroc_mean,
            'Macro F1': f1_mean,
            'ECE': ece,
            'CKA': cka_mean,
            'CFR': cfr_mean,
            'Mean ΔP': delta_p_mean,
            'Blank Drop': blank_drop_mean,
            'Blank Accuracy': blank_acc_mean,
            'Neutral Accuracy': neutral_acc_mean,
        }
    
    # Find best for each metric
    best_accuracy = max(model_data.items(), key=lambda x: x[1]['Accuracy'])
    best_auroc = max(model_data.items(), key=lambda x: x[1]['AUROC'])
    best_f1 = max(model_data.items(), key=lambda x: x[1]['Macro F1'])
    best_calibration = min(model_data.items(), key=lambda x: x[1]['ECE'])
    
    # Best multimodal fusion: lowest healthy CKA without collapse (0.85 < CKA < 0.95)
    healthy_fusion = {k: v for k, v in model_data.items() if 0.85 < v['CKA'] < 0.95}
    if healthy_fusion:
        best_fusion = min(healthy_fusion.items(), key=lambda x: x[1]['CKA'])
    else:
        best_fusion = min(model_data.items(), key=lambda x: abs(x[1]['CKA'] - 0.90))
    
    lowest_cfr = min(model_data.items(), key=lambda x: x[1]['CFR'])
    lowest_delta_p = min(model_data.items(), key=lambda x: x[1]['Mean ΔP'])
    lowest_blank_drop = min(model_data.items(), key=lambda x: x[1]['Blank Drop'])
    
    # Most blind to counterfactual text: lowest blank accuracy drop (more blind)
    most_blind = min(model_data.items(), key=lambda x: x[1]['Blank Drop'])
    
    # Calculate majority class baseline for warning
    # BCC class has 845 samples out of 2298 total
    majority_baseline = 845 / 2298 * 100
    
    print(f"- Best Accuracy: {best_accuracy[0]} ({format_metric(best_accuracy[1]['Accuracy'])}%)")
    print(f"- Best AUROC: {best_auroc[0]} ({format_metric(best_auroc[1]['AUROC'])})")
    print(f"- Best Macro F1: {best_f1[0]} ({format_metric(best_f1[1]['Macro F1'])})")
    print(f"- Best Calibration (lowest ECE): {best_calibration[0]} ({format_metric(best_calibration[1]['ECE'])})")
    print(f"- Best Multimodal Fusion: {best_fusion[0]} (CKA: {format_metric(best_fusion[1]['CKA'])})")
    print(f"- Lowest CFR: {lowest_cfr[0]} ({format_metric(lowest_cfr[1]['CFR'])})")
    print(f"- Lowest Mean ΔP: {lowest_delta_p[0]} ({format_metric(lowest_delta_p[1]['Mean ΔP'])})")
    print(f"- Lowest Blank Accuracy Drop: {lowest_blank_drop[0]} ({format_metric(lowest_blank_drop[1]['Blank Drop'])}%)")
    print(f"- Most Semantically Blind to Counterfactual Text: {most_blind[0]} (Blank Drop: {format_metric(most_blind[1]['Blank Drop'])}%)")
    
    # Warning for models with blank accuracy below majority baseline
    print(f"\n⚠️  MAJORITY BASELINE WARNING (BCC class: {majority_baseline:.2f}%):")
    for model_name, data in model_data.items():
        blank_a = data['Blank Accuracy']
        neutral_a = data['Neutral Accuracy']
        if blank_a < majority_baseline:
            neutral_note = f", Neutral={neutral_a:.2f}%" if neutral_a is not None else ""
            print(f"  ❌ {model_name}: Blank-∅ accuracy ({blank_a:.2f}%{neutral_note}) BELOW majority baseline ({majority_baseline:.2f}%) "
                  f"— empty-string tokenizer edge case suspected")
        else:
            print(f"  ✅ {model_name}: Blank accuracy ({blank_a:.2f}%) above majority baseline")

# Main execution
if __name__ == "__main__":
    # Redirect output to file
    import sys
    original_stdout = sys.stdout
    
    with open('evaluation_final_results_6.txt', 'w') as f:
        sys.stdout = f
        
        # Print individual model reports (only models present in results)
        available_models = list(progress_data['results'].keys())
        for model_name in available_models:
            print_model_report(model_name)
        
        # Print comparison tables
        print_comparison_table('Accuracy')
        print_comparison_table('AUROC')
        print_comparison_table('Macro F1')
        
        # Print best model summary
        print_best_model_summary()
    
    # Restore stdout
    sys.stdout = original_stdout
    print("Results saved to evaluation_final_results_6.txt")
