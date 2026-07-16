import pandas as pd

df = pd.read_csv('results/task15_ddi_demographic_patching.csv')

def compute_aggregate(group):
    baseline_disparity = group['light_true_logit'].mean() - group['dark_true_logit'].mean()
    
    # Numerators
    main_shift = group['main_light_to_dark_true_logit'].mean() - group['dark_true_logit'].mean()
    same_tone_shift = group['same_tone_dark_to_dark_true_logit'].mean() - group['dark_true_logit'].mean()
    random_class_shift = group['random_class_light_to_dark_true_logit'].mean() - group['dark_true_logit'].mean()
    reverse_shift = group['reverse_dark_to_light_true_logit'].mean() - group['light_true_logit'].mean()
    
    # Pred flips
    main_flip = (group['main_patched_pred'] != group['dark_pred']).mean()
    same_tone_flip = (group['same_tone_patched_pred'] != group['dark_pred']).mean()
    
    return pd.Series({
        'n': len(group),
        'aggregate_main_recovery_pct': (main_shift / baseline_disparity) * 100,
        'aggregate_same_tone_recovery_pct': (same_tone_shift / baseline_disparity) * 100,
        'aggregate_random_class_recovery_pct': (random_class_shift / baseline_disparity) * 100,
        'aggregate_reverse_recovery_pct': (reverse_shift / (-baseline_disparity)) * 100,
        'main_pred_flip_rate': main_flip,
        'specificity_gap': ((main_shift - same_tone_shift) / baseline_disparity) * 100
    })

summary = df.groupby('head').apply(compute_aggregate).reset_index()

# Sort by specificity gap to find the most causally specific heads
summary = summary.sort_values(by='specificity_gap', ascending=False)

print("Stable Aggregate Recovery Metrics (Ratio of Means):")
print(summary.to_string(index=False))

summary.to_csv('results/task15_ddi_demographic_patching_stable_summary.csv', index=False)
