import pandas as pd
import numpy as np
import os
import re

np.random.seed(42)

# Load data exact same way as dataset.py
df = pd.read_csv('milk10k_train.csv')

# 1. Filter out NAN diagnostic
df = df[df['diagnostic'].astype(str).str.upper() != 'NAN'].reset_index(drop=True)

# 2. Extract lesion ID via regex
df['lesion_id'] = df['filepath'].apply(
    lambda x: re.search(r'(IL_\d+)', str(x)).group(1) if re.search(r'(IL_\d+)', str(x)) else None
)
df = df.dropna(subset=['lesion_id']).reset_index(drop=True)

# 3. Deterministic split boundary
all_lesions = sorted(df['lesion_id'].unique())
n_val_lesions = int(0.15 * len(all_lesions))
val_lesion_set = set(all_lesions[-n_val_lesions:])

df_val_det = df[df['lesion_id'].isin(val_lesion_set)]
df_train_det = df[~df['lesion_id'].isin(val_lesion_set)]

classes = sorted(df['diagnostic'].unique())

print(f"Total images: {len(df)}")
print(f"Total unique lesions: {len(all_lesions)}")
print(f"Val lesions: {len(val_lesion_set)} ({len(df_val_det)} images)")
print(f"Train lesions: {len(all_lesions) - len(val_lesion_set)} ({len(df_train_det)} images)")

det_class_counts = df_val_det['diagnostic'].value_counts()
det_class_props = det_class_counts / len(df_val_det)

print("\nDeterministic Val Set Proportions:")
for c in classes:
    cnt = det_class_counts.get(c, 0)
    prop = det_class_props.get(c, 0.0)
    print(f"  {c}: count={cnt}, prop={prop:.4f}")

# 4. Monte Carlo 10,000 random lesion-disjoint splits
n_iterations = 10000
random_props = {c: [] for c in classes}
random_counts = {c: [] for c in classes}

all_lesions_arr = np.array(all_lesions)

for i in range(n_iterations):
    val_lesions_rand = np.random.choice(all_lesions_arr, size=n_val_lesions, replace=False)
    val_set_rand = set(val_lesions_rand)
    df_val_rand = df[df['lesion_id'].isin(val_set_rand)]
    
    counts = df_val_rand['diagnostic'].value_counts()
    n_images_rand = len(df_val_rand)
    
    for c in classes:
        cnt = counts.get(c, 0)
        random_counts[c].append(cnt)
        random_props[c].append(cnt / n_images_rand)

print("\n10,000 Random Splits Audit Results:")
audit_rows = []

max_z_score = 0.0
for c in classes:
    props = np.array(random_props[c])
    counts = np.array(random_counts[c])
    
    mean_prop = np.mean(props)
    std_prop = np.std(props)
    
    det_prop = det_class_props.get(c, 0.0)
    det_cnt = det_class_counts.get(c, 0)
    
    z_score = (det_prop - mean_prop) / std_prop if std_prop > 0 else 0.0
    if abs(z_score) > abs(max_z_score):
        max_z_score = z_score
        
    p_lower = np.percentile(props, 2.5)
    p_upper = np.percentile(props, 97.5)
    
    # Tail probability (p-value for being as or more extreme)
    p_tail = np.mean(np.abs(props - mean_prop) >= np.abs(det_prop - mean_prop))
    
    print(f"Class {c:4s}: Det={det_cnt:4d} ({det_prop*100:5.2f}%) | "
          f"Random Mean={mean_prop*100:5.2f}% ± {std_prop*100:4.2f}% | "
          f"95% CI=[{p_lower*100:5.2f}%, {p_upper*100:5.2f}%] | "
          f"Z-Score={z_score:+.4f} | p_tail={p_tail:.4f}")
    
    audit_rows.append({
        'class': c,
        'deterministic_count': det_cnt,
        'deterministic_prop': det_prop,
        'random_mean_prop': mean_prop,
        'random_std_prop': std_prop,
        'ci_95_lower': p_lower,
        'ci_95_upper': p_upper,
        'z_score': z_score,
        'p_tail': p_tail
    })

print(f"\nMaximum absolute z-score across all classes: {abs(max_z_score):.4f}")

# Save CSV to extended_tables as referenced in paper
os.makedirs('extended_tables', exist_ok=True)
audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv('extended_tables/task29_lesion_split_distribution_audit.csv', index=False)
print("Saved audit CSV to extended_tables/task29_lesion_split_distribution_audit.csv")
