import pandas as pd
import re
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder

from config import cfg

def parse_history(text):
    text = str(text)
    age = np.nan
    sex = "unknown"
    site = "unknown"
    
    # Extract age
    age_match = re.search(r'age (\d+)', text)
    if age_match:
        age = float(age_match.group(1))
        
    # Extract sex
    if text.startswith('Female'):
        sex = 'female'
    elif text.startswith('Male'):
        sex = 'male'
        
    # Extract site
    site_match = re.search(r'lesion on the (.*?)\.', text)
    if site_match:
        site = site_match.group(1).strip()
        
    return pd.Series({'age_approx': age, 'sex': sex, 'anatom_site_general': site})

def main():
    print("Loading PAD-UFES-20 data...")
    df = pd.read_csv(cfg.paths.pad_ufes_csv)
    
    print("Extracting tabular features from clinical_history...")
    extracted_features = df['clinical_history'].apply(parse_history)
    df = pd.concat([df, extracted_features], axis=1)
    
    # Save the simple CSV
    out_csv = Path(cfg.paths.pad_ufes_csv).with_name('pad_ufes_20_tabular.csv')
    df[['age_approx', 'sex', 'anatom_site_general', 'diagnostic']].to_csv(out_csv, index=False)
    print(f"Saved extracted features to {out_csv}")
    
    # Prepare for training
    X = df[['age_approx', 'sex', 'anatom_site_general']].copy()
    y = df['diagnostic']
    
    # Handle missing age
    X['age_approx'] = X['age_approx'].fillna(X['age_approx'].median())
    
    # Encode labels for strict AUROC tracking
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['age_approx']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['sex', 'anatom_site_general'])
        ])

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nStarting 5-Fold Cross Validation on Tabular Features...")
    print("Features: Age, Sex, Anatomical Site")
    print("Target: 6-Class Diagnosis")
    print("=" * 60)

    for name, model in models.items():
        clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        
        # We use macro metrics to match the paper
        scores = cross_validate(clf, X, y_encoded, cv=cv, 
                              scoring={'accuracy': 'accuracy', 
                                       'auroc': 'roc_auc_ovr',
                                       'f1_macro': 'f1_macro'})
        
        acc_mean = np.mean(scores['test_accuracy'])
        acc_std = np.std(scores['test_accuracy'])
        auroc_mean = np.mean(scores['test_auroc'])
        auroc_std = np.std(scores['test_auroc'])
        f1_mean = np.mean(scores['test_f1_macro'])
        f1_std = np.std(scores['test_f1_macro'])
        
        print(f"Model: {name}")
        print(f"  Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
        print(f"  AUROC (Macro OVR): {auroc_mean:.4f} ± {auroc_std:.4f}")
        print(f"  F1 (Macro): {f1_mean:.4f} ± {f1_std:.4f}")
        print("-" * 60)

if __name__ == '__main__':
    main()
