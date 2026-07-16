import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def train_tabular_baseline():
    print("Loading data...")
    # Load training data
    train_df = pd.read_csv('milk10k_train.csv')
    test_df = pd.read_csv('pad_ufes_20_test.csv')
    
    # We need to extract the tabular features.
    # From prepare_data.py, the columns might be 'age_approx', 'sex', 'site'
    # Wait, milk10k_train.csv might have missing values or different column names.
    # Let's check available columns by matching 'age', 'sex', 'anatom'
    
    # Safely get age
    age_col = next((c for c in train_df.columns if 'age' in c.lower()), None)
    # Safely get sex
    sex_col = next((c for c in train_df.columns if 'sex' in c.lower() or 'gender' in c.lower()), None)
    # Safely get site
    site_col = next((c for c in train_df.columns if 'site' in c.lower() or 'anatom' in c.lower()), None)
    
    print(f"Using columns: Age='{age_col}', Sex='{sex_col}', Site='{site_col}'")
    
    if not age_col or not sex_col or not site_col:
        print("Could not find all required metadata columns in training set.")
        return
        
    features = [age_col, sex_col, site_col]
    
    # Fill NAs
    train_df[age_col] = pd.to_numeric(train_df[age_col], errors='coerce').fillna(50)
    train_df[sex_col] = train_df[sex_col].fillna('unknown').astype(str)
    train_df[site_col] = train_df[site_col].fillna('unknown').astype(str)
    
    test_df[age_col] = pd.to_numeric(test_df[age_col], errors='coerce').fillna(50)
    test_df[sex_col] = test_df[sex_col].fillna('unknown').astype(str)
    test_df[site_col] = test_df[site_col].fillna('unknown').astype(str)
    
    # Target label is 'diagnostic' according to prepare_data.py
    y_train = train_df['diagnostic']
    y_test = test_df['diagnostic']
    
    X_train = train_df[features]
    X_test = test_df[features]
    
    # Create Pipeline
    numeric_features = [age_col]
    categorical_features = [sex_col, site_col]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])
                          
    print("Training Tabular Baseline (Logistic Regression)...")
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print("\n📊 --- Tabular Baseline Report ---")
    print(f"Metadata Only Accuracy (Logistic Regression): {acc*100:.2f}%")
    print("-" * 35)

if __name__ == "__main__":
    train_tabular_baseline()
