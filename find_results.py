import os
import pandas as pd

for root, dirs, files in os.walk('results'):
    for file in files:
        if file.endswith('.csv'):
            path = os.path.join(root, file)
            df = pd.read_csv(path)
            print(f"\n--- {file} ---")
            print(df.head())
