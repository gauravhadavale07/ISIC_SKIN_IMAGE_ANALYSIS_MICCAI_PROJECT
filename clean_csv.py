import pandas as pd


def main():
    print("🧹 Sweeping out the literal 'NAN' strings...")

    df = pd.read_csv('./milk10k_train.csv')
    print(f"Original shape: {df.shape}")

    clean_df = df[~df['diagnostic'].astype(str).str.upper().isin(['NAN', 'NULL', 'NONE'])].dropna(subset=['diagnostic'])

    valid_classes = ['MEL', 'BCC', 'SCC', 'ACK', 'NEV', 'SEK']
    clean_df = clean_df[clean_df['diagnostic'].isin(valid_classes)]

    print(f"Cleaned shape: {clean_df.shape}")
    print(f"Classes remaining: {clean_df['diagnostic'].unique().tolist()}")

    clean_df.to_csv('./milk10k_train.csv', index=False)
    print("✅ CSV is now 100% MICCAI compliant. Ready for PyTorch!")


if __name__ == "__main__":
    main()
