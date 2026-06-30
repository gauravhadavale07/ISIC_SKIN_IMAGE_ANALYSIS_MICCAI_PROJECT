import os
import glob
import pandas as pd


def map_diagnosis(diag):
    diag = str(diag).lower()
    if 'melanoma' in diag:
        return 'MEL'
    if 'basal cell carcinoma' in diag:
        return 'BCC'
    if 'squamous cell carcinoma' in diag:
        return 'SCC'
    if 'actinic keratosis' in diag:
        return 'ACK'
    if 'nevus' in diag:
        return 'NEV'
    if 'seborrheic keratosis' in diag or 'bkl' in diag:
        return 'SEK'
    return 'UNKNOWN'


def build_text(row):
    def safe_str(value, default):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        text = str(value).strip()
        if text.lower() in ('nan', 'none', ''):
            return default
        return text

    sex = safe_str(row.get('sex'), 'Patient').capitalize()
    age = safe_str(row.get('age_approx'), 'unknown age').replace('.0', '')
    site = safe_str(row.get('site'), 'the skin').lower()
    base = f"{sex}, age {age}, presents with a lesion on {site}."

    feats = []
    if float(row.get('MONET_erythema', 0)) > 0.5:
        feats.append("distinct erythema")
    if float(row.get('MONET_pigmented', 0)) > 0.5:
        feats.append("irregular pigmentation")
    if float(row.get('MONET_vasculature_vessels', 0)) > 0.5:
        feats.append("visible vascular structures")
    if float(row.get('MONET_ulceration_crust', 0)) > 0.5:
        feats.append("surface ulceration or crusting")

    if feats:
        return base + " Dermatoscopic evaluation indicates " + ", ".join(feats) + "."
    return base + " No prominent secondary morphological features were confidently identified."


def main():
    print("🚀 INITIATING SURGICAL MILK10K BUILDER...")

    raw_dir = "./data/raw_milk10k"
    output_csv = "./milk10k_train.csv"

    all_imgs = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        all_imgs.extend(glob.glob(os.path.join(raw_dir, '**', ext), recursive=True))
    img_map = {os.path.splitext(os.path.basename(p))[0].strip(): p for p in all_imgs}

    meta_df = pd.read_csv(os.path.join(raw_dir, "MILK10k_Training_Metadata.csv"))
    supp_df = pd.read_csv(os.path.join(raw_dir, "MILK10k_Training_Supplement.csv"))

    df = pd.merge(meta_df, supp_df, on='isic_id', how='inner')
    print(f"📋 Merged dataset shape: {df.shape}")

    df['clean_id'] = df['isic_id'].astype(str).str.strip()
    df['filepath'] = df['clean_id'].map(img_map)
    valid_df = df.dropna(subset=['filepath']).copy()
    print(f"✅ Found {len(valid_df)} perfect image matches.")

    valid_df['diagnostic'] = valid_df['diagnosis_full'].apply(map_diagnosis)
    valid_df = valid_df[valid_df['diagnostic'] != 'UNKNOWN']

    print("✍️ Synthesizing true MONET clinical history...")
    valid_df['clinical_history'] = valid_df.apply(build_text, axis=1)

    final_df = valid_df[['filepath', 'clinical_history', 'diagnostic']]
    final_df.to_csv(output_csv, index=False)

    print(f"✅ SUCCESS! {len(final_df)} clean, labeled records saved to {output_csv}.")
    print("🚀 YOU ARE READY! Run: python run_experiment.py")


if __name__ == "__main__":
    main()
