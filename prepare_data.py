import os
import glob
import zipfile
import subprocess
import pandas as pd
from typing import List
from config import cfg

def run_command(command: str):
    print(f"⚙️ Executing: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"❌ Error executing {command}:\n{process.stderr}")
    else:
        print("✅ Success.")

def download_from_kaggle(dataset_slug: str, download_dir: str):
    if os.path.exists(download_dir) and len(os.listdir(download_dir)) > 0:
        print(f"📦 Data already exists in {download_dir}. Skipping download.")
        return

    os.makedirs(download_dir, exist_ok=True)
    print(f"📥 Downloading {dataset_slug} to {download_dir}...")
    run_command(f"kaggle datasets download -d {dataset_slug} -p {download_dir}")

    zip_files = glob.glob(f"{download_dir}/*.zip")
    if zip_files:
        zip_path = zip_files[0]
        print(f"🗜️ Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        os.remove(zip_path)
        print("✅ Extraction complete.")


def _safe_str(value, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def _map_column(df: pd.DataFrame, target_col: str, candidates: List[str]):
    src_col = next((c for c in candidates if c in df.columns), None)
    if src_col is not None:
        df[target_col] = df[src_col]


def build_clinical_history_monet(row: pd.Series) -> str:
    sex = _safe_str(row.get('sex'), 'Patient').capitalize()
    age = _safe_str(row.get('age_approx'), 'unknown age')
    site = _safe_str(row.get('site'), 'the skin').lower()

    base_desc = f"{sex}, age {age}, presents with a lesion on {site}."

    clinical_features = []
    if float(row.get('MONET_erythema', 0)) > 0.5: clinical_features.append("distinct erythema")
    if float(row.get('MONET_pigmented', 0)) > 0.5: clinical_features.append("irregular pigmentation")
    if float(row.get('MONET_vasculature_vessels', 0)) > 0.5: clinical_features.append("visible vascular structures")
    if float(row.get('MONET_ulceration_crust', 0)) > 0.5: clinical_features.append("surface ulceration or crusting")

    if clinical_features:
        feature_text = " Dermatoscopic evaluation indicates " + ", ".join(clinical_features) + "."
    else:
        feature_text = " No prominent secondary morphological features (erythema, ulceration, or abnormal vasculature) were confidently identified."

    return base_desc + feature_text


def process_milk10k(raw_dir: str, output_csv: str):
    print(f"\n🔍 Processing TRUE MILK10k from {raw_dir}...")

    all_imgs = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        all_imgs.extend(glob.glob(os.path.join(raw_dir, '**', ext), recursive=True))

    img_map = {os.path.splitext(os.path.basename(p))[0].strip(): p for p in all_imgs}

    meta_files = glob.glob(f"{raw_dir}/**/*.csv", recursive=True)
    if not meta_files: 
        print("❌ No CSV metadata found.")
        return

    df = None
    for f in meta_files:
        temp_df = pd.read_csv(f)
        
        # STRICT FIX 1: Prioritize Image IDs to guarantee correct index merging. 
        # Removed 'lesion_id' to prevent orphaned labels.
        id_cols_to_try = ['isic_id', 'image_id', 'img_id', 'image', 'isic', 'image_name']
        id_col = next((col for col in id_cols_to_try if col in temp_df.columns), None)

        if id_col:
            temp_df['clean_id'] = temp_df[id_col].astype(str).apply(lambda x: os.path.splitext(x)[0].strip())
            temp_df = temp_df.set_index('clean_id')
            
            temp_df = temp_df.groupby(level=0).first()
            
            if df is None:
                df = temp_df
            else:
                df = df.combine_first(temp_df)

    df = df.reset_index()
    if 'index' in df.columns and 'clean_id' not in df.columns:
         df = df.rename(columns={'index': 'clean_id'})
    elif 'clean_id' not in df.columns:
         df['clean_id'] = df.index
    
    print(f"📋 Master Combined Dataset Shape: {df.shape}")

    df['filepath'] = df['clean_id'].map(img_map)
    valid_df = df.dropna(subset=['filepath']).copy()
    print(f"✅ Found {len(valid_df)} images exactly matching the CSV on disk.")

    if len(valid_df) == 0: return

    # STRICT FIX 2: Case-insensitive label hunting
    target_cols = ['MEL', 'BCC', 'SCCKA', 'AKIEC', 'NV', 'BKL', 'DF', 'VASC', 'SCC', 'ACK', 'NEV', 'SEK']
    existing = [c for c in valid_df.columns if c.upper() in target_cols]

    if existing:
        # Safely force numeric to avoid string-comparison crashes on '1.0' vs 0
        numeric_targets = valid_df[existing].apply(pd.to_numeric, errors='coerce').fillna(0)
        row_maxes = numeric_targets.max(axis=1)
        valid_mask = row_maxes > 0 
        valid_df.loc[valid_mask, 'diagnostic_raw'] = numeric_targets.loc[valid_mask].idxmax(axis=1)
    else:
        possible_cols = ['diagnosis', 'dx', 'benign_malignant', 'target', 'diagnostic']
        diag_col = next((col for col in valid_df.columns if col.lower() in possible_cols), None)

        if diag_col:
            valid_df['diagnostic_raw'] = valid_df[diag_col]
        else:
            print("❌ Could not find diagnostic columns in ANY merged CSV.")
            return

    # Expand mapping to catch all edge cases safely
    acronym_map = {
        "sccka": "SCC", "akiec": "ACK", "nv": "NEV", "bkl": "SEK",
        "melanoma": "MEL", "basal cell carcinoma": "BCC", "nevus": "NEV",
        "squamous cell carcinoma": "SCC", "actinic keratosis": "ACK",
        "seborrheic keratosis": "SEK", "benign": "NEV",
        "mel": "MEL", "bcc": "BCC", "scc": "SCC", "ack": "ACK", "nev": "NEV", "sek": "SEK"
    }

    valid_df['diagnostic'] = valid_df['diagnostic_raw'].astype(str).str.lower().replace(acronym_map).str.upper()

    print("✍️ Synthesizing rich MONET clinical history...")
    valid_df['clinical_history'] = valid_df.apply(build_clinical_history_monet, axis=1)

    final_df = valid_df[['filepath', 'clinical_history', 'diagnostic']].dropna(subset=['diagnostic'])
    final_df = final_df[final_df['diagnostic'] != 'NAN']
    final_df.to_csv(output_csv, index=False)
    print(f"✅ MILK10k Built: {len(final_df)} valid records saved to {output_csv}!")


def process_pad_ufes(raw_dir: str, output_csv: str):
    print(f"\n🔍 Processing PAD-UFES-20 from {raw_dir}...")
    meta_files = glob.glob(f"{raw_dir}/**/metadata.csv", recursive=True)
    if not meta_files: return

    target_csv = max(meta_files, key=os.path.getsize)
    df = pd.read_csv(target_csv)

    _map_column(df, 'sex', ['gender', 'sex'])
    _map_column(df, 'age_approx', ['age', 'patient_age', 'Age'])
    _map_column(df, 'anatom_site_general', ['region', 'anatom_site_general', 'site'])

    df['clinical_history'] = df.apply(
        lambda row: f"{_safe_str(row.get('sex'), 'Patient').capitalize()}, "
                    f"age {_safe_str(row.get('age_approx'), 'unknown')}, "
                    f"presents with a lesion on the {_safe_str(row.get('anatom_site_general'), 'skin').lower()}.",
        axis=1
    )

    all_imgs = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        all_imgs.extend(glob.glob(os.path.join(raw_dir, '**', ext), recursive=True))

    img_map = {os.path.splitext(os.path.basename(p))[0].strip(): p for p in all_imgs}

    df['clean_id'] = df['img_id'].astype(str).apply(lambda x: os.path.splitext(x)[0].strip())
    df['filepath'] = df['clean_id'].map(img_map)

    valid_df = df.dropna(subset=['filepath']).copy()

    valid_df['diagnostic'] = valid_df['diagnostic'].str.upper().replace({"BKL": "SEK", "BOD": "SCC"})
    final_df = valid_df[['filepath', 'clinical_history', 'diagnostic']].dropna(subset=['diagnostic'])
    final_df.to_csv(output_csv, index=False)
    print(f"✅ PAD-UFES-20 Built: {len(final_df)} valid records saved!")


if __name__ == "__main__":
    print("🚀 LAUNCHING MONET-AWARE DATA PIPELINE\n")

    MILK10K_SLUG  = "able23/skincancermilk10k"
    PAD_UFES_SLUG = "mahdavi1202/skin-cancer"

    raw_milk_dir = "./data/raw_milk10k"
    raw_pad_dir  = "./data/raw_pad_ufes"

    download_from_kaggle(MILK10K_SLUG,  raw_milk_dir)
    download_from_kaggle(PAD_UFES_SLUG, raw_pad_dir)

    process_milk10k(raw_dir=raw_milk_dir, output_csv=cfg.paths.milk10k_csv)
    process_pad_ufes(raw_dir=raw_pad_dir, output_csv=cfg.paths.pad_ufes_csv)

    print("\n🏁 Data Pipeline Complete. Ready for dataset.py!")