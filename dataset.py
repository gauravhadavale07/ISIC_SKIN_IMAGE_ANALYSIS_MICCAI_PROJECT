import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer
from torchvision import transforms
from config import cfg


def get_transforms(*args, **kwargs):
    """
    Returns standard ImageNet normalization transforms.
    Accepts *args and **kwargs to safely handle if run_experiment.py passes 'train' or 'val'.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class MultimodalDermatologyDataset(Dataset):
    # FIX: added a `tokenizer` parameter (an already-instantiated HF tokenizer
    # object). Previously run_experiment.py called this constructor with
    # `tokenizer=tokenizer`, but the only matching parameter name here was
    # `tokenizer_name`, so the real tokenizer object was silently swallowed by
    # **kwargs and a BRAND NEW tokenizer was instantiated from the hardcoded
    # default string on every single dataset split (train/val/test) instead of
    # reusing the one tokenizer loaded once in run_experiment.py. 
    def __init__(self, csv_file, img_dir=None, tokenizer=None,
                 tokenizer_name="emilyalsentzer/Bio_ClinicalBERT", transform=None,
<<<<<<< HEAD
                 max_length=128, split="all", **kwargs):
=======
                 max_length=128, **kwargs):
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
        """
        Multimodal dataset for dermoscopic images and clinical history text.

        Args:
<<<<<<< HEAD
            split: 'train', 'val', or 'all'. Used to perform strict lesion-disjoint splits.
=======
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
            tokenizer: an already-instantiated HuggingFace tokenizer. If
                provided, it is used directly (preferred — avoids reloading
                the tokenizer once per dataset split, and guarantees every
                split shares the exact same vocabulary/config).
            tokenizer_name: fallback HF model string, only used if
                `tokenizer` is not provided.
        """
<<<<<<< HEAD
        import re
=======
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
        print(f"📦 Loading dataset from: {csv_file}")
        self.df = pd.read_csv(csv_file)

        # =================================================================
        # --- DEFENSIVE DATA SCRUBBING ---
        # Filter out literal 'NAN' strings that bypassed dropna() during preprocessing
        self.df = self.df[self.df['diagnostic'].astype(str).str.upper() != 'NAN']
        self.df = self.df.reset_index(drop=True) 
<<<<<<< HEAD

        # --- LESION-DISJOINT SPLITTING LOGIC ---
        if split in ["train", "val"]:
            # Extract lesion ID from filepath
            self.df['lesion_id'] = self.df['filepath'].apply(
                lambda x: re.search(r'(IL_\d+)', str(x)).group(1) if re.search(r'(IL_\d+)', str(x)) else None
            )
            # Filter out any rows without a valid lesion ID
            self.df = self.df.dropna(subset=['lesion_id'])
            
            # Group all lesions and determine the 85/15 boundary
            all_lesions = sorted(self.df['lesion_id'].unique())
            n_val_lesions = int(0.15 * len(all_lesions))
            val_lesion_set = set(all_lesions[-n_val_lesions:])
            
            # Apply the split
            if split == "val":
                self.df = self.df[self.df['lesion_id'].isin(val_lesion_set)]
            elif split == "train":
                self.df = self.df[~self.df['lesion_id'].isin(val_lesion_set)]
                
            self.df = self.df.reset_index(drop=True)
            print(f"   ↳ {split.upper()} SPLIT: {len(self.df)} images (lesion-disjoint)")
=======
>>>>>>> 0555f8e631286ee37d47a1d638ba93ce7e343a20
        # =================================================================

        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        # FIX: label_map now sourced from cfg.data.LABEL_MAP (single source of
        # truth) instead of being redefined inline here.
        self.label_map = cfg.data.LABEL_MAP

        if transform is None:
            self.transform = get_transforms()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ==========================================
        # 1. VISION PATHWAY
        # ==========================================
        # We implicitly trust 'filepath' because fast_build.py verified it
        img_path = row['filepath']

        # .convert('RGB') is critical to prevent crashes on RGBA/Grayscale images
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # ==========================================
        # 2. TEXT PATHWAY (CLINICAL HISTORY)
        # ==========================================
        text = str(row['clinical_history'])
        if text.strip().lower() in ('nan', 'none'):
            text = cfg.audit.blank_string
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Squeeze removes the extra batch dimension added by return_tensors='pt'
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # ==========================================
        # 3. LABEL EXTRACTION
        # ==========================================
        label_str = str(row['diagnostic']).strip().upper()
        label_idx = self.label_map[label_str]
        label = torch.tensor(label_idx, dtype=torch.long)

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }