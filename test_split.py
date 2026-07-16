import sys
from dataset import MultimodalDermatologyDataset
from transformers import AutoTokenizer
from config import cfg

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)

print("Loading train...")
train_ds = MultimodalDermatologyDataset(
    csv_file=cfg.paths.milk10k_csv, img_dir="", tokenizer=tokenizer, split="train"
)

print("Loading val...")
val_ds = MultimodalDermatologyDataset(
    csv_file=cfg.paths.milk10k_csv, img_dir="", tokenizer=tokenizer, split="val"
)

train_lesions = set(train_ds.df['lesion_id'])
val_lesions = set(val_ds.df['lesion_id'])

overlap = train_lesions.intersection(val_lesions)
print(f"Overlap lesions: {len(overlap)}")
print(f"Disjoint: {train_lesions.isdisjoint(val_lesions)}")

if not train_lesions.isdisjoint(val_lesions):
    sys.exit(1)
