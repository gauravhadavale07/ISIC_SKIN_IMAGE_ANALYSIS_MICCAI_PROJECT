import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import MultimodalDermatologyDataset, get_transforms
from config import cfg

tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
dataset = MultimodalDermatologyDataset("./milk10k_clean_val_temp.csv", "./data/raw_milk10k/", tokenizer, get_transforms())
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

start = time.time()
print("Getting first batch...")
batch = next(iter(loader))
print(f"Batch loaded in {time.time() - start:.2f} seconds!")
