import torch
import time
from config import cfg
from dataset import MultimodalDermatologyDataset
from torch.utils.data import DataLoader
from transforms import get_train_transforms
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone, clean_up_tokenization_spaces=True)
ds = MultimodalDermatologyDataset(
    csv_file=cfg.paths.milk10k_csv,
    img_dir="",
    tokenizer=tokenizer,
    transform=get_train_transforms()
)
loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=2)

print("Starting benchmark...")
start = time.time()
for i, batch in enumerate(loader):
    # just load data
    if i == 50:
        break
end = time.time()
print(f"Data loading it/s: {50 / (end - start)}")
