import os, sys, torch
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from torch.utils.data import DataLoader, random_split
from models.image_only import ImageOnlyClassifier
from evaluate import Evaluator

def build_milk10k_val_loader(seed):
    dataset = MultimodalDermatologyDataset(
        csv_file="./milk10k_train.csv",
        img_dir="./data/raw_milk10k/",
        transform=get_transforms(),
        is_ood=False
    )
    n = len(dataset)
    train_size = int(0.85 * n)
    val_size = n - train_size
    gen = torch.Generator().manual_seed(seed)
    _, val_sub = random_split(dataset, [train_size, val_size], generator=gen)
    return DataLoader(val_sub, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
accs = []
for seed in [456, 789, 1337]:
    model = ImageOnlyClassifier().to(device)
    ckpt = torch.load(f"./checkpoints/ImageOnly_seed_{seed}/best_model.pth", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    val_loader = build_milk10k_val_loader(seed)
    acc = Evaluator(model, device).evaluate(val_loader)['Accuracy']
    accs.append(acc)
    print(f"Seed {seed}: {acc*100:.2f}%")
print(f"Mean: {sum(accs)/len(accs)*100:.2f}%")
