import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from config import cfg


SPLIT_SEEDS = [101, 202, 303, 404, 505]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TextRowsDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if text.strip().lower() in ("nan", "none"):
            text = cfg.audit.blank_string
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_milk10k() -> pd.DataFrame:
    df = pd.read_csv(cfg.paths.milk10k_csv)
    df = df[df["diagnostic"].astype(str).str.upper() != "NAN"].copy()
    df["diagnostic"] = df["diagnostic"].astype(str).str.upper().str.strip()
    df["lesion_id"] = df["filepath"].astype(str).apply(
        lambda x: re.search(r"(IL_\d+)", x).group(1) if re.search(r"(IL_\d+)", x) else None
    )
    df = df.dropna(subset=["lesion_id"]).reset_index(drop=True)
    df["label"] = df["diagnostic"].map(cfg.data.LABEL_MAP)
    if df["label"].isna().any():
        missing = sorted(df.loc[df["label"].isna(), "diagnostic"].unique())
        raise ValueError(f"Unknown labels in MILK10k CSV: {missing}")
    df["label"] = df["label"].astype(int)
    return df


def stratified_lesion_split(df: pd.DataFrame, seed: int, val_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    lesion_df = (
        df.groupby("lesion_id")["diagnostic"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    val_lesions = []
    for _, group in lesion_df.groupby("diagnostic", sort=True):
        lesions = group["lesion_id"].to_numpy()
        k = max(1, int(round(val_frac * len(lesions))))
        val_lesions.extend(rng.choice(lesions, size=k, replace=False).tolist())
    val_lesions = np.array(sorted(val_lesions))
    train_lesions = np.array(sorted(set(lesion_df["lesion_id"]) - set(val_lesions)))
    return train_lesions, val_lesions


@torch.no_grad()
def cache_text_embeddings(df: pd.DataFrame, device: torch.device, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone)
    encoder = AutoModel.from_pretrained(cfg.model.text_backbone).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    dataset = TextRowsDataset(
        texts=df["clinical_history"].tolist(),
        labels=df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=cfg.data.max_text_len,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    features, labels = [], []
    for batch in tqdm(loader, desc="Caching BioClinicalBERT CLS embeddings"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        features.append(outputs.last_hidden_state[:, 0, :].detach().cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


class LinearTextHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(cfg.model.text_dim, cfg.model.num_classes)

    def forward(self, x):
        return self.classifier(x)


@dataclass
class SplitResult:
    split_seed: int
    n_train_lesions: int
    n_val_lesions: int
    n_train_images: int
    n_val_images: int
    val_majority_class: str
    val_majority_accuracy: float
    textonly_accuracy: float
    macro_f1: float
    weighted_f1: float
    best_epoch: int
    best_val_loss: float


def train_one_split(
    features: torch.Tensor,
    labels: torch.Tensor,
    df: pd.DataFrame,
    split_seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> SplitResult:
    set_seed(split_seed)
    train_lesions, val_lesions = stratified_lesion_split(df, split_seed)
    train_mask = df["lesion_id"].isin(train_lesions).to_numpy()
    val_mask = df["lesion_id"].isin(val_lesions).to_numpy()
    assert not set(train_lesions).intersection(set(val_lesions))

    x_train, y_train = features[train_mask], labels[train_mask]
    x_val, y_val = features[val_mask], labels[val_mask]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    model = LinearTextHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.train.weight_decay)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.train.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step()
            scheduler.step()

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += float(loss.item()) * yb.size(0)
                val_n += yb.size(0)
        val_loss = val_loss_sum / max(1, val_n)
        print(f"split={split_seed} epoch={epoch} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            logits = model(xb.to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

    y_true = y_val.numpy()
    val_df = df.loc[val_mask].copy()
    majority_class = val_df["diagnostic"].value_counts().idxmax()
    majority_acc = float((val_df["diagnostic"] == majority_class).mean() * 100.0)

    return SplitResult(
        split_seed=split_seed,
        n_train_lesions=int(len(train_lesions)),
        n_val_lesions=int(len(val_lesions)),
        n_train_images=int(train_mask.sum()),
        n_val_images=int(val_mask.sum()),
        val_majority_class=str(majority_class),
        val_majority_accuracy=majority_acc,
        textonly_accuracy=float(accuracy_score(y_true, preds) * 100.0),
        macro_f1=float(f1_score(y_true, preds, average="macro", zero_division=0) * 100.0),
        weighted_f1=float(f1_score(y_true, preds, average="weighted", zero_division=0) * 100.0),
        best_epoch=int(best_epoch),
        best_val_loss=float(best_loss),
    )


def summarize(results: List[SplitResult]) -> Dict[str, object]:
    acc = np.array([r.textonly_accuracy for r in results], dtype=float)
    maj = np.array([r.val_majority_accuracy for r in results], dtype=float)
    diff = acc - maj

    def mean_ci(x):
        if len(x) < 2:
            return float(x.mean()), float("nan"), float("nan")
        sem = stats.sem(x)
        h = sem * stats.t.ppf(0.975, len(x) - 1)
        return float(x.mean()), float(x.mean() - h), float(x.mean() + h)

    t_stat, p_val = stats.ttest_1samp(diff, 0.0)
    return {
        "n_splits": len(results),
        "split_seeds": [r.split_seed for r in results],
        "textonly_accuracy_mean_ci": mean_ci(acc),
        "majority_accuracy_mean_ci": mean_ci(maj),
        "textonly_minus_majority_pp_mean_ci": mean_ci(diff),
        "textonly_minus_majority_t": float(t_stat),
        "textonly_minus_majority_p": float(p_val),
        "interpretation": (
            "Repeated random stratified lesion-disjoint splits test whether the "
            "MILK10k Text-Only baseline exceeds validation-set majority prediction "
            "after lesion leakage is removed."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=cfg.train.epochs)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=cfg.train.learning_rate)
    parser.add_argument("--seeds", type=int, nargs="*", default=SPLIT_SEEDS)
    args = parser.parse_args()

    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Task 30 Text-Only random lesion-disjoint CV on {device}")
    print(f"Split seeds: {args.seeds}")
    print(f"Training head epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}")

    df = load_milk10k()
    features, labels = cache_text_embeddings(df, device=device, batch_size=args.embed_batch_size)

    results = []
    for seed in args.seeds:
        result = train_one_split(
            features=features,
            labels=labels,
            df=df,
            split_seed=seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        print(result)
        results.append(result)

    rows = [asdict(r) for r in results]
    out_csv = os.path.join(cfg.paths.results_dir, "task30_textonly_random_lesion_cv.csv")
    out_json = os.path.join(cfg.paths.results_dir, "task30_textonly_random_lesion_cv_summary.json")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = summarize(results)
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)

    print("\nTask 30 summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out_csv}")
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
