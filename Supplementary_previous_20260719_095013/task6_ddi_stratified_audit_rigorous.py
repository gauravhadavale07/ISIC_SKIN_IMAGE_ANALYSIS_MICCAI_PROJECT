import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F

from config import cfg
from evaluate import Evaluator
from models.image_only import ImageOnlyClassifier
from models.text_only import TextOnlyClassifier
from models.late_fusion import LateFusionClassifier
from models.gmu import GMUClassifier
from models.cross_attention import CrossAttentionT2VClassifier, CrossAttentionV2TClassifier

import warnings
warnings.filterwarnings("ignore")

# Disease mapping based on our 6-class config
DDI_DISEASE_MAPPING = {
    'actinic-keratosis': 'ACK',
    'basal-cell-carcinoma': 'BCC',
    'basal-cell-carcinoma-superficial': 'BCC',
    'basal-cell-carcinoma-nodular': 'BCC',
    'melanoma': 'MEL',
    'melanoma-in-situ': 'MEL',
    'melanoma-acral-lentiginous': 'MEL',
    'nodular-melanoma-(nm)': 'MEL',
    'melanocytic-nevi': 'NEV',
    'blue-nevus': 'NEV',
    'epidermal-nevus': 'NEV',
    'dysplastic-nevus': 'NEV',
    'atypical-spindle-cell-nevus-of-reed': 'NEV',
    'pigmented-spindle-cell-nevus-of-reed': 'NEV',
    'squamous-cell-carcinoma': 'SCC',
    'squamous-cell-carcinoma-in-situ': 'SCC',
    'squamous-cell-carcinoma-keratoacanthoma': 'SCC',
    'seborrheic-keratosis': 'SEK',
    'seborrheic-keratosis-irritated': 'SEK'
}

class DDIDataset(Dataset):
    def __init__(self, df, img_dir, tokenizer, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = 128
        self.label_map = cfg.data.LABEL_MAP

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load Image
        img_name = row['DDI_file']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Text input: Structured unknown string to mimic training distribution
        text = "Age: unknown, Sex: unknown, Anatomical Site: unknown"
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Label mapping
        label_str = row['miccai_class']
        label = self.label_map[label_str]
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def bootstrap_metrics(y_true, y_pred, y_prob, n_classes, n_bootstraps=2000, seed=42):
    """Calculates 95% CI for Accuracy, Macro-F1 and AUROC using bootstrap resampling."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    acc_scores = []
    f1_scores = []
    auroc_scores = []
    
    for _ in range(n_bootstraps):
        indices = rng.choice(n, size=n, replace=True)
        y_t = y_true[indices]
        y_p = y_pred[indices]
        y_pr = y_prob[indices]
        
        present_classes = np.unique(y_t)
        if len(present_classes) > 1:
            try:
                acc = np.mean(y_t == y_p)
                f1 = f1_score(y_t, y_p, average='macro')
                
                auc_sum = 0
                for c in present_classes:
                    y_t_binary = (y_t == c).astype(int)
                    auc_sum += roc_auc_score(y_t_binary, y_pr[:, c])
                auroc = auc_sum / len(present_classes)
                
                acc_scores.append(acc)
                f1_scores.append(f1)
                auroc_scores.append(auroc)
            except ValueError:
                pass
        
    if not acc_scores:
         return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
         
    acc_scores = np.array(acc_scores)
    f1_scores = np.array(f1_scores)
    auroc_scores = np.array(auroc_scores)
    
    acc_mean = np.mean(acc_scores)
    acc_lower = np.percentile(acc_scores, 2.5)
    acc_upper = np.percentile(acc_scores, 97.5)
    
    f1_mean = np.mean(f1_scores)
    f1_lower = np.percentile(f1_scores, 2.5)
    f1_upper = np.percentile(f1_scores, 97.5)
    
    auroc_mean = np.mean(auroc_scores)
    auroc_lower = np.percentile(auroc_scores, 2.5)
    auroc_upper = np.percentile(auroc_scores, 97.5)
    
    return acc_mean, acc_lower, acc_upper, f1_mean, f1_lower, f1_upper, auroc_mean, auroc_lower, auroc_upper

def evaluate_model_on_df(model, df, img_dir, tokenizer, device):
    """Evaluates the model and returns true labels, pred labels, and probabilities to compute CIs."""
    dataset = DDIDataset(df, img_dir, tokenizer, get_eval_transforms())
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=cfg.train.use_amp):
                logits, _, _ = model(imgs, input_ids, attn_mask)
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)
                
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def main():
    print("======================================================================")
    print("TASK 6: SKIN-TONE STRATIFICATION AUDIT ON DDI (RIGOROUS METRICS)")
    print("======================================================================")
    
    # 1. Load and Harmonize DDI Metadata
    metadata_path = "data/ddi/ddidiversedermatologyimages/ddi_metadata.csv"
    img_dir = "data/ddi/ddidiversedermatologyimages/Images"
    
    if not os.path.exists(metadata_path):
        print(f"[ERROR] DDI metadata not found at {metadata_path}")
        return
        
    df = pd.read_csv(metadata_path)
    print(f"Total initial DDI records: {len(df)}")
    
    # Map diseases
    df['miccai_class'] = df['disease'].map(DDI_DISEASE_MAPPING)
    
    # Filter OOD
    df_clean = df.dropna(subset=['miccai_class']).reset_index(drop=True)
    print(f"Total DDI records after filtering to 6 MICCAI classes: {len(df_clean)}")
    
    bins = {
        12: "FST I/II (Light)",
        34: "FST III/IV (Medium)",
        56: "FST V/VI (Dark)"
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(cfg.model.text_backbone)
    n_classes = len(cfg.data.LABEL_MAP)
    
    architectures = {
        "Image-Only": (ImageOnlyClassifier, "ImageOnly"),
        "Late Fusion": (LateFusionClassifier, "Late_Fusion"),
        "GMU": (GMUClassifier, "GMU_Baseline"),
        "Cross-Attention (T->V)": (CrossAttentionT2VClassifier, "Cross-Attention_T→V"),
        "Cross-Attention (V->T)": (CrossAttentionV2TClassifier, "Cross-Attention_V→T")
    }
    
    results_records = []
    
    for arch_name, (ModelClass, prefix) in architectures.items():
        print(f"\nEvaluating {arch_name}...")
        
        # We'll use seed 456 (or 1337)
        ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"{prefix}_seed_456", "best_model.pth")
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"{prefix}_seed_1337", "best_model.pth")
            
        if not os.path.exists(ckpt_path):
            print(f"  [ERROR] Checkpoint not found for {arch_name}, skipping.")
            continue
            
        model = ModelClass().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        
        # 1. Overall metrics
        y_true, y_pred, y_prob = evaluate_model_on_df(model, df_clean, img_dir, tokenizer, device)
        
        # Save raw predictions
        os.makedirs("results", exist_ok=True)
        pred_path = f"results/ddi_raw_preds_{prefix}_seed_456.npz"
        np.savez(pred_path, y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        
        acc, acc_lo, acc_hi, f1, f1_lo, f1_hi, auroc, auroc_lo, auroc_hi = bootstrap_metrics(y_true, y_pred, y_prob, n_classes)
        
        print(f"  Overall: Acc {acc*100:.1f}%, F1 {f1*100:.1f}%, AUROC {auroc:.3f}")
        
        results_records.append({
            "Architecture": arch_name,
            "Skin_Tone_Bin": "Overall",
            "N": len(df_clean),
            "Accuracy (%)": acc * 100,
            "Acc_CI_Lower": acc_lo * 100,
            "Acc_CI_Upper": acc_hi * 100,
            "Macro_F1": f1 * 100,
            "F1_CI_Lower": f1_lo * 100,
            "F1_CI_Upper": f1_hi * 100,
            "AUROC": auroc,
            "AUROC_CI_Lower": auroc_lo,
            "AUROC_CI_Upper": auroc_hi
        })
        
        # 2. Stratified metrics
        for tone_code, tone_name in bins.items():
            df_bin = df_clean[df_clean['skin_tone'] == tone_code]
            if len(df_bin) == 0:
                continue
                
            mask = df_clean['skin_tone'] == tone_code
            y_true_bin = y_true[mask]
            y_pred_bin = y_pred[mask]
            y_prob_bin = y_prob[mask]
            
            bin_acc, bin_acc_lo, bin_acc_hi, bin_f1, bin_f1_lo, bin_f1_hi, bin_auroc, bin_auroc_lo, bin_auroc_hi = bootstrap_metrics(y_true_bin, y_pred_bin, y_prob_bin, n_classes)
            print(f"    {tone_name} (N={len(df_bin)}): Acc {bin_acc*100:.1f}%, F1 {bin_f1*100:.1f}%, AUROC {bin_auroc:.3f}")
            
            results_records.append({
                "Architecture": arch_name,
                "Skin_Tone_Bin": tone_name,
                "N": len(df_bin),
                "Accuracy (%)": bin_acc * 100,
                "Acc_CI_Lower": bin_acc_lo * 100,
                "Acc_CI_Upper": bin_acc_hi * 100,
                "Macro_F1": bin_f1 * 100,
                "F1_CI_Lower": bin_f1_lo * 100,
                "F1_CI_Upper": bin_f1_hi * 100,
                "AUROC": bin_auroc,
                "AUROC_CI_Lower": bin_auroc_lo,
                "AUROC_CI_Upper": bin_auroc_hi
            })
            
    # Save results
    os.makedirs('results', exist_ok=True)
    out_csv = 'results/task6_ddi_stratified_audit_rigorous.csv'
    res_df = pd.DataFrame(results_records)
    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ Rigorous Task 6 complete! Stratified audit results saved to: {out_csv}")

if __name__ == "__main__":
    main()
