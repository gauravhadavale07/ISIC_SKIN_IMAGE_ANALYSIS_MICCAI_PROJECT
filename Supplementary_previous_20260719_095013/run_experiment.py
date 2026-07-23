# run_experiment.py
import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd # Import pandas to check CSV contents

# Import our custom configuration and pipeline components
from config import cfg
from dataset import MultimodalDermatologyDataset, get_transforms
from transforms import get_train_transforms, get_eval_transforms
from trainer import MultimodalTrainer, set_seed
from evaluate import Evaluator
from counterfactual import CounterfactualAuditor
from cka import CKAAuditor
from statistical_analyzer import StatisticalAnalyzer  # <--- FIXED IMPORT

# Import Architectures
from models.late_fusion import LateFusionClassifier
from models.cross_attention import CrossAttentionClassifier, CrossAttentionV2TClassifier, CrossAttentionT2VClassifier
from models.gmu import GMUClassifier


def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def checkpoint_is_compatible(model, checkpoint_path, device) -> bool:
    if not os.path.exists(checkpoint_path):
        return False
    try:
        load_checkpoint(model, checkpoint_path, device)
        return True
    except RuntimeError as e:
        print(f"⚠️  Incompatible checkpoint at {checkpoint_path}: {e}")
        return False


def main_logic():
    print("🚀 INITIALIZING MICCAI MULTI-SEED EXPERIMENT PIPELINE")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Compute Device: {device}")
    
    # Initialize the global statistical aggregator
    stats_analyzer = StatisticalAnalyzer()
    
    # Load the tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_backbone, clean_up_tokenization_spaces=True)
    
    # Load the pure OOD Test Set (PAD-UFES-20)
    print("\n📦 Loading OOD Evaluation Dataset (PAD-UFES-20)...")
    test_ds = MultimodalDermatologyDataset(
        csv_file=cfg.paths.pad_ufes_csv,
        img_dir="",  # FIX: Prevent duplicate path generation
        tokenizer=tokenizer,
        transform=get_eval_transforms()
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Define the architectures we want to compare
    architectures = {
        "Late Fusion": LateFusionClassifier,
        "GMU Baseline": GMUClassifier,
        "Cross-Attention V→T": CrossAttentionV2TClassifier,
        "Cross-Attention T→V": CrossAttentionT2VClassifier
    }

    # =========================================================================
    # THE MASTER SEED LOOP
    # =========================================================================
    for seed in cfg.seeds:
        print("\n" + "=" * 65)
        print(f"🌱 COMMENCING RUN WITH SEED: {seed}")
        print("=" * 65)
        
        # 1. Lock down all randomness for this run
        set_seed(seed)
        
        # 2. FIX: Prevent the Transform Leak and Ensure Data Exists
        # We instantiate TWO distinct dataset objects to ensure our training 
        # augmentations aren't overwritten by our validation transforms.
        
        # Check if milk10k_train.csv exists and has data
        milk10k_csv_path = cfg.paths.milk10k_csv
        if not os.path.exists(milk10k_csv_path):
            raise FileNotFoundError(f"milk10k_train.csv not found at {milk10k_csv_path}. Please run prepare_data.py or fast_build.py first.")
        
        try:
            milk10k_df = pd.read_csv(milk10k_csv_path)
            if milk10k_df.empty:
                raise ValueError(f"{milk10k_csv_path} is empty. Please ensure data preparation was successful.")
            if 'diagnostic' not in milk10k_df.columns or milk10k_df['diagnostic'].isnull().all():
                 # This check is more robust than just checking for empty, ensuring the crucial column exists and has values
                 raise ValueError(f"{milk10k_csv_path} does not contain valid diagnostic information or is missing the 'diagnostic' column. Please check data preparation.")
        except Exception as e:
            print(f"Error reading or validating {milk10k_csv_path}: {e}")
            print("Attempting to use fast_build.py output as a fallback...")
            # If prepare_data.py failed, fast_build.py might have created it.
            # This is a safety net, but ideally prepare_data.py should be fixed.
            # For now, we assume fast_build.py works and creates the file.
            # If fast_build.py also failed, the FileNotFoundError above would have already triggered.
            # If the file exists but is malformed in a way pandas can't read, it's a deeper issue.
            # We re-raise if it's still problematic.
            raise e 
        
        # Instantiate datasets only if the CSV is valid and non-empty
        train_ds = MultimodalDermatologyDataset(
            csv_file=milk10k_csv_path,
            img_dir="",  # FIX: Prevent duplicate path generation
            tokenizer=tokenizer,
            transform=get_train_transforms(),
            split="train"
        )
        
        val_ds = MultimodalDermatologyDataset(
            csv_file=milk10k_csv_path,
            img_dir="",  # FIX: Prevent duplicate path generation
            tokenizer=tokenizer,
            transform=get_eval_transforms(),
            split="val"
        )
        
        # Ensure train_ds has samples before calculating split sizes
        if len(train_ds) == 0:
             raise ValueError(f"Dataset loaded from {milk10k_csv_path} resulted in 0 samples after initialization. Please check the CSV file and dataset loading logic.")
        
        # DataLoader creation is now safe
        train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0, pin_memory=True)

        # =====================================================================
        # THE ARCHITECTURE LOOP
        # =====================================================================
        for model_name, ModelClass in architectures.items():
            run_identifier = f"{model_name.replace(' ', '_')}_seed_{seed}"
            run_key = f"{seed}:{model_name}"

            if stats_analyzer.is_complete(run_key):
                print(f"\n⏭️  Skipping {model_name} (Seed {seed}) — already completed.")
                continue

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    print(f"\n⚙️  Initializing {model_name} (Seed {seed}) [Attempt {attempt}/{max_retries}]...")
                    
                    model = ModelClass()
                    best_model_path = os.path.join(cfg.paths.checkpoint_dir, run_identifier, 'best_model.pth')
                    
                    # Extract only trainable parameters (Fusion layers & Classifier)
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    
                    optimizer = torch.optim.AdamW(
                        trainable_params, 
                        lr=cfg.train.learning_rate, 
                        weight_decay=cfg.train.weight_decay
                    )
                    
                    # Calculate total training steps for the linear warmup scheduler
                    total_steps = len(train_loader) * cfg.train.epochs
                    warmup_steps = int(cfg.train.warmup_ratio * total_steps)
                    
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer, 
                        num_warmup_steps=warmup_steps, 
                        num_training_steps=total_steps
                    )
                    
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    # -------------------------------------------------------------
                    # PHASE 1: TRAINING
                    # -------------------------------------------------------------
                    trainer = MultimodalTrainer(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        criterion=criterion,
                        device=device,
                        run_name=run_identifier
                    )
                    
                    if checkpoint_is_compatible(model, best_model_path, device):
                        print(f"📂 Found compatible checkpoint at {best_model_path} — skipping training.")
                    else:
                        if os.path.exists(best_model_path):
                            print("♻️  Removing stale/incompatible checkpoint before retraining.")
                            os.remove(best_model_path)
                        trainer.fit()
                    
                    # -------------------------------------------------------------
                    # PHASE 2: OOD EVALUATION & AUDITING
                    # -------------------------------------------------------------
                    load_checkpoint(model, best_model_path, device)
                    
                    print(f"\n🩺 Commencing Zero-Shot OOD Audit on PAD-UFES-20...")
                    
                    # A. Standard Clinical Metrics
                    evaluator = Evaluator(model, device)
                    std_results = evaluator.evaluate(test_loader)
                    
                    # B. Counterfactual Semantic Audit
                    cf_auditor = CounterfactualAuditor(model, tokenizer, device)
                    cf_results = cf_auditor.run_audit(test_loader)
                    
                    # C. Latent Space Geometric Audit (CKA)
                    cka_auditor = CKAAuditor(model, device)
                    cka_results = cka_auditor.run_audit(test_loader)
                    
                    # Combine all metrics into a single dictionary
                    combined_metrics = {**std_results, **cf_results, **cka_results}
                    
                    # Save to global statistics tracker
                    stats_analyzer.add_run(model_name, combined_metrics, run_key)
                    evaluator.print_report(std_results, prefix=f"{model_name} (Seed {seed})")
                    cf_auditor.print_report(cf_results)
                    cka_auditor.print_report(cka_results, model_name=model_name)
                    
                    # Success - break out of retry loop
                    break
                    
                except Exception as e:
                    import traceback
                    print(f"\n❌ ERROR during {model_name} (Seed {seed}) on attempt {attempt}: {e}")
                    traceback.print_exc()
                    if attempt == max_retries:
                        print(f"⚠️ PERMANENT FAILURE for {model_name} (Seed {seed}) after {max_retries} attempts. Skipping.")
                        # Could log to a failure file here if needed
                    else:
                        print("⏳ Waiting 30 seconds before retrying...")
                        import time
                        time.sleep(30)

        # Print intermediate results after each seed
        print(f"\n{'='*65}")
        print(f"📊 INTERMEDIATE RESULTS AFTER SEED {seed}")
        print(f"{'='*65}")
        stats_analyzer.print_report()

    # =========================================================================
    # THE FINAL STATISTICAL REPORT
    # =========================================================================
    print(f"\n{'='*65}")
    print(f"🏁 FINAL MULTI-SEED STATISTICAL REPORT (ALL SEEDS)")
    print(f"{'='*65}")
    stats_analyzer.print_report()

import modal
import modal.mount

app = modal.App("miccai-multimodal-pipeline")

vol_checkpoints = modal.Volume.from_name("miccai-checkpoints", create_if_missing=True)
vol_results = modal.Volume.from_name("miccai-results", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch", "torchvision", "transformers", "pandas", "scikit-learn", "timm", "scipy", "pillow", "tqdm", "networkx"
).workdir("/root/project").add_local_dir(
    "/tmp/miccai_code_fresh", 
    remote_path="/root/project"
).add_local_dir(
    "data", 
    remote_path="/root/project/data"
).add_local_file(
    "milk10k_train.csv",
    remote_path="/root/project/milk10k_train.csv"
).add_local_file(
    "pad_ufes_20_test.csv",
    remote_path="/root/project/pad_ufes_20_test.csv"
)

@app.function(
    gpu="H200",
    image=image,
    volumes={
        "/root/project/checkpoints": vol_checkpoints,
        "/root/project/results": vol_results
    },
    timeout=86400
)
def run_heavy_workload():
    main_logic()

@app.local_entrypoint()
def main():
    run_heavy_workload.remote()

if __name__ == "__main__":
    modal.runner.deploy_stub = lambda *args, **kwargs: None # just in case
    # Execution is handled by modal run, so we don't need to call main() directly here,
    # but having it at the bottom is standard.
