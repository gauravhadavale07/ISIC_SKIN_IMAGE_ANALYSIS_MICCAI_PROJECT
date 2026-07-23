import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from config import cfg

def set_seed(seed: int = 42):
    """
    Locks all random number generators to ensure 100% reproducible experiments.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultimodalTrainer:
    """
    Robust PyTorch Training Engine.
    Handles AMP, Gradient Clipping, Early Stopping, and Checkpointing.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, criterion, device, run_name):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.run_name = run_name
        
        # Output paths
        self.save_dir = os.path.join(cfg.paths.checkpoint_dir, run_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # AMP Scaler for mixed precision
        self.scaler = GradScaler(self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda")
        
        # State tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{cfg.train.epochs} [Train]")
        
        for batch in pbar:
            # Move tensors to GPU
            imgs = batch["image"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with Automatic Mixed Precision
            with autocast(device_type=self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda"):
                # Our models return (logits, fused_representation, vis_cls)
                logits, _, _ = self.model(imgs, input_ids, attn_mask)
                loss = self.criterion(logits, labels)
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Unscale gradients to apply clipping safely
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.max_grad_norm)
            
            # Optimizer & Scheduler steps
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100.*correct/total:.2f}%", 'LR': f"{current_lr:.2e}"})
            
        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def _validate_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}/{cfg.train.epochs} [Val]")
        
        for batch in pbar:
            imgs = batch["image"].to(self.device, non_blocking=True)
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attn_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, enabled=cfg.train.use_amp and self.device.type == "cuda"):
                logits, _, _ = self.model(imgs, input_ids, attn_mask)
                loss = self.criterion(logits, labels)
                
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100.*correct/total:.2f}%"})
            
        return total_loss / len(self.val_loader), 100. * correct / total

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Saves model state, optimizer state, and scaler for resume capability."""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'val_loss': val_loss
        }
        
        # Always save the latest epoch
        last_path = os.path.join(self.save_dir, 'last_model.pth')
        torch.save(state, last_path)
        
        # Save explicitly if it's the best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(state, best_path)
            print(f"🌟 New best model saved! (Val Loss: {val_loss:.4f})")

    def fit(self):
        """Main training loop orchestrator."""
        print(f"\n🚀 Starting Training Run: {self.run_name}")
        print("-" * 50)
        
        for epoch in range(1, cfg.train.epochs + 1):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._validate_epoch(epoch)
            
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Early Stopping & Checkpointing Logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_loss, is_best=False)
                print(f"⚠️ Validation loss did not improve. Patience: {self.patience_counter}/{cfg.train.patience}")
                
            if self.patience_counter >= cfg.train.patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}! Model has stopped learning.")
                break
                
        print("\n🏁 Training Complete.")
        print(f"🏆 Best Validation Loss: {self.best_val_loss:.4f}")
        print("-" * 50)