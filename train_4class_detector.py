"""
FoodGuard 4-Class Detector Training Script
===========================================

Improvements over baseline:
  - Focal Loss        : focuses on hard/confusing examples
  - Degradation Augs  : JPEG, blur, noise to avoid compression overfitting
  - Gradient Clipping : prevents NaN spikes with AMP + accumulation
  - EMA Weights       : smoother model → better FPR at inference

Classes (ImageFolder alphabetical order):
  [0] compressed_ai
  [1] edited_ai
  [2] perfect_ai
  [3] real

Target: ≤5% FPR on real images
"""

import io
import os
import json
import random
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image, ImageFilter

import timm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    # Paths
    DATA_ROOT = Path("e:/BML/Semester-VI/Prj-3/dataset_4class")
    CHECKPOINT_DIR = Path("checkpoints/food_detector")
    
    # Model
    MODEL_NAME = "efficientnet_b3"
    NUM_CLASSES = 4
    IMAGE_SIZE = 512
    
    # Training
    BATCH_SIZE = 16  # Physical batch size (fits in 12GB VRAM)
    ACCUMULATION_STEPS = 2  # Effective batch size = 16 * 2 = 32
    NUM_WORKERS = 4    # Windows paging file limit — can't safely spawn more PyTorch workers
    USE_COMPILE = False # torch.compile() disabled — Triton backend not supported on Windows
    EPOCHS = 30
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 3  # Linear LR warmup
    LABEL_SMOOTHING = 0.1
    DROPOUT = 0.3
    GRAD_CLIP = 1.0        # Max gradient norm (prevents NaN with AMP)
    EMA_DECAY  = 0.9998    # EMA weight decay (higher = slower but smoother)
    
    # Early stopping
    EARLY_STOP_PATIENCE = 5
    
    # Class weights — MUST match ImageFolder alphabetical index order:
    #   [0] compressed_ai  5,000 images  → highest weight (smallest class)
    #   [1] edited_ai       8,000 images  → medium-high weight
    #   [2] perfect_ai     11,173 images  → near-baseline weight
    #   [3] real           12,000 images  → slight boost to reduce FPR on real
    # Inverse-frequency: w_i = max_count / count_i  (then normalised)
    CLASS_WEIGHTS = torch.tensor([2.4, 1.5, 1.1, 1.5])  # [compressed_ai, edited_ai, perfect_ai, real]
    
    # FPR target
    TARGET_FPR = 0.05
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Mixed precision

Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Class names — MUST match ImageFolder alphabetical order!
# Verified against: ImageFolder(dataset_4class/train).classes
CLASS_NAMES = ["compressed_ai", "edited_ai", "perfect_ai", "real"]
REAL_CLASS_INDEX = 3  # 'real' is last alphabetically → index 3

# =============================================================================
# FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples so the model focuses on
    hard edge-cases (e.g. compressed_ai that looks almost real).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 is the standard value from the original paper.
    Class weights (alpha) are passed in to handle imbalance.
    Label smoothing is baked in via cross_entropy.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight          = weight
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # Standard cross-entropy gives log(p_t)
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        # p_t = exp(-CE)
        pt = torch.exp(-ce)
        focal_weight = (1.0 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# =============================================================================
# EMA (Exponential Moving Average of model weights)
# =============================================================================

class ModelEMA:
    """Maintains a shadow copy of model weights as an EMA.
    Use ema.apply() before evaluation, ema.restore() to go back to training.
    """
    def __init__(self, model, decay=0.9998):
        self.decay  = decay
        self.shadow = deepcopy(model.state_dict())
        # Move shadow to same device
        for k in self.shadow:
            self.shadow[k] = self.shadow[k].float()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.float()

    def apply(self, model):
        """Swap model weights with EMA shadow (call before eval)."""
        self._backup = deepcopy(model.state_dict())
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()})

    def restore(self, model):
        """Restore original training weights (call after eval)."""
        model.load_state_dict(self._backup)

# =============================================================================
# DATA LOADING
# =============================================================================

class DegradationAugment:
    """Randomly applies forensic-relevant degradations to PIL images.

    AI detection depends on high-frequency noise patterns.  Standard spatial
    augmentations (flip/rotate) do NOT change these patterns.  These transforms
    teach the model the difference between:
      - Real camera sensor noise
      - AI diffusion model noise
      - JPEG blocking artifacts at various quality levels
    """
    def __call__(self, img):
        r = random.random()
        if r < 0.25:
            # JPEG re-compression at random quality (40-95)
            quality = random.randint(40, 95)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            img = Image.open(buf).copy()
        elif r < 0.40:
            # Gaussian blur (simulates slight focus loss / social media processing)
            radius = random.uniform(0.3, 1.2)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        elif r < 0.50:
            # Unsharp mask (sharpening, common AI post-processing)
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
        # else: no degradation (50% of the time — keep originals too)
        return img


def get_transforms(is_train=True):
    """Transforms with degradation augmentations for AI forensics training."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            # Spatial augmentations
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
            # Forensic degradation augmentations (key for AI detection)
            DegradationAugment(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders():
    """Create train/val/test loaders from folder structure."""
    train_dataset = ImageFolder(
        Config.DATA_ROOT / "train",
        transform=get_transforms(is_train=True)
    )
    val_dataset = ImageFolder(
        Config.DATA_ROOT / "val",
        transform=get_transforms(is_train=False)
    )
    test_dataset = ImageFolder(
        Config.DATA_ROOT / "test",
        transform=get_transforms(is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS,
        pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS,
        pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS,
        pin_memory=True, persistent_workers=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# =============================================================================
# MODEL
# =============================================================================

def create_model():
    """Load EfficientNet-B3 via timm with dropout for regularization."""
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
        drop_rate=Config.DROPOUT,
    )
    model = model.to(Config.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Dropout: {Config.DROPOUT}")
    print(f"Trainable parameters: {num_params:,}")

    # torch.compile() — fuses ops into optimised Triton kernels for the GPU
    # 'reduce-overhead' minimises Python interpreter overhead per step
    if Config.USE_COMPILE:
        print("Compiling model with torch.compile(mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled. First epoch will be slower (compilation step).")

    return model

# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, ema=None):
    """Train one epoch with AMP, gradient accumulation, clipping, and EMA."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)

        # Forward pass with Mixed Precision
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / Config.ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        # Step optimizer every ACCUMULATION_STEPS batches
        if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
            # Unscale → clip → step  (gradient clipping MUST happen after unscale)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # Update EMA shadow weights after each optimiser step
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * Config.ACCUMULATION_STEPS
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # Handle remaining gradients at end of epoch
    if (batch_idx + 1) % Config.ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if ema is not None:
            ema.update(model)

    return total_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    """Validate model (Step 8)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100.0 * correct / total

# =============================================================================
# EVALUATION (Step 9 & 10)
# =============================================================================

@torch.no_grad()
def evaluate_fpr(model, loader):
    """Compute confusion matrix and FPR on real class.
    
    Returns:
        confusion_matrix, fpr_on_real, accuracy
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in loader:
        images = images.to(Config.DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    # FPR on real = (Real predicted as AI) / (Total Real)
    # REAL_CLASS_INDEX = 3 (ImageFolder alphabetical: compressed_ai=0, edited_ai=1, perfect_ai=2, real=3)
    real_idx = REAL_CLASS_INDEX
    total_real = cm[real_idx].sum()
    real_predicted_as_ai = total_real - cm[real_idx, real_idx]
    fpr = real_predicted_as_ai / total_real if total_real > 0 else 0.0
    
    accuracy = np.trace(cm) / cm.sum()
    
    return cm, fpr, accuracy


@torch.no_grad()
def calibrate_threshold(model, loader):
    """Step 10: Find threshold for real class that achieves ≤5% FPR.
    
    Returns:
        optimal_threshold, fpr_at_threshold
    """
    model.eval()
    all_probs_real = []
    all_is_real = []
    
    for images, labels in loader:
        images = images.to(Config.DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        prob_real = probs[:, REAL_CLASS_INDEX]  # Use correct real class index
        
        all_probs_real.extend(prob_real.cpu().numpy())
        all_is_real.extend((labels == REAL_CLASS_INDEX).numpy())
    
    all_probs_real = np.array(all_probs_real)
    all_is_real = np.array(all_is_real)
    
    # Try different thresholds
    thresholds = np.linspace(0.5, 0.99, 100)
    best_threshold = 0.85
    best_fpr = 1.0
    
    for thresh in thresholds:
        # Predict real if prob_real > thresh
        pred_real = all_probs_real > thresh
        
        # FPR = (True real predicted as AI) / (Total true real)
        true_real_mask = all_is_real == 1
        if true_real_mask.sum() == 0:
            continue
        
        false_ai = (~pred_real) & true_real_mask
        fpr = false_ai.sum() / true_real_mask.sum()
        
        if fpr <= Config.TARGET_FPR and fpr < best_fpr:
            best_threshold = thresh
            best_fpr = fpr
    
    return best_threshold, best_fpr


def save_confusion_matrix(cm, path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plt.yticks(tick_marks, CLASS_NAMES)
    
    # Annotate cells
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_training_curves(history, path):
    """Plot and save training curves (loss, accuracy, FPR)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[1].set_title('Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # FPR
    axes[2].plot(epochs, [f*100 for f in history['val_fpr']], 'g-', label='Val FPR')
    axes[2].axhline(y=Config.TARGET_FPR*100, color='r', linestyle='--', label=f'Target ({Config.TARGET_FPR*100}%)')
    axes[2].set_title('FPR on Real Class (%)')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    print("=" * 60)
    print("FOODGUARD 4-CLASS DETECTOR TRAINING")
    print("=" * 60)
    
    # Step 1: Check GPU
    print(f"\nDevice: {Config.DEVICE}")
    if Config.DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"AMP: {Config.USE_AMP}")
    
    # Step 2 & 3: Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Step 4: Create model
    print("\n" + "=" * 60)
    print("MODEL")
    print("=" * 60)
    model = create_model()
    
    # Focal Loss with class weights (replaces CrossEntropyLoss)
    criterion = FocalLoss(
        weight=Config.CLASS_WEIGHTS.to(Config.DEVICE),
        gamma=2.0,
        label_smoothing=Config.LABEL_SMOOTHING
    )
    print(f"Loss: FocalLoss (gamma=2.0, label_smoothing={Config.LABEL_SMOOTHING})")

    # EMA — maintains a shadow copy of weights for stable evaluation
    ema = ModelEMA(model, decay=Config.EMA_DECAY)
    print(f"EMA decay: {Config.EMA_DECAY}")
    print(f"Grad clip: {Config.GRAD_CLIP}")
    
    # Optimizer & Scheduler with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Warmup + Cosine Annealing scheduler
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, total_iters=Config.WARMUP_EPOCHS
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS - Config.WARMUP_EPOCHS
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[Config.WARMUP_EPOCHS]
    )
    
    # Mixed Precision Scaler
    scaler = GradScaler('cuda') if Config.USE_AMP else None
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_fpr": []
    }
    
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop with early stopping
    print("\n" + "=" * 60)
    print("TRAINING")
    print(f"Effective batch size: {Config.BATCH_SIZE * Config.ACCUMULATION_STEPS}")
    print(f"Focal Loss gamma:    2.0")
    print(f"Grad clip max_norm:  {Config.GRAD_CLIP}")
    print(f"EMA decay:           {Config.EMA_DECAY}")
    print(f"Early stopping:      patience={Config.EARLY_STOP_PATIENCE}")
    print("=" * 60)
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, ema)
        # Evaluate using EMA weights for a stable val metric
        ema.apply(model)
        val_loss, val_acc = validate(model, val_loader, criterion)
        cm, val_fpr, _ = evaluate_fpr(model, val_loader)
        ema.restore(model)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # (FPR already computed above using EMA weights)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_fpr"].append(val_fpr)
        
        # Overfitting gap
        gap = train_acc - val_acc
        
        print(f"Epoch {epoch:2d}/{Config.EPOCHS} | "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}% | "
              f"FPR: {val_fpr*100:.2f}% | "
              f"Gap: {gap:.1f}% | "
              f"LR: {current_lr:.6f}")
        
        # Save best model (by val loss for better generalization)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            # Save both raw and EMA weights
            torch.save(model.state_dict(), Config.CHECKPOINT_DIR / "best.pth")
            torch.save(ema.shadow, Config.CHECKPOINT_DIR / "best_ema.pth")
            print(f"  [OK] Best model saved (val_loss: {val_loss:.4f}) [raw + EMA]")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"  [FAIL] Early stopping triggered (patience={Config.EARLY_STOP_PATIENCE})")
                break
    
    # Threshold calibration
    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION")
    print("=" * 60)
    # Load EMA weights for calibration (more stable than raw weights)
    ema_state = torch.load(Config.CHECKPOINT_DIR / "best_ema.pth")
    model.load_state_dict({k: v.to(Config.DEVICE) for k, v in ema_state.items()})
    threshold, fpr = calibrate_threshold(model, val_loader)
    print(f"Optimal Threshold: {threshold:.3f}")
    print(f"FPR at Threshold:  {fpr*100:.2f}%")
    
    # Test on unseen data
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    cm_test, fpr_test, acc_test = evaluate_fpr(model, test_loader)
    print(f"Test Accuracy: {acc_test*100:.2f}%")
    print(f"Test FPR (Real): {fpr_test*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm_test)
    
    # Save confusion matrix
    save_confusion_matrix(cm_test, Config.CHECKPOINT_DIR / "confusion_matrix.png")
    
    # Save final model
    # Save final EMA model as the deployment checkpoint
    torch.save(model.state_dict(), Config.CHECKPOINT_DIR / "food_ai_detector.pth")
    
    # Save training curves
    save_training_curves(history, Config.CHECKPOINT_DIR / "training_curves.png")
    
    # Save metadata
    metadata = {
        "model": Config.MODEL_NAME,
        "num_classes": Config.NUM_CLASSES,
        "image_size": Config.IMAGE_SIZE,
        "threshold": float(threshold),
        "test_accuracy": float(acc_test),
        "test_fpr": float(fpr_test),
        "class_names": CLASS_NAMES,
        "real_class_index": REAL_CLASS_INDEX,
        "dropout": Config.DROPOUT,
        "label_smoothing": Config.LABEL_SMOOTHING,
        "effective_batch_size": Config.BATCH_SIZE * Config.ACCUMULATION_STEPS,
        "epochs_trained": epoch,
        "best_val_loss": float(best_val_loss),
        "trained_at": datetime.now().isoformat()
    }
    with open(Config.CHECKPOINT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Epochs trained:    {epoch}")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy:     {acc_test*100:.2f}%")
    print(f"Test FPR (Real):   {fpr_test*100:.2f}%")
    print(f"Train-Val Gap:     {history['train_acc'][-1] - history['val_acc'][-1]:.1f}%")
    
    if fpr_test <= Config.TARGET_FPR:
        print(f"[OK] TARGET ACHIEVED: FPR ≤ {Config.TARGET_FPR*100}%")
    else:
        print(f"[FAIL] TARGET MISSED: FPR > {Config.TARGET_FPR*100}%")
    
    print(f"\nModel saved: {Config.CHECKPOINT_DIR / 'food_ai_detector.pth'}")
    print(f"Threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
