"""
FoodGuard 4-Class Detector Training Script
===========================================

Following the 12-step prototype specification for enterprise-grade training.

Classes:
  0: real
  1: perfect_ai
  2: compressed_ai
  3: edited_ai

Target: ≤5% FPR on real images
"""

import os
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import timm
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
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
    BATCH_SIZE = 16  # Adjust based on 12GB VRAM
    EPOCHS = 20
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    
    # Class weights (1.2 for real to reduce FPR)
    CLASS_WEIGHTS = torch.tensor([1.2, 1.0, 1.0, 1.0])
    
    # FPR target
    TARGET_FPR = 0.05
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Mixed precision

Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Class names (must match folder order)
CLASS_NAMES = ["real", "perfect_ai", "compressed_ai", "edited_ai"]

# =============================================================================
# DATA LOADING
# =============================================================================

def get_transforms(is_train=True):
    """Get transforms following Step 3 spec: minimal augmentation."""
    if is_train:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1),  # Slight jitter only
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
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
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

# =============================================================================
# MODEL
# =============================================================================

def create_model():
    """Step 4: Load EfficientNet-B3 via timm."""
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=True,
        num_classes=Config.NUM_CLASSES
    )
    model = model.to(Config.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Trainable parameters: {num_params:,}")
    
    return model

# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler):
    """Train for one epoch with AMP (Step 5 & 8)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Step 5: Mixed Precision
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
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
        
        with autocast():
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
    real_idx = 0
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
        prob_real = probs[:, 0]  # Probability of real class
        
        all_probs_real.extend(prob_real.cpu().numpy())
        all_is_real.extend((labels == 0).numpy())
    
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
    
    # Step 6: Loss with class weights
    criterion = nn.CrossEntropyLoss(weight=Config.CLASS_WEIGHTS.to(Config.DEVICE))
    
    # Step 7: Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )
    
    # Step 5: Mixed Precision Scaler
    scaler = GradScaler() if Config.USE_AMP else None
    
    # Training history
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_fpr": []
    }
    
    best_val_acc = 0.0
    
    # Step 8: Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        # Step 9: Evaluate FPR
        cm, val_fpr, _ = evaluate_fpr(model, val_loader)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_fpr"].append(val_fpr)
        
        print(f"Epoch {epoch:2d}/{Config.EPOCHS} | "
              f"Train: {train_loss:.4f} / {train_acc:.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:.2f}% | "
              f"FPR: {val_fpr*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.CHECKPOINT_DIR / "best.pth")
            print(f"  ✓ Best model saved")
    
    # Step 10: Threshold calibration
    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION")
    print("=" * 60)
    model.load_state_dict(torch.load(Config.CHECKPOINT_DIR / "best.pth"))
    threshold, fpr = calibrate_threshold(model, val_loader)
    print(f"Optimal Threshold: {threshold:.3f}")
    print(f"FPR at Threshold:  {fpr*100:.2f}%")
    
    # Step 11: Test on unseen data
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
    
    # Step 12: Save final model
    torch.save(model.state_dict(), Config.CHECKPOINT_DIR / "food_ai_detector.pth")
    
    # Save metadata
    metadata = {
        "model": Config.MODEL_NAME,
        "num_classes": Config.NUM_CLASSES,
        "image_size": Config.IMAGE_SIZE,
        "threshold": float(threshold),
        "test_accuracy": float(acc_test),
        "test_fpr": float(fpr_test),
        "class_names": CLASS_NAMES,
        "trained_at": datetime.now().isoformat()
    }
    with open(Config.CHECKPOINT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy:     {acc_test*100:.2f}%")
    print(f"Test FPR (Real):   {fpr_test*100:.2f}%")
    
    if fpr_test <= Config.TARGET_FPR:
        print(f"✓ TARGET ACHIEVED: FPR ≤ {Config.TARGET_FPR*100}%")
    else:
        print(f"✗ TARGET MISSED: FPR > {Config.TARGET_FPR*100}%")
    
    print(f"\nModel saved: {Config.CHECKPOINT_DIR / 'food_ai_detector.pth'}")
    print(f"Threshold: {threshold:.3f}")


if __name__ == "__main__":
    main()
