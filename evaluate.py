"""
Evaluate trained Food Classifier

Usage:
    python evaluate.py --checkpoint checkpoints/best.pt --dataset food_101
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data import FoodDataset, get_val_transforms
from src.models import FoodClassifier


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Food Classifier')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='food_101',
                        help='Dataset to evaluate on')
    parser.add_argument('--data-root', type=str, default='e:/BML/Semester-VI/Prj-3',
                        help='Data root directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Run evaluation and collect predictions."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, classes, output_path, top_n=20):
    """Plot confusion matrix for top N most confused classes."""
    # Find most confused classes
    np.fill_diagonal(cm, 0)
    confusion_sums = cm.sum(axis=1) + cm.sum(axis=0)
    top_indices = confusion_sums.argsort()[-top_n:]
    
    # Extract sub-matrix
    cm_subset = cm[np.ix_(top_indices, top_indices)]
    class_subset = [classes[i] for i in top_indices]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_subset, yticklabels=class_subset)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Top {top_n} Most Confused Classes)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def main():
    args = parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load test dataset
    print(f"\nLoading {args.dataset} test set...")
    transform = get_val_transforms(224)
    test_dataset = FoodDataset(
        root_dir=args.data_root,
        dataset_name=args.dataset,
        split='test',
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    num_classes = test_dataset.num_classes
    classes = test_dataset.classes
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = FoodClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    print("\nRunning evaluation...")
    preds, labels = evaluate(model, test_loader, device)
    
    # Calculate metrics
    accuracy = (preds == labels).mean() * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    
    # Classification report
    report = classification_report(labels, preds, target_names=classes, output_dict=True)
    
    # Save report
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
        f.write(classification_report(labels, preds, target_names=classes))
    print(f"Classification report saved to: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, classes, cm_path)
    
    # Top-5 and Bottom-5 classes by accuracy
    per_class_acc = []
    for i, cls in enumerate(classes):
        mask = labels == i
        if mask.sum() > 0:
            acc = (preds[mask] == i).mean() * 100
            per_class_acc.append((cls, acc, mask.sum()))
    
    per_class_acc.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 Classes (by accuracy):")
    for cls, acc, n in per_class_acc[:5]:
        print(f"  {cls}: {acc:.1f}% (n={n})")
    
    print("\nBottom 5 Classes (by accuracy):")
    for cls, acc, n in per_class_acc[-5:]:
        print(f"  {cls}: {acc:.1f}% (n={n})")


if __name__ == '__main__':
    main()
