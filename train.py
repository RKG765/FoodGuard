"""
Train Food Classifier

Usage:
    python train.py --dataset food_101 --epochs 50 --batch-size 32
"""

import argparse
import yaml
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data import FoodDataset, create_data_loaders, get_train_transforms, get_val_transforms
from src.models import create_model, Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Food Classifier')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset', type=str, default='food_101',
                        choices=['food_101', 'uecfood256', 'indian_food'],
                        help='Dataset to train on')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--architecture', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'efficientnet_b1', 'resnet50', 'resnet101'],
                        help='Model architecture')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config if exists
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get data root from config or default
    data_root = config.get('data', {}).get('root', 'e:/BML/Semester-VI/Prj-3')
    image_size = config.get('data', {}).get('image_size', 224)
    
    # Create transforms
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    # Create data loaders
    print(f"\nLoading {args.dataset} dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        root_dir=data_root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=4,
        train_transform=train_transform,
        val_transform=val_transform
    )
    
    # Get number of classes from dataset
    num_classes = train_loader.dataset.num_classes
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    print(f"\nCreating {args.architecture} model...")
    model = create_model(
        num_classes=num_classes,
        architecture=args.architecture,
        pretrained=True,
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=device == 'cuda',
        early_stopping_patience=5
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    history = trainer.train()
    
    print(f"\nTraining complete!")
    print(f"Best checkpoint saved to: {args.checkpoint_dir}/best.pt")


if __name__ == '__main__':
    main()
