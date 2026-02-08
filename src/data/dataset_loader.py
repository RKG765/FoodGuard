"""
Unified Food Dataset Loader
Supports: Food-101, UECFOOD256, Aircrowd, Indian Food datasets
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class FoodDataset(Dataset):
    """Unified dataset loader for multiple food image datasets."""
    
    SUPPORTED_DATASETS = ['food_101', 'uecfood256', 'aircrowd', 'indian_food']
    
    def __init__(
        self,
        root_dir: str,
        dataset_name: str = 'food_101',
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        """
        Args:
            root_dir: Path to the project root (e.g., e:/BML/Semester-VI/Prj-3)
            dataset_name: One of 'food_101', 'uecfood256', 'aircrowd', 'indian_food'
            split: 'train', 'val', or 'test'
            transform: Optional transforms to apply to images
        """
        self.root_dir = Path(root_dir)
        self.dataset_name = dataset_name.lower()
        self.split = split
        self.transform = transform
        
        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Dataset must be one of {self.SUPPORTED_DATASETS}")
        
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset based on type."""
        if self.dataset_name == 'food_101':
            self._load_food101()
        elif self.dataset_name == 'uecfood256':
            self._load_uecfood256()
        elif self.dataset_name == 'indian_food':
            self._load_indian_food()
        else:
            raise NotImplementedError(f"{self.dataset_name} loader not implemented")
    
    def _load_food101(self):
        """Load Food-101 dataset."""
        base_path = self.root_dir / 'food_101' / 'food-101' / 'food-101'
        meta_path = base_path / 'meta'
        images_path = base_path / 'images'
        
        # Load classes
        classes_file = meta_path / 'classes.txt'
        with open(classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load split file
        split_file = meta_path / f'{self.split}.txt'
        if not split_file.exists():
            # If val split doesn't exist, create from train
            if self.split == 'val':
                split_file = meta_path / 'train.txt'
            else:
                split_file = meta_path / 'test.txt'
        
        with open(split_file, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Handle train/val split (80/20 of train.txt)
        if self.split in ['train', 'val']:
            train_file = meta_path / 'train.txt'
            with open(train_file, 'r') as f:
                all_train = [line.strip() for line in f.readlines()]
            
            split_idx = int(len(all_train) * 0.85)
            if self.split == 'train':
                lines = all_train[:split_idx]
            else:
                lines = all_train[split_idx:]
        
        # Build samples list
        for line in lines:
            class_name = line.split('/')[0]
            img_path = images_path / f"{line}.jpg"
            if img_path.exists():
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def _load_uecfood256(self):
        """Load UECFOOD256 dataset."""
        base_path = self.root_dir / 'UECFOOD256'
        
        # Load category mapping
        category_file = base_path / 'category.txt'
        self.classes = []
        self.class_to_idx = {}
        
        if category_file.exists():
            with open(category_file, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        idx = int(parts[0]) - 1  # 0-indexed
                        class_name = parts[1]
                        self.classes.append(class_name)
                        self.class_to_idx[class_name] = idx
        
        # Load images from category folders
        all_samples = []
        for folder in base_path.iterdir():
            if folder.is_dir() and folder.name.isdigit():
                class_idx = int(folder.name) - 1
                for img_file in folder.glob('*.jpg'):
                    all_samples.append((str(img_file), class_idx))
        
        # Split samples
        self.samples = self._split_samples(all_samples)
    
    def _load_indian_food(self):
        """Load Indian Food dataset."""
        base_path = self.root_dir / 'indian_food_data' / 'image_for _cuisines' / 'data'
        
        # Get unique classes from folder structure or file names
        all_samples = []
        class_set = set()
        
        for img_file in base_path.glob('*.jpg'):
            # Extract class from filename (e.g., "dal_makhani_123.jpg" -> "dal_makhani")
            name_parts = img_file.stem.rsplit('_', 1)
            if len(name_parts) > 1 and name_parts[-1].isdigit():
                class_name = name_parts[0]
            else:
                class_name = img_file.stem
            class_set.add(class_name)
            all_samples.append((str(img_file), class_name))
        
        self.classes = sorted(list(class_set))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Convert class names to indices
        all_samples = [(path, self.class_to_idx[cls]) for path, cls in all_samples]
        self.samples = self._split_samples(all_samples)
    
    def _split_samples(self, samples: List) -> List:
        """Split samples into train/val/test (70/15/15)."""
        n = len(samples)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        if self.split == 'train':
            return samples[:train_end]
        elif self.split == 'val':
            return samples[train_end:val_end]
        else:  # test
            return samples[val_end:]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)


def create_data_loaders(
    root_dir: str,
    dataset_name: str = 'food_101',
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = FoodDataset(root_dir, dataset_name, 'train', train_transform)
    val_dataset = FoodDataset(root_dir, dataset_name, 'val', val_transform)
    test_dataset = FoodDataset(root_dir, dataset_name, 'test', val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick test
    dataset = FoodDataset(
        root_dir='e:/BML/Semester-VI/Prj-3',
        dataset_name='food_101',
        split='train'
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"First 5 classes: {dataset.classes[:5]}")
