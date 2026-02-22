"""
3-Class Detector Dataset Loader
================================

Reads dataset_index.csv and creates a 3-class dataset:
  Class 0: real           (pristine real food images)
  Class 1: ai_generated   (AI-generated images: raw, compressed, degraded)
  Class 2: manipulated    (edited real images with AI modifications)

Supports stratified train/val/test splits and dual-stream (RGB + ELA).
"""

import csv
import random
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image

from .ela import compute_ela


LABEL_MAP = {
    "real": 0,
    "ai_generated": 1,
    "manipulated": 2,
}

CLASS_NAMES = ["real", "ai_generated", "manipulated"]


class DetectorDataset(Dataset):
    """
    3-class detector dataset from dataset_index.csv.

    Returns (rgb_image, ela_image, label) when dual_stream=True,
    or (rgb_image, label) when dual_stream=False.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        rgb_transform: Optional[Callable] = None,
        ela_transform: Optional[Callable] = None,
        dual_stream: bool = True,
        split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
        ela_quality: int = 90,
    ):
        """
        Args:
            root_dir: Project root (e.g. e:/BML/Semester-VI/Prj-3)
            split: 'train', 'val', or 'test'
            rgb_transform: Transforms for RGB stream
            ela_transform: Transforms for ELA stream (applied after ELA computation)
            dual_stream: If True, returns (rgb, ela, label); else (rgb, label)
            split_ratios: (train, val, test) ratios
            seed: Random seed for reproducible splits
            ela_quality: JPEG quality for ELA computation
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.rgb_transform = rgb_transform
        self.ela_transform = ela_transform
        self.dual_stream = dual_stream
        self.ela_quality = ela_quality

        self.classes = CLASS_NAMES
        self.class_to_idx = LABEL_MAP
        self.samples: List[Tuple[str, int]] = []

        # Load and split
        all_samples = self._load_csv()
        self.samples = self._stratified_split(all_samples, split, split_ratios, seed)

    def _load_csv(self) -> List[Tuple[str, int]]:
        """Load dataset_index.csv and return (path, label) pairs.

        CSV Format:
          file_path, label, source, subtype
        Where label is: 0 (real), 1 (ai_generated), 2 (manipulated)
        """
        csv_path = self.root_dir / "dataset_index.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"dataset_index.csv not found at {csv_path}. "
                "Run: python scripts/build_detector_csv.py"
            )

        samples = []
        skipped = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    label = int(row["label"].strip())
                    if label not in [0, 1, 2]:
                        continue

                    # Use forward-slash path from CSV directly
                    img_path = self.root_dir / row["file_path"].replace("/", "\\")
                    if not img_path.exists():
                        skipped += 1
                        continue

                    samples.append((str(img_path), label))
                except (ValueError, KeyError):
                    continue

        if skipped > 0:
            print(f"  [DetectorDataset] Skipped {skipped} missing files")

        return samples

    def _stratified_split(
        self,
        samples: List[Tuple[str, int]],
        split: str,
        ratios: Tuple[float, float, float],
        seed: int,
    ) -> List[Tuple[str, int]]:
        """Stratified split preserving class ratios."""
        rng = random.Random(seed)

        # Group by label
        by_label: Dict[int, List[Tuple[str, int]]] = {}
        for path, label in samples:
            by_label.setdefault(label, []).append((path, label))

        result = []
        train_r, val_r, _ = ratios

        for label, items in by_label.items():
            rng.shuffle(items)
            n = len(items)
            train_end = int(n * train_r)
            val_end = int(n * (train_r + val_r))

            if split == "train":
                result.extend(items[:train_end])
            elif split == "val":
                result.extend(items[train_end:val_end])
            else:  # test
                result.extend(items[val_end:])

        rng.shuffle(result)
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        # RGB stream
        rgb = self.rgb_transform(image) if self.rgb_transform else image

        if self.dual_stream:
            # ELA stream
            ela_img = compute_ela(image, quality=self.ela_quality)
            ela = self.ela_transform(ela_img) if self.ela_transform else ela_img
            return rgb, ela, label
        else:
            return rgb, label

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_label_counts(self) -> Dict[str, int]:
        """Return count of samples per label."""
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss function."""
        counts = [0] * len(CLASS_NAMES)
        for _, label in self.samples:
            counts[label] += 1
        total = sum(counts)
        weights = [total / (len(counts) * c) if c > 0 else 0.0 for c in counts]
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        counts = [0] * len(CLASS_NAMES)
        for _, label in self.samples:
            counts[label] += 1
        class_weights = [1.0 / c if c > 0 else 0.0 for c in counts]
        return [class_weights[label] for _, label in self.samples]


def create_detector_loaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    rgb_transform_train=None,
    rgb_transform_val=None,
    ela_transform_train=None,
    ela_transform_val=None,
    dual_stream: bool = True,
    use_weighted_sampler: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test loaders with optional weighted sampling."""

    train_ds = DetectorDataset(
        root_dir, "train", rgb_transform_train, ela_transform_train, dual_stream
    )
    val_ds = DetectorDataset(
        root_dir, "val", rgb_transform_val, ela_transform_val, dual_stream
    )
    test_ds = DetectorDataset(
        root_dir, "test", rgb_transform_val, ela_transform_val, dual_stream
    )

    # Weighted sampler for imbalanced classes
    if use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
