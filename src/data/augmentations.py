"""
Data Augmentation Transforms for Food Image Classification
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 224):
    """
    Training transforms with augmentation.
    
    Includes:
    - Random resized crop
    - Horizontal flip
    - Color jitter
    - Normalization (ImageNet stats)
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


def get_val_transforms(image_size: int = 224):
    """
    Validation/test transforms (no augmentation).
    
    Includes:
    - Resize and center crop
    - Normalization (ImageNet stats)
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),  # 256 for 224
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_inference_transforms(image_size: int = 224):
    """Alias for validation transforms."""
    return get_val_transforms(image_size)


# Inverse normalization for visualization
INVERSE_NORMALIZE = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
