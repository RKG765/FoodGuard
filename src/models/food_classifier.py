"""
Food Image Classifier using EfficientNet-B0
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class FoodClassifier(nn.Module):
    """
    EfficientNet-B0 based food image classifier with custom classification head.
    """
    
    SUPPORTED_ARCHITECTURES = ['efficientnet_b0', 'efficientnet_b1', 'resnet50', 'resnet101']
    
    def __init__(
        self,
        num_classes: int = 101,
        architecture: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: Number of food categories to classify
            architecture: Backbone architecture name
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout rate for classification head
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        
        # Load backbone
        self.backbone, in_features = self._create_backbone(architecture, pretrained)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def _create_backbone(self, architecture: str, pretrained: bool):
        """Create backbone network and return it with feature dimension."""
        weights = 'IMAGENET1K_V1' if pretrained else None
        
        if architecture == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        elif architecture == 'efficientnet_b1':
            model = models.efficientnet_b1(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            
        elif architecture == 'resnet50':
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            
        elif architecture == 'resnet101':
            model = models.resnet101(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            
        else:
            raise ValueError(f"Architecture must be one of {self.SUPPORTED_ARCHITECTURES}")
        
        return model, in_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Logits of shape (B, num_classes)
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def freeze_backbone(self):
        """Freeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Return number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(
    num_classes: int = 101,
    architecture: str = 'efficientnet_b0',
    pretrained: bool = True,
    device: Optional[str] = None
) -> FoodClassifier:
    """
    Factory function to create and initialize model.
    
    Args:
        num_classes: Number of output classes
        architecture: Model architecture name
        pretrained: Use ImageNet pretrained weights
        device: Target device ('cuda', 'cpu', or None for auto)
    
    Returns:
        Initialized FoodClassifier model
    """
    model = FoodClassifier(
        num_classes=num_classes,
        architecture=architecture,
        pretrained=pretrained
    )
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    print(f"Created {architecture} model with {model.get_num_params():,} trainable parameters")
    print(f"Device: {device}")
    
    return model


if __name__ == '__main__':
    # Quick test
    model = create_model(num_classes=101)
    x = torch.randn(2, 3, 224, 224).to(next(model.parameters()).device)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
