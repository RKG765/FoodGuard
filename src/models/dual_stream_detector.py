"""
Dual-Stream Detector for AI Food Image Detection
=================================================

Architecture:
  Stream A (RGB):   EfficientNet-B2 on original image
  Stream B (ELA):   EfficientNet-B2 on Error Level Analysis image
  Fusion:           Concatenate features -> Dense -> 3-class output

Classes:
  0: real
  1: ai_generated
  2: manipulated
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class DualStreamDetector(nn.Module):
    """
    Dual-stream network combining RGB and forensic (ELA) features.

    Stream A processes the raw RGB image.
    Stream B processes the ELA image (compression artifacts).
    Features are concatenated and passed through a fusion classifier.
    """

    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = "efficientnet_b2",
        pretrained: bool = True,
        dropout: float = 0.4,
        freeze_backbone_initially: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stream A: RGB
        self.stream_rgb, rgb_features = self._create_backbone(backbone, pretrained)

        # Stream B: ELA (forensic)
        self.stream_ela, ela_features = self._create_backbone(backbone, pretrained)

        total_features = rgb_features + ela_features

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_features),
            nn.Dropout(p=dropout),
            nn.Linear(total_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone_initially:
            self.freeze_backbones()

    def _create_backbone(self, name: str, pretrained: bool):
        """Create a backbone and return (model, feature_dim)."""
        weights = "IMAGENET1K_V1" if pretrained else None

        if name == "efficientnet_b2":
            model = models.efficientnet_b2(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()

        elif name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=weights)
            in_features = model.classifier[1].in_features
            model.classifier = nn.Identity()

        elif name == "convnext_tiny":
            model = models.convnext_tiny(weights=weights)
            in_features = model.classifier[2].in_features
            model.classifier = nn.Identity()

        elif name == "resnet50":
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported backbone: {name}")

        return model, in_features

    def forward(self, rgb: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            rgb: RGB image tensor (B, 3, H, W)
            ela: ELA image tensor (B, 3, H, W)

        Returns:
            Logits (B, num_classes)
        """
        feat_rgb = self.stream_rgb(rgb)
        feat_ela = self.stream_ela(ela)

        # Concatenate features
        fused = torch.cat([feat_rgb, feat_ela], dim=1)

        logits = self.classifier(fused)
        return logits

    def freeze_backbones(self):
        """Freeze both backbone parameters."""
        for param in self.stream_rgb.parameters():
            param.requires_grad = False
        for param in self.stream_ela.parameters():
            param.requires_grad = False
        print("  Backbones frozen")

    def unfreeze_backbones(self):
        """Unfreeze both backbone parameters."""
        for param in self.stream_rgb.parameters():
            param.requires_grad = True
        for param in self.stream_ela.parameters():
            param.requires_grad = True
        print("  Backbones unfrozen")

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class SingleStreamDetector(nn.Module):
    """
    Fallback single-stream detector (RGB only).
    For quick testing or when ELA is not needed.
    """

    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = "efficientnet_b2",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "efficientnet_b2":
            self.backbone = models.efficientnet_b2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
