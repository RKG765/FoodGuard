"""
Focal Loss for Class-Imbalanced Classification
===============================================

Focal Loss down-weights well-classified examples and focuses on hard ones.
Particularly useful when class distribution is heavily imbalanced.

L(p_t) = -(1 - p_t)^gamma * log(p_t)

Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Args:
        alpha: Per-class weights tensor of shape (num_classes,), or None for uniform.
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to CrossEntropyLoss.
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (B, C)
            targets: Class indices (B,)

        Returns:
            Focal loss scalar
        """
        # Compute log softmax for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        # Gather class probabilities
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Label smoothing
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / inputs.size(1)
            targets_one_hot = targets_one_hot * (1.0 - self.label_smoothing) + smooth

        # Focal weight: (1 - p_t)^gamma
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1.0 - pt) ** self.gamma

        # Cross-entropy
        ce_loss = -(targets_one_hot * log_probs).sum(dim=1)

        # Apply focal weight
        loss = focal_weight * ce_loss

        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
