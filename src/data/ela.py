"""
Error Level Analysis (ELA) for Forensic Image Detection
========================================================

ELA highlights regions where compression levels differ,
revealing edits or AI generation artifacts.

Usage:
    from src.data.ela import compute_ela, ELATransform
"""

import io
import numpy as np
from PIL import Image, ImageChops
import torch
from torchvision import transforms


def compute_ela(image: Image.Image, quality: int = 90, scale: int = 15) -> Image.Image:
    """
    Compute Error Level Analysis of an image.

    Process:
      1. Re-save image at given JPEG quality
      2. Compute pixel-wise difference with original
      3. Scale up the difference for visibility

    Args:
        image: PIL Image (RGB)
        quality: JPEG recompression quality (lower = more visible artifacts)
        scale: Multiplier for the difference image

    Returns:
        ELA image as PIL Image (RGB)
    """
    # Re-compress at target quality
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # Compute absolute difference
    ela = ImageChops.difference(image, recompressed)

    # Scale up for visibility
    extrema = ela.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1

    scale_factor = 255.0 / max_diff * scale
    ela = ela.point(lambda x: min(int(x * scale_factor), 255))

    return ela


class ELATransform:
    """
    Torchvision-compatible transform that computes ELA.

    Usage in transforms.Compose:
        transforms.Compose([
            ELATransform(quality=90, scale=15),
            transforms.Resize(224),
            transforms.ToTensor(),
            ...
        ])
    """

    def __init__(self, quality: int = 90, scale: int = 15):
        self.quality = quality
        self.scale = scale

    def __call__(self, image: Image.Image) -> Image.Image:
        return compute_ela(image, quality=self.quality, scale=self.scale)

    def __repr__(self):
        return f"ELATransform(quality={self.quality}, scale={self.scale})"
