"""Image preprocessing for the OAD ViT model."""

from __future__ import annotations

import numpy as np
from PIL import Image

_IMAGE_SIZE = 224


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """PIL Image -> normalized CHW float32 array."""
    img = img.convert("RGB").resize((_IMAGE_SIZE, _IMAGE_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)  # HWC -> CHW


def preprocess_path(image_path: str) -> np.ndarray:
    """Load image from path, resize to 224x224, normalize to [-1, 1], return CHW."""
    return preprocess_pil(Image.open(image_path))
