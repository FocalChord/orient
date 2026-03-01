"""Orientation detection: enum, result type, angle mapping, verification pass."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from ._model import get_model
from ._preprocess import preprocess_path, preprocess_pil


class Orientation(IntEnum):
    """Detected content orientation as clockwise rotation needed to correct."""

    CORRECT = 0
    CW_90 = 90
    CW_180 = 180
    CCW_90 = 270

    @property
    def exif_orientation(self) -> int:
        """Map to EXIF Orientation tag value."""
        return {0: 1, 90: 6, 180: 3, 270: 8}[self.value]

    @property
    def label(self) -> str:
        return {0: "Correct", 90: "90° CW", 180: "180°", 270: "90° CCW"}[self.value]

    @property
    def is_correct(self) -> bool:
        return self == Orientation.CORRECT


@dataclass
class Result:
    """Detection result for a single image."""

    orientation: Orientation
    confidence: float
    angle: float

    @property
    def needs_rotation(self) -> bool:
        return self.orientation != Orientation.CORRECT

    @property
    def is_correct(self) -> bool:
        return self.orientation == Orientation.CORRECT

    def __repr__(self) -> str:
        return (
            f"Result(orientation={self.orientation.name}, "
            f"confidence={self.confidence:.2f}, "
            f"angle={self.angle:.1f})"
        )


def _angle_to_orientation(angle: float) -> tuple[Orientation, float]:
    """Convert predicted angle to nearest Orientation and confidence."""
    angle = angle % 360
    nearest = round(angle / 90) * 90 % 360
    dist = min(angle % 90, 90 - angle % 90)
    confidence = 1.0 - (dist / 45.0)
    orientation_map = {
        0: Orientation.CORRECT,
        90: Orientation.CW_90,
        180: Orientation.CW_180,
        270: Orientation.CCW_90,
    }
    return orientation_map[nearest], confidence


def _verify_direction(model, image_or_path: Union[str, Path, Image.Image]) -> Orientation:
    """For 90°/270° predictions, rotate both ways and pick the direction
    that makes the image look most upright (closest to 0°).
    """
    if isinstance(image_or_path, Image.Image):
        img = image_or_path
    else:
        img = Image.open(image_or_path)

    img_cw = img.rotate(-90, expand=True)
    img_ccw = img.rotate(90, expand=True)

    batch = np.stack([preprocess_pil(img_cw), preprocess_pil(img_ccw)])
    preds = model.predict(batch, verbose=0).flatten()

    dist_cw = min(abs(preds[0] % 360), abs(360 - preds[0] % 360))
    dist_ccw = min(abs(preds[1] % 360), abs(360 - preds[1] % 360))

    return Orientation.CW_90 if dist_cw < dist_ccw else Orientation.CCW_90


def detect_single(image: Union[str, Path, Image.Image]) -> Result:
    """Run OAD model on a single image with direction verification."""
    model = get_model()

    if isinstance(image, Image.Image):
        arr = preprocess_pil(image)
    else:
        arr = preprocess_path(str(image))

    batch = np.expand_dims(arr, 0)
    pred = model.predict(batch, verbose=0)[0][0]

    orientation, confidence = _angle_to_orientation(pred)

    if orientation in (Orientation.CW_90, Orientation.CCW_90):
        orientation = _verify_direction(model, image)

    return Result(orientation=orientation, confidence=confidence, angle=float(pred))


def detect_batch(image_paths: list[Union[str, Path]]) -> list[Result]:
    """Run OAD model on a batch of images with direction verification."""
    model = get_model()
    batch = np.stack([preprocess_path(str(p)) for p in image_paths])
    preds = model.predict(batch, verbose=0).flatten()

    results = []
    for image_path, pred in zip(image_paths, preds):
        orientation, confidence = _angle_to_orientation(pred)

        if orientation in (Orientation.CW_90, Orientation.CCW_90):
            orientation = _verify_direction(model, image_path)

        results.append(Result(
            orientation=orientation,
            confidence=confidence,
            angle=float(pred),
        ))

    return results
