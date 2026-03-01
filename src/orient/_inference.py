"""Orientation detection: enum, result type, angle mapping, verification pass."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from ._model import get_model
from ._preprocess import preprocess_path, preprocess_pil

_MAX_WORKERS = min(8, os.cpu_count() or 4)


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
    path: Path | None = None

    @property
    def needs_rotation(self) -> bool:
        return self.orientation != Orientation.CORRECT

    @property
    def is_correct(self) -> bool:
        return self.orientation == Orientation.CORRECT

    def __repr__(self) -> str:
        parts = [
            f"orientation={self.orientation.name}",
            f"confidence={self.confidence:.2f}",
            f"angle={self.angle:.1f}",
        ]
        if self.path is not None:
            parts.append(f"path={self.path}")
        return f"Result({', '.join(parts)})"


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
        image_path = None
    else:
        arr = preprocess_path(str(image))
        image_path = Path(image)

    batch = np.expand_dims(arr, 0)
    pred = model.predict(batch, verbose=0)[0][0]

    orientation, confidence = _angle_to_orientation(pred)

    if orientation in (Orientation.CW_90, Orientation.CCW_90):
        orientation = _verify_direction(model, image)

    return Result(orientation=orientation, confidence=confidence, angle=float(pred), path=image_path)


def detect_batch(
    image_paths: list[Union[str, Path]],
    *,
    batch_size: int = 32,
) -> list[Result]:
    """Run OAD model on a batch of images with direction verification.

    Processes in chunks of *batch_size* to cap peak memory. Preprocessing
    within each chunk is parallelised across threads (Pillow releases the GIL).
    """
    model = get_model()
    results: list[Result] = []

    for start in range(0, len(image_paths), batch_size):
        chunk_paths = image_paths[start : start + batch_size]

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            arrays = list(pool.map(lambda p: preprocess_path(str(p)), chunk_paths))

        batch = np.stack(arrays)
        preds = model.predict(batch, verbose=0).flatten()

        for image_path, pred in zip(chunk_paths, preds):
            orientation, confidence = _angle_to_orientation(pred)

            if orientation in (Orientation.CW_90, Orientation.CCW_90):
                orientation = _verify_direction(model, image_path)

            results.append(Result(
                orientation=orientation,
                confidence=confidence,
                angle=float(pred),
                path=Path(image_path),
            ))

    return results
