"""Image rotation — pure Python, no external tools required.

Methods:
  "exif"      — Set EXIF Orientation tag via piexif (metadata only, truly lossless)
  "transpose" — Rotate pixels via Pillow transpose (re-encodes JPEG)
"""

from __future__ import annotations

from pathlib import Path

import piexif
from PIL import Image

from ._inference import Orientation


def rotate_exif(image_path: Path, orientation: Orientation) -> bool:
    """Set EXIF Orientation tag. Truly lossless — only touches metadata bytes."""
    if orientation == Orientation.CORRECT:
        return False

    try:
        exif_dict = piexif.load(str(image_path))
    except piexif.InvalidImageDataError:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}}

    exif_dict["0th"][piexif.ImageIFD.Orientation] = orientation.exif_orientation
    piexif.insert(piexif.dump(exif_dict), str(image_path))
    return True


def rotate_transpose(image_path: Path, orientation: Orientation) -> bool:
    """Rotate pixels using Pillow. Re-encodes JPEG but requires no external tools."""
    if orientation == Orientation.CORRECT:
        return False

    transpose_op = {
        Orientation.CW_90: Image.Transpose.ROTATE_270,
        Orientation.CW_180: Image.Transpose.ROTATE_180,
        Orientation.CCW_90: Image.Transpose.ROTATE_90,
    }[orientation]

    img = Image.open(image_path)
    rotated = img.transpose(transpose_op)
    rotated.save(image_path, quality=95)
    return True


def apply_rotation(
    image_path: Path,
    orientation: Orientation,
    method: str = "exif",
) -> bool:
    """Apply rotation using the specified method."""
    if method == "exif":
        return rotate_exif(image_path, orientation)
    elif method == "transpose":
        return rotate_transpose(image_path, orientation)
    else:
        raise ValueError(f"Unknown rotation method: {method!r}. Use 'exif' or 'transpose'.")
