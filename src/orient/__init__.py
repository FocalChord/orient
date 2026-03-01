"""orient - Detect and fix orientation of images using deep learning."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image

from ._inference import Orientation, Result, detect_batch, detect_single
from ._rotation import apply_rotation

__all__ = ["detect", "fix", "Orientation", "Result"]


def detect(
    input: Union[str, Path, Image.Image, list[str], list[Path]],
) -> Union[Result, list[Result]]:
    """Detect image orientation.

    Accepts a file path (str/Path), a PIL Image, or a list of paths.
    Returns a single Result for single inputs, or a list for lists.
    """
    if isinstance(input, list):
        return detect_batch([str(p) for p in input])
    return detect_single(input)


def fix(
    input: Union[str, Path, list[str], list[Path]],
    *,
    method: str = "exiftool",
) -> Union[Result, list[Result]]:
    """Detect orientation and apply lossless rotation to fix it.

    Accepts a file path (str/Path) or a list of paths.
    Returns the detection Result(s) so you can inspect what happened.
    """
    if isinstance(input, list):
        paths = [Path(p) for p in input]
        results = detect_batch([str(p) for p in paths])
        for path, result in zip(paths, results):
            if result.needs_rotation:
                apply_rotation(path, result.orientation, method)
        return results

    path = Path(input)
    result = detect_single(str(path))
    if result.needs_rotation:
        apply_rotation(path, result.orientation, method)
    return result
