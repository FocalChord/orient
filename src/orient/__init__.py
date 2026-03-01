"""orient - Detect and fix orientation of images using deep learning."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union

from PIL import Image

from ._discover import discover_jpegs
from ._inference import Orientation, Result, detect_batch, detect_single
from ._rotation import apply_rotation

__all__ = ["detect", "fix", "Orientation", "Result"]

_MAX_WORKERS = min(8, os.cpu_count() or 4)


def detect(
    input: Union[str, Path, Image.Image, list[str], list[Path]],
    *,
    batch_size: int = 32,
    recursive: bool = True,
) -> Union[Result, list[Result]]:
    """Detect image orientation.

    Accepts a file path (str/Path), a PIL Image, a directory path,
    or a list of paths.  Returns a single Result for single-image inputs,
    or a list for directories and lists.
    """
    if isinstance(input, list):
        return detect_batch([str(p) for p in input], batch_size=batch_size)

    if isinstance(input, (str, Path)) and Path(input).is_dir():
        paths = discover_jpegs(input, recursive=recursive)
        if not paths:
            return []
        return detect_batch([str(p) for p in paths], batch_size=batch_size)

    return detect_single(input)


def _apply_rotations_parallel(
    results: list[Result],
    method: str,
) -> None:
    """Apply rotations in parallel via ThreadPoolExecutor."""
    to_rotate = [(r.path, r.orientation) for r in results if r.needs_rotation and r.path]
    if not to_rotate:
        return

    def _rotate(item: tuple[Path, Orientation]) -> None:
        apply_rotation(item[0], item[1], method)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        list(pool.map(_rotate, to_rotate))


def fix(
    input: Union[str, Path, list[str], list[Path]],
    *,
    method: str = "exif",
    batch_size: int = 32,
    recursive: bool = True,
) -> Union[Result, list[Result]]:
    """Detect orientation and apply rotation to fix it.

    Accepts a file path (str/Path), a directory path, or a list of paths.
    Returns the detection Result(s) so you can inspect what happened.

    Methods:
      "exif"      — Set EXIF Orientation tag (metadata only, truly lossless)
      "transpose" — Rotate pixels via Pillow (re-encodes JPEG)
    """
    if isinstance(input, list):
        paths = [Path(p) for p in input]
        results = detect_batch([str(p) for p in paths], batch_size=batch_size)
        _apply_rotations_parallel(results, method)
        return results

    path = Path(input)
    if path.is_dir():
        paths = discover_jpegs(path, recursive=recursive)
        if not paths:
            return []
        results = detect_batch([str(p) for p in paths], batch_size=batch_size)
        _apply_rotations_parallel(results, method)
        return results

    result = detect_single(str(path))
    if result.needs_rotation:
        apply_rotation(path, result.orientation, method)
    return result
