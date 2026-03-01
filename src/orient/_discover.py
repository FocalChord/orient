"""JPEG file discovery for directory-level processing."""

from __future__ import annotations

from pathlib import Path

_JPEG_EXTENSIONS = {".jpg", ".jpeg"}


def discover_jpegs(directory: str | Path, *, recursive: bool = True) -> list[Path]:
    """Find JPEG files in a directory, sorted by name.

    Args:
        directory: Path to search.
        recursive: If True (default), search subdirectories too.

    Returns:
        Sorted list of JPEG file paths.

    Raises:
        NotADirectoryError: If *directory* is not an existing directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    glob = directory.rglob if recursive else directory.glob
    paths = [p for p in glob("*") if p.suffix.lower() in _JPEG_EXTENSIONS]
    paths.sort(key=lambda p: p.name)
    return paths
