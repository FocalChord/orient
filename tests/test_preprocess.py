"""Tests for image preprocessing."""

import numpy as np
from PIL import Image

from orient._preprocess import preprocess_path, preprocess_pil


def _make_image(width=400, height=300):
    """Create a random RGB test image."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_preprocess_pil_shape():
    img = _make_image()
    result = preprocess_pil(img)
    assert result.shape == (3, 224, 224)


def test_preprocess_pil_dtype():
    img = _make_image()
    result = preprocess_pil(img)
    assert result.dtype == np.float32


def test_preprocess_pil_range():
    img = _make_image()
    result = preprocess_pil(img)
    assert result.min() >= -1.0
    assert result.max() <= 1.0


def test_preprocess_pil_converts_rgba():
    img = _make_image().convert("RGBA")
    result = preprocess_pil(img)
    assert result.shape == (3, 224, 224)


def test_preprocess_pil_converts_grayscale():
    img = _make_image().convert("L")
    result = preprocess_pil(img)
    assert result.shape == (3, 224, 224)


def test_preprocess_path(tmp_path):
    img = _make_image()
    path = tmp_path / "test.jpg"
    img.save(path)
    result = preprocess_path(str(path))
    assert result.shape == (3, 224, 224)
    assert result.dtype == np.float32


def test_preprocess_pil_deterministic():
    img = _make_image()
    a = preprocess_pil(img)
    b = preprocess_pil(img)
    np.testing.assert_array_equal(a, b)
