"""Tests for rotation module."""

import numpy as np
import piexif
import pytest
from PIL import Image

from orient._inference import Orientation
from orient._rotation import apply_rotation, rotate_exif, rotate_transpose


def _make_jpeg(path, width=100, height=80):
    """Create a real JPEG file for testing."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


class TestRotateExif:
    def test_correct_is_noop(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path)
        assert rotate_exif(path, Orientation.CORRECT) is False

    def test_sets_exif_orientation_cw90(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path)
        assert rotate_exif(path, Orientation.CW_90) is True

        exif = piexif.load(str(path))
        assert exif["0th"][piexif.ImageIFD.Orientation] == 6

    def test_sets_exif_orientation_180(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path)
        rotate_exif(path, Orientation.CW_180)

        exif = piexif.load(str(path))
        assert exif["0th"][piexif.ImageIFD.Orientation] == 3

    def test_sets_exif_orientation_ccw90(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path)
        rotate_exif(path, Orientation.CCW_90)

        exif = piexif.load(str(path))
        assert exif["0th"][piexif.ImageIFD.Orientation] == 8


class TestRotateTranspose:
    def test_correct_is_noop(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path, width=100, height=80)
        assert rotate_transpose(path, Orientation.CORRECT) is False

        # Dimensions unchanged
        img = Image.open(path)
        assert img.size == (100, 80)

    def test_rotates_cw90(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path, width=100, height=80)
        assert rotate_transpose(path, Orientation.CW_90) is True

        img = Image.open(path)
        assert img.size == (80, 100)

    def test_rotates_180(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path, width=100, height=80)
        rotate_transpose(path, Orientation.CW_180)

        img = Image.open(path)
        assert img.size == (100, 80)

    def test_rotates_ccw90(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path, width=100, height=80)
        rotate_transpose(path, Orientation.CCW_90)

        img = Image.open(path)
        assert img.size == (80, 100)


class TestApplyRotation:
    def test_default_is_exif(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path)
        apply_rotation(path, Orientation.CW_90)

        exif = piexif.load(str(path))
        assert exif["0th"][piexif.ImageIFD.Orientation] == 6

    def test_transpose_method(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path, width=100, height=80)
        apply_rotation(path, Orientation.CW_90, method="transpose")

        img = Image.open(path)
        assert img.size == (80, 100)

    def test_invalid_method(self, tmp_path):
        path = tmp_path / "test.jpg"
        _make_jpeg(path)
        with pytest.raises(ValueError, match="Unknown rotation method"):
            apply_rotation(path, Orientation.CW_90, method="magic")
