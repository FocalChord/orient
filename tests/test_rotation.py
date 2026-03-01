"""Tests for rotation module."""

import shutil
from unittest.mock import patch, MagicMock

import pytest

from orient._inference import Orientation
from orient._rotation import apply_rotation, rotate_exiftool, rotate_jpegtran


class TestRotateExiftool:
    def test_correct_is_noop(self, tmp_path):
        path = tmp_path / "test.jpg"
        path.write_bytes(b"fake")
        assert rotate_exiftool(path, Orientation.CORRECT) is False

    @patch("orient._rotation._check_tool", return_value=False)
    def test_raises_if_not_installed(self, mock_check, tmp_path):
        path = tmp_path / "test.jpg"
        path.write_bytes(b"fake")
        with pytest.raises(RuntimeError, match="exiftool not found"):
            rotate_exiftool(path, Orientation.CW_90)


class TestRotateJpegtran:
    def test_correct_is_noop(self, tmp_path):
        path = tmp_path / "test.jpg"
        path.write_bytes(b"fake")
        assert rotate_jpegtran(path, Orientation.CORRECT) is False

    @patch("orient._rotation._check_tool", return_value=False)
    def test_raises_if_not_installed(self, mock_check, tmp_path):
        path = tmp_path / "test.jpg"
        path.write_bytes(b"fake")
        with pytest.raises(RuntimeError, match="jpegtran not found"):
            rotate_jpegtran(path, Orientation.CW_90)


class TestApplyRotation:
    def test_invalid_method(self, tmp_path):
        path = tmp_path / "test.jpg"
        path.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unknown rotation method"):
            apply_rotation(path, Orientation.CW_90, method="magic")
