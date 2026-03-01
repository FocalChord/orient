"""Tests for orientation types and angle mapping."""

from pathlib import Path

import pytest

from orient._inference import Orientation, Result, _angle_to_orientation


class TestOrientation:
    def test_values(self):
        assert Orientation.CORRECT == 0
        assert Orientation.CW_90 == 90
        assert Orientation.CW_180 == 180
        assert Orientation.CCW_90 == 270

    def test_labels(self):
        assert Orientation.CORRECT.label == "Correct"
        assert Orientation.CW_90.label == "90° CW"
        assert Orientation.CW_180.label == "180°"
        assert Orientation.CCW_90.label == "90° CCW"

    def test_exif_orientation(self):
        assert Orientation.CORRECT.exif_orientation == 1
        assert Orientation.CW_90.exif_orientation == 6
        assert Orientation.CW_180.exif_orientation == 3
        assert Orientation.CCW_90.exif_orientation == 8

    def test_is_correct(self):
        assert Orientation.CORRECT.is_correct is True
        assert Orientation.CW_90.is_correct is False
        assert Orientation.CW_180.is_correct is False
        assert Orientation.CCW_90.is_correct is False


class TestResult:
    def test_needs_rotation_correct(self):
        r = Result(orientation=Orientation.CORRECT, confidence=0.95, angle=1.2)
        assert r.needs_rotation is False
        assert r.is_correct is True

    def test_needs_rotation_cw90(self):
        r = Result(orientation=Orientation.CW_90, confidence=0.90, angle=88.5)
        assert r.needs_rotation is True
        assert r.is_correct is False

    def test_path_default_none(self):
        r = Result(orientation=Orientation.CORRECT, confidence=0.95, angle=1.2)
        assert r.path is None

    def test_path_populated(self):
        r = Result(orientation=Orientation.CW_90, confidence=0.90, angle=88.5, path=Path("photo.jpg"))
        assert r.path == Path("photo.jpg")

    def test_repr_without_path(self):
        r = Result(orientation=Orientation.CW_90, confidence=0.93, angle=82.4)
        s = repr(r)
        assert "CW_90" in s
        assert "0.93" in s
        assert "82.4" in s
        assert "path" not in s

    def test_repr_with_path(self):
        r = Result(orientation=Orientation.CW_90, confidence=0.93, angle=82.4, path=Path("photo.jpg"))
        s = repr(r)
        assert "CW_90" in s
        assert "photo.jpg" in s


class TestAngleToOrientation:
    @pytest.mark.parametrize("angle,expected", [
        (0.0, Orientation.CORRECT),
        (2.5, Orientation.CORRECT),
        (-3.0, Orientation.CORRECT),
        (358.0, Orientation.CORRECT),
    ])
    def test_correct(self, angle, expected):
        orientation, _ = _angle_to_orientation(angle)
        assert orientation == expected

    @pytest.mark.parametrize("angle,expected", [
        (90.0, Orientation.CW_90),
        (85.0, Orientation.CW_90),
        (95.0, Orientation.CW_90),
    ])
    def test_cw90(self, angle, expected):
        orientation, _ = _angle_to_orientation(angle)
        assert orientation == expected

    @pytest.mark.parametrize("angle,expected", [
        (180.0, Orientation.CW_180),
        (175.0, Orientation.CW_180),
        (185.0, Orientation.CW_180),
    ])
    def test_cw180(self, angle, expected):
        orientation, _ = _angle_to_orientation(angle)
        assert orientation == expected

    @pytest.mark.parametrize("angle,expected", [
        (270.0, Orientation.CCW_90),
        (265.0, Orientation.CCW_90),
        (275.0, Orientation.CCW_90),
        (-90.0, Orientation.CCW_90),
    ])
    def test_ccw90(self, angle, expected):
        orientation, _ = _angle_to_orientation(angle)
        assert orientation == expected

    def test_confidence_exact(self):
        _, confidence = _angle_to_orientation(90.0)
        assert confidence == 1.0

    def test_confidence_off_by_10(self):
        _, confidence = _angle_to_orientation(80.0)
        assert 0.7 < confidence < 0.8

    def test_confidence_at_boundary(self):
        _, confidence = _angle_to_orientation(45.0)
        assert confidence == pytest.approx(0.0)
