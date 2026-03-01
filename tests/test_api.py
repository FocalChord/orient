"""Tests for the public API (detect/fix) with mocked model."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

import orient
from orient._inference import Orientation, Result


def _make_image(width=400, height=300):
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _mock_model():
    """Create a mock model that predicts 0° (correct orientation)."""
    model = MagicMock()
    model.predict.return_value = np.array([[0.5]])
    return model


def _mock_model_90():
    """Create a mock model that predicts ~90° then verifies CW."""
    model = MagicMock()
    # First call: initial prediction of 90°
    # Second call: verification pass — CW (close to 0) vs CCW (far from 0)
    model.predict.side_effect = [
        np.array([[88.0]]),              # initial: ~90°
        np.array([[2.0], [175.0]]),      # verify: CW=2° (closer to 0), CCW=175°
    ]
    return model


@patch("orient._inference.get_model")
def test_detect_single_path(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    img = _make_image()
    path = tmp_path / "test.jpg"
    img.save(path)

    result = orient.detect(str(path))
    assert isinstance(result, Result)
    assert result.orientation == Orientation.CORRECT
    assert result.path == path


@patch("orient._inference.get_model")
def test_detect_single_pathlib(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    img = _make_image()
    path = tmp_path / "test.jpg"
    img.save(path)

    result = orient.detect(path)
    assert isinstance(result, Result)


@patch("orient._inference.get_model")
def test_detect_pil_image(mock_get_model):
    mock_get_model.return_value = _mock_model()
    img = _make_image()

    result = orient.detect(img)
    assert isinstance(result, Result)
    assert result.is_correct
    assert result.path is None


@patch("orient._inference.get_model")
def test_detect_batch(mock_get_model, tmp_path):
    model = MagicMock()
    model.predict.side_effect = [
        np.array([[0.5], [89.0]]),       # batch prediction
        np.array([[3.0], [170.0]]),      # verification for 89°
    ]
    mock_get_model.return_value = model

    paths = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jpg"
        _make_image().save(p)
        paths.append(str(p))

    results = orient.detect(paths)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, Result) for r in results)


@patch("orient._inference.get_model")
def test_detect_with_verification(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model_90()
    img = _make_image()
    path = tmp_path / "test.jpg"
    img.save(path)

    result = orient.detect(str(path))
    assert result.orientation == Orientation.CW_90
    assert result.needs_rotation is True


@patch("orient.apply_rotation")
@patch("orient._inference.get_model")
def test_fix_applies_rotation(mock_get_model, mock_apply, tmp_path):
    mock_get_model.return_value = _mock_model_90()
    mock_apply.return_value = True

    img = _make_image()
    path = tmp_path / "test.jpg"
    img.save(path)

    result = orient.fix(str(path))
    assert result.needs_rotation
    mock_apply.assert_called_once_with(path, Orientation.CW_90, "exif")


@patch("orient.apply_rotation")
@patch("orient._inference.get_model")
def test_fix_skips_correct(mock_get_model, mock_apply, tmp_path):
    mock_get_model.return_value = _mock_model()
    img = _make_image()
    path = tmp_path / "test.jpg"
    img.save(path)

    result = orient.fix(str(path))
    assert result.is_correct
    mock_apply.assert_not_called()


@patch("orient.apply_rotation")
@patch("orient._inference.get_model")
def test_fix_transpose_method(mock_get_model, mock_apply, tmp_path):
    mock_get_model.return_value = _mock_model_90()
    mock_apply.return_value = True

    img = _make_image()
    path = tmp_path / "test.jpg"
    img.save(path)

    result = orient.fix(str(path), method="transpose")
    mock_apply.assert_called_once_with(path, Orientation.CW_90, "transpose")


@patch("orient.apply_rotation")
@patch("orient._inference.get_model")
def test_fix_batch(mock_get_model, mock_apply, tmp_path):
    model = MagicMock()
    model.predict.side_effect = [
        np.array([[0.5], [89.0]]),
        np.array([[3.0], [170.0]]),
    ]
    mock_get_model.return_value = model
    mock_apply.return_value = True

    paths = []
    for i in range(2):
        p = tmp_path / f"test_{i}.jpg"
        _make_image().save(p)
        paths.append(str(p))

    results = orient.fix(paths)
    assert len(results) == 2
    # Only the rotated one should trigger apply_rotation
    assert mock_apply.call_count == 1


# --- Directory tests ---


def _save_jpegs(directory, count=3):
    """Save test JPEGs into a directory, return sorted paths."""
    paths = []
    for i in range(count):
        p = directory / f"img_{i:02d}.jpg"
        _make_image().save(p)
        paths.append(p)
    return paths


@patch("orient._inference.get_model")
def test_detect_directory(mock_get_model, tmp_path):
    model = MagicMock()
    model.predict.return_value = np.array([[0.5], [0.5], [0.5]])
    mock_get_model.return_value = model
    _save_jpegs(tmp_path, count=3)

    results = orient.detect(str(tmp_path))
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(r.path is not None for r in results)


@patch("orient._inference.get_model")
def test_detect_directory_pathlib(mock_get_model, tmp_path):
    model = MagicMock()
    model.predict.return_value = np.array([[0.5], [0.5]])
    mock_get_model.return_value = model
    _save_jpegs(tmp_path, count=2)

    results = orient.detect(tmp_path)
    assert len(results) == 2


@patch("orient._inference.get_model")
def test_detect_directory_recursive(mock_get_model, tmp_path):
    model = MagicMock()
    model.predict.return_value = np.array([[0.5], [0.5]])
    mock_get_model.return_value = model

    sub = tmp_path / "sub"
    sub.mkdir()
    _make_image().save(tmp_path / "top.jpg")
    _make_image().save(sub / "deep.jpg")

    results = orient.detect(tmp_path, recursive=True)
    assert len(results) == 2


@patch("orient._inference.get_model")
def test_detect_directory_non_recursive(mock_get_model, tmp_path):
    model = MagicMock()
    model.predict.return_value = np.array([[0.5]])
    mock_get_model.return_value = model

    sub = tmp_path / "sub"
    sub.mkdir()
    _make_image().save(tmp_path / "top.jpg")
    _make_image().save(sub / "deep.jpg")

    results = orient.detect(tmp_path, recursive=False)
    assert len(results) == 1


@patch("orient._inference.get_model")
def test_detect_directory_empty(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    results = orient.detect(tmp_path)
    assert results == []


@patch("orient.apply_rotation")
@patch("orient._inference.get_model")
def test_fix_directory(mock_get_model, mock_apply, tmp_path):
    model = MagicMock()
    model.predict.side_effect = [
        np.array([[0.5], [89.0]]),          # batch: first correct, second ~90°
        np.array([[2.0], [175.0]]),         # verification for 89°
    ]
    mock_get_model.return_value = model
    mock_apply.return_value = True
    _save_jpegs(tmp_path, count=2)

    results = orient.fix(str(tmp_path))
    assert len(results) == 2
    assert mock_apply.call_count == 1


@patch("orient._inference.get_model")
def test_fix_directory_empty(mock_get_model, tmp_path):
    mock_get_model.return_value = _mock_model()
    results = orient.fix(tmp_path)
    assert results == []


@patch("orient._inference.get_model")
def test_detect_batch_sets_path(mock_get_model, tmp_path):
    model = MagicMock()
    model.predict.return_value = np.array([[0.5], [0.5]])
    mock_get_model.return_value = model

    paths = _save_jpegs(tmp_path, count=2)
    results = orient.detect([str(p) for p in paths])
    for r, expected in zip(results, paths):
        assert r.path == expected
