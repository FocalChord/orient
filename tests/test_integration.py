"""Integration test — runs the real model. Skipped if weights aren't available."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image

import orient
from orient._inference import Orientation

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _model_available() -> bool:
    """Check if model weights are cached (don't trigger a download in CI)."""
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(
            "focalchord/film-rotation", "weights/model-vit-ang-loss.h5"
        )
        return isinstance(result, str)  # returns path string if cached
    except Exception:
        return False


requires_model = pytest.mark.skipif(
    not _model_available(),
    reason="Model weights not cached — run `orient.detect()` once to download",
)


@requires_model
def test_detects_rotated_portrait(tmp_path):
    """Take a normal landscape image, rotate it 90° CW to simulate a
    portrait shot stored sideways, and verify orient detects it.
    """
    src = EXAMPLES_DIR / "cat.jpg"
    test_img = tmp_path / "cat_rotated.jpg"

    # Rotate 90° CW to simulate sideways portrait
    img = Image.open(src)
    original_size = img.size  # (800, 550) landscape
    assert original_size[0] > original_size[1], "source should be landscape"

    rotated = img.rotate(-90, expand=True)
    rotated.save(test_img, quality=95)

    # The rotated image is now portrait dimensions (550, 800) but content is sideways
    rotated_size = Image.open(test_img).size
    assert rotated_size[1] > rotated_size[0], "rotated image should be portrait dims"

    # Model should detect it needs rotation
    result = orient.detect(str(test_img))
    assert result.needs_rotation, f"Expected rotation needed, got {result}"
    assert result.orientation in (Orientation.CW_90, Orientation.CCW_90), (
        f"Expected 90° rotation, got {result.orientation.label}"
    )


@requires_model
def test_fix_makes_landscape_from_rotated_portrait(tmp_path):
    """Rotate a landscape image sideways, then fix() it and verify
    the dimensions flip back to landscape.
    """
    src = EXAMPLES_DIR / "cat.jpg"
    test_img = tmp_path / "cat_rotated.jpg"

    img = Image.open(src)
    original_size = img.size  # (800, 550)

    # Rotate 90° CW — now (550, 800), content sideways
    rotated = img.rotate(-90, expand=True)
    rotated.save(test_img, quality=95)

    before = Image.open(test_img).size
    assert before[1] > before[0], "should start as portrait"

    # Fix it
    result = orient.fix(str(test_img), method="transpose")
    assert result.needs_rotation

    # After fix, dimensions should have flipped back to landscape
    after = Image.open(test_img).size
    assert after[0] > after[1], (
        f"Expected landscape after fix, got {after[0]}x{after[1]}"
    )


@requires_model
def test_correct_image_not_rotated(tmp_path):
    """A normal upright image should be detected as correct."""
    src = EXAMPLES_DIR / "landscape.jpg"
    test_img = tmp_path / "landscape.jpg"
    shutil.copy(src, test_img)

    result = orient.detect(str(test_img))
    assert result.is_correct, f"Expected correct orientation, got {result}"


@requires_model
def test_detect_pil_image_integration():
    """Verify detect() works with a PIL Image (no file path)."""
    img = Image.open(EXAMPLES_DIR / "coffee.jpg")
    result = orient.detect(img)
    assert isinstance(result, orient.Result)
    # A normal stock photo should be detected as correct
    assert result.is_correct, f"Expected correct orientation, got {result}"
