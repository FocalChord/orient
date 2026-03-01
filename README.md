# orient

Detect and fix image orientation using a deep learning model ([Deep-OAD](https://github.com/e-bug/Deep-OAD) ViT).

## Install

```bash
pip install orient
```

## Usage

```python
import orient

# Detect orientation
result = orient.detect("photo.jpg")
result.orientation      # orient.Orientation.CW_90
result.confidence       # 0.93
result.angle            # 82.4
result.needs_rotation   # True
result.is_correct       # False

# Batch detection
results = orient.detect(["a.jpg", "b.jpg"])

# PIL Image input
from PIL import Image
result = orient.detect(Image.open("photo.jpg"))

# Detect + apply lossless rotation
orient.fix("photo.jpg")
orient.fix("photo.jpg", method="jpegtran")
orient.fix(["a.jpg", "b.jpg"])
```

## How it works

Uses a fine-tuned ViT model to predict the rotation angle of an image. For 90/270 degree predictions, a verification pass rotates the image both ways and picks the direction that looks most upright.

Model weights (~990 MB) are automatically downloaded from [Hugging Face](https://huggingface.co/focalchord/film-rotation) on first use.

## Rotation methods

- **exiftool** (default) - Sets the EXIF Orientation tag. Truly lossless. Requires `exiftool`.
- **jpegtran** - Lossless DCT rotation. Requires `jpegtran` (from libjpeg-turbo).

## See also

- [auto-orient](https://github.com/FocalChord/auto-orient) - CLI tool for bulk processing
- [Deep-OAD](https://github.com/e-bug/Deep-OAD) - The underlying orientation angle detection model
