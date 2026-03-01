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

# Folder detection — finds all JPEGs recursively
results = orient.detect("photos/")
for r in results:
    print(f"{r.path.name}: {r.orientation.label}")

# Folder options
orient.detect("photos/", batch_size=16, recursive=False)

# PIL Image input
from PIL import Image
result = orient.detect(Image.open("photo.jpg"))

# Detect + fix orientation
orient.fix("photo.jpg")                       # set EXIF tag (default, lossless)
orient.fix("photo.jpg", method="transpose")   # rotate pixels via Pillow
orient.fix(["a.jpg", "b.jpg"])

# Fix an entire folder
orient.fix("photos/")
```

## How it works

Uses a fine-tuned ViT model to predict the rotation angle of an image. For 90/270 degree predictions, a verification pass rotates the image both ways and picks the direction that looks most upright.

Model weights (~990 MB) are automatically downloaded from [Hugging Face](https://huggingface.co/focalchord/film-rotation) on first use.

## Rotation methods

- **exif** (default) — Sets the EXIF Orientation tag via piexif. Truly lossless (metadata only). No external tools needed.
- **transpose** — Rotates pixels using Pillow. Re-encodes JPEG but works everywhere. No external tools needed.

## See also

- [auto-orient](https://github.com/FocalChord/auto-orient) - CLI tool for bulk processing
- [Deep-OAD](https://github.com/e-bug/Deep-OAD) - The underlying orientation angle detection model
