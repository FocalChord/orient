"""ViT model loading with automatic weight download from Hugging Face."""

from __future__ import annotations

import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_IMAGE_SIZE = 224
_HF_REPO = "focalchord/film-rotation"
_HF_FILENAME = "weights/model-vit-ang-loss.h5"

_model = None


def get_model():
    """Build ViT architecture and load fine-tuned OAD weights.

    Weights are auto-downloaded from Hugging Face on first use (~990 MB)
    and cached locally by huggingface_hub.
    """
    global _model
    if _model is not None:
        return _model

    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, TFAutoModel
    from tf_keras import layers as L
    from tf_keras.models import Model

    weights_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)

    config = AutoConfig.from_pretrained("google/vit-base-patch16-224")
    vit_base = TFAutoModel.from_config(config)

    img_input = L.Input(shape=(3, _IMAGE_SIZE, _IMAGE_SIZE))
    x = vit_base(img_input)
    y = L.Dense(1, activation="linear")(x[-1])
    model = Model(img_input, y)
    model.load_weights(weights_path)

    _model = model
    return _model
