"""
Image model wrapper — EfficientNetB3, ONNX Runtime edition.

This is the ONNX-optimized replacement for the original TensorFlow/Keras
image_model.py. It loads the exported image_model.onnx and performs
identical preprocessing and inference, but uses onnxruntime instead of
TensorFlow, reducing memory usage from ~1.5 GB to ~40 MB.

The preprocessing (EfficientNet preprocess_input) is reimplemented in
pure NumPy — no TensorFlow dependency required at inference time.
"""
import io
import json
import logging

import numpy as np
from PIL import Image

from app import config

logger = logging.getLogger("smart_dermatology.image_model")

_session = None
_classes = None
_img_size = None
_input_name = None


def load():
    """Load the ONNX model + label classes + preprocessing metadata once, at startup."""
    global _session, _classes, _img_size, _input_name

    import onnxruntime as ort

    logger.info("Loading ONNX image model from %s", config.IMAGE_MODEL_ONNX_PATH)

    # Use CPUExecutionProvider (default); no GPU needed on Koyeb free tier
    _session = ort.InferenceSession(
        config.IMAGE_MODEL_ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )
    _input_name = _session.get_inputs()[0].name

    with open(config.IMAGE_LABELS_PATH) as f:
        label_data = json.load(f)
    _classes = label_data["classes"]

    with open(config.IMAGE_META_PATH) as f:
        meta = json.load(f)
    _img_size = tuple(meta["img_size"])

    logger.info("ONNX Image model ready. classes=%s img_size=%s", _classes, _img_size)
    return _classes


def is_loaded() -> bool:
    return _session is not None


def get_classes():
    return _classes


def _efficientnet_preprocess(arr: np.ndarray) -> np.ndarray:
    """
    Pure-NumPy reimplementation of tf.keras.applications.efficientnet.preprocess_input.

    EfficientNet uses 'torch'-style preprocessing (mode='torch' in Keras):
        1. Scale pixel values from [0, 255] to [0, 1]
        2. Normalize using ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    """
    arr = arr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return arr


def predict(image_bytes: bytes):
    """
    Args:
        image_bytes: raw bytes of an uploaded JPG/PNG image.

    Returns:
        (probs: np.ndarray[num_classes], predicted_label: str, confidence: float 0-100)
    """
    if _session is None:
        raise RuntimeError("Image model not loaded yet")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(_img_size)
    arr = _efficientnet_preprocess(np.expand_dims(np.array(img), axis=0))

    probs = _session.run(None, {_input_name: arr})[0][0]
    pred_idx = int(np.argmax(probs))
    predicted_label = _classes[pred_idx]
    confidence = round(float(np.max(probs)) * 100, 2)
    return probs, predicted_label, confidence
