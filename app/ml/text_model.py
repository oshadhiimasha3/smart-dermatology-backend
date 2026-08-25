"""
Text model wrapper — DistilBERT V2 multi-task, ONNX Runtime edition.

This is the ONNX-optimized replacement for the original PyTorch text_model.py.
It loads the exported text_model.onnx and performs identical tokenization,
inference, temperature scaling, and concern thresholding — but uses
onnxruntime instead of PyTorch, reducing memory from ~1+ GB to ~80 MB.

The HuggingFace tokenizer is kept (it is lightweight, ~2 MB) because
it must produce the exact same token IDs the model was trained on.
"""

import json
import logging

import numpy as np

from app import config


logger = logging.getLogger("smart_dermatology.text_model")


_session = None
_tokenizer = None

_main_classes = None
_concern_classes = None

_max_length = None

# V2 calibration
_main_temperature = 1.0

# Backward-compatible global fallback
_concern_threshold = 0.5

# V2 individual thresholds
_concern_thresholds = {}


def load():
    """
    Load Text Model V2 (ONNX), tokenizer, labels,
    calibration temperature and concern thresholds.
    """

    global _session
    global _tokenizer
    global _main_classes
    global _concern_classes
    global _max_length
    global _main_temperature
    global _concern_threshold
    global _concern_thresholds

    import onnxruntime as ort
    from tokenizers import Tokenizer

    # ---------------------------------------------------------
    # Load label metadata
    # ---------------------------------------------------------

    with open(config.TEXT_LABELS_PATH, encoding="utf-8") as f:
        label_data = json.load(f)

    _main_classes = label_data["main_condition_classes"]
    _concern_classes = label_data["additional_concern_classes"]

    # ---------------------------------------------------------
    # Load V2 model metadata
    # ---------------------------------------------------------

    with open(config.TEXT_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    _max_length = int(
        meta.get("max_length", 128)
    )

    # ---------------------------------------------------------
    # V2 main-condition calibration
    # ---------------------------------------------------------

    _main_temperature = float(
        meta.get("main_temperature", 1.0)
    )

    # Protect against invalid metadata
    if _main_temperature <= 0:
        logger.warning(
            "Invalid main_temperature=%s. Falling back to 1.0",
            _main_temperature
        )
        _main_temperature = 1.0

    # ---------------------------------------------------------
    # V2 additional-concern thresholds
    # ---------------------------------------------------------

    _concern_threshold = float(
        meta.get("concern_threshold", 0.5)
    )

    _concern_thresholds = meta.get(
        "concern_thresholds",
        {}
    )

    if not isinstance(_concern_thresholds, dict):
        logger.warning(
            "concern_thresholds missing or invalid. "
            "Using global threshold %.2f for all concerns.",
            _concern_threshold
        )
        _concern_thresholds = {}

    # ---------------------------------------------------------
    # Load tokenizer (lightweight, uses the HuggingFace
    # tokenizers library — NOT the full transformers package)
    # ---------------------------------------------------------

    logger.info(
        "Loading tokenizer from %s",
        config.TOKENIZER_DIR
    )

    # Use the fast tokenizer file directly (tokenizer.json)
    # This avoids importing the heavy `transformers` package.
    import os
    tokenizer_json_path = os.path.join(config.TOKENIZER_DIR, "tokenizer.json")
    _tokenizer = Tokenizer.from_file(tokenizer_json_path)

    # Configure padding and truncation to match the training setup
    from tokenizers import processors
    _tokenizer.enable_truncation(max_length=_max_length)
    _tokenizer.enable_padding(length=_max_length, pad_id=0, pad_token="[PAD]")

    # ---------------------------------------------------------
    # Load ONNX model
    # ---------------------------------------------------------

    logger.info(
        "Loading ONNX text model from %s",
        config.TEXT_MODEL_ONNX_PATH
    )

    _session = ort.InferenceSession(
        config.TEXT_MODEL_ONNX_PATH,
        providers=["CPUExecutionProvider"],
    )

    # ---------------------------------------------------------
    # Logging
    # ---------------------------------------------------------

    logger.info(
        "Text Model V2 (ONNX) ready. "
        "main_temperature=%.4f, "
        "global_concern_threshold=%.2f, "
        "individual_thresholds=%d",
        _main_temperature,
        _concern_threshold,
        len(_concern_thresholds)
    )

    logger.info(
        "Main classes=%s",
        _main_classes
    )

    logger.info(
        "Concern classes=%s",
        _concern_classes
    )

    return _main_classes, _concern_classes


def is_loaded() -> bool:
    return _session is not None


def get_classes():
    return _main_classes, _concern_classes


def get_temperature() -> float:
    """
    Returns the validation-derived main-condition
    calibration temperature.
    """
    return _main_temperature


def get_concern_thresholds():
    """
    Returns individual concern thresholds.
    """
    return dict(_concern_thresholds)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (pure NumPy)."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid function (pure NumPy)."""
    return 1.0 / (1.0 + np.exp(-x))


def predict(text: str):
    """
    Predict the main skin condition and additional concerns.

    Args:
        text:
            Free-form user description of skin concerns.

    Returns:
        (
            main_probs:
                np.ndarray with one calibrated probability
                for each of the 9 main classes,

            predicted_label:
                predicted main-condition label,

            confidence:
                calibrated model confidence from 0-100,

            additional_concerns:
                list of detected additional concerns
        )
    """

    if _session is None:
        raise RuntimeError(
            "Text model not loaded yet"
        )

    if text is None or not str(text).strip():
        raise ValueError(
            "Text description cannot be empty"
        )

    # ---------------------------------------------------------
    # Tokenisation (using the fast HuggingFace tokenizer)
    # ---------------------------------------------------------

    encoded = _tokenizer.encode(str(text))
    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

    # ---------------------------------------------------------
    # ONNX Runtime inference
    # ---------------------------------------------------------

    main_logits, concern_logits = _session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )

    # Results are numpy arrays of shape (1, num_classes)
    main_logits = main_logits[0]       # shape: (9,)
    concern_logits = concern_logits[0] # shape: (16,)

    # ---------------------------------------------------------
    # V2 MAIN-CONDITION CALIBRATION
    # ---------------------------------------------------------

    calibrated_main_logits = (
        main_logits / _main_temperature
    )

    main_probs = _softmax(
        calibrated_main_logits
    )

    concern_probs = _sigmoid(
        concern_logits
    )

    # ---------------------------------------------------------
    # Main condition
    # ---------------------------------------------------------

    pred_idx = int(
        np.argmax(main_probs)
    )

    predicted_label = _main_classes[pred_idx]

    confidence = round(
        float(main_probs[pred_idx]) * 100,
        2
    )

    # ---------------------------------------------------------
    # Additional concerns — V2 individual thresholds
    # ---------------------------------------------------------

    concerns = []

    for i, probability in enumerate(concern_probs):

        concern_name = _concern_classes[i]

        threshold = float(
            _concern_thresholds.get(
                concern_name,
                _concern_threshold
            )
        )

        if probability >= threshold:
            concerns.append(concern_name)

    return (
        main_probs.astype(np.float32),
        predicted_label,
        confidence,
        concerns
    )