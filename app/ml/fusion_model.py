"""
Fusion Model V2 wrapper.

Final deployed strategy:

    Global Weighted Late Fusion

    Image weight = 0.50
    Text weight  = 0.50

Fusion V2 improvements:
- Uses validation-selected fusion weights.
- Applies validation-derived temperature scaling to image probabilities.
- Text probabilities are already calibrated inside text_model.py.
- Normalises the final fused probability vector.

The image and text base models themselves remain unchanged.

IMPORTANT:
Fusion evaluation was performed using constructed same-class image/text
pairs because genuinely paired patient image-text records were unavailable.
The reported fusion results therefore represent experimental system
evaluation and are not clinical accuracy estimates.
"""

import json
import logging

import numpy as np

from app import config


logger = logging.getLogger(
    "smart_dermatology.fusion_model"
)


_classes = None
_best_method = None
_num_classes = None

_wa_w_img = None
_wa_w_txt = None

_image_temperature = 1.0


def _temperature_scale_probs(
    probs: np.ndarray,
    temperature: float
) -> np.ndarray:
    """
    Apply temperature scaling directly to an existing
    probability distribution.

    Equivalent to:
        softmax(log(probabilities) / temperature)

    This allows the saved EfficientNet probability output
    to use the validation-derived image temperature without
    modifying or retraining the image model.
    """

    probs = np.asarray(
        probs,
        dtype=np.float64
    )

    if probs.ndim != 1:
        raise ValueError(
            "Expected a 1D image probability vector"
        )

    if temperature <= 0:
        raise ValueError(
            "Image temperature must be greater than zero"
        )

    eps = 1e-12

    probs = np.clip(
        probs,
        eps,
        1.0
    )

    # Convert probability distribution back to
    # log-probability space.
    log_probs = np.log(probs)

    scaled_logits = (
        log_probs / temperature
    )

    # Numerical stability
    scaled_logits -= np.max(
        scaled_logits
    )

    exp_values = np.exp(
        scaled_logits
    )

    calibrated = (
        exp_values /
        np.sum(exp_values)
    )

    return calibrated.astype(
        np.float32
    )


def load():
    """
    Load final Fusion V2 metadata.
    """

    global _classes
    global _best_method
    global _num_classes
    global _wa_w_img
    global _wa_w_txt
    global _image_temperature

    # ---------------------------------------------------------
    # Load metadata
    # ---------------------------------------------------------

    with open(
        config.FUSION_META_PATH,
        encoding="utf-8"
    ) as f:
        meta = json.load(f)

    _classes = meta["classes"]

    _num_classes = len(
        _classes
    )

    _best_method = meta[
        "best_method"
    ]

    # ---------------------------------------------------------
    # Image calibration
    # ---------------------------------------------------------

    _image_temperature = float(
        meta.get(
            "image_temperature",
            1.0
        )
    )

    if _image_temperature <= 0:
        raise ValueError(
            "Invalid image_temperature "
            "in fusion_meta.json"
        )

    # ---------------------------------------------------------
    # Final method
    # ---------------------------------------------------------

    if _best_method != "Weighted Average":
        raise ValueError(
            "Fusion V2 backend expects "
            "'Weighted Average'. "
            f"Found: {_best_method}"
        )

    weights = meta[
        "weighted_average_weights"
    ]

    # Final V2 uses GLOBAL weights,
    # not manually overridden per-class weights.
    if "per_class" in weights:
        raise ValueError(
            "Final Fusion V2 should use global "
            "validation-selected weights, not "
            "the old manual per-class weighting."
        )

    _wa_w_img = float(
        weights["image"]
    )

    _wa_w_txt = float(
        weights["text"]
    )

    # ---------------------------------------------------------
    # Validate weights
    # ---------------------------------------------------------

    if _wa_w_img < 0 or _wa_w_txt < 0:
        raise ValueError(
            "Fusion weights cannot be negative"
        )

    weight_sum = (
        _wa_w_img +
        _wa_w_txt
    )

    if weight_sum <= 0:
        raise ValueError(
            "Fusion weights must sum to "
            "a positive value"
        )

    # Normalise just in case metadata contains
    # very small floating-point differences.
    _wa_w_img /= weight_sum
    _wa_w_txt /= weight_sum

    logger.info(
        "Fusion V2 ready. "
        "method=%s | image_weight=%.2f | "
        "text_weight=%.2f | image_temperature=%.4f",
        _best_method,
        _wa_w_img,
        _wa_w_txt,
        _image_temperature
    )

    logger.info(
        "Fusion classes=%s",
        _classes
    )

    return (
        _classes,
        _best_method
    )


def is_loaded() -> bool:
    return _classes is not None


def get_classes():
    return _classes


def get_method():
    return _best_method


def get_weights():
    return {
        "image": _wa_w_img,
        "text": _wa_w_txt
    }


def get_image_temperature():
    return _image_temperature


def fuse(
    img_probs: np.ndarray,
    txt_probs: np.ndarray
) -> np.ndarray:
    """
    Fuse image and text probability vectors.

    Pipeline:

        EfficientNet raw probabilities
                ↓
        image temperature calibration
                ↓
        50% image

        Text Model V2 probabilities
        (already temperature calibrated)
                ↓
        50% text

                ↓
        weighted late fusion
                ↓
        normalised final probabilities
    """

    if _classes is None:
        raise RuntimeError(
            "Fusion model not loaded yet"
        )

    # ---------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------

    img_probs = np.asarray(
        img_probs,
        dtype=np.float32
    ).reshape(-1)

    txt_probs = np.asarray(
        txt_probs,
        dtype=np.float32
    ).reshape(-1)

    if len(img_probs) != _num_classes:
        raise ValueError(
            f"Expected {_num_classes} image "
            f"probabilities, got {len(img_probs)}"
        )

    if len(txt_probs) != _num_classes:
        raise ValueError(
            f"Expected {_num_classes} text "
            f"probabilities, got {len(txt_probs)}"
        )

    if not np.isfinite(img_probs).all():
        raise ValueError(
            "Image probabilities contain "
            "invalid values"
        )

    if not np.isfinite(txt_probs).all():
        raise ValueError(
            "Text probabilities contain "
            "invalid values"
        )

    if np.any(img_probs < 0):
        raise ValueError(
            "Image probabilities cannot "
            "contain negative values"
        )

    if np.any(txt_probs < 0):
        raise ValueError(
            "Text probabilities cannot "
            "contain negative values"
        )

    # ---------------------------------------------------------
    # Normalise original distributions
    # ---------------------------------------------------------

    img_sum = float(
        np.sum(img_probs)
    )

    txt_sum = float(
        np.sum(txt_probs)
    )

    if img_sum <= 0:
        raise ValueError(
            "Image probability vector "
            "has zero total probability"
        )

    if txt_sum <= 0:
        raise ValueError(
            "Text probability vector "
            "has zero total probability"
        )

    img_probs = (
        img_probs / img_sum
    )

    txt_probs = (
        txt_probs / txt_sum
    )

    # ---------------------------------------------------------
    # Fusion V2 image calibration
    # ---------------------------------------------------------

    calibrated_img_probs = (
        _temperature_scale_probs(
            img_probs,
            _image_temperature
        )
    )

    # Text probabilities arriving here are already
    # calibrated by app/ml/text_model.py.

    # ---------------------------------------------------------
    # Global weighted late fusion
    # ---------------------------------------------------------

    fused_probs = (
        _wa_w_img *
        calibrated_img_probs
        +
        _wa_w_txt *
        txt_probs
    )

    # ---------------------------------------------------------
    # Final normalisation
    # ---------------------------------------------------------

    fused_sum = float(
        np.sum(fused_probs)
    )

    if fused_sum <= 0:
        raise RuntimeError(
            "Fusion produced an invalid "
            "probability vector"
        )

    fused_probs = (
        fused_probs /
        fused_sum
    )

    return fused_probs.astype(
        np.float32
    )