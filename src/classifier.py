"""
classifier.py — Mask Classification Module

Loads the trained MobileNetV2-based mask classifier and provides
a simple predict() interface that accepts a face crop (numpy BGR array)
and returns a (label, confidence) tuple.

Design notes:
- Model is loaded once and cached on first call (lazy init).
- Input preprocessing matches training pipeline exactly.
- Thread-safe for single-threaded inference loop.
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np

from src.config import ClassificationConfig

# Lazy TensorFlow import to keep startup time low when TF is not needed
_model_cache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Classifier class
# ─────────────────────────────────────────────────────────────────────────────

class MaskClassifier:
    """
    MobileNetV2-based binary mask classifier.

    Attributes:
        config: ClassificationConfig instance.
        model:  Loaded Keras model (loaded lazily on first predict call).
    """

    def __init__(self, config: ClassificationConfig) -> None:
        self._config = config
        self._model = None  # loaded lazily

    # ── Private helpers ──────────────────────────────────────────────────────

    def _load_model(self):
        """Load (or retrieve from cache) the Keras model."""
        model_path = self._config.model_path
        if model_path in _model_cache:
            return _model_cache[model_path]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at: {model_path}\n"
                "Run  python train/train.py  to train and save the model first."
            )

        # Import TF here to avoid penalising startup for non-inference runs
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # type: ignore

        # Suppress verbose TF logging
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        model = keras.models.load_model(model_path)
        _model_cache[model_path] = model
        return model

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess a BGR face crop for MobileNetV2 inference.

        Steps:
        1. Resize to model input size (224×224)
        2. Convert BGR → RGB
        3. Apply MobileNetV2 preprocess_input (scales to [-1, 1])
        4. Add batch dimension → (1, H, W, 3)
        """
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore

        h, w = self._config.input_size
        resized = cv2.resize(face_bgr, (w, h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32)
        arr = preprocess_input(arr)
        return np.expand_dims(arr, axis=0)

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(self, face_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Classify a face crop as 'Mask' or 'No Mask'.

        Args:
            face_bgr: BGR face crop (numpy array). Must be non-empty.

        Returns:
            Tuple of (label: str, confidence: float [0.0–1.0]).
            label is one of self._config.labels.

        Raises:
            ValueError: If face_bgr is empty or None.
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("face_bgr must be a non-empty numpy array.")

        if self._model is None:
            self._model = self._load_model()

        blob = self._preprocess(face_bgr)
        preds = self._model.predict(blob, verbose=0)  # shape: (1, 2)

        # preds[0] = [prob_mask, prob_no_mask]
        mask_prob = float(preds[0][0])
        no_mask_prob = float(preds[0][1])

        if mask_prob >= no_mask_prob:
            label = self._config.labels[0]   # "Mask"
            confidence = mask_prob
        else:
            label = self._config.labels[1]   # "No Mask"
            confidence = no_mask_prob

        return label, round(confidence, 4)

    def is_compliant(self, label: str) -> bool:
        """Return True if the label represents mask compliance."""
        return label == self._config.labels[0]

    def __repr__(self) -> str:
        return (
            f"MaskClassifier(model={self._config.model_path!r}, "
            f"threshold={self._config.confidence_threshold})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Draw utility (lives here to keep main.py clean)
# ─────────────────────────────────────────────────────────────────────────────

# Colour constants (BGR)
_COLOR_MASK = (0, 200, 80)       # green  — compliant
_COLOR_NO_MASK = (0, 60, 220)    # red    — violation
_COLOR_UNCERTAIN = (0, 165, 255) # orange — below threshold

_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_SCALE = 0.65
_THICKNESS = 2


def draw_prediction(
    frame: np.ndarray,
    box,                      # BoundingBox namedtuple
    label: str,
    confidence: float,
    threshold: float = 0.70,
) -> np.ndarray:
    """
    Draw a bounding box and label overlay on the frame in-place.

    Args:
        frame:      BGR image to annotate.
        box:        BoundingBox(x, y, w, h) of the face.
        label:      Classification label string.
        confidence: Classifier confidence [0.0–1.0].
        threshold:  Below this → orange "uncertain" colour.

    Returns:
        Annotated frame (same object, modified in-place).
    """
    x, y, w, h = box.x, box.y, box.w, box.h

    if confidence < threshold:
        color = _COLOR_UNCERTAIN
        display_label = f"? {label} ({confidence:.0%})"
    elif label == "Mask":
        color = _COLOR_MASK
        display_label = f"✓ Mask ({confidence:.0%})"
    else:
        color = _COLOR_NO_MASK
        display_label = f"✗ No Mask ({confidence:.0%})"

    # Bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, _THICKNESS)

    # Label background pill
    (text_w, text_h), baseline = cv2.getTextSize(
        display_label, _FONT, _FONT_SCALE, _THICKNESS
    )
    label_y = y - 10 if y - 10 > text_h else y + h + text_h + 10
    cv2.rectangle(
        frame,
        (x, label_y - text_h - baseline),
        (x + text_w + 6, label_y + baseline),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        display_label,
        (x + 3, label_y),
        _FONT,
        _FONT_SCALE,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame
