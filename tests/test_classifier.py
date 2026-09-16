"""
test_classifier.py — Unit tests for the MaskClassifier module.

Tests classifier preprocessing, output types, confidence range,
and draw_prediction helper — without requiring a real model file
by using a mock Keras model.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ClassificationConfig
from src.classifier import MaskClassifier, draw_prediction
from src.detector import BoundingBox


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cls_config():
    return ClassificationConfig(
        model_path="models/mask_classifier.h5",
        input_size=(224, 224),
        confidence_threshold=0.70,
        labels=["Mask", "No Mask"],
    )


@pytest.fixture
def classifier(cls_config):
    return MaskClassifier(cls_config)


@pytest.fixture
def face_bgr():
    """A 100×100 BGR image simulating a face crop."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return img


@pytest.fixture
def mock_model_mask():
    """Mock Keras model that always predicts 'Mask' with 0.95 confidence."""
    m = MagicMock()
    m.predict.return_value = np.array([[0.95, 0.05]])
    return m


@pytest.fixture
def mock_model_no_mask():
    """Mock Keras model that always predicts 'No Mask' with 0.92 confidence."""
    m = MagicMock()
    m.predict.return_value = np.array([[0.08, 0.92]])
    return m


# ── Initialisation tests ──────────────────────────────────────────────────────

class TestMaskClassifierInit:
    def test_repr(self, classifier):
        r = repr(classifier)
        assert "MaskClassifier" in r
        assert "mask_classifier.h5" in r

    def test_model_not_loaded_at_init(self, classifier):
        assert classifier._model is None


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_output_shape(self, classifier, face_bgr):
        blob = classifier._preprocess(face_bgr)
        assert blob.shape == (1, 224, 224, 3), f"Got {blob.shape}"

    def test_output_dtype(self, classifier, face_bgr):
        blob = classifier._preprocess(face_bgr)
        assert blob.dtype == np.float32

    def test_values_in_mobilenet_range(self, classifier, face_bgr):
        """MobileNetV2 preprocess_input scales to [-1, 1]."""
        blob = classifier._preprocess(face_bgr)
        assert blob.min() >= -1.1, f"Min value {blob.min()} out of range"
        assert blob.max() <= 1.1, f"Max value {blob.max()} out of range"


# ── Prediction tests (mocked model) ──────────────────────────────────────────

class TestPredictMask:
    def test_predict_mask_label(self, classifier, face_bgr, mock_model_mask):
        classifier._model = mock_model_mask
        label, conf = classifier.predict(face_bgr)
        assert label == "Mask"
        assert 0.0 <= conf <= 1.0
        assert conf == pytest.approx(0.95, abs=1e-3)

    def test_predict_no_mask_label(self, classifier, face_bgr, mock_model_no_mask):
        classifier._model = mock_model_no_mask
        label, conf = classifier.predict(face_bgr)
        assert label == "No Mask"
        assert conf == pytest.approx(0.92, abs=1e-3)

    def test_confidence_rounded_to_4dp(self, classifier, face_bgr, mock_model_mask):
        classifier._model = mock_model_mask
        _, conf = classifier.predict(face_bgr)
        # Confidence should have at most 4 decimal places
        assert conf == round(conf, 4)

    def test_empty_face_raises_value_error(self, classifier):
        with pytest.raises(ValueError, match="non-empty"):
            classifier.predict(np.array([]))

    def test_none_face_raises_value_error(self, classifier):
        with pytest.raises(ValueError, match="non-empty"):
            classifier.predict(None)  # type: ignore[arg-type]


class TestIsCompliant:
    def test_mask_is_compliant(self, classifier):
        assert classifier.is_compliant("Mask") is True

    def test_no_mask_not_compliant(self, classifier):
        assert classifier.is_compliant("No Mask") is False

    def test_unknown_label(self, classifier):
        assert classifier.is_compliant("Unknown") is False


# ── Model loading tests ───────────────────────────────────────────────────────

class TestModelLoading:
    def test_missing_model_raises_file_not_found(self, cls_config, face_bgr):
        cfg = ClassificationConfig(
            model_path="nonexistent_model.h5",
            input_size=(224, 224),
            confidence_threshold=0.70,
            labels=["Mask", "No Mask"],
        )
        clf = MaskClassifier(cfg)
        with pytest.raises(FileNotFoundError, match="Model not found"):
            clf.predict(face_bgr)


# ── draw_prediction tests ─────────────────────────────────────────────────────

class TestDrawPrediction:
    def test_returns_ndarray(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = BoundingBox(x=100, y=100, w=80, h=80)
        result = draw_prediction(frame, box, "Mask", 0.95)
        assert isinstance(result, np.ndarray)
        assert result.shape == frame.shape

    def test_mask_draws_green_area(self):
        """Green pixels (BGR: 0,200,80) should appear after drawing 'Mask'."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = BoundingBox(x=100, y=100, w=80, h=80)
        draw_prediction(frame, box, "Mask", 0.95, threshold=0.70)
        # Check that some non-zero pixels were drawn
        assert frame.sum() > 0

    def test_no_mask_draws_red_area(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = BoundingBox(x=100, y=100, w=80, h=80)
        draw_prediction(frame, box, "No Mask", 0.88, threshold=0.70)
        assert frame.sum() > 0

    def test_uncertain_below_threshold(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        box = BoundingBox(x=100, y=100, w=80, h=80)
        # Low confidence → orange colour path
        draw_prediction(frame, box, "No Mask", 0.50, threshold=0.70)
        assert frame.sum() > 0
