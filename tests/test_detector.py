"""
test_detector.py — Unit tests for the face detection module.

Tests cover:
- HaarCascadeDetector initialization
- Empty frame handling
- BoundingBox output type/values
- crop_face utility function
- Factory function routing
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DetectionConfig
from src.detector import (
    BoundingBox,
    HaarCascadeDetector,
    crop_face,
    create_detector,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def haar_config():
    return DetectionConfig(
        backend="haar",
        haar_scale_factor=1.1,
        haar_min_neighbors=5,
        haar_min_size=(30, 30),
    )


@pytest.fixture
def detector(haar_config):
    return HaarCascadeDetector(haar_config)


@pytest.fixture
def blank_frame():
    """All-black 480×640 frame — no faces expected."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_face_frame():
    """
    A grey face-like circle on a white background.
    Haar Cascade may or may not detect it; we test that output is well-formed.
    """
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    # Draw a flesh-toned ellipse to simulate a face region
    cv2.ellipse(frame, (320, 240), (80, 100), 0, 0, 360, (180, 160, 140), -1)
    # Eyes
    cv2.circle(frame, (290, 210), 12, (50, 50, 50), -1)
    cv2.circle(frame, (350, 210), 12, (50, 50, 50), -1)
    return frame


# ── Initialization tests ──────────────────────────────────────────────────────

class TestHaarCascadeDetectorInit:
    def test_creates_successfully(self, haar_config):
        det = HaarCascadeDetector(haar_config)
        assert det is not None

    def test_repr_contains_class_name(self, detector):
        assert "HaarCascadeDetector" in repr(detector)


# ── Detection tests ───────────────────────────────────────────────────────────

class TestHaarCascadeDetectorDetect:
    def test_blank_frame_returns_empty_list(self, detector, blank_frame):
        boxes = detector.detect(blank_frame)
        assert isinstance(boxes, list)
        assert len(boxes) == 0

    def test_none_frame_returns_empty(self, detector):
        boxes = detector.detect(None)  # type: ignore[arg-type]
        assert boxes == []

    def test_empty_array_returns_empty(self, detector):
        boxes = detector.detect(np.array([]))
        assert boxes == []

    def test_output_type_is_bounding_box(self, detector, synthetic_face_frame):
        boxes = detector.detect(synthetic_face_frame)
        for box in boxes:
            assert isinstance(box, BoundingBox)
            assert box.w > 0
            assert box.h > 0
            assert box.x >= 0
            assert box.y >= 0

    def test_multi_face_frame_multiple_boxes(self, detector):
        """Two side-by-side face regions — detector may find 0..N boxes."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        for cx in [160, 480]:
            cv2.ellipse(frame, (cx, 240), (60, 80), 0, 0, 360, (180, 160, 140), -1)
        boxes = detector.detect(frame)
        # We only assert output is well-formed, not specific count
        assert isinstance(boxes, list)


# ── crop_face tests ───────────────────────────────────────────────────────────

class TestCropFace:
    def test_basic_crop(self, blank_frame):
        box = BoundingBox(x=100, y=100, w=80, h=80)
        crop = crop_face(blank_frame, box, padding=0.0)
        assert crop.shape == (80, 80, 3)

    def test_crop_with_padding(self, blank_frame):
        box = BoundingBox(x=100, y=100, w=80, h=80)
        crop = crop_face(blank_frame, box, padding=0.2)
        # With 20% padding, crop should be larger than raw box
        assert crop.shape[0] >= 80
        assert crop.shape[1] >= 80

    def test_crop_edge_clamped(self, blank_frame):
        """Box near frame edge should clamp, not crash."""
        box = BoundingBox(x=600, y=440, w=80, h=80)
        crop = crop_face(blank_frame, box, padding=0.0)
        assert crop.size >= 0  # may be small but not an error


# ── Factory tests ─────────────────────────────────────────────────────────────

class TestCreateDetector:
    def test_haar_backend(self, haar_config):
        det = create_detector(haar_config)
        assert isinstance(det, HaarCascadeDetector)

    def test_unknown_backend_raises(self):
        cfg = DetectionConfig(backend="nonexistent")
        with pytest.raises(ValueError, match="Unknown detection backend"):
            create_detector(cfg)

    def test_case_insensitive_backend(self):
        cfg = DetectionConfig(backend="HAAR")
        det = create_detector(cfg)
        assert isinstance(det, HaarCascadeDetector)
