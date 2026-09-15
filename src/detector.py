"""
detector.py — Face Detection Module

Detects and localises human faces in a BGR frame (numpy array).
Supports two backends selectable via config.yaml:
  - "haar"      : OpenCV Haar Cascade (default, lightest CPU load)
  - "mediapipe" : Google MediaPipe Face Detection (more accurate)

Returns a list of BoundingBox named-tuples for each detected face.
"""

from __future__ import annotations

import os
from collections import namedtuple
from typing import List

import cv2
import numpy as np

# Optional import — mediapipe may not be installed in all envs
try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

from src.config import DetectionConfig

# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────

BoundingBox = namedtuple("BoundingBox", ["x", "y", "w", "h"])

# Path to OpenCV's bundled Haar cascade XML
_HAAR_CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,  # type: ignore[attr-defined]
    "haarcascade_frontalface_default.xml",
)


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseDetector:
    """Abstract face detector interface."""

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ─────────────────────────────────────────────────────────────────────────────
# Haar Cascade Detector
# ─────────────────────────────────────────────────────────────────────────────

class HaarCascadeDetector(BaseDetector):
    """
    Face detector using OpenCV Haar Cascade.

    Lightweight, CPU-only, zero external dependencies beyond opencv.
    Works well for frontal faces in reasonable lighting.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._cascade = cv2.CascadeClassifier(_HAAR_CASCADE_PATH)
        if self._cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from: {_HAAR_CASCADE_PATH}\n"
                "Ensure opencv-python is correctly installed."
            )

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Detect faces in a BGR frame.

        Args:
            frame: OpenCV BGR image (H x W x 3).

        Returns:
            List of BoundingBox(x, y, w, h) — may be empty.
        """
        if frame is None or frame.size == 0:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improve contrast

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._config.haar_scale_factor,
            minNeighbors=self._config.haar_min_neighbors,
            minSize=self._config.haar_min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return []

        return [BoundingBox(x, y, w, h) for x, y, w, h in faces]


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Detector
# ─────────────────────────────────────────────────────────────────────────────

class MediaPipeDetector(BaseDetector):
    """
    Face detector backed by Google MediaPipe Face Detection.

    More accurate than Haar Cascade but requires the mediapipe package.
    Uses model_selection=0 (short-range, ≤2m) for real-time webcam use.
    """

    def __init__(self, config: DetectionConfig) -> None:
        if not _MP_AVAILABLE:
            raise ImportError(
                "mediapipe is not installed. "
                "Run: pip install mediapipe  OR switch to backend: haar"
            )
        self._config = config
        mp_face = mp.solutions.face_detection  # type: ignore[attr-defined]
        self._detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return []

        boxes: List[BoundingBox] = []
        for detection in results.detections:
            bb = detection.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            # Clamp to frame boundaries
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            if bw > 0 and bh > 0:
                boxes.append(BoundingBox(x, y, bw, bh))

        return boxes

    def __del__(self):
        if hasattr(self, "_detector"):
            self._detector.close()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_detector(config: DetectionConfig) -> BaseDetector:
    """
    Factory function — returns the appropriate detector based on config.

    Args:
        config: DetectionConfig with backend field.

    Returns:
        A BaseDetector instance (HaarCascadeDetector or MediaPipeDetector).

    Raises:
        ValueError: If backend name is unrecognised.
    """
    backend = config.backend.lower().strip()
    if backend == "haar":
        return HaarCascadeDetector(config)
    elif backend == "mediapipe":
        return MediaPipeDetector(config)
    else:
        raise ValueError(
            f"Unknown detection backend: '{backend}'. "
            "Valid options: 'haar', 'mediapipe'"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_face(frame: np.ndarray, box: BoundingBox, padding: float = 0.15) -> np.ndarray:
    """
    Crop and return the face region from a frame, with optional padding.

    Args:
        frame:   BGR image.
        box:     BoundingBox to crop.
        padding: Fractional padding around the box (default 15%).

    Returns:
        Cropped BGR face image. May be empty if box is out-of-frame.
    """
    h, w = frame.shape[:2]
    pad_x = int(box.w * padding)
    pad_y = int(box.h * padding)
    x1 = max(0, box.x - pad_x)
    y1 = max(0, box.y - pad_y)
    x2 = min(w, box.x + box.w + pad_x)
    y2 = min(h, box.y + box.h + pad_y)
    return frame[y1:y2, x1:x2]
