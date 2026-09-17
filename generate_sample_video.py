"""
generate_sample_video.py — Create a synthetic fallback sample video.

Generates a 10-second 30fps video (assets/sample_video.mp4) with
animated text overlays simulating a real camera feed, so main.py has
something to fall back to when no webcam is available.

Run once from the project root:
  python generate_sample_video.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

OUT_PATH = Path("assets") / "sample_video.mp4"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 640, 480
FPS = 30
DURATION_S = 10
TOTAL_FRAMES = FPS * DURATION_S

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))

font = cv2.FONT_HERSHEY_DUPLEX
np.random.seed(42)

print(f"Generating {DURATION_S}s sample video …")
for i in range(TOTAL_FRAMES):
    # Dark gradient background
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    frame[:] = (13, 17, 23)

    # Animated scan-line effect
    scan_y = int((i / TOTAL_FRAMES) * HEIGHT * 3) % HEIGHT
    cv2.line(frame, (0, scan_y), (WIDTH, scan_y), (30, 50, 70), 1)

    # Simulated camera grid overlay
    for gx in range(0, WIDTH, 80):
        cv2.line(frame, (gx, 0), (gx, HEIGHT), (20, 30, 40), 1)
    for gy in range(0, HEIGHT, 60):
        cv2.line(frame, (0, gy), (WIDTH, gy), (20, 30, 40), 1)

    # Corner brackets
    for cx, cy in [(60, 60), (WIDTH-60, 60), (60, HEIGHT-60), (WIDTH-60, HEIGHT-60)]:
        cv2.rectangle(frame, (cx-45, cy-35), (cx+45, cy+35), (88, 166, 255), 1)

    # Simulated face placeholder
    face_cx, face_cy = 320, 240
    jitter_x = int(np.sin(i * 0.05) * 15)
    jitter_y = int(np.cos(i * 0.07) * 10)
    cv2.ellipse(frame, (face_cx + jitter_x, face_cy + jitter_y),
                (75, 90), 0, 0, 360, (160, 140, 120), -1)
    # Mask overlay (green rectangle covering lower face)
    mx, my = face_cx + jitter_x - 55, face_cy + jitter_y + 10
    cv2.rectangle(frame, (mx, my), (mx + 110, my + 60), (20, 80, 30), -1)
    cv2.rectangle(frame, (mx, my), (mx + 110, my + 60), (63, 185, 80), 2)

    # Status text
    ts = f"Frame: {i+1:04d}  |  FPS: 30"
    cv2.putText(frame, ts, (10, 20), font, 0.42, (139, 148, 158), 1, cv2.LINE_AA)
    cv2.putText(frame, "DEMO — No Webcam Fallback Feed", (10, HEIGHT - 10),
                font, 0.40, (88, 166, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Mask Compliance Monitor", (10, 45),
                font, 0.55, (88, 166, 255), 1, cv2.LINE_AA)

    # Pulsing bounding box on simulated face
    pulse = int(abs(np.sin(i * 0.1)) * 3)
    cv2.rectangle(frame,
                  (face_cx + jitter_x - 80 - pulse, face_cy + jitter_y - 95 - pulse),
                  (face_cx + jitter_x + 80 + pulse, face_cy + jitter_y + 95 + pulse),
                  (63, 185, 80), 2)
    cv2.putText(frame, "✓ Mask (97%)",
                (face_cx + jitter_x - 75, face_cy + jitter_y - 100),
                font, 0.50, (63, 185, 80), 1, cv2.LINE_AA)

    writer.write(frame)

writer.release()
print(f"[OK] Sample video saved: {OUT_PATH}  ({TOTAL_FRAMES} frames)")
