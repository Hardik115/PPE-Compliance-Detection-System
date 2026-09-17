"""
main.py — Application Entry Point

Orchestrates the full detection pipeline:
  webcam / video / image  →  detector  →  classifier  →  logger  →  display

Usage
-----
  python -m src.main                          # default webcam (source=0)
  python -m src.main --source assets/sample_video.mp4
  python -m src.main --source photo.jpg --mode image
  python -m src.main --mode report            # generate report and exit

Keyboard shortcuts (live mode)
-------------------------------
  q  — quit
  r  — generate compliance report immediately
  s  — save current frame snapshot manually
  p  — pause / resume
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── ensure src/ is importable when run from project root ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import get_config
from src.detector import create_detector, crop_face
from src.classifier import MaskClassifier, draw_prediction
from src.logger import ViolationLogger
from src.report_gen import generate_report


# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

def _draw_hud(frame: np.ndarray, stats: dict, paused: bool) -> np.ndarray:
    """Draw a heads-up display showing frame stats."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent top bar
    cv2.rectangle(overlay, (0, 0), (w, 52), (13, 17, 23), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    # Title
    cv2.putText(frame, "Mask Compliance Monitor", (12, 22),
                font, 0.55, (88, 166, 255), 1, cv2.LINE_AA)
    # Stats row
    fps_txt = f"FPS: {stats.get('fps', 0):.1f}"
    face_txt = f"Faces: {stats.get('faces', 0)}"
    viol_txt = f"Violations: {stats.get('violations', 0)}"
    rate_txt = f"Rate: {stats.get('compliance_rate', 100):.0f}%"
    status_txt = "⏸ PAUSED" if paused else "● LIVE"
    status_color = (255, 165, 0) if paused else (63, 185, 80)

    cv2.putText(frame, fps_txt,   (12,  42), font, 0.42, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(frame, face_txt,  (100, 42), font, 0.42, (201, 209, 217), 1, cv2.LINE_AA)
    cv2.putText(frame, viol_txt,  (180, 42), font, 0.42, (248, 81, 73),   1, cv2.LINE_AA)
    cv2.putText(frame, rate_txt,  (320, 42), font, 0.42, (63, 185, 80),   1, cv2.LINE_AA)
    cv2.putText(frame, status_txt,(w - 100, 22), font, 0.42, status_color, 1, cv2.LINE_AA)

    # Key hint bar (bottom)
    hint = "  q: quit   r: report   s: snapshot   p: pause"
    cv2.rectangle(frame, (0, h - 22), (w, h), (13, 17, 23), cv2.FILLED)
    cv2.putText(frame, hint, (8, h - 7), font, 0.36, (139, 148, 158), 1, cv2.LINE_AA)

    return frame


def _resize_to_width(frame: np.ndarray, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    scale = target_w / w
    return cv2.resize(frame, (target_w, int(h * scale)))


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline core
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(source, cfg, headless: bool = False):
    """
    Main detection loop.

    Args:
        source:   int (webcam index) or str (file path).
        cfg:      AppConfig.
        headless: If True, skip cv2.imshow (for testing / CI).
    """
    detector = create_detector(cfg.detection)
    classifier = MaskClassifier(cfg.classification)
    vlogger = ViolationLogger(cfg.logging)

    session_id = vlogger.start_session(cfg.input.camera_id)

    # Try to open source; fall back to sample video
    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        fallback = cfg.input.fallback_video
        print(f"⚠  Could not open source '{source}'. Trying fallback: {fallback}")
        cap = cv2.VideoCapture(str(fallback))
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open video source '{source}' or fallback '{fallback}'.\n"
                "Connect a webcam or provide a valid video path."
            )

    frame_idx = 0
    violation_count = 0
    total_faces = 0
    frame_skip = cfg.input.frame_skip
    threshold = cfg.classification.confidence_threshold
    log_compliant = cfg.logging.log_compliant
    cam_id = cfg.input.camera_id

    paused = False
    fps_timer = time.time()
    fps_count = 0
    fps = 0.0

    print("✅ Detection running. Press 'q' to quit, 'r' for report, 'p' to pause.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("📹 Stream ended.")
                break

            frame_idx += 1
            fps_count += 1

            # FPS calculation
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = fps_count / elapsed
                fps_count = 0
                fps_timer = time.time()

            # Process every Nth frame
            if frame_idx % frame_skip != 0:
                continue

            # ── Detection ──────────────────────────────────────────────────
            boxes = detector.detect(frame)
            total_faces += len(boxes)

            # ── Classification per face ────────────────────────────────────
            for box in boxes:
                face_crop = crop_face(frame, box)
                if face_crop.size == 0:
                    continue

                try:
                    label, confidence = classifier.predict(face_crop)
                except Exception as exc:
                    print(f"⚠  Classifier error: {exc}")
                    continue

                is_violation = (label == cfg.classification.labels[1])  # "No Mask"

                # Log violations (and optionally compliant faces)
                if is_violation or log_compliant:
                    vlogger.log_violation(
                        session_id=session_id,
                        face_crop=face_crop if is_violation else np.array([]),
                        label=label,
                        confidence=confidence,
                        frame_index=frame_idx,
                        camera_id=cam_id,
                    )

                if is_violation:
                    violation_count += 1

                # Annotate frame
                draw_prediction(frame, box, label, confidence, threshold)

        # ── Resize & HUD ───────────────────────────────────────────────────
        if not headless:
            display = _resize_to_width(frame, cfg.input.display_width)
            logged = vlogger.get_summary()
            stats = {
                "fps": fps,
                "faces": len(boxes) if not paused else 0,
                "violations": logged["no_mask_count"],
                "compliance_rate": logged["compliance_rate"],
            }
            display = _draw_hud(display, stats, paused)
            cv2.imshow("Mask Compliance Monitor", display)

        # ── Key handling ───────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("r"):
            generate_report(vlogger, cfg.reporting.output_dir)
        elif key == ord("s") and not paused:
            snap_path = Path(cfg.logging.snapshot_dir) / f"manual_{frame_idx}.jpg"
            cv2.imwrite(str(snap_path), frame)
            print(f"📸 Snapshot saved: {snap_path}")

    # ── Teardown ───────────────────────────────────────────────────────────
    vlogger.end_session(session_id, total_frames=frame_idx)
    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    print(f"\n📊 Session summary: {frame_idx} frames, {violation_count} violations logged.")


def run_image_mode(image_path: str, cfg):
    """Process a single static image and display result."""
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    detector = create_detector(cfg.detection)
    classifier = MaskClassifier(cfg.classification)
    vlogger = ViolationLogger(cfg.logging)
    session_id = vlogger.start_session(cfg.input.camera_id)

    boxes = detector.detect(frame)
    print(f"Detected {len(boxes)} face(s).")

    for box in boxes:
        face_crop = crop_face(frame, box)
        if face_crop.size == 0:
            continue
        label, confidence = classifier.predict(face_crop)
        print(f"  → {label} ({confidence:.1%})")
        draw_prediction(frame, box, label, confidence, cfg.classification.confidence_threshold)
        if label == cfg.classification.labels[1]:
            vlogger.log_violation(session_id, face_crop, label, confidence)

    vlogger.end_session(session_id, total_frames=1)

    out_path = Path(image_path).stem + "_result.jpg"
    cv2.imwrite(out_path, frame)
    print(f"Result saved: {out_path}")

    cv2.imshow("Result", _resize_to_width(frame, cfg.input.display_width))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Mask/PPE Compliance Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", default=None,
        help="Input source: 0 for webcam, or path to video/image. "
             "Default: value from config.yaml",
    )
    parser.add_argument(
        "--mode", choices=["detect", "image", "report"], default="detect",
        help="'detect': live detection loop (default)  "
             "'image': process a single image  "
             "'report': generate report and exit",
    )
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml (auto-detected if omitted)"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without display (useful for servers / CI)"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = get_config(args.config)

    if args.mode == "report":
        vlogger = ViolationLogger(cfg.logging)
        generate_report(vlogger, cfg.reporting.output_dir)
        return

    # Resolve source
    if args.source is not None:
        try:
            source = int(args.source)
        except ValueError:
            source = args.source
    else:
        source = cfg.input.source

    if args.mode == "image":
        run_image_mode(str(source), cfg)
    else:
        run_detection(source, cfg, headless=args.headless)


if __name__ == "__main__":
    main()
