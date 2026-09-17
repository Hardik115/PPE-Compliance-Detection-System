# Mask / PPE Compliance Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)](https://opencv.org)
[![CPU-Only](https://img.shields.io/badge/Hardware-CPU--Only-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)]()

An automated, CPU-only computer vision pipeline that detects human faces in a
webcam/video feed, classifies whether each person is wearing a mask, logs
violations to a SQLite database, and generates a rich HTML compliance report.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🎯 Face Detection | OpenCV Haar Cascade (default) or MediaPipe |
| 🧠 Mask Classification | MobileNetV2 fine-tuned CNN — CPU inference ≤ 200 ms/frame |
| 📊 Live HUD | Real-time FPS, violation count, compliance rate overlay |
| 🗃 Violation Logging | SQLite with timestamps, confidence scores, face crop snapshots |
| 📑 HTML Report | Embedded matplotlib charts (pie, hourly bar, confidence histogram) |
| 📹 Webcam Fallback | Automatically falls back to `assets/sample_video.mp4` |
| ⌨ Keyboard Controls | `q` quit · `r` report · `s` snapshot · `p` pause |

---

## 🗂 Project Structure

```
mask-compliance-detector/
├── src/
│   ├── config.py          # Typed config loader (reads config.yaml)
│   ├── detector.py        # Face detection — Haar Cascade / MediaPipe
│   ├── classifier.py      # MobileNetV2 mask classifier + draw helper
│   ├── logger.py          # SQLite violation logger (thread-safe)
│   ├── report_gen.py      # HTML + TXT report generator with charts
│   └── main.py            # Entry point — live detection, image mode, report mode
├── train/
│   ├── train.py           # Two-phase MobileNetV2 training script
│   └── download_dataset.py# Auto-download & organise dataset
├── tests/
│   ├── test_detector.py
│   ├── test_classifier.py
│   ├── test_logger.py
│   └── test_report_gen.py
├── data/                  # Dataset (populated by download_dataset.py)
│   ├── with_mask/
│   └── without_mask/
├── models/
│   └── mask_classifier.h5 # Trained model (produced by train.py)
├── logs/
│   ├── violations.db      # SQLite database
│   └── snapshots/         # Saved violation face crops
├── docs/                  # Architecture & UML diagrams, training curves
├── reports/               # Generated compliance reports (.txt + .html)
├── assets/
│   └── sample_video.mp4   # Webcam fallback demo video
├── config.yaml            # ← All user-tunable settings live here
├── requirements.txt
├── generate_diagrams.py   # Generates all 5 project diagrams
└── generate_sample_video.py
```

---

## ⚡ Quick Start

### 1 — Install dependencies

```bash
# (Recommended) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

> **GPU note**: `requirements.txt` installs `tensorflow-cpu`. No CUDA required.

### 2 — Generate the sample fallback video

```bash
python generate_sample_video.py
```

### 3 — Download dataset & train the model

```bash
# Download ~190 MB dataset from GitHub, then train (fast mode ≈ 5 min on CPU)
python train/train.py --fast

# Full training (recommended for best accuracy, ≈ 30-60 min on CPU)
python train/train.py
```

Training produces:
- `models/mask_classifier.h5` — saved Keras model
- `docs/training_curves.png` — accuracy & loss plots

### 4 — Run the detector

```bash
# Live webcam (falls back to sample video if no camera found)
python -m src.main

# Process a video file
python -m src.main --source path/to/video.mp4

# Process a single image
python -m src.main --source photo.jpg --mode image

# Generate report from existing logs (no camera needed)
python -m src.main --mode report
```

### 5 — Generate all project diagrams

```bash
python generate_diagrams.py
# Saves: docs/architecture_diagram.png, uml_use_case.png,
#        uml_class.png, uml_sequence.png, er_diagram.png
```

### 6 — Run tests

```bash
pytest tests/ -v --tb=short
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## ⚙️ Configuration (`config.yaml`)

All parameters are in one place — no code changes needed:

```yaml
detection:
  backend: "haar"           # "haar" | "mediapipe"
  haar_scale_factor: 1.1
  haar_min_neighbors: 5

classification:
  confidence_threshold: 0.70  # below → orange "uncertain" box

input:
  source: 0                   # 0 = webcam; or path to file
  frame_skip: 2               # process every 2nd frame (speed vs accuracy)

logging:
  save_snapshots: true        # save face crop images for each violation
  log_compliant: false        # log only violations (false) or all detections

training:
  epochs_head: 10
  epochs_finetune: 5
  batch_size: 32
```

---

## 🎮 Keyboard Controls (Live Mode)

| Key | Action |
|---|---|
| `q` | Quit and save session |
| `r` | Generate HTML + TXT report immediately |
| `s` | Save current frame as a manual snapshot |
| `p` | Pause / resume video feed |

---

## 📊 Report Output

Reports are saved to `reports/` as `report_YYYYMMDD_HHMMSS.html` and `.txt`.

The HTML report includes:
- **Summary cards** — total events, compliant count, violations, compliance rate
- **Donut chart** — Compliant vs. Violations
- **Bar chart** — Violations per hour (UTC)
- **Histogram** — Classifier confidence score distribution
- **Violation table** — Last 50 events with timestamp, label, confidence, camera ID

---

## 🏗 Architecture

```
Input (Webcam / Video / Image)
         │
         ▼
  ┌─────────────────┐
  │  Face Detection │  detector.py — OpenCV Haar Cascade
  │  (Module 1)     │
  └────────┬────────┘
           │ face crops (BoundingBox list)
           ▼
  ┌─────────────────────┐
  │  Mask Classifier    │  classifier.py — MobileNetV2 (CPU)
  │  (Module 2)         │  → "Mask" / "No Mask" + confidence
  └────────┬────────────┘
           │ violations
           ▼
  ┌─────────────────────┐
  │  Logger (Module 3)  │  logger.py — SQLite + snapshot save
  └────────┬────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  Report Generator   │  report_gen.py — HTML + TXT + charts
  └─────────────────────┘
```

---

## 🧪 Test Coverage

| Module | Tests | Coverage Areas |
|---|---|---|
| `detector.py` | 12 | init, blank/None frames, BoundingBox, crop_face, factory |
| `classifier.py` | 14 | preprocessing shape/dtype/range, mocked model, draw_prediction |
| `logger.py` | 18 | DB init, sessions, violations, snapshots, summary, concurrency |
| `report_gen.py` | 14 | txt/html paths, content, charts (b64), edge cases |

---

## 📐 Database Schema

```sql
CREATE TABLE sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT    NOT NULL,   -- ISO 8601 UTC
    ended_at      TEXT,
    total_frames  INTEGER DEFAULT 0,
    camera_id     TEXT    DEFAULT 'cam0'
);

CREATE TABLE violations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER REFERENCES sessions(id),
    timestamp     TEXT    NOT NULL,
    label         TEXT    NOT NULL,   -- "Mask" | "No Mask"
    confidence    REAL    NOT NULL,
    snapshot_path TEXT,
    camera_id     TEXT    DEFAULT 'cam0',
    frame_index   INTEGER
);
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Frame capture, Haar Cascade, image I/O |
| `tensorflow-cpu` | MobileNetV2 model inference |
| `mediapipe` | Optional alternate face detector |
| `numpy` | Array operations |
| `matplotlib` | Report charts |
| `Jinja2` | HTML report template engine |
| `PyYAML` | Config file parsing |
| `tqdm` | Dataset download progress bar |
| `requests` | Dataset HTTP download |
| `pytest` | Unit testing |

---

## 📄 License

MIT License — free to use, modify, and distribute.
