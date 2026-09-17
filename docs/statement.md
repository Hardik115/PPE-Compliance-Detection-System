# Problem Statement

## Mask / PPE Compliance Detection System

**Document Type:** Project Problem Statement  
**Version:** 1.0  
**Date:** September 2026  

---

## 1. Background and Motivation

The enforcement of mask and personal protective equipment (PPE) compliance in
workplaces, hospitals, manufacturing facilities, and public spaces is a
critical safety requirement. In high-risk environments — operating theatres,
cleanrooms, food-processing plants, and pandemic-response zones — mask
non-compliance directly correlates with elevated infection rates and
regulatory violations that can result in financial penalties or facility
shutdowns.

Manual monitoring by safety officers is:

- **Inconsistent** — Human attention lapses; officers cannot watch every
  entrance and corridor simultaneously.
- **Labour-intensive** — Dedicated headcount for compliance monitoring is
  costly and difficult to scale.
- **Not Scalable** — Expanding to a new building or shift requires
  proportional increases in staffing.
- **Lag-prone** — By the time a violation is noticed and addressed, the
  non-compliant individual may have already entered a restricted area.
- **Undocumented** — Without automated logging, there is no audit trail for
  compliance review or incident investigation.

---

## 2. Problem Definition

> *How can we automate real-time face-mask compliance monitoring across
> multiple camera feeds, on commodity CPU hardware, without requiring
> specialist GPU infrastructure, while generating a full audit trail of
> violations and on-demand compliance reports?*

---

## 3. Scope

| In Scope | Out of Scope |
|---|---|
| Webcam / video file input | Multi-camera network management |
| Face detection (OpenCV / MediaPipe) | Other PPE types (gloves, helmets) |
| Binary mask classification (Mask / No Mask) | Re-identification across cameras |
| SQLite violation logging + snapshots | Cloud deployment / streaming |
| HTML compliance report with charts | Mobile app interface |
| CPU-only inference | GPU-accelerated training at scale |

---

## 4. Stakeholders

| Role | Interest |
|---|---|
| **Safety Officer** | Real-time violation alerts, live annotated feed |
| **Facility Administrator** | Compliance rate reports, audit logs |
| **IT Department** | Minimal hardware requirements, easy deployment |
| **Regulatory Body** | Timestamped evidence of compliance enforcement |

---

## 5. Objectives

1. **Detect** human faces in real time from a webcam or pre-recorded video
   feed using lightweight OpenCV-based methods.
2. **Classify** each detected face crop as *Mask-Compliant* or
   *Non-Compliant* using a MobileNetV2-based CNN with ≥ 95% test accuracy.
3. **Log** every violation with ISO 8601 timestamp, classifier confidence,
   face crop image, and camera identifier to a persistent SQLite database.
4. **Report** compliance statistics on demand — total events, violation
   count, compliance rate, and per-hour violation trend — in both plain-text
   and interactive HTML formats.

---

## 6. Functional Requirements

### FR-1 — Face Detection Module
- The system SHALL detect and localise human faces in each processed frame.
- Detection SHALL return a bounding box `(x, y, w, h)` for each face.
- The system SHALL handle frames with zero detected faces without crashing.
- The system SHALL handle frames with multiple detected faces simultaneously.
- Detection backend SHALL be selectable: Haar Cascade or MediaPipe.

### FR-2 — Mask Classification Module
- The system SHALL classify each face crop as `"Mask"` or `"No Mask"`.
- Classification SHALL use a MobileNetV2-based CNN trained on a public
  face-mask dataset.
- The system SHALL output a confidence score `[0.0–1.0]` with each label.
- Predictions below the configured confidence threshold SHALL be marked
  as "uncertain" (orange bounding box).

### FR-3 — Logging & Storage Module
- The system SHALL persist each violation record to an SQLite database.
- Each record SHALL include: timestamp (UTC), label, confidence score,
  snapshot path, camera ID, and frame index.
- The system SHALL track monitoring sessions (start/end time, total frames).
- The system SHALL optionally save a JPEG snapshot of each violating face.

### FR-4 — Report Generation Module
- The system SHALL generate a plain-text summary report on demand.
- The system SHALL generate an HTML report with embedded charts (no server
  required — fully self-contained).
- Charts SHALL include: compliance donut, violations-per-hour bar chart,
  and classifier confidence histogram.
- Reports SHALL be timestamped and saved to the `reports/` directory.

### FR-5 — User Interface
- The system SHALL display a real-time annotated video feed with coloured
  bounding boxes (green = Mask, red = No Mask, orange = uncertain).
- The system SHALL overlay a HUD showing FPS, face count, violation count,
  and compliance rate.
- The system SHALL support keyboard controls: quit, pause, snapshot, report.
- The system SHALL fall back to a sample video when no webcam is detected.

---

## 7. Non-Functional Requirements

| ID | Requirement | Metric |
|---|---|---|
| NFR-1 | **Performance** | Inference ≤ 200 ms/frame on CPU |
| NFR-2 | **Accuracy** | ≥ 95% classification accuracy on test set |
| NFR-3 | **Reliability** | Zero crashes on empty/multi-face frames |
| NFR-4 | **Portability** | Runs on Windows / Linux / macOS with Python 3.9+ |
| NFR-5 | **Maintainability** | Modular codebase; each module independently testable |
| NFR-6 | **Usability** | Single config file; one command to start |
| NFR-7 | **Data Integrity** | All violations persisted atomically; WAL journal mode |
| NFR-8 | **Resource** | No GPU required; CUDA-free inference |

---

## 8. Technical Constraints

- **Language:** Python 3.9+
- **CV Library:** OpenCV ≥ 4.8
- **ML Framework:** TensorFlow/Keras ≥ 2.13 (CPU-only build)
- **Model Architecture:** MobileNetV2 (ImageNet pre-trained, fine-tuned)
- **Database:** SQLite 3 (via Python `sqlite3` standard library)
- **Hardware:** x86-64 or ARM CPU, ≥ 4 GB RAM, no GPU
- **OS:** Windows 10+, Ubuntu 20.04+, macOS 12+

---

## 9. Deliverables

| Deliverable | Location |
|---|---|
| Source code (7 modules) | `src/`, `train/` |
| Unit tests (4 suites) | `tests/` |
| Trained model | `models/mask_classifier.h5` |
| System architecture diagram | `docs/architecture_diagram.png` |
| UML Use Case diagram | `docs/uml_use_case.png` |
| UML Class diagram | `docs/uml_class.png` |
| UML Sequence diagram | `docs/uml_sequence.png` |
| ER diagram | `docs/er_diagram.png` |
| Training curves | `docs/training_curves.png` |
| README | `docs/README.md` |
| Problem statement | `docs/statement.md` |
| Sample compliance report | `reports/` (generated at runtime) |

---

## 10. Success Criteria

The project is considered successful when:

1. The detection + classification pipeline processes at ≥ 5 FPS on a
   standard laptop CPU.
2. The classifier achieves ≥ 95% accuracy on the held-out test split.
3. All 4 test suites pass with ≥ 90% code coverage on core modules.
4. A compliance HTML report is generated with correct statistics and
   embedded charts within 5 seconds of the `r` keypress.
5. The system runs end-to-end from `python -m src.main` with a single
   `pip install -r requirements.txt` setup step.
