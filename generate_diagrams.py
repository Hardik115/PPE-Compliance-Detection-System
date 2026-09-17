"""
generate_diagrams.py — Programmatically generate all 5 project diagrams.

Produces:
  docs/architecture_diagram.png
  docs/uml_use_case.png
  docs/uml_class.png
  docs/uml_sequence.png
  docs/er_diagram.png

Run from project root:
  python generate_diagrams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

DOCS = Path(__file__).parent / "docs"
DOCS.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG        = "#0d1117"
SURFACE   = "#161b22"
BORDER    = "#21262d"
BOX_BLUE  = "#1f6feb"
BOX_DARK  = "#0f3460"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"
TEXT      = "#c9d1d9"
TEXT_DIM  = "#8b949e"
ORANGE    = "#db6d28"


def _fig(w=14, h=8):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, label, sublabel="", color=BOX_BLUE, fontsize=9, radius=0.015):
    fancy = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        linewidth=1.5, edgecolor=ACCENT, facecolor=color, zorder=3,
    )
    ax.add_patch(fancy)
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha="center", va="center", color=TEXT,
            fontsize=fontsize, fontweight="bold", zorder=4)
    if sublabel:
        ax.text(x, y - 0.022, sublabel,
                ha="center", va="center", color=TEXT_DIM,
                fontsize=fontsize - 1.5, zorder=4, style="italic")


def _arrow(ax, x1, y1, x2, y2, color=ACCENT, lw=1.8, label="", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color, lw=lw,
                    linestyle=ls,
                    connectionstyle="arc3,rad=0.0",
                ), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.018, label, ha="center", va="bottom",
                color=ACCENT, fontsize=7, zorder=5)


def _title(ax, t):
    ax.text(0.5, 0.97, t, ha="center", va="top",
            color=ACCENT, fontsize=13, fontweight="bold", zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# 1. System Architecture Diagram
# ─────────────────────────────────────────────────────────────────────────────

def architecture_diagram():
    fig, ax = _fig(15, 9)
    _title(ax, "Mask/PPE Compliance Detection System — Architecture")

    # Input sources
    _box(ax, 0.10, 0.72, 0.14, 0.09, "[CAM] Webcam Feed", "OpenCV VideoCapture", BOX_DARK, 8)
    _box(ax, 0.10, 0.55, 0.14, 0.09, "[IMG] Static Image", "JPEG / PNG / MP4", BOX_DARK, 8)
    ax.text(0.10, 0.63, "OR", ha="center", va="center", color=TEXT_DIM, fontsize=8)

    # Module 1
    _box(ax, 0.29, 0.635, 0.16, 0.22, "Module 1\nFace Detection",
         "detector.py\nHaar Cascade / MediaPipe", BOX_BLUE, 8.5)

    # Module 2
    _box(ax, 0.50, 0.635, 0.16, 0.22, "Module 2\nMask Classifier",
         "classifier.py\nMobileNetV2 CNN (CPU)", BOX_BLUE, 8.5)

    # Outputs from Module 2
    _box(ax, 0.69, 0.75, 0.10, 0.07, "✓  Mask", "Compliant", "#0a3a0a", 8)
    _box(ax, 0.69, 0.52, 0.10, 0.07, "✗  No Mask", "Violation", "#3a0a0a", 8)

    # Module 3 — Logger
    _box(ax, 0.84, 0.635, 0.14, 0.22, "Module 3\nLogger",
         "logger.py\nSQLite DB\n+ Snapshots", BOX_BLUE, 8.5)

    # Support boxes bottom row
    _box(ax, 0.18, 0.22, 0.14, 0.09, "config.py", "config.yaml loader", BOX_DARK, 8)
    _box(ax, 0.36, 0.22, 0.14, 0.09, "main.py", "Orchestrator + HUD", BOX_DARK, 8)
    _box(ax, 0.55, 0.22, 0.14, 0.09, "report_gen.py", "HTML + Charts", BOX_DARK, 8)
    _box(ax, 0.74, 0.22, 0.14, 0.09, "train.py", "MobileNetV2 Training", BOX_DARK, 8)

    # Arrows — main pipeline
    _arrow(ax, 0.17, 0.72, 0.21, 0.68)
    _arrow(ax, 0.17, 0.55, 0.21, 0.60)
    _arrow(ax, 0.37, 0.635, 0.42, 0.635, label="face crops")
    _arrow(ax, 0.58, 0.73, 0.64, 0.76)
    _arrow(ax, 0.58, 0.54, 0.64, 0.52)
    _arrow(ax, 0.74, 0.52, 0.77, 0.58, color=RED, label="log violation")
    _arrow(ax, 0.77, 0.635, 0.84, 0.635, label="", color=RED)

    # Arrows to bottom
    _arrow(ax, 0.36, 0.27, 0.36, 0.26, color=TEXT_DIM)
    _arrow(ax, 0.84, 0.525, 0.55, 0.27, color=YELLOW, label="query")

    # Legend
    for i, (color, lbl) in enumerate([
        (ACCENT, "Data Flow"), (RED, "Violation Path"), (YELLOW, "Report Query")
    ]):
        ax.plot([0.05 + i*0.13], [0.10], "o-", color=color, markersize=5)
        ax.text(0.07 + i*0.13, 0.10, lbl, color=TEXT_DIM, fontsize=7.5, va="center")

    fig.savefig(str(DOCS / "architecture_diagram.png"), dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] architecture_diagram.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. UML Use Case Diagram
# ─────────────────────────────────────────────────────────────────────────────

def uml_use_case():
    fig, ax = _fig(14, 10)
    _title(ax, "UML Use Case Diagram — Mask Compliance Detection System")

    # System boundary
    system_rect = FancyBboxPatch((0.22, 0.06), 0.58, 0.82,
                                  boxstyle="round,pad=0.01",
                                  linewidth=2, edgecolor=ACCENT,
                                  facecolor=SURFACE, zorder=1)
    ax.add_patch(system_rect)
    ax.text(0.51, 0.90, "Mask Compliance Detection System",
            ha="center", va="center", color=ACCENT, fontsize=10, fontweight="bold")

    def use_case(x, y, text, w=0.17, h=0.065):
        ell = mpatches.Ellipse((x, y), w, h,
                                linewidth=1.4, edgecolor=BOX_BLUE,
                                facecolor="#0f2040", zorder=3)
        ax.add_patch(ell)
        ax.text(x, y, text, ha="center", va="center",
                color=TEXT, fontsize=7.5, fontweight="bold",
                zorder=4, wrap=True, multialignment="center")

    def actor(x, y, name):
        # Head
        ax.add_patch(plt.Circle((x, y + 0.055), 0.022, color=ACCENT, zorder=3))
        # Body
        ax.plot([x, x], [y + 0.033, y - 0.025], color=ACCENT, lw=2, zorder=3)
        # Arms
        ax.plot([x - 0.03, x + 0.03], [y + 0.01, y + 0.01], color=ACCENT, lw=2, zorder=3)
        # Legs
        ax.plot([x, x - 0.025], [y - 0.025, y - 0.065], color=ACCENT, lw=2, zorder=3)
        ax.plot([x, x + 0.025], [y - 0.025, y - 0.065], color=ACCENT, lw=2, zorder=3)
        ax.text(x, y - 0.085, name, ha="center", va="top",
                color=TEXT, fontsize=8, fontweight="bold")

    # Actors
    actor(0.10, 0.62, "Safety\nOfficer")
    actor(0.10, 0.28, "Administrator")
    actor(0.90, 0.65, "Camera\n(System)")

    # Use cases
    use_case(0.51, 0.77, "Monitor Live Feed")
    use_case(0.51, 0.63, "Detect Faces\nin Frame")
    use_case(0.51, 0.49, "Classify\nMask / No Mask")
    use_case(0.51, 0.35, "Log Violation\nto Database")
    use_case(0.51, 0.21, "Save Violation\nSnapshot")
    use_case(0.72, 0.77, "Generate\nCompliance Report")
    use_case(0.72, 0.63, "View Violation\nHistory")
    use_case(0.72, 0.49, "Configure\nSystem Settings")

    # Actor → UC associations
    def assoc(x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color=TEXT_DIM, lw=1.2, zorder=2)

    assoc(0.13, 0.66, 0.42, 0.77)
    assoc(0.13, 0.60, 0.63, 0.77)
    assoc(0.13, 0.64, 0.63, 0.63)
    assoc(0.13, 0.30, 0.63, 0.49)
    assoc(0.13, 0.32, 0.63, 0.77)

    assoc(0.87, 0.65, 0.60, 0.63)

    # Include / extend arrows (dashed)
    def include(x1, y1, x2, y2, lbl="«include»"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=YELLOW,
                                   lw=1.2, linestyle="dashed"), zorder=2)
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.02, my, lbl, color=YELLOW, fontsize=6.5)

    include(0.51, 0.745, 0.51, 0.66)
    include(0.51, 0.595, 0.51, 0.52)
    include(0.51, 0.465, 0.51, 0.385, "«extend»")
    include(0.51, 0.315, 0.51, 0.245)

    # Legend
    ax.plot([0.24, 0.32], [0.03, 0.03], color=TEXT_DIM, lw=1.2)
    ax.text(0.33, 0.03, "Association", color=TEXT_DIM, fontsize=7.5, va="center")
    ax.plot([0.45, 0.53], [0.03, 0.03], color=YELLOW, lw=1.2, linestyle="dashed")
    ax.text(0.54, 0.03, "«include» / «extend»", color=YELLOW, fontsize=7.5, va="center")

    fig.savefig(str(DOCS / "uml_use_case.png"), dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] uml_use_case.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. UML Class Diagram
# ─────────────────────────────────────────────────────────────────────────────

def uml_class():
    fig, ax = _fig(16, 10)
    _title(ax, "UML Class Diagram — Mask Compliance Detection System")

    def class_box(ax, x, y, name, stereotypes, attrs, methods, w=0.20, color=BOX_BLUE):
        line_h = 0.032
        n_lines = 1 + len(attrs) + len(methods) + 2  # name + dividers
        total_h = n_lines * line_h + 0.01

        # Outer box
        rect = FancyBboxPatch((x - w/2, y - total_h/2), w, total_h,
                               boxstyle="round,pad=0.005",
                               linewidth=1.5, edgecolor=ACCENT, facecolor=SURFACE, zorder=3)
        ax.add_patch(rect)

        cur_y = y + total_h/2 - line_h

        # Stereotype and name header
        if stereotypes:
            ax.text(x, cur_y, stereotypes, ha="center", va="center",
                    color=TEXT_DIM, fontsize=6.5, style="italic", zorder=4)
            cur_y -= line_h
        ax.text(x, cur_y, name, ha="center", va="center",
                color=ACCENT, fontsize=8.5, fontweight="bold", zorder=4)
        cur_y -= line_h * 0.6
        # Divider
        ax.plot([x - w/2 + 0.005, x + w/2 - 0.005], [cur_y, cur_y],
                color=BORDER, lw=0.8, zorder=4)
        cur_y -= line_h * 0.4

        # Attributes
        for a in attrs:
            ax.text(x - w/2 + 0.01, cur_y, a, ha="left", va="center",
                    color=TEXT, fontsize=7, zorder=4, family="monospace")
            cur_y -= line_h
        # Divider
        ax.plot([x - w/2 + 0.005, x + w/2 - 0.005], [cur_y + line_h*0.3, cur_y + line_h*0.3],
                color=BORDER, lw=0.8, zorder=4)

        # Methods
        for m in methods:
            ax.text(x - w/2 + 0.01, cur_y, m, ha="left", va="center",
                    color=GREEN, fontsize=7, zorder=4, family="monospace")
            cur_y -= line_h

        return total_h

    # Classes
    class_box(ax, 0.15, 0.72, "BaseDetector", "«abstract»",
               ["# config: DetectionConfig"],
               ["+ detect(frame) → List[BoundingBox]"])
    class_box(ax, 0.15, 0.45, "HaarCascadeDetector", "",
               ["- _cascade: CascadeClassifier"],
               ["+ detect(frame) → List[BoundingBox]"])
    class_box(ax, 0.15, 0.20, "MediaPipeDetector", "",
               ["- _detector: FaceDetection"],
               ["+ detect(frame) → List[BoundingBox]",
                "- __del__()"])

    class_box(ax, 0.43, 0.72, "MaskClassifier", "",
               ["- _config: ClassificationConfig",
                "- _model: keras.Model"],
               ["+ predict(face_bgr) → (str, float)",
                "+ is_compliant(label) → bool",
                "- _load_model()",
                "- _preprocess(face) → ndarray"])

    class_box(ax, 0.43, 0.32, "BoundingBox", "«namedtuple»",
               ["x: int", "y: int", "w: int", "h: int"], [])

    class_box(ax, 0.70, 0.72, "ViolationLogger", "",
               ["- _config: LoggingConfig",
                "- _db_path: str",
                "- _lock: threading.Lock"],
               ["+ start_session() → int",
                "+ end_session(id, frames)",
                "+ log_violation(...) → int",
                "+ get_violations() → List",
                "+ get_summary() → Dict",
                "- _init_db()",
                "- _save_snapshot()"])

    class_box(ax, 0.70, 0.28, "ReportGenerator", "«module»",
               [],
               ["+ generate_report(logger, out_dir)",
                "- _chart_pie(summary) → str",
                "- _chart_violations_by_hour()",
                "- _chart_confidence_histogram()"])

    class_box(ax, 0.88, 0.72, "AppConfig", "«dataclass»",
               ["detection: DetectionConfig",
                "classification: ClassificationConfig",
                "input: InputConfig",
                "logging: LoggingConfig",
                "reporting: ReportingConfig"],
               ["+ load_config() → AppConfig",
                "+ get_config() → AppConfig"], w=0.19)

    # Inheritance arrows
    for child_y in [0.45, 0.20]:
        _arrow(ax, 0.15, child_y + 0.07, 0.15, 0.62, color=YELLOW)
    ax.text(0.12, 0.55, "inherits", color=YELLOW, fontsize=6.5, rotation=90)

    # Associations
    _arrow(ax, 0.26, 0.45, 0.33, 0.68, color=TEXT_DIM, label="uses")
    _arrow(ax, 0.53, 0.65, 0.60, 0.68, color=TEXT_DIM, label="logs to")
    _arrow(ax, 0.70, 0.58, 0.70, 0.36, color=ORANGE, label="queries")
    _arrow(ax, 0.79, 0.72, 0.88, 0.72, color=TEXT_DIM, label="configured by")

    fig.savefig(str(DOCS / "uml_class.png"), dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] uml_class.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. UML Sequence Diagram
# ─────────────────────────────────────────────────────────────────────────────

def uml_sequence():
    fig, ax = _fig(16, 10)
    _title(ax, "UML Sequence Diagram — Frame Processing Pipeline")

    actors = ["main.py", "detector.py", "classifier.py", "logger.py", "report_gen.py"]
    xs = [0.10, 0.28, 0.47, 0.66, 0.85]
    colors = [BOX_BLUE, BOX_BLUE, BOX_BLUE, BOX_BLUE, BOX_DARK]

    TOP_Y = 0.88
    BOT_Y = 0.05

    # Lifeline headers
    for x, name, col in zip(xs, actors, colors):
        fancy = FancyBboxPatch((x - 0.07, TOP_Y - 0.035), 0.14, 0.07,
                               boxstyle="round,pad=0.005",
                               linewidth=1.4, edgecolor=ACCENT, facecolor=col, zorder=3)
        ax.add_patch(fancy)
        ax.text(x, TOP_Y, name, ha="center", va="center",
                color=TEXT, fontsize=8.5, fontweight="bold", zorder=4)
        # Dashed lifeline
        ax.plot([x, x], [TOP_Y - 0.035, BOT_Y], color=BORDER, lw=1.2,
                linestyle="dashed", zorder=1)

    # Activation box helper
    def active(x, y1, y2, color=BOX_BLUE):
        ax.add_patch(FancyBboxPatch((x - 0.012, y2), 0.024, y1 - y2,
                                    boxstyle="square,pad=0",
                                    linewidth=1, edgecolor=ACCENT, facecolor=color, zorder=2))

    # Message arrow helper
    def msg(x1, x2, y, label, ret=False, color=ACCENT):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(
                        arrowstyle="-|>" if not ret else "<|-",
                        color=color, lw=1.5,
                        linestyle="-" if not ret else "dashed",
                    ), zorder=4)
        ax.text((x1 + x2) / 2, y + 0.015, label,
                ha="center", va="bottom", color=TEXT if not ret else TEXT_DIM,
                fontsize=7.5, zorder=5)

    # Sequence steps (y positions from top to bottom)
    steps = [
        # (from_x, to_x, y, label, is_return)
        (xs[0], xs[0], 0.80, "read_frame(cap)", False),
        (xs[0], xs[1], 0.74, "detect(frame)", False),
        (xs[1], xs[0], 0.68, "→ [BoundingBox, ...]", True),
        (xs[0], xs[0], 0.63, "crop_face(frame, box)", False),
        (xs[0], xs[2], 0.57, "predict(face_crop)", False),
        (xs[2], xs[0], 0.51, "→ (label, confidence)", True),
        (xs[0], xs[0], 0.46, "draw_prediction(frame, ...)", False),
        (xs[0], xs[3], 0.40, "log_violation(session_id, ...)", False),
        (xs[3], xs[3], 0.34, "INSERT INTO violations", False),
        (xs[3], xs[3], 0.29, "_save_snapshot(face_crop)", False),
        (xs[3], xs[0], 0.24, "→ row_id", True),
        (xs[0], xs[0], 0.19, "imshow(annotated_frame)", False),
        (xs[0], xs[4], 0.13, "generate_report(logger, out)", False),
        (xs[4], xs[3], 0.08, "get_summary() / get_violations()", False),
    ]

    # Activation regions
    active(xs[0], 0.82, 0.11)
    active(xs[1], 0.76, 0.66)
    active(xs[2], 0.59, 0.49)
    active(xs[3], 0.42, 0.22)
    active(xs[4], 0.15, 0.06)

    for x1, x2, y, lbl, ret in steps:
        if x1 == x2:
            ax.annotate("", xy=(x1 + 0.045, y - 0.025), xytext=(x1 + 0.012, y),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2,
                                        connectionstyle="arc3,rad=-0.5"), zorder=4)
            ax.text(x1 + 0.07, y - 0.012, lbl, color=ORANGE, fontsize=7, va="center")
        else:
            msg(x1, x2, y, lbl, ret, color=TEXT_DIM if ret else ACCENT)

    # Note box
    note_x, note_y = 0.50, 0.92
    ax.add_patch(FancyBboxPatch((note_x - 0.13, note_y - 0.025), 0.26, 0.05,
                                boxstyle="round,pad=0.005",
                                linewidth=1, edgecolor=YELLOW, facecolor="#1a1600", zorder=3))
    ax.text(note_x, note_y, "Loop: per frame, per detected face | Press 'r' to generate report",
            ha="center", va="center", color=YELLOW, fontsize=7, zorder=4)

    fig.savefig(str(DOCS / "uml_sequence.png"), dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] uml_sequence.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. ER Diagram
# ─────────────────────────────────────────────────────────────────────────────

def er_diagram():
    fig, ax = _fig(13, 8)
    _title(ax, "Entity-Relationship Diagram — SQLite Schema")

    def entity(ax, x, y, name, pk, attrs, w=0.28):
        line_h = 0.052
        total_h = (1 + len(attrs) + 1) * line_h  # header + attrs + pk
        top = y + total_h / 2

        # Outer rect
        ax.add_patch(FancyBboxPatch((x - w/2, y - total_h/2), w, total_h,
                                    boxstyle="round,pad=0.008",
                                    linewidth=1.8, edgecolor=ACCENT,
                                    facecolor=SURFACE, zorder=3))

        # Header
        ax.add_patch(FancyBboxPatch((x - w/2, top - line_h), w, line_h,
                                    boxstyle="square,pad=0",
                                    linewidth=0, facecolor=BOX_BLUE, zorder=3))
        ax.text(x, top - line_h/2, name, ha="center", va="center",
                color=TEXT, fontsize=9.5, fontweight="bold", zorder=4)

        cur_y = top - line_h - line_h * 0.8

        # PK
        ax.text(x - w/2 + 0.015, cur_y, pk, ha="left", va="center",
                color=YELLOW, fontsize=8, fontweight="bold", zorder=4, family="monospace")
        ax.plot([x - w/2 + 0.01, x + w/2 - 0.01], [cur_y - line_h*0.35, cur_y - line_h*0.35],
                color=BORDER, lw=0.8)
        cur_y -= line_h

        for a in attrs:
            is_fk = a.startswith("FK")
            ax.text(x - w/2 + 0.015, cur_y, a, ha="left", va="center",
                    color=ORANGE if is_fk else TEXT, fontsize=8, zorder=4, family="monospace")
            cur_y -= line_h

    entity(ax, 0.28, 0.52, "sessions",
           "PK  id  INTEGER",
           ["started_at   TEXT  NOT NULL",
            "ended_at     TEXT",
            "total_frames INTEGER",
            "camera_id    TEXT"],
           w=0.34)

    entity(ax, 0.73, 0.52, "violations",
           "PK  id  INTEGER",
           ["FK  session_id  INTEGER",
            "timestamp   TEXT  NOT NULL",
            "label       TEXT  NOT NULL",
            "confidence  REAL  NOT NULL",
            "snapshot_path TEXT",
            "camera_id   TEXT",
            "frame_index INTEGER"],
           w=0.34)

    # Relationship line
    ax.plot([0.45, 0.56], [0.52, 0.52], color=ACCENT, lw=2, zorder=2)
    # Crow's foot (many side — violations)
    for dy in [-0.015, 0, 0.015]:
        ax.plot([0.56, 0.575], [0.52 + dy, 0.52], color=ACCENT, lw=1.5)
    # Single bar (one side — sessions)
    ax.plot([0.45, 0.45], [0.505, 0.535], color=ACCENT, lw=2)

    ax.text(0.505, 0.545, "1", color=ACCENT, fontsize=11, fontweight="bold", ha="center")
    ax.text(0.505, 0.49, "has many", color=TEXT_DIM, fontsize=8, ha="center")
    ax.text(0.62, 0.545, "N", color=ACCENT, fontsize=11, fontweight="bold", ha="center")

    # Indexes note
    ax.add_patch(FancyBboxPatch((0.54, 0.15), 0.38, 0.13,
                                boxstyle="round,pad=0.008",
                                linewidth=1, edgecolor=YELLOW, facecolor="#110e00", zorder=3))
    ax.text(0.73, 0.265, "Indexes",
            ha="center", va="center", color=YELLOW, fontsize=8.5, fontweight="bold")
    ax.text(0.73, 0.225, "idx_violations_timestamp  ON violations(timestamp)",
            ha="center", va="center", color=TEXT_DIM, fontsize=7.5, family="monospace")
    ax.text(0.73, 0.185, "idx_violations_label       ON violations(label)",
            ha="center", va="center", color=TEXT_DIM, fontsize=7.5, family="monospace")

    # Legend
    for xi, col, lbl in [(0.08, YELLOW, "PK — Primary Key"),
                          (0.24, ORANGE, "FK — Foreign Key")]:
        ax.add_patch(plt.Rectangle((xi - 0.005, 0.08), 0.01, 0.025, color=col))
        ax.text(xi + 0.018, 0.0925, lbl, color=TEXT_DIM, fontsize=8, va="center")

    fig.savefig(str(DOCS / "er_diagram.png"), dpi=130,
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("[OK] er_diagram.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating diagrams...")
    architecture_diagram()
    uml_use_case()
    uml_class()
    uml_sequence()
    er_diagram()
    print(f"\nAll diagrams saved to: {DOCS}")
