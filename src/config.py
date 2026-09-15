"""
config.py — Central configuration loader for the Mask Compliance Detector.

Reads config.yaml from project root and exposes a typed Config dataclass.
All other modules import from here instead of reading YAML directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    backend: str = "haar"
    haar_scale_factor: float = 1.1
    haar_min_neighbors: int = 5
    haar_min_size: Tuple[int, int] = (60, 60)


@dataclass
class ClassificationConfig:
    model_path: str = "models/mask_classifier.h5"
    input_size: Tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.70
    labels: List[str] = field(default_factory=lambda: ["Mask", "No Mask"])


@dataclass
class InputConfig:
    source: object = 0          # int (webcam) or str (file path)
    fallback_video: str = "assets/sample_video.mp4"
    frame_skip: int = 2
    display_width: int = 900
    camera_id: str = "cam0"


@dataclass
class LoggingConfig:
    db_path: str = "logs/violations.db"
    snapshot_dir: str = "logs/snapshots"
    save_snapshots: bool = True
    log_compliant: bool = False


@dataclass
class ReportingConfig:
    output_dir: str = "reports"
    include_charts: bool = True
    chart_dpi: int = 120


@dataclass
class TrainingConfig:
    data_dir: str = "data"
    model_save_path: str = "models/mask_classifier.h5"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs_head: int = 10
    epochs_finetune: int = 5
    learning_rate_head: float = 0.001
    learning_rate_finetune: float = 0.0001
    validation_split: float = 0.15
    test_split: float = 0.10
    augmentation: bool = True
    dataset_url: str = (
        "https://github.com/chandrikadeb7/Face-Mask-Detection/"
        "archive/refs/heads/master.zip"
    )
    dataset_zip: str = "data/raw_dataset.zip"


@dataclass
class AppConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    input: InputConfig = field(default_factory=InputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    project_root: Path = field(default_factory=Path.cwd)


# ─────────────────────────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────────────────────────

def _find_config_file() -> Path:
    """Walk up from CWD to find config.yaml."""
    search = Path.cwd()
    for _ in range(5):
        candidate = search / "config.yaml"
        if candidate.exists():
            return candidate
        search = search.parent
    raise FileNotFoundError(
        "config.yaml not found. Run the app from the project root."
    )


def load_config(config_path: str | None = None) -> AppConfig:
    """
    Load and parse config.yaml, returning a fully-typed AppConfig.

    Args:
        config_path: Optional explicit path to config.yaml.
                     If None, auto-discovers from CWD upwards.
    """
    if config_path:
        cfg_file = Path(config_path)
    else:
        cfg_file = _find_config_file()

    with open(cfg_file, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    project_root = cfg_file.parent

    def _get(section: dict, *keys, default=None):
        """Safe nested get."""
        d = section
        for k in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(k, default)
        return d

    det_raw = raw.get("detection", {})
    cls_raw = raw.get("classification", {})
    inp_raw = raw.get("input", {})
    log_raw = raw.get("logging", {})
    rep_raw = raw.get("reporting", {})
    trn_raw = raw.get("training", {})

    # Resolve relative paths against project root
    def resolve(p: str) -> str:
        path = Path(p)
        return str(project_root / path) if not path.is_absolute() else p

    detection = DetectionConfig(
        backend=det_raw.get("backend", "haar"),
        haar_scale_factor=float(det_raw.get("haar_scale_factor", 1.1)),
        haar_min_neighbors=int(det_raw.get("haar_min_neighbors", 5)),
        haar_min_size=tuple(det_raw.get("haar_min_size", [60, 60])),
    )

    classification = ClassificationConfig(
        model_path=resolve(cls_raw.get("model_path", "models/mask_classifier.h5")),
        input_size=tuple(cls_raw.get("input_size", [224, 224])),
        confidence_threshold=float(cls_raw.get("confidence_threshold", 0.70)),
        labels=cls_raw.get("labels", ["Mask", "No Mask"]),
    )

    source_raw = inp_raw.get("source", 0)
    # Keep as int if it looks like a device index
    try:
        source = int(source_raw)
    except (ValueError, TypeError):
        source = str(source_raw)

    input_cfg = InputConfig(
        source=source,
        fallback_video=resolve(inp_raw.get("fallback_video", "assets/sample_video.mp4")),
        frame_skip=int(inp_raw.get("frame_skip", 2)),
        display_width=int(inp_raw.get("display_width", 900)),
        camera_id=str(inp_raw.get("camera_id", "cam0")),
    )

    logging_cfg = LoggingConfig(
        db_path=resolve(log_raw.get("db_path", "logs/violations.db")),
        snapshot_dir=resolve(log_raw.get("snapshot_dir", "logs/snapshots")),
        save_snapshots=bool(log_raw.get("save_snapshots", True)),
        log_compliant=bool(log_raw.get("log_compliant", False)),
    )

    reporting_cfg = ReportingConfig(
        output_dir=resolve(rep_raw.get("output_dir", "reports")),
        include_charts=bool(rep_raw.get("include_charts", True)),
        chart_dpi=int(rep_raw.get("chart_dpi", 120)),
    )

    training_cfg = TrainingConfig(
        data_dir=resolve(trn_raw.get("data_dir", "data")),
        model_save_path=resolve(trn_raw.get("model_save_path", "models/mask_classifier.h5")),
        image_size=tuple(trn_raw.get("image_size", [224, 224])),
        batch_size=int(trn_raw.get("batch_size", 32)),
        epochs_head=int(trn_raw.get("epochs_head", 10)),
        epochs_finetune=int(trn_raw.get("epochs_finetune", 5)),
        learning_rate_head=float(trn_raw.get("learning_rate_head", 0.001)),
        learning_rate_finetune=float(trn_raw.get("learning_rate_finetune", 0.0001)),
        validation_split=float(trn_raw.get("validation_split", 0.15)),
        test_split=float(trn_raw.get("test_split", 0.10)),
        augmentation=bool(trn_raw.get("augmentation", True)),
        dataset_url=trn_raw.get("dataset_url", ""),
        dataset_zip=resolve(trn_raw.get("dataset_zip", "data/raw_dataset.zip")),
    )

    return AppConfig(
        detection=detection,
        classification=classification,
        input=input_cfg,
        logging=logging_cfg,
        reporting=reporting_cfg,
        training=training_cfg,
        project_root=project_root,
    )


# Module-level singleton for convenience
_cached_config: AppConfig | None = None


def get_config(config_path: str | None = None) -> AppConfig:
    """Return cached config (loads once per process)."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config(config_path)
    return _cached_config
