"""
logger.py — Violation Logging Module (SQLite-backed)

Persists every mask violation (and optionally compliant events) to a local
SQLite database. Provides a clean query API consumed by report_gen.py.

Schema
------
violations  : one row per detected face classification event flagged as violation
sessions    : one row per application run (start/end time, total frames)

All timestamps are stored as ISO 8601 UTC strings.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np

from src.config import LoggingConfig

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT    NOT NULL,
    ended_at      TEXT,
    total_frames  INTEGER DEFAULT 0,
    camera_id     TEXT    DEFAULT 'cam0'
);

CREATE TABLE IF NOT EXISTS violations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER REFERENCES sessions(id),
    timestamp     TEXT    NOT NULL,
    label         TEXT    NOT NULL,          -- "Mask" | "No Mask"
    confidence    REAL    NOT NULL,
    snapshot_path TEXT,                      -- path to saved face crop or NULL
    camera_id     TEXT    DEFAULT 'cam0',
    frame_index   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON violations(timestamp);
CREATE INDEX IF NOT EXISTS idx_violations_label     ON violations(label);
"""


# ─────────────────────────────────────────────────────────────────────────────
# ViolationLogger
# ─────────────────────────────────────────────────────────────────────────────

class ViolationLogger:
    """
    Thread-safe SQLite-backed logger for mask compliance events.

    Usage::

        logger = ViolationLogger(config)
        session_id = logger.start_session()
        logger.log_violation(session_id, face_crop, "No Mask", 0.92, frame_idx=10)
        logger.end_session(session_id, total_frames=500)
    """

    def __init__(self, config: LoggingConfig) -> None:
        self._config = config
        self._db_path = config.db_path
        self._snapshot_dir = Path(config.snapshot_dir)
        self._lock = threading.Lock()

        # Ensure directories exist
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    # ── DB helpers ───────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        """Yield a short-lived connection. Auto-commits on success."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_DDL)

    # ── Session management ───────────────────────────────────────────────────

    def start_session(self, camera_id: Optional[str] = None) -> int:
        """
        Record a new monitoring session.

        Returns:
            session_id (int) to pass to subsequent log_violation calls.
        """
        cam = camera_id or self._config.__class__.__name__
        cam = camera_id or "cam0"
        now = _utcnow()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (started_at, camera_id) VALUES (?, ?)",
                (now, cam),
            )
            return cur.lastrowid  # type: ignore[return-value]

    def end_session(self, session_id: int, total_frames: int = 0) -> None:
        """Mark a session as ended and record total frames processed."""
        now = _utcnow()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at=?, total_frames=? WHERE id=?",
                (now, total_frames, session_id),
            )

    # ── Violation logging ────────────────────────────────────────────────────

    def log_violation(
        self,
        session_id: int,
        face_crop: np.ndarray,
        label: str,
        confidence: float,
        frame_index: int = 0,
        camera_id: str = "cam0",
    ) -> int:
        """
        Log a classification event (violation or compliant, depending on config).

        Args:
            session_id:  Active session ID from start_session().
            face_crop:   BGR numpy array of the detected face (may be empty).
            label:       "Mask" or "No Mask".
            confidence:  Classifier confidence score.
            frame_index: Frame counter for reference.
            camera_id:   Camera identifier string.

        Returns:
            Row ID of the inserted violation record.
        """
        now = _utcnow()
        snapshot_path: Optional[str] = None

        # Save snapshot image if configured
        if self._config.save_snapshots and face_crop is not None and face_crop.size > 0:
            snapshot_path = self._save_snapshot(face_crop, now)

        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO violations
                   (session_id, timestamp, label, confidence, snapshot_path,
                    camera_id, frame_index)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (session_id, now, label, confidence, snapshot_path,
                 camera_id, frame_index),
            )
            return cur.lastrowid  # type: ignore[return-value]

    # ── Query API ────────────────────────────────────────────────────────────

    def get_violations(self, label: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve violation records, optionally filtered by label.

        Args:
            label: If provided, filter to "Mask" or "No Mask".

        Returns:
            List of dicts with keys: id, timestamp, label, confidence,
            snapshot_path, camera_id, frame_index, session_id.
        """
        with self._connect() as conn:
            if label:
                rows = conn.execute(
                    "SELECT * FROM violations WHERE label=? ORDER BY timestamp DESC",
                    (label,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM violations ORDER BY timestamp DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def get_summary(self) -> Dict[str, Any]:
        """
        Compute an aggregate compliance summary from the violations table.

        Returns:
            Dict with keys:
              total_logged, mask_count, no_mask_count, compliance_rate,
              latest_violation, violations_by_hour (list of dicts)
        """
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
            mask_cnt = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE label='Mask'"
            ).fetchone()[0]
            no_mask_cnt = conn.execute(
                "SELECT COUNT(*) FROM violations WHERE label='No Mask'"
            ).fetchone()[0]
            latest = conn.execute(
                "SELECT timestamp FROM violations WHERE label='No Mask' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            # Group violations per hour for chart
            by_hour_rows = conn.execute(
                """SELECT strftime('%Y-%m-%d %H:00', timestamp) AS hour,
                          COUNT(*) AS count
                   FROM violations
                   WHERE label='No Mask'
                   GROUP BY hour
                   ORDER BY hour""",
            ).fetchall()

        compliance_rate = (mask_cnt / total * 100) if total > 0 else 100.0
        return {
            "total_logged": total,
            "mask_count": mask_cnt,
            "no_mask_count": no_mask_cnt,
            "compliance_rate": round(compliance_rate, 2),
            "latest_violation": latest[0] if latest else None,
            "violations_by_hour": [dict(r) for r in by_hour_rows],
        }

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Return all session records."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Snapshot helper ──────────────────────────────────────────────────────

    def _save_snapshot(self, face_crop: np.ndarray, timestamp: str) -> str:
        """Save face crop to disk; return the file path."""
        safe_ts = timestamp.replace(":", "-").replace(".", "-")
        filename = f"violation_{safe_ts}.jpg"
        filepath = self._snapshot_dir / filename
        cv2.imwrite(str(filepath), face_crop)
        return str(filepath)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
