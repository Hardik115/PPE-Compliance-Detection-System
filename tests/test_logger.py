"""
test_logger.py — Unit tests for the ViolationLogger module.

Tests cover:
- Database initialisation and schema creation
- Session start/end lifecycle
- Violation insertion with and without snapshots
- Query API: get_violations(), get_summary(), get_sessions()
- Thread safety (basic concurrent inserts)
"""

import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LoggingConfig
from src.logger import ViolationLogger


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_logger(tmp_path):
    """ViolationLogger backed by a temporary directory (auto-cleaned)."""
    cfg = LoggingConfig(
        db_path=str(tmp_path / "test_violations.db"),
        snapshot_dir=str(tmp_path / "snapshots"),
        save_snapshots=True,
        log_compliant=False,
    )
    return ViolationLogger(cfg)


@pytest.fixture
def tmp_logger_no_snap(tmp_path):
    """ViolationLogger with snapshots disabled."""
    cfg = LoggingConfig(
        db_path=str(tmp_path / "test_no_snap.db"),
        snapshot_dir=str(tmp_path / "snapshots"),
        save_snapshots=False,
        log_compliant=False,
    )
    return ViolationLogger(cfg)


@pytest.fixture
def face_crop():
    """Random BGR face-crop image."""
    return np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)


# ── Schema / init tests ───────────────────────────────────────────────────────

class TestDatabaseInit:
    def test_db_file_created(self, tmp_logger, tmp_path):
        assert (tmp_path / "test_violations.db").exists()

    def test_snapshot_dir_created(self, tmp_logger, tmp_path):
        assert (tmp_path / "snapshots").is_dir()

    def test_empty_violations_on_start(self, tmp_logger):
        rows = tmp_logger.get_violations()
        assert rows == []

    def test_empty_sessions_on_start(self, tmp_logger):
        rows = tmp_logger.get_sessions()
        assert rows == []


# ── Session lifecycle tests ───────────────────────────────────────────────────

class TestSessionLifecycle:
    def test_start_session_returns_int(self, tmp_logger):
        sid = tmp_logger.start_session("cam0")
        assert isinstance(sid, int)
        assert sid >= 1

    def test_multiple_sessions_unique_ids(self, tmp_logger):
        sid1 = tmp_logger.start_session("cam0")
        sid2 = tmp_logger.start_session("cam1")
        assert sid1 != sid2

    def test_end_session_updates_record(self, tmp_logger):
        sid = tmp_logger.start_session("cam0")
        tmp_logger.end_session(sid, total_frames=100)
        sessions = tmp_logger.get_sessions()
        assert len(sessions) == 1
        s = sessions[0]
        assert s["total_frames"] == 100
        assert s["ended_at"] is not None


# ── Violation logging tests ───────────────────────────────────────────────────

class TestViolationLogging:
    def test_log_violation_inserts_row(self, tmp_logger, face_crop):
        sid = tmp_logger.start_session()
        row_id = tmp_logger.log_violation(sid, face_crop, "No Mask", 0.92, frame_index=5)
        assert isinstance(row_id, int)
        rows = tmp_logger.get_violations()
        assert len(rows) == 1

    def test_logged_fields_match(self, tmp_logger, face_crop):
        sid = tmp_logger.start_session()
        tmp_logger.log_violation(sid, face_crop, "No Mask", 0.88, frame_index=10, camera_id="cam1")
        rows = tmp_logger.get_violations()
        r = rows[0]
        assert r["label"] == "No Mask"
        assert r["confidence"] == pytest.approx(0.88, abs=1e-4)
        assert r["frame_index"] == 10
        assert r["camera_id"] == "cam1"

    def test_snapshot_saved_when_enabled(self, tmp_logger, tmp_path, face_crop):
        sid = tmp_logger.start_session()
        tmp_logger.log_violation(sid, face_crop, "No Mask", 0.91)
        rows = tmp_logger.get_violations()
        snap_path = rows[0]["snapshot_path"]
        assert snap_path is not None
        assert Path(snap_path).exists()

    def test_snapshot_not_saved_when_disabled(self, tmp_logger_no_snap, face_crop):
        sid = tmp_logger_no_snap.start_session()
        tmp_logger_no_snap.log_violation(sid, face_crop, "No Mask", 0.91)
        rows = tmp_logger_no_snap.get_violations()
        assert rows[0]["snapshot_path"] is None

    def test_empty_crop_no_crash(self, tmp_logger):
        sid = tmp_logger.start_session()
        row_id = tmp_logger.log_violation(sid, np.array([]), "No Mask", 0.85)
        assert isinstance(row_id, int)

    def test_multiple_violations_ordered_desc(self, tmp_logger, face_crop):
        sid = tmp_logger.start_session()
        for i in range(5):
            tmp_logger.log_violation(sid, face_crop, "No Mask", 0.80 + i * 0.01, frame_index=i)
        rows = tmp_logger.get_violations()
        assert len(rows) == 5
        # Should be ordered by timestamp descending
        timestamps = [r["timestamp"] for r in rows]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_filter_by_label(self, tmp_logger, face_crop):
        sid = tmp_logger.start_session()
        tmp_logger.log_violation(sid, face_crop, "No Mask", 0.92)
        tmp_logger.log_violation(sid, np.array([]), "Mask", 0.95)
        no_mask_rows = tmp_logger.get_violations(label="No Mask")
        mask_rows = tmp_logger.get_violations(label="Mask")
        assert len(no_mask_rows) == 1
        assert len(mask_rows) == 1


# ── Summary API tests ─────────────────────────────────────────────────────────

class TestGetSummary:
    def test_empty_db_summary(self, tmp_logger):
        summary = tmp_logger.get_summary()
        assert summary["total_logged"] == 0
        assert summary["mask_count"] == 0
        assert summary["no_mask_count"] == 0
        assert summary["compliance_rate"] == 100.0
        assert summary["latest_violation"] is None

    def test_compliance_rate_calculation(self, tmp_logger, face_crop):
        sid = tmp_logger.start_session()
        for _ in range(8):
            tmp_logger.log_violation(sid, np.array([]), "Mask", 0.95)
        for _ in range(2):
            tmp_logger.log_violation(sid, face_crop, "No Mask", 0.90)
        summary = tmp_logger.get_summary()
        assert summary["total_logged"] == 10
        assert summary["mask_count"] == 8
        assert summary["no_mask_count"] == 2
        assert summary["compliance_rate"] == pytest.approx(80.0, abs=0.1)

    def test_violations_by_hour_key_exists(self, tmp_logger):
        summary = tmp_logger.get_summary()
        assert "violations_by_hour" in summary
        assert isinstance(summary["violations_by_hour"], list)


# ── Thread-safety test ────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_inserts_no_data_loss(self, tmp_logger, face_crop):
        sid = tmp_logger.start_session()
        errors = []

        def insert_violations(n):
            try:
                for i in range(n):
                    tmp_logger.log_violation(sid, face_crop, "No Mask", 0.90, frame_index=i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_violations, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        rows = tmp_logger.get_violations()
        assert len(rows) == 50  # 5 threads × 10 inserts
