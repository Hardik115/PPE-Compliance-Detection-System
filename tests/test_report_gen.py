"""
test_report_gen.py — Unit tests for the report generator module.

Tests cover:
- Text report generation output path and content
- HTML report generation and structure
- Chart generation (base64 output validation)
- Edge cases: empty database, all compliant, all violations
"""

import sys
import re
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LoggingConfig, ReportingConfig
from src.logger import ViolationLogger
from src.report_gen import generate_report, _chart_pie, _chart_violations_by_hour, _chart_confidence_histogram


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_logger(tmp_path):
    cfg = LoggingConfig(
        db_path=str(tmp_path / "test.db"),
        snapshot_dir=str(tmp_path / "snaps"),
        save_snapshots=False,
        log_compliant=False,
    )
    return ViolationLogger(cfg)


@pytest.fixture
def populated_logger(tmp_logger):
    """Logger pre-populated with 7 mask + 3 no-mask events."""
    sid = tmp_logger.start_session("cam0")
    face = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    for _ in range(7):
        tmp_logger.log_violation(sid, np.array([]), "Mask", 0.95)
    for _ in range(3):
        tmp_logger.log_violation(sid, face, "No Mask", 0.88)
    tmp_logger.end_session(sid, total_frames=200)
    return tmp_logger


@pytest.fixture
def report_dir(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    return str(d)


# ── generate_report tests ─────────────────────────────────────────────────────

class TestGenerateReport:
    def test_returns_dict_with_paths(self, populated_logger, report_dir):
        result = generate_report(populated_logger, report_dir)
        assert "txt_path" in result
        assert "html_path" in result

    def test_txt_file_created(self, populated_logger, report_dir):
        result = generate_report(populated_logger, report_dir)
        assert Path(result["txt_path"]).exists()

    def test_html_file_created(self, populated_logger, report_dir):
        result = generate_report(populated_logger, report_dir)
        assert Path(result["html_path"]).exists()

    def test_txt_contains_compliance_rate(self, populated_logger, report_dir):
        result = generate_report(populated_logger, report_dir)
        txt = Path(result["txt_path"]).read_text()
        assert "Compliance rate" in txt
        assert "70.0%" in txt  # 7/10

    def test_html_contains_key_elements(self, populated_logger, report_dir):
        result = generate_report(populated_logger, report_dir)
        html = Path(result["html_path"]).read_text()
        assert "<!DOCTYPE html>" in html
        assert "Mask Compliance Report" in html
        assert "Compliant" in html
        assert "Violations" in html

    def test_html_contains_embedded_charts(self, populated_logger, report_dir):
        result = generate_report(populated_logger, report_dir)
        html = Path(result["html_path"]).read_text()
        # Base64-encoded PNGs should be embedded
        assert "data:image/png;base64," in html

    def test_html_violation_table_has_rows(self, populated_logger, report_dir):
        result = generate_report(populated_logger, report_dir)
        html = Path(result["html_path"]).read_text()
        assert "No Mask" in html

    def test_empty_db_generates_report(self, tmp_logger, report_dir):
        """Report generation should not crash on empty database."""
        result = generate_report(tmp_logger, report_dir)
        assert Path(result["txt_path"]).exists()
        txt = Path(result["txt_path"]).read_text()
        assert "100.0%" in txt  # 100% compliant when no events

    def test_output_dir_created_if_missing(self, populated_logger, tmp_path):
        new_dir = str(tmp_path / "new_report_dir")
        result = generate_report(populated_logger, new_dir)
        assert Path(new_dir).is_dir()

    def test_report_filenames_include_timestamp(self, populated_logger, report_dir):
        result = generate_report(populated_logger, report_dir)
        txt_name = Path(result["txt_path"]).name
        html_name = Path(result["html_path"]).name
        # Should match pattern: report_YYYYMMDD_HHMMSS.txt
        assert re.match(r"report_\d{8}_\d{6}\.txt", txt_name)
        assert re.match(r"report_\d{8}_\d{6}\.html", html_name)


# ── Chart function tests ──────────────────────────────────────────────────────

class TestChartFunctions:
    def test_chart_pie_returns_nonempty_b64(self):
        summary = {
            "mask_count": 7, "no_mask_count": 3,
            "violations_by_hour": []
        }
        b64 = _chart_pie(summary)
        assert isinstance(b64, str)
        assert len(b64) > 100  # non-trivial base64 string

    def test_chart_pie_all_compliant(self):
        summary = {"mask_count": 10, "no_mask_count": 0, "violations_by_hour": []}
        b64 = _chart_pie(summary)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_chart_pie_all_violations(self):
        summary = {"mask_count": 0, "no_mask_count": 10, "violations_by_hour": []}
        b64 = _chart_pie(summary)
        assert isinstance(b64, str)

    def test_chart_hourly_empty_data(self):
        summary = {"violations_by_hour": []}
        b64 = _chart_violations_by_hour(summary)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_chart_hourly_with_data(self):
        summary = {
            "violations_by_hour": [
                {"hour": "2026-09-17 09:00", "count": 3},
                {"hour": "2026-09-17 10:00", "count": 7},
            ]
        }
        b64 = _chart_violations_by_hour(summary)
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_chart_confidence_histogram_no_violations(self):
        b64 = _chart_confidence_histogram([])
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_chart_confidence_histogram_with_data(self):
        violations = [
            {"label": "No Mask", "confidence": 0.85},
            {"label": "No Mask", "confidence": 0.92},
            {"label": "Mask", "confidence": 0.97},
        ]
        b64 = _chart_confidence_histogram(violations)
        assert isinstance(b64, str)
        assert len(b64) > 100
