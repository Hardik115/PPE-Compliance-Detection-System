"""
report_gen.py — Compliance Report Generator

Queries the SQLite database and produces two artefacts:
  1. A timestamped plain-text summary (.txt)
  2. A rich HTML report with embedded matplotlib charts (.html)

Usage::
    python -m src.report_gen                  # uses default config.yaml
    python -m src.report_gen --out reports/   # override output directory
"""

from __future__ import annotations

import argparse
import base64
import io
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.config import get_config
from src.logger import ViolationLogger

# ─────────────────────────────────────────────────────────────────────────────
# Chart generators
# ─────────────────────────────────────────────────────────────────────────────

_DARK_BG = "#0d1117"
_ACCENT = "#58a6ff"
_GREEN = "#3fb950"
_RED = "#f85149"
_TEXT = "#c9d1d9"
_GRID = "#21262d"


def _apply_dark_style(ax, fig):
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", color=_GRID, linewidth=0.7, linestyle="--")


def _fig_to_base64(fig) -> str:
    """Encode a matplotlib Figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _chart_pie(summary: Dict[str, Any]) -> str:
    """Compliance vs. Violation donut chart."""
    mask = summary["mask_count"]
    no_mask = summary["no_mask_count"]
    if mask == 0 and no_mask == 0:
        mask, no_mask = 1, 0  # show 100% compliant placeholder

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        [mask, no_mask],
        labels=["Compliant", "Violations"],
        autopct="%1.1f%%",
        colors=[_GREEN, _RED],
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": _DARK_BG, "linewidth": 2},
        textprops={"color": _TEXT, "fontsize": 10},
    )
    for at in autotexts:
        at.set_color(_DARK_BG)
        at.set_fontweight("bold")
    ax.set_title("Compliance Rate", color=_TEXT, fontsize=12, pad=12)
    _apply_dark_style(ax, fig)
    ax.set_facecolor(_DARK_BG)
    return _fig_to_base64(fig)


def _chart_violations_by_hour(summary: Dict[str, Any]) -> str:
    """Bar chart — violations per hour."""
    data = summary.get("violations_by_hour", [])
    if not data:
        hours = ["No data"]
        counts = [0]
    else:
        hours = [d["hour"].split(" ")[1] for d in data]  # extract HH:00
        counts = [d["count"] for d in data]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(hours, counts, color=_RED, alpha=0.85, edgecolor=_DARK_BG, linewidth=0.5)
    ax.set_xlabel("Hour (UTC)", fontsize=10)
    ax.set_ylabel("Violations", fontsize=10)
    ax.set_title("Violations per Hour", fontsize=12, color=_TEXT)
    # Value labels on bars
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(val),
            ha="center", va="bottom", color=_TEXT, fontsize=8,
        )
    plt.xticks(rotation=30, ha="right")
    _apply_dark_style(ax, fig)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _chart_confidence_histogram(violations) -> str:
    """Histogram of classifier confidence scores for No Mask events."""
    no_mask_confidences = [
        v["confidence"] for v in violations if v["label"] == "No Mask"
    ]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if no_mask_confidences:
        ax.hist(no_mask_confidences, bins=15, color=_ACCENT, edgecolor=_DARK_BG,
                alpha=0.9, linewidth=0.5)
    else:
        ax.text(0.5, 0.5, "No violations yet", ha="center", va="center",
                transform=ax.transAxes, color=_TEXT, fontsize=11)
    ax.set_xlabel("Confidence Score", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Violation Confidence Distribution", fontsize=12)
    _apply_dark_style(ax, fig)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Mask Compliance Report — {generated_at}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #21262d;
    --text: #c9d1d9; --text-dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif;
         font-size: 14px; line-height: 1.6; padding: 2rem; }}
  h1 {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 600; color: var(--text); margin: 1.5rem 0 0.75rem; }}
  .subtitle {{ color: var(--text-dim); font-size: 0.85rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
           gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 10px; padding: 1.25rem; text-align: center; }}
  .card .value {{ font-size: 2.2rem; font-weight: 700; }}
  .card .label {{ color: var(--text-dim); font-size: 0.78rem; text-transform: uppercase;
                  letter-spacing: 0.06em; margin-top: 0.25rem; }}
  .green {{ color: var(--green); }} .red {{ color: var(--red); }}
  .accent {{ color: var(--accent); }} .yellow {{ color: var(--yellow); }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
             gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-box {{ background: var(--surface); border: 1px solid var(--border);
                border-radius: 10px; padding: 1rem; text-align: center; }}
  .chart-box img {{ max-width: 100%; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface);
           border-radius: 10px; overflow: hidden; }}
  thead {{ background: #1f2937; }}
  th, td {{ padding: 0.65rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
        letter-spacing: 0.05em; color: var(--text-dim); }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 20px;
            font-size: 0.75rem; font-weight: 600; }}
  .badge-red {{ background: rgba(248,81,73,0.2); color: var(--red); }}
  .badge-green {{ background: rgba(63,185,80,0.2); color: var(--green); }}
  footer {{ margin-top: 3rem; color: var(--text-dim); font-size: 0.78rem; text-align: center; }}
</style>
</head>
<body>
<h1>🛡 Mask Compliance Report</h1>
<p class="subtitle">Generated: {generated_at} UTC &nbsp;|&nbsp; Camera: {camera_id}</p>

<div class="grid">
  <div class="card"><div class="value accent">{total_logged}</div>
    <div class="label">Total Events</div></div>
  <div class="card"><div class="value green">{mask_count}</div>
    <div class="label">Compliant (Mask)</div></div>
  <div class="card"><div class="value red">{no_mask_count}</div>
    <div class="label">Violations (No Mask)</div></div>
  <div class="card"><div class="value {rate_color}">{compliance_rate}%</div>
    <div class="label">Compliance Rate</div></div>
</div>

<h2>📊 Visualisations</h2>
<div class="charts">
  <div class="chart-box"><img src="data:image/png;base64,{chart_pie}" alt="Compliance pie"/></div>
  <div class="chart-box"><img src="data:image/png;base64,{chart_hourly}" alt="Hourly violations"/></div>
  <div class="chart-box"><img src="data:image/png;base64,{chart_conf}" alt="Confidence distribution"/></div>
</div>

<h2>📋 Recent Violation Log (last 50)</h2>
<table>
<thead><tr>
  <th>#</th><th>Timestamp (UTC)</th><th>Label</th>
  <th>Confidence</th><th>Frame</th><th>Camera</th>
</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>

<footer>Mask/PPE Compliance Detector &mdash; Auto-generated report</footer>
</body>
</html>
"""


def _build_table_rows(violations) -> str:
    rows = []
    for v in violations[:50]:
        label = v["label"]
        badge_cls = "badge-red" if label == "No Mask" else "badge-green"
        conf_pct = f"{v['confidence']:.1%}"
        rows.append(
            f"<tr>"
            f"<td>{v['id']}</td>"
            f"<td>{v['timestamp']}</td>"
            f"<td><span class='badge {badge_cls}'>{label}</span></td>"
            f"<td>{conf_pct}</td>"
            f"<td>{v.get('frame_index', '—')}</td>"
            f"<td>{v.get('camera_id', '—')}</td>"
            f"</tr>"
        )
    return "\n".join(rows) if rows else "<tr><td colspan='6'>No violations logged yet.</td></tr>"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(logger: ViolationLogger, output_dir: str) -> Dict[str, str]:
    """
    Generate both .txt and .html compliance reports.

    Args:
        logger:     ViolationLogger instance with data.
        output_dir: Directory where reports are saved.

    Returns:
        Dict with keys 'txt_path' and 'html_path'.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = logger.get_summary()
    violations = logger.get_violations()
    sessions = logger.get_sessions()

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    camera_id = sessions[0]["camera_id"] if sessions else "cam0"

    # ── Plain text report ────────────────────────────────────────────────────
    txt_lines = [
        "=" * 60,
        "   MASK / PPE COMPLIANCE REPORT",
        f"   Generated : {generated_at} UTC",
        f"   Camera    : {camera_id}",
        "=" * 60,
        "",
        f"  Total events logged  : {summary['total_logged']}",
        f"  Compliant (Mask)     : {summary['mask_count']}",
        f"  Violations (No Mask) : {summary['no_mask_count']}",
        f"  Compliance rate      : {summary['compliance_rate']}%",
        f"  Latest violation     : {summary['latest_violation'] or 'None'}",
        "",
        "  Violations per hour:",
    ]
    for row in summary["violations_by_hour"]:
        txt_lines.append(f"    {row['hour']}  →  {row['count']} violation(s)")
    txt_lines += ["", "=" * 60, "  (End of report)", "=" * 60]

    txt_path = out / f"report_{stamp}.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    # ── HTML report ──────────────────────────────────────────────────────────
    rate = summary["compliance_rate"]
    rate_color = "green" if rate >= 90 else ("yellow" if rate >= 70 else "red")

    html = _HTML_TEMPLATE.format(
        generated_at=generated_at,
        camera_id=camera_id,
        total_logged=summary["total_logged"],
        mask_count=summary["mask_count"],
        no_mask_count=summary["no_mask_count"],
        compliance_rate=summary["compliance_rate"],
        rate_color=rate_color,
        chart_pie=_chart_pie(summary),
        chart_hourly=_chart_violations_by_hour(summary),
        chart_conf=_chart_confidence_histogram(violations),
        table_rows=_build_table_rows(violations),
    )

    html_path = out / f"report_{stamp}.html"
    html_path.write_text(html, encoding="utf-8")

    print(f"\n✅ Reports saved:")
    print(f"   TXT  → {txt_path}")
    print(f"   HTML → {html_path}")
    return {"txt_path": str(txt_path), "html_path": str(html_path)}


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate mask compliance report.")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--out", default=None, help="Override output directory")
    args = parser.parse_args()

    cfg = get_config(args.config)
    out_dir = args.out or cfg.reporting.output_dir

    logger = ViolationLogger(cfg.logging)
    generate_report(logger, out_dir)


if __name__ == "__main__":
    main()
