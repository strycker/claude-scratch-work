"""Unit tests for scripts/build_platform_data.py's check_price_coverage() —
the pure assertion that replaced the column-name substring heuristic
responsible for printing BUILD OK over a 0x0 daily_raw checkpoint.

``scripts`` is on pytest's pythonpath (see pyproject.toml
``[tool.pytest.ini_options] pythonpath``), same pattern as
tests/test_scripts_weekly_report.py.
"""
from __future__ import annotations

import build_platform_data as build
import pandas as pd


def test_check_price_coverage_reports_failure_for_zero_row_frame():
    frame = pd.DataFrame(columns=["SPY", "QQQ"])
    msg = build.check_price_coverage(frame)
    assert isinstance(msg, str)
    assert msg  # non-empty string, not just truthy


def test_check_price_coverage_reports_failure_for_rows_but_no_columns():
    frame = pd.DataFrame(index=pd.date_range("2020-01-01", periods=5))
    msg = build.check_price_coverage(frame)
    assert isinstance(msg, str)
    assert msg


def test_check_price_coverage_reports_failure_for_absent_checkpoint():
    msg = build.check_price_coverage(None)
    assert isinstance(msg, str)
    assert msg


def test_check_price_coverage_passes_for_populated_frame():
    frame = pd.DataFrame(
        {"SPY": [400.0, 401.0, 402.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    msg = build.check_price_coverage(frame)
    assert msg is None
