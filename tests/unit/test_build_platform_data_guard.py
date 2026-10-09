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
import pytest


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


# ── fred_daily_raw in the build (plan 08.2-03, ruling A1) ────────────────────


def _drive_main(monkeypatch, tmp_path, fetch):
    """Run build.main() with every source patched: no network, no tracked write, no .env read."""
    import logging

    import dotenv

    import trading_crab_lib.platform.checkpoints as platform_ckpt
    import trading_crab_lib.platform.honesty.holdout as holdout_mod
    import trading_crab_lib.platform.ingestion.macro_daily as macro_daily
    import trading_crab_lib.platform.transforms_monthly as transforms_monthly
    from trading_crab_lib.checkpoints import CheckpointManager

    monkeypatch.setenv("FRED_API_KEY", "x")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    cm = CheckpointManager(checkpoint_dir=tmp_path / "platform")
    cm.save(pd.DataFrame({"SPY": [400.0, 401.0]}, index=pd.date_range("2026-09-01", periods=2)), "daily_raw")
    spine_cfgs = []

    def spine(cfg):
        spine_cfgs.append(cfg)
        return pd.DataFrame({"x": [1.0, 2.0]}, index=pd.date_range("2020-10-31", periods=2, freq="ME"))

    monkeypatch.setattr(transforms_monthly, "build_monthly_spine", spine)
    monkeypatch.setattr(platform_ckpt, "get_platform_checkpoint_manager", lambda: cm)
    monkeypatch.setattr(holdout_mod, "assert_dev_checkpoint_within_boundary", lambda *a, **k: None)
    monkeypatch.setattr(macro_daily, "fetch_fred_daily", fetch)
    logging.getLogger("build_platform_data").propagate = True
    return build.main([]), spine_cfgs


def test_main_fetches_fred_daily_once_with_the_build_cfg(monkeypatch, tmp_path):
    calls = []

    def fetch(cfg):
        calls.append(cfg)
        return pd.DataFrame({"fred_daaa": [4.5], "fred_dbaa": [5.5]}, index=pd.to_datetime(["2026-09-30"]))

    code, spine_cfgs = _drive_main(monkeypatch, tmp_path, fetch)
    assert code == 0
    assert len(calls) == 1
    assert calls[0] is spine_cfgs[0]


# 08.4 (2026-10-05): D-T8 supersedes ruling A1; under build.fail_loud a failed FRED daily fetch exits 1
@pytest.mark.parametrize("failure", ["raise", "empty"])
def test_a_failed_fred_daily_fetch_fails_the_build(monkeypatch, tmp_path, caplog, failure):
    import logging

    def ok(cfg):
        return pd.DataFrame({"fred_daaa": [4.5], "fred_dbaa": [5.5]}, index=pd.to_datetime(["2026-09-30"]))

    def bad(cfg):
        if failure == "raise":
            raise ConnectionError("FRED unreachable")
        return pd.DataFrame()

    ok_code, _ = _drive_main(monkeypatch, tmp_path / "ok", ok)
    with caplog.at_level(logging.ERROR, logger="build_platform_data"):
        bad_code, _ = _drive_main(monkeypatch, tmp_path / "bad", bad)
    assert ok_code == 0
    assert bad_code == 1
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert any("DAAA/DBAA" in m and "python scripts/build_platform_data.py" in m for m in errors), errors
