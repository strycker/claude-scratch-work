"""Mocked unit tests for platform/ingestion/macro_daily.py — daily FRED
credit ingestion (L4-04, Phase 4 RESEARCH Pitfall 2).

All network access is mocked — no real HTTP/FRED calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_daily_fred_series(n_days: int = 40) -> pd.Series:
    """A synthetic sub-monthly (business-day) series — keeping every row
    proves no resample happened (a resample to month-end would collapse
    this down to ~1-2 rows)."""
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.Series(np.linspace(3.0, 5.0, len(idx)), index=idx)


# ── assemble_fred_daily — pure function, no I/O ─────────────────────────────


def test_assemble_fred_daily_returns_two_column_frame_with_configured_names():
    s1 = _make_daily_fred_series(10)
    s2 = _make_daily_fred_series(10) + 1.0
    from trading_crab_lib.platform.ingestion.macro_daily import assemble_fred_daily

    df = assemble_fred_daily({"fred_daaa": s1, "fred_dbaa": s2})
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["fred_daaa", "fred_dbaa"]


def test_assemble_fred_daily_preserves_sub_monthly_index_no_resample():
    from trading_crab_lib.platform.ingestion.macro_daily import assemble_fred_daily

    s = _make_daily_fred_series(40)  # ~40 business days, spans under 2 months
    df = assemble_fred_daily({"fred_daaa": s})
    # A resample to month-end would collapse this to ~2 rows — asserting we
    # keep (nearly) all 40 rows proves no resample occurred.
    assert len(df) >= 38


def test_assemble_fred_daily_empty_map_returns_empty_frame():
    from trading_crab_lib.platform.ingestion.macro_daily import assemble_fred_daily

    df = assemble_fred_daily({})
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ── fetch_fred_daily — per-series graceful failure + checkpoint persistence ─


def _make_fred_daily_cfg():
    return {
        "fred_daily": {
            "api_key": "fake_key_for_testing",
            "series": {
                "DAAA": {"name": "fred_daaa", "shift": False},
                "DBAA": {"name": "fred_dbaa", "shift": False},
            },
        },
        "data": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
    }


@patch("trading_crab_lib.platform.ingestion.macro_daily.get_platform_checkpoint_manager")
@patch("trading_crab_lib.platform.ingestion.macro_daily.Fred")
def test_fetch_fred_daily_basic(mock_fred_cls, mock_get_cm):
    from trading_crab_lib.platform.ingestion.macro_daily import fetch_fred_daily

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = _make_daily_fred_series(40)
    mock_fred_cls.return_value = mock_fred
    mock_cm = MagicMock()
    mock_get_cm.return_value = mock_cm

    df = fetch_fred_daily(_make_fred_daily_cfg())
    assert isinstance(df, pd.DataFrame)
    assert "fred_daaa" in df.columns
    assert "fred_dbaa" in df.columns
    assert len(df) >= 38  # no resample — near-full sub-monthly row count preserved


@patch("trading_crab_lib.platform.ingestion.macro_daily.get_platform_checkpoint_manager")
@patch("trading_crab_lib.platform.ingestion.macro_daily.Fred")
def test_fetch_fred_daily_saves_fred_daily_raw_checkpoint(mock_fred_cls, mock_get_cm):
    from trading_crab_lib.platform.ingestion.macro_daily import fetch_fred_daily

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = _make_daily_fred_series(40)
    mock_fred_cls.return_value = mock_fred
    mock_cm = MagicMock()
    mock_get_cm.return_value = mock_cm

    fetch_fred_daily(_make_fred_daily_cfg())

    mock_cm.save.assert_called_once()
    saved_df, saved_name = mock_cm.save.call_args[0]
    assert saved_name == "fred_daily_raw"
    assert isinstance(saved_df, pd.DataFrame)


@patch("trading_crab_lib.platform.ingestion.macro_daily.get_platform_checkpoint_manager")
@patch("trading_crab_lib.platform.ingestion.macro_daily.Fred")
def test_fetch_fred_daily_handles_single_series_failure(mock_fred_cls, mock_get_cm):
    """A raised exception inside one series' fetch is caught, logged at
    WARNING, and the other series still lands in the frame."""
    from trading_crab_lib.platform.ingestion.macro_daily import fetch_fred_daily

    mock_fred = MagicMock()

    def _side_effect(series_id, **kwargs):
        if series_id == "DAAA":
            raise OSError("API rate limit")
        return _make_daily_fred_series(40)

    mock_fred.get_series.side_effect = _side_effect
    mock_fred_cls.return_value = mock_fred
    mock_get_cm.return_value = MagicMock()

    df = fetch_fred_daily(_make_fred_daily_cfg())
    assert "fred_dbaa" in df.columns
    assert "fred_daaa" not in df.columns


def test_fetch_fred_daily_missing_api_key_returns_empty_frame():
    from trading_crab_lib.platform.ingestion.macro_daily import fetch_fred_daily

    cfg = {
        "fred_daily": {"api_key": None, "series": {"DAAA": {"name": "fred_daaa"}}},
        "fred_monthly": {"api_key": None},
        "data": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
    }
    df = fetch_fred_daily(cfg)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_fetch_fred_daily_no_series_configured_returns_empty_frame():
    from trading_crab_lib.platform.ingestion.macro_daily import fetch_fred_daily

    cfg = {
        "fred_daily": {"api_key": "fake_key_for_testing", "series": {}},
        "data": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
    }
    df = fetch_fred_daily(cfg)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


@patch("trading_crab_lib.platform.ingestion.macro_daily.Fred")
def test_fetch_fred_daily_falls_back_to_fred_monthly_api_key(mock_fred_cls):
    """fred_daily has no api_key of its own — falls back to fred_monthly's
    (both read from the same FRED_API_KEY env var per config.py)."""
    from trading_crab_lib.platform.ingestion.macro_daily import fetch_fred_daily

    mock_fred = MagicMock()
    mock_fred.get_series.return_value = _make_daily_fred_series(40)
    mock_fred_cls.return_value = mock_fred

    cfg = {
        "fred_daily": {"series": {"DAAA": {"name": "fred_daaa"}}},  # no api_key key
        "fred_monthly": {"api_key": "fallback_key"},
        "data": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
    }
    with patch(
        "trading_crab_lib.platform.ingestion.macro_daily.get_platform_checkpoint_manager",
        return_value=MagicMock(),
    ):
        df = fetch_fred_daily(cfg)
    mock_fred_cls.assert_called_once_with(api_key="fallback_key")
    assert "fred_daaa" in df.columns


# ── report_fred_daily — artifact persistence ────────────────────────────────


def test_report_fred_daily_writes_parquet_artifact(tmp_path):
    from trading_crab_lib.platform.ingestion.macro_daily import report_fred_daily

    assemble_frame = pd.DataFrame(
        {"fred_daaa": [3.0, 3.1]}, index=pd.bdate_range("2024-01-01", periods=2)
    )
    path = report_fred_daily(assemble_frame, output_dir=tmp_path)
    assert path.exists()
    assert path.suffix == ".parquet"


def test_report_fred_daily_empty_frame_writes_schema_stable_artifact(tmp_path):
    from trading_crab_lib.platform.ingestion.macro_daily import report_fred_daily

    path = report_fred_daily(pd.DataFrame(), output_dir=tmp_path)
    assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
