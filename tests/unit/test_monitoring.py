"""Tests for pipeline monitoring helpers (Phase C1)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.monitoring import (
    format_completeness_table,
    validate_date_range,
    DateRangeReport,
    count_source_columns,
    SourceRowCounts,
    compute_feature_quality,
    FeatureQualityReport,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def quarterly_df():
    """20-quarter DataFrame with 3 columns."""
    idx = pd.date_range("2000-03-31", periods=20, freq="QE")
    return pd.DataFrame(
        {"col_a": range(20), "col_b": np.random.default_rng(42).random(20), "col_c": range(20)},
        index=idx,
    )


@pytest.fixture
def sample_cfg():
    """Minimal config with FRED + multpl + macrotrends + ETF keys."""
    return {
        "fred": {
            "series": {
                "GDP": {"name": "fred_gdp", "shift": True},
                "CPIAUCSL": {"name": "fred_cpi", "shift": False},
            }
        },
        "multpl": {
            "datasets": [
                ["sp500", "S&P 500", "https://example.com", "number"],
                ["us_cpi", "US CPI", "https://example.com", "number"],
            ]
        },
        "macrotrends": {
            "series": [
                {"name": "gold_spot", "url": "https://example.com"},
            ]
        },
        "features": {
            "asset_price_columns": ["SPY", "TLT"],
        },
    }


# ── C1.1: format_completeness_table ──────────────────────────────────────


class TestFormatCompletenessTable:
    def test_pass_report_formatting(self, quarterly_df):
        from trading_crab_lib.ingestion import ingestion_completeness_report

        report = ingestion_completeness_report(
            quarterly_df, expected_columns=["col_a", "col_b", "col_c"]
        )
        text = format_completeness_table(report)
        assert "PASS" in text
        assert "3/3" in text

    def test_fail_report_shows_nan_table(self):
        from trading_crab_lib.ingestion import ingestion_completeness_report

        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame(
            {
                "good": range(10),
                "bad": [np.nan] * 6 + [1.0, 2.0, 3.0, 4.0],
            },
            index=idx,
        )
        report = ingestion_completeness_report(df, nan_threshold=0.5)
        text = format_completeness_table(report)
        assert "FAIL" in text
        assert "Top NaN columns" in text
        assert "bad" in text

    def test_no_nan_columns_no_table(self, quarterly_df):
        from trading_crab_lib.ingestion import ingestion_completeness_report

        report = ingestion_completeness_report(quarterly_df)
        text = format_completeness_table(report)
        assert "Top NaN columns" not in text


# ── C1.2: validate_date_range ─────────────────────────────────────────────


class TestValidateDateRange:
    def test_current_data_passes(self):
        idx = pd.date_range("2020-03-31", periods=25, freq="QE")
        df = pd.DataFrame({"x": range(25)}, index=idx)
        # Use a reference date that's within the data range
        ref = idx[-1] + pd.DateOffset(days=30)
        report = validate_date_range(df, reference_date=ref)
        assert report.passed
        assert report.quarters_behind <= 1

    def test_stale_data_warns(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        # Reference 10 years after data ends
        ref = datetime(2012, 6, 30)
        report = validate_date_range(df, reference_date=ref)
        assert not report.passed
        assert report.quarters_behind > 1

    def test_empty_df_fails(self):
        df = pd.DataFrame()
        report = validate_date_range(df)
        assert not report.passed

    def test_stale_series_detected(self):
        idx = pd.date_range("2020-03-31", periods=20, freq="QE")
        df = pd.DataFrame(
            {
                "fresh": range(20),
                "stale": [1.0] * 5 + [np.nan] * 15,
            },
            index=idx,
        )
        report = validate_date_range(
            df,
            reference_date=idx[-1],
            stale_threshold_quarters=2,
        )
        assert any("stale" == col for col, _ in report.stale_series)

    def test_summary_format(self):
        idx = pd.date_range("2020-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        report = validate_date_range(df, reference_date=idx[-1])
        text = report.summary()
        assert "Date range" in text

    def test_quarters_behind_calculation(self):
        idx = pd.date_range("2020-03-31", periods=4, freq="QE")
        df = pd.DataFrame({"x": range(4)}, index=idx)
        # Reference 2 years after last data point
        ref = datetime(2023, 3, 31)
        report = validate_date_range(df, reference_date=ref)
        assert report.quarters_behind >= 2


# ── C1.3: count_source_columns ────────────────────────────────────────────


class TestCountSourceColumns:
    def test_identifies_fred_columns(self, sample_cfg):
        df = pd.DataFrame(
            {"fred_gdp": [1], "fred_cpi": [2], "sp500": [3]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        assert counts.fred == 2
        assert counts.multpl == 1
        assert counts.total_columns == 3

    def test_identifies_etf_columns(self, sample_cfg):
        df = pd.DataFrame(
            {"etf_spy": [1], "etf_tlt": [2], "fred_gdp": [3]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        assert counts.etf == 2
        assert counts.fred == 1

    def test_identifies_macrotrends_columns(self, sample_cfg):
        df = pd.DataFrame(
            {"gold_spot": [1], "sp500": [2]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        assert counts.macrotrends == 1
        assert counts.multpl == 1

    def test_other_columns_counted(self, sample_cfg):
        df = pd.DataFrame(
            {"market_code": [1], "unknown_col": [2], "fred_gdp": [3]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        assert counts.other == 2  # market_code + unknown_col
        assert counts.fred == 1

    def test_summary_format(self, sample_cfg):
        df = pd.DataFrame(
            {"fred_gdp": [1], "sp500": [2]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        text = counts.summary()
        assert "FRED API" in text
        assert "multpl.com" in text
        assert "TOTAL" in text

    def test_empty_config(self):
        df = pd.DataFrame(
            {"a": [1], "b": [2]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, {})
        assert counts.other == 2
        assert counts.fred == 0


# ── C1.4: compute_feature_quality ─────────────────────────────────────────


class TestComputeFeatureQuality:
    def test_basic_quality_report(self):
        idx = pd.date_range("2000-03-31", periods=20, freq="QE")
        df = pd.DataFrame(
            {
                "feat_a": np.random.default_rng(0).random(20),
                "feat_b": np.random.default_rng(1).random(20) * 100,
                "feat_c": np.random.default_rng(2).random(20),
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        assert report.n_rows == 20
        assert report.n_cols == 3
        assert len(report.top_variance_columns) == 3
        assert len(report.top_correlation_pairs) <= 3

    def test_nan_detection(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame(
            {
                "clean": range(10),
                "dirty": [np.nan, np.nan, np.nan] + list(range(7)),
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        assert report.nan_counts["dirty"] == 3
        assert report.nan_counts["clean"] == 0
        assert report.top_nan_columns[0][0] == "dirty"

    def test_market_code_excluded(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame(
            {
                "feat": range(10),
                "market_code": [0] * 10,
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        assert report.n_cols == 1
        assert "market_code" not in report.nan_counts

    def test_variance_ranking(self):
        idx = pd.date_range("2000-03-31", periods=50, freq="QE")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "low_var": rng.random(50) * 0.01,
                "high_var": rng.random(50) * 1000,
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        assert report.top_variance_columns[0][0] == "high_var"

    def test_correlation_pairs(self):
        idx = pd.date_range("2000-03-31", periods=50, freq="QE")
        x = np.arange(50, dtype=float)
        df = pd.DataFrame(
            {
                "a": x,
                "b": x * 2 + 1,  # perfectly correlated with a
                "c": np.random.default_rng(0).random(50),  # uncorrelated
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        # a and b should be the most correlated pair
        top_pair = report.top_correlation_pairs[0]
        assert {top_pair[0], top_pair[1]} == {"a", "b"}
        assert abs(top_pair[2]) > 0.99

    def test_summary_string(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"x": range(10), "y": range(10)}, index=idx)
        report = compute_feature_quality(df)
        text = report.summary()
        assert "Feature quality" in text
        assert "NaN cells" in text
        assert "Top-5 highest-variance" in text

    def test_single_column(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        report = compute_feature_quality(df)
        assert report.n_cols == 1
        assert report.top_correlation_pairs == []

    def test_empty_df(self):
        df = pd.DataFrame()
        report = compute_feature_quality(df)
        assert report.n_rows == 0
        assert report.n_cols == 0
