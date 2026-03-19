"""Tests for ingestion completeness report (P23)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab.ingestion import ingestion_completeness_report, CompletenessReport


@pytest.fixture
def sample_df():
    idx = pd.date_range("2000-03-31", periods=20, freq="QE")
    return pd.DataFrame(
        {"col_a": range(20), "col_b": range(20), "col_c": range(20)},
        index=idx,
    )


class TestIngestCompleteness:
    def test_all_present_passes(self, sample_df):
        report = ingestion_completeness_report(
            sample_df, expected_columns=["col_a", "col_b", "col_c"]
        )
        assert report.passed
        assert report.total_columns == 3
        assert report.missing_columns == []

    def test_missing_column_fails(self, sample_df):
        report = ingestion_completeness_report(
            sample_df, expected_columns=["col_a", "col_b", "col_c", "col_d"]
        )
        assert not report.passed
        assert "col_d" in report.missing_columns

    def test_extra_columns_noted(self, sample_df):
        report = ingestion_completeness_report(
            sample_df, expected_columns=["col_a"]
        )
        assert "col_b" in report.extra_columns
        assert "col_c" in report.extra_columns

    def test_high_nan_column_fails(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame({
            "good": range(10),
            "bad": [np.nan] * 8 + [1.0, 2.0],
        }, index=idx)
        report = ingestion_completeness_report(df, nan_threshold=0.5)
        assert not report.passed
        assert "bad" in report.high_nan_columns
        assert "good" not in report.high_nan_columns

    def test_no_expected_columns_nan_only(self, sample_df):
        report = ingestion_completeness_report(sample_df)
        assert report.passed
        assert report.expected_columns == 3

    def test_empty_df(self):
        df = pd.DataFrame()
        report = ingestion_completeness_report(df, expected_columns=["x"])
        assert not report.passed
        assert "x" in report.missing_columns

    def test_summary_string(self, sample_df):
        report = ingestion_completeness_report(
            sample_df, expected_columns=["col_a", "col_b", "col_c"]
        )
        s = report.summary()
        assert "3/3" in s
        assert "PASS" in s

    def test_summary_fail_string(self, sample_df):
        report = ingestion_completeness_report(
            sample_df, expected_columns=["col_a", "col_b", "col_c", "col_d"]
        )
        s = report.summary()
        assert "FAIL" in s
        assert "MISSING" in s
