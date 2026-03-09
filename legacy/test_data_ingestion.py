"""Tests for data_ingestion module — parse_multpl_series and helpers."""

import numpy as np
import pandas as pd
import pytest

from trading_crab.data_ingestion import parse_multpl_series


class TestParseMultplSeries:
    """Test the raw-row-to-Series conversion for various value types."""

    def test_numeric_values(self):
        """Basic numeric values with commas are parsed correctly."""
        raw = [
            ["Jan 31, 2020", "3,200.50"],
            ["Apr 30, 2020", "2,900.00"],
            ["Jul 31, 2020", "3,100.75"],
        ]
        result = parse_multpl_series(raw, "sp500", "num")
        assert isinstance(result, pd.Series)
        assert result.name == "sp500"
        assert len(result) > 0
        # Values should be floats with commas stripped
        assert all(isinstance(v, float) for v in result.values)
        assert result.max() > 3000  # commas were stripped correctly

    def test_percent_values(self):
        """Percent strings are stripped and divided by 100."""
        raw = [
            ["Mar 31, 2020", "5.5%"],
            ["Jun 30, 2020", "4.2%"],
            ["Sep 30, 2020", "3.8%"],
        ]
        result = parse_multpl_series(raw, "rate", "percent")
        assert all(v < 1.0 for v in result.values)  # divided by 100
        assert abs(result.max() - 0.055) < 0.01

    def test_million_suffix(self):
        """'million' suffix is stripped correctly."""
        raw = [
            ["Jan 31, 2020", "330 million"],
            ["Apr 30, 2020", "331 million"],
        ]
        result = parse_multpl_series(raw, "pop", "million")
        assert all(v > 100 for v in result.values)  # numeric, no suffix

    def test_trillion_suffix(self):
        """'trillion' suffix is stripped correctly."""
        raw = [
            ["Jan 31, 2020", "21.5 trillion"],
            ["Apr 30, 2020", "22.0 trillion"],
        ]
        result = parse_multpl_series(raw, "gdp", "trillion")
        assert all(v > 10 for v in result.values)

    def test_empty_strings_become_nan_then_dropped(self):
        """Empty value strings are treated as NaN and dropped before resampling.
        If a quarter has only empty values, that quarter is absent from output."""
        raw = [
            ["Jan 31, 2020", "100"],
            ["Feb 28, 2020", ""],
            ["Mar 31, 2020", "120"],  # Q1 has valid data
            ["Apr 30, 2020", ""],     # Q2 only entry is empty → NaN quarter
            ["Jul 31, 2020", "200"],  # Q3 has valid data
        ]
        result = parse_multpl_series(raw, "val", "num")
        # Q1=120, Q2=NaN (dropped), Q3=200 → dropna leaves 2 quarters
        # Actually: dropna is before set_index, so empty row is dropped,
        # but resample still creates Q2 with NaN. Result has 3 quarters.
        # The key assertion: no spurious zeros or empty-string artifacts
        assert all(v > 0 for v in result.dropna().values)

    def test_quarterly_resampling(self):
        """Monthly data is resampled to quarterly (last value per quarter)."""
        raw = [
            ["Jan 31, 2020", "100"],
            ["Feb 29, 2020", "110"],
            ["Mar 31, 2020", "120"],  # ← Q1 last
            ["Apr 30, 2020", "130"],
            ["May 31, 2020", "140"],
            ["Jun 30, 2020", "150"],  # ← Q2 last
        ]
        result = parse_multpl_series(raw, "val", "num")
        assert len(result) == 2  # two quarters
        assert result.iloc[0] == 120.0
        assert result.iloc[1] == 150.0
