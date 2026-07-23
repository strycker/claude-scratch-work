"""Unit tests for trading_crab_lib.platform.assets.vol (L3-02).

Follows the incumbent tests/unit/test_platform_gap_lag.py structure: a class
per behavior, docstring per test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.assets.vol import (
    DAILY_ANNUALIZATION,
    MONTHLY_ANNUALIZATION,
    ewma_vol,
    latest_ewma_vol,
    report_vol,
)


def _reference_ewma_std(returns: pd.Series, halflife: float) -> pd.Series:
    """Hand-rolled RiskMetrics-style EWMA std, independent of ewma_vol's own
    implementation, used as the correctness oracle."""
    alpha = 1 - math.exp(-math.log(2) / halflife)
    return returns.ewm(alpha=alpha, min_periods=2).std()


# ── ewma_vol ──────────────────────────────────────────────────────────────────────


class TestEwmaVol:
    def test_matches_hand_rolled_reference_daily(self):
        """ewma_vol matches a hand-computed EWMA std (alpha = 1 - exp(-ln(2)/h))
        times sqrt(annualization_factor) on a fixed synthetic series."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.01, 100))
        halflife = 11.2

        result = ewma_vol(returns, halflife=halflife, annualization_factor=DAILY_ANNUALIZATION)
        expected = _reference_ewma_std(returns, halflife) * math.sqrt(DAILY_ANNUALIZATION)

        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_annualization_scales_by_sqrt_af(self):
        """Daily uses annualization_factor=252, monthly uses 12; the returned
        value scales by sqrt(af)."""
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0, 0.02, 50))

        daily = ewma_vol(returns, halflife=6.0, annualization_factor=DAILY_ANNUALIZATION)
        monthly = ewma_vol(returns, halflife=6.0, annualization_factor=MONTHLY_ANNUALIZATION)

        ratio = (daily / monthly).dropna()
        expected_ratio = math.sqrt(DAILY_ANNUALIZATION) / math.sqrt(MONTHLY_ANNUALIZATION)
        assert ratio.apply(lambda x: x == pytest.approx(expected_ratio)).all()

    def test_first_element_nan_due_to_min_periods(self):
        """min_periods=2 semantics: the first element of any series is NaN."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        result = ewma_vol(returns, halflife=5.0, annualization_factor=DAILY_ANNUALIZATION)
        assert math.isnan(result.iloc[0])

    def test_constant_return_series_has_zero_vol_no_error(self):
        """A constant-return series has ~0 volatility, no divide-by-anything error."""
        returns = pd.Series([0.005] * 10)
        result = ewma_vol(returns, halflife=4.0, annualization_factor=MONTHLY_ANNUALIZATION)
        assert result.iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_uses_ewm_not_manual_alpha_loop(self):
        """Grep-style check: implementation must call .ewm(halflife=...)."""
        import inspect

        from trading_crab_lib.platform.assets import vol as vol_module

        source = inspect.getsource(vol_module.ewma_vol)
        assert ".ewm(" in source
        assert "halflife" in source


# ── latest_ewma_vol ───────────────────────────────────────────────────────────────


class TestLatestEwmaVol:
    def test_returns_last_non_nan_value_as_float(self):
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(0, 0.01, 30))
        result = latest_ewma_vol(returns, halflife=5.0, annualization_factor=DAILY_ANNUALIZATION)
        assert isinstance(result, float)
        assert result > 0

    def test_length_one_series_returns_nan_not_crash(self):
        """min_periods=2 semantics — the first element is NaN, so a length-1
        series returns NaN, not a crash."""
        returns = pd.Series([0.01])
        result = latest_ewma_vol(returns, halflife=5.0, annualization_factor=DAILY_ANNUALIZATION)
        assert math.isnan(result)


# ── report_vol ────────────────────────────────────────────────────────────────────


class TestReportVol:
    def test_persists_schema_stable_parquet(self, tmp_path):
        vols = {"SPY": 0.18, "TLT": 0.09}
        report_dir = tmp_path / "reports"

        path = report_vol(vols, halflife=11.2, output_dir=report_dir)

        assert path.exists()
        df = pd.read_parquet(path)
        assert "asset" in df.columns
        assert "ewma_vol_annualized" in df.columns
        assert "halflife" in df.columns
        assert len(df) == 2
        assert set(df["asset"]) == {"SPY", "TLT"}

    def test_empty_vols_schema_stable(self, tmp_path):
        report_dir = tmp_path / "reports"
        path = report_vol({}, halflife=11.2, output_dir=report_dir)
        df = pd.read_parquet(path)
        assert "asset" in df.columns
        assert len(df) == 0
