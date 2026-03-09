"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def quarterly_index():
    """20 quarter-end dates starting 2000-Q1."""
    return pd.date_range("2000-03-31", periods=20, freq="QE")


@pytest.fixture
def raw_macro_df(quarterly_index):
    """Minimal macro DataFrame with all columns needed by add_cross_ratios."""
    rng = np.random.default_rng(0)
    n = len(quarterly_index)
    return pd.DataFrame(
        {
            "sp500":     rng.uniform(800, 4000, n),
            "sp500_adj": rng.uniform(800, 4000, n),
            "dividend":  rng.uniform(10, 60, n),
            "div_yield": rng.uniform(0.01, 0.05, n),
            "gdp":       rng.uniform(8000, 22000, n),
            "cpi":       rng.uniform(150, 280, n),
            "fred_gdp":  rng.uniform(8000, 22000, n),
            "fred_gnp":  rng.uniform(7500, 21000, n),
            "fred_baa":  rng.uniform(3.0, 9.0, n),
            "fred_aaa":  rng.uniform(2.5, 8.0, n),
            "fred_cpi":  rng.uniform(150, 280, n),
        },
        index=quarterly_index,
    )


@pytest.fixture
def cluster_labels(quarterly_index):
    """Integer cluster labels cycling 0–4."""
    return pd.Series(
        np.tile(np.arange(5), len(quarterly_index) // 5 + 1)[: len(quarterly_index)],
        index=quarterly_index,
        name="balanced_cluster",
    )


@pytest.fixture
def asset_prices(quarterly_index):
    """Synthetic quarterly asset prices for 3 tickers."""
    rng = np.random.default_rng(1)
    n = len(quarterly_index)
    return pd.DataFrame(
        {
            "SPY": rng.uniform(80, 400, n),
            "GLD": rng.uniform(50, 200, n),
            "TLT": rng.uniform(70, 150, n),
        },
        index=quarterly_index,
    )
