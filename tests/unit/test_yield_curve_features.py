from __future__ import annotations

import pandas as pd

from market_regime.transforms import add_yield_curve_features


def test_add_yield_curve_features_computes_spreads() -> None:
    idx = pd.date_range("2000-01-01", periods=3, freq="QE-DEC")
    df = pd.DataFrame(
        {
            "fred_gs10": [5.0, 4.0, 3.0],
            "fred_gs2": [3.0, 2.5, 2.0],
            "fred_tb3ms": [2.0, 1.5, 1.0],
        },
        index=idx,
    )
    out = add_yield_curve_features(df)
    assert "yc_10y_2y" in out.columns
    assert "yc_10y_3m" in out.columns
    assert "yc_2y_3m" in out.columns

    pd.testing.assert_series_equal(
        out["yc_10y_2y"],
        pd.Series([2.0, 1.5, 1.0], index=idx, name="yc_10y_2y"),
    )
    pd.testing.assert_series_equal(
        out["yc_10y_3m"],
        pd.Series([3.0, 2.5, 2.0], index=idx, name="yc_10y_3m"),
    )
    pd.testing.assert_series_equal(
        out["yc_2y_3m"],
        pd.Series([1.0, 1.0, 1.0], index=idx, name="yc_2y_3m"),
    )


def test_add_yield_curve_features_handles_missing_columns() -> None:
    idx = pd.date_range("2000-01-01", periods=2, freq="QE-DEC")
    df = pd.DataFrame({"fred_gs10": [5.0, 4.0]}, index=idx)
    out = add_yield_curve_features(df)
    # No spreads when required inputs are missing
    assert "yc_10y_2y" not in out
    assert "yc_10y_3m" not in out
    assert "yc_2y_3m" not in out
