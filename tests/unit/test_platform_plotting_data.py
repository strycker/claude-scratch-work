"""Tests for src/trading_crab_lib/platform/plotting/data.py."""

from __future__ import annotations

# matplotlib.use("Agg") must precede pyplot import — import order is intentional.
# pylint: disable=wrong-import-position,wrong-import-order
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from trading_crab_lib.platform.plotting import data as pdata  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order


@pytest.fixture
def monthly_df() -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.standard_normal((40, 5)),
        index=idx,
        columns=["a", "b", "c", "d", "e"],
    )
    # Stagger first-valid dates so the sort-by-first-valid path is exercised.
    df.loc[df.index[:5], "c"] = np.nan
    df.loc[df.index[10:], "e"] = np.nan
    return df


class TestPlotCoverageTimeline:
    def test_does_not_crash(self, monthly_df, tmp_path):
        save_path = tmp_path / "coverage.png"
        fig = pdata.plot_coverage_timeline(monthly_df, title="monthly_raw column coverage", save_path=save_path)
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()

    def test_empty_input(self):
        fig = pdata.plot_coverage_timeline(pd.DataFrame())
        assert isinstance(fig, plt.Figure)

    def test_max_columns_truncates_without_crashing(self, monthly_df):
        fig = pdata.plot_coverage_timeline(monthly_df, max_columns=2)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes[0].get_yticklabels()) == 2
