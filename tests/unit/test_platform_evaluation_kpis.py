"""Unit tests for trading_crab_lib.platform.evaluation.kpis (EVAL-04/§8.9).

Follows the incumbent test_platform_gap_lag.py structure: a class per
behavior, docstring per test. All computation runs against hand-built
synthetic return series (no network, no checkpoint dependency) with a known
answer computed by hand in each test's docstring/comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation.kpis import (
    crisis_capture_ratio,
    cvar,
    max_drawdown_and_duration,
    terminal_log_wealth,
)
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF

# ── terminal_log_wealth ──────────────────────────────────────────────────────────


class TestTerminalLogWealth:
    def test_matches_log1p_sum(self):
        """terminal_log_wealth(returns) == np.log1p(returns).sum() for a known
        monthly return sequence."""
        idx = pd.date_range("2020-01-31", periods=6, freq="ME")
        returns = pd.Series([0.02, -0.01, 0.03, 0.00, -0.02, 0.015], index=idx)

        result = terminal_log_wealth(returns)

        assert result == pytest.approx(np.log1p(returns).sum())

    def test_all_zero_returns_is_zero(self):
        """A flat (all-zero) return series has terminal log wealth of exactly 0.0."""
        idx = pd.date_range("2020-01-31", periods=4, freq="ME")
        returns = pd.Series([0.0, 0.0, 0.0, 0.0], index=idx)

        assert terminal_log_wealth(returns) == pytest.approx(0.0)


# ── max_drawdown_and_duration ────────────────────────────────────────────────────


class TestMaxDrawdown:
    def test_known_worst_drawdown_and_duration(self):
        """Hand-built curve with exactly one drawdown episode: cum starts at
        1.0 (peak), falls for 4 periods, then recovers to a new high on
        period 5 (drawdown returns to 0) and stays flat.

        returns:      [0.00, -0.10, -0.10, 0.05, 0.05, 0.30, 0.00]
        cum:          [1.00,  0.90,  0.81, 0.8505, 0.893025, 1.1609325, 1.1609325]
        running_max:  [1.00,  1.00,  1.00, 1.00,   1.00,     1.1609325, 1.1609325]
        drawdown:     [0.00, -0.10, -0.19, -0.1495, -0.106975, 0.00, 0.00]

        Worst drawdown magnitude is -0.19 at index 2 (period "1980-02" in
        this synthetic index). The longest run below the prior peak (drawdown
        < 0) spans indices 1-4 inclusive == 4 periods; index 0 sits exactly
        at the peak (drawdown == 0, not underwater) and indices 5-6 have
        already recovered to a new high (drawdown == 0 again).
        """
        idx = pd.date_range("1980-01-31", periods=7, freq="ME")
        returns = pd.Series([0.00, -0.10, -0.10, 0.05, 0.05, 0.30, 0.00], index=idx)

        result = max_drawdown_and_duration(returns)

        assert result["max_drawdown"] == pytest.approx(-0.19)
        assert result["duration_months"] == 4

    def test_never_recovers_duration_runs_to_end_of_series(self):
        """A drawdown that never recovers to a new high before the series ends
        has its underwater run extend all the way to the last period."""
        idx = pd.date_range("1980-01-31", periods=4, freq="ME")
        returns = pd.Series([0.10, -0.20, 0.00, 0.00], index=idx)
        # cum: 1.10, 0.88, 0.88, 0.88; running_max: 1.10 throughout after peak.
        # drawdown: 0.0, -0.2, -0.2, -0.2 -> underwater run is indices 1-3 == 3.

        result = max_drawdown_and_duration(returns)

        assert result["max_drawdown"] == pytest.approx(-0.20)
        assert result["duration_months"] == 3

    def test_flat_series_has_zero_drawdown_and_zero_duration(self):
        """A monotonically non-negative return series never dips below its
        running peak — max_drawdown is 0.0 and duration is 0 periods."""
        idx = pd.date_range("1980-01-31", periods=3, freq="ME")
        returns = pd.Series([0.01, 0.02, 0.0], index=idx)

        result = max_drawdown_and_duration(returns)

        assert result["max_drawdown"] == pytest.approx(0.0)
        assert result["duration_months"] == 0


# ── cvar ──────────────────────────────────────────────────────────────────────────


class TestCVaR:
    def test_default_alpha_averages_worst_tail(self):
        """cvar(returns, alpha=0.05) == mean of the worst floor(alpha*n) returns
        (down-tail convention). n=20 -> floor(0.05*20) == 1, so CVaR is exactly
        the single worst observed return."""
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.005, 0.03, 20))
        worst_one = float(np.sort(returns.to_numpy())[0])

        result = cvar(returns)

        assert result == pytest.approx(worst_one)

    def test_custom_alpha_averages_worst_k(self):
        """A known 8-value sequence with alpha=0.25 -> floor(0.25*8) == 2:
        CVaR is the mean of the two worst returns."""
        returns = pd.Series([0.05, -0.30, 0.02, -0.10, 0.01, 0.03, -0.02, 0.04])
        worst_two_mean = float(np.mean([-0.30, -0.10]))

        result = cvar(returns, alpha=0.25)

        assert result == pytest.approx(worst_two_mean)


# ── crisis_capture_ratio ──────────────────────────────────────────────────────────


class TestCrisisCaptureBounded:
    def test_computes_strategy_over_benchmark_cumulative_per_window(self):
        """crisis_capture_ratio computes strategy-cumulative / benchmark-cumulative
        (down-capture convention, A6) over each configured in-sample window."""
        idx = pd.date_range("1973-01-31", periods=24, freq="ME")
        rng = np.random.default_rng(3)
        strategy_returns = pd.Series(rng.normal(-0.01, 0.02, 24), index=idx)
        benchmark_returns = pd.Series(rng.normal(-0.02, 0.03, 24), index=idx)
        windows = [{"name": "1973-74_oil_shock", "start": "1973-01-01", "end": "1974-12-31"}]

        result = crisis_capture_ratio(strategy_returns, benchmark_returns, windows)

        window = pd.Timestamp("1973-01-01"), pd.Timestamp("1974-12-31")
        strat_cum = (1 + strategy_returns.loc[window[0]:window[1]]).prod() - 1
        bench_cum = (1 + benchmark_returns.loc[window[0]:window[1]]).prod() - 1
        assert result["1973-74_oil_shock"] == pytest.approx(strat_cum / bench_cum)

    def test_window_end_past_cutoff_raises(self):
        """A window whose end date exceeds the holdout cutoff (2020-12-31) must
        never silently be included (Pitfall 4) — it raises ValueError."""
        idx = pd.date_range("2019-01-31", periods=36, freq="ME")
        strategy_returns = pd.Series(0.01, index=idx)
        benchmark_returns = pd.Series(0.01, index=idx)
        windows = [{"name": "2020-22_covid_and_beyond", "start": "2020-01-01", "end": "2022-12-31"}]

        with pytest.raises(ValueError):
            crisis_capture_ratio(strategy_returns, benchmark_returns, windows)

    def test_window_end_exactly_at_cutoff_is_accepted(self):
        """A window ending exactly at the cutoff (2020-12-31) is the boundary
        case that must be accepted, not rejected."""
        idx = pd.date_range("2008-01-31", periods=36, freq="ME")
        strategy_returns = pd.Series(0.01, index=idx)
        benchmark_returns = pd.Series(0.01, index=idx)
        windows = [{"name": "boundary_window", "start": "2008-01-01", "end": DEFAULT_HOLDOUT_CUTOFF}]

        result = crisis_capture_ratio(strategy_returns, benchmark_returns, windows)

        assert "boundary_window" in result

    def test_multiple_windows_all_computed(self):
        """Multiple in-sample windows all produce a result keyed by window name."""
        idx = pd.date_range("1973-01-31", periods=500, freq="ME")
        rng = np.random.default_rng(11)
        strategy_returns = pd.Series(rng.normal(0.005, 0.02, 500), index=idx)
        benchmark_returns = pd.Series(rng.normal(0.006, 0.03, 500), index=idx)
        windows = [
            {"name": "1973-74_oil_shock", "start": "1973-01-01", "end": "1974-12-31"},
            {"name": "2000-02_dotcom_bust", "start": "2000-01-01", "end": "2002-12-31"},
            {"name": "2008-09_gfc", "start": "2008-01-01", "end": "2009-12-31"},
        ]

        result = crisis_capture_ratio(strategy_returns, benchmark_returns, windows)

        assert set(result.keys()) == {"1973-74_oil_shock", "2000-02_dotcom_bust", "2008-09_gfc"}
