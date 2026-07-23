"""Unit tests for trading_crab_lib.platform.tripwire.monitor (L4-04).

Follows the incumbent tests/unit/test_platform_gap_lag.py structure: a class
per behavior, docstring per test. The headline invariant is the OR-logic
truth table (T-04-11): escalation is COUNT-driven over the 3 independent
signals, never signal-identity-driven — swapping WHICH signal fired must
not change the tier.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib import ROOT
from trading_crab_lib.platform.tripwire import monitor
from trading_crab_lib.platform.tripwire.monitor import (
    TripwireEscalation,
    credit_spread_velocity,
    escalate,
    realized_vol_spike,
    run_tripwire,
    spy_drawdown_from_peak,
)

# ── escalate — headline OR-logic truth table ────────────────────────────────


class TestEscalateOrLogic:
    @pytest.mark.parametrize(
        "vol_spike,credit_velocity,spy_drawdown,expected",
        [
            (False, False, False, TripwireEscalation.NONE),
            (True, False, False, TripwireEscalation.RUN_WEEKLY_SCORING_EARLY),
            (False, True, False, TripwireEscalation.RUN_WEEKLY_SCORING_EARLY),
            (False, False, True, TripwireEscalation.RUN_WEEKLY_SCORING_EARLY),
            (True, True, False, TripwireEscalation.TIER1_DERISK_REVIEW),
            (True, False, True, TripwireEscalation.TIER1_DERISK_REVIEW),
            (False, True, True, TripwireEscalation.TIER1_DERISK_REVIEW),
            (True, True, True, TripwireEscalation.TIER1_DERISK_REVIEW),
        ],
    )
    def test_all_8_combinations(self, vol_spike, credit_velocity, spy_drawdown, expected):
        """0 triggers -> NONE, exactly 1 -> RUN_WEEKLY_SCORING_EARLY (all 3
        single-True cases), 2 or 3 -> TIER1_DERISK_REVIEW."""
        assert escalate(vol_spike, credit_velocity, spy_drawdown) == expected

    def test_count_driven_not_identity_driven(self):
        """The three single-True cases and the three two-True cases each
        produce identical tiers regardless of WHICH signal is True —
        swapping which family fired must not change the outcome."""
        single_true_results = {
            escalate(True, False, False),
            escalate(False, True, False),
            escalate(False, False, True),
        }
        assert single_true_results == {TripwireEscalation.RUN_WEEKLY_SCORING_EARLY}

        two_true_results = {
            escalate(True, True, False),
            escalate(True, False, True),
            escalate(False, True, True),
        }
        assert two_true_results == {TripwireEscalation.TIER1_DERISK_REVIEW}


# ── realized_vol_spike (vol family) ──────────────────────────────────────────


class TestRealizedVolSpike:
    def test_fires_when_recent_window_much_more_volatile_than_baseline(self):
        rng = np.random.default_rng(0)
        baseline = list(rng.normal(0, 0.001, 20))
        recent = list(rng.normal(0, 0.05, 5))
        daily_returns = pd.Series(baseline + recent)

        assert realized_vol_spike(
            daily_returns, short_window=5, baseline_window=20, ratio_threshold=1.5, halflife=3.0
        ) is True

    def test_quiet_when_recent_and_baseline_similar_vol(self):
        rng = np.random.default_rng(0)
        baseline = list(rng.normal(0, 0.01, 20))
        recent = list(rng.normal(0, 0.01, 5))
        daily_returns = pd.Series(baseline + recent)

        assert realized_vol_spike(
            daily_returns, short_window=5, baseline_window=20, ratio_threshold=1.5, halflife=3.0
        ) is False

    def test_too_short_series_returns_false_not_crash(self):
        daily_returns = pd.Series([0.01, 0.02])
        assert realized_vol_spike(
            daily_returns, short_window=5, baseline_window=20, ratio_threshold=1.5, halflife=3.0
        ) is False

    def test_reuses_ewma_vol_no_duplicate_decay_loop(self):
        """Grep-style check: must call ewma_vol(...) — do not re-implement
        the EWMA decay math (platform/assets/vol.py is the single source)."""
        source = inspect.getsource(monitor.realized_vol_spike)
        assert "ewma_vol(" in source


# ── credit_spread_velocity (credit family) ──────────────────────────────────


class TestCreditSpreadVelocity:
    def test_fires_on_widening_spread_over_lookback(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        daaa = pd.Series([4.0] * 10, index=idx)
        # BBA-AAA spread widens from 100bps to 130bps -> +30bps over 5 days
        dbaa = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0, 5.05, 5.10, 5.15, 5.20, 5.30], index=idx)

        assert credit_spread_velocity(daaa, dbaa, lookback_days=5, bps_threshold=25) is True

    def test_quiet_on_flat_spread(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        daaa = pd.Series([4.0] * 10, index=idx)
        dbaa = pd.Series([5.0] * 10, index=idx)

        assert credit_spread_velocity(daaa, dbaa, lookback_days=5, bps_threshold=25) is False

    def test_quiet_on_narrowing_spread(self):
        idx = pd.bdate_range("2024-01-01", periods=10)
        daaa = pd.Series([4.0] * 10, index=idx)
        dbaa = pd.Series([5.30, 5.20, 5.15, 5.10, 5.05, 5.0, 5.0, 5.0, 5.0, 5.0], index=idx)

        assert credit_spread_velocity(daaa, dbaa, lookback_days=5, bps_threshold=25) is False

    def test_too_short_series_returns_false_not_crash(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        daaa = pd.Series([4.0] * 3, index=idx)
        dbaa = pd.Series([5.0] * 3, index=idx)

        assert credit_spread_velocity(daaa, dbaa, lookback_days=5, bps_threshold=25) is False


# ── spy_drawdown_from_peak (price family) ────────────────────────────────────


class TestSpyDrawdownFromPeak:
    def test_fires_on_drawdown_past_threshold(self):
        prices = pd.Series([100, 105, 110, 108, 100, 96, 88])  # -20% from peak of 110
        assert spy_drawdown_from_peak(prices, drawdown_threshold=0.10) is True

    def test_quiet_near_peak(self):
        prices = pd.Series([100, 105, 110, 109, 108, 107])  # ~-2.7% from peak
        assert spy_drawdown_from_peak(prices, drawdown_threshold=0.10) is False

    def test_quiet_at_new_peak(self):
        prices = pd.Series([100, 105, 110, 115, 120])  # currently AT the peak
        assert spy_drawdown_from_peak(prices, drawdown_threshold=0.10) is False


# ── run_tripwire — orchestrator ──────────────────────────────────────────────


class TestRunTripwire:
    def _synthetic_series(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2024-01-01", periods=90)
        spy_prices = pd.Series(np.linspace(400, 420, 90), index=idx)
        daily_returns = pd.Series(rng.normal(0, 0.005, 90))
        daaa = pd.Series([4.0] * 90, index=idx)
        dbaa = pd.Series([5.0] * 90, index=idx)
        return daily_returns, daaa, dbaa, spy_prices

    def test_returns_escalation_on_injected_series(self):
        daily_returns, daaa, dbaa, spy_prices = self._synthetic_series()
        result = run_tripwire(
            {}, daily_returns=daily_returns, daaa=daaa, dbaa=dbaa, spy_prices=spy_prices
        )
        assert isinstance(result, TripwireEscalation)

    def test_reads_thresholds_from_cfg_tripwire_section(self):
        """A cfg override for spy_drawdown_pct changes the outcome — proves
        thresholds are actually read from cfg.get('tripwire', {})."""
        idx = pd.bdate_range("2024-01-01", periods=30)
        spy_prices = pd.Series(list(range(100, 130)), index=idx)  # steadily rising, no dip
        daily_returns = pd.Series([0.001] * 30)
        daaa = pd.Series([4.0] * 30, index=idx)
        dbaa = pd.Series([5.0] * 30, index=idx)

        cfg = {"tripwire": {"spy_drawdown_pct": 0.0}}  # any dip at all trips it
        # force a tiny dip at the end
        spy_prices.iloc[-1] = spy_prices.iloc[-2] - 1

        result = run_tripwire(
            cfg, daily_returns=daily_returns, daaa=daaa, dbaa=dbaa, spy_prices=spy_prices
        )
        assert isinstance(result, TripwireEscalation)
        assert result != TripwireEscalation.NONE

    def test_reads_daily_raw_and_fred_daily_raw_when_series_omitted(self):
        mock_cm = MagicMock()
        idx = pd.bdate_range("2024-01-01", periods=90)
        daily_raw = pd.DataFrame({"SPY": np.linspace(400, 410, 90)}, index=idx)
        fred_daily_raw = pd.DataFrame(
            {"fred_daaa": [4.0] * 90, "fred_dbaa": [5.0] * 90}, index=idx
        )

        def _load(name):
            return {"daily_raw": daily_raw, "fred_daily_raw": fred_daily_raw}[name]

        mock_cm.load.side_effect = _load

        result = run_tripwire({}, cm=mock_cm)

        assert isinstance(result, TripwireEscalation)
        loaded_names = [call.args[0] for call in mock_cm.load.call_args_list]
        assert "daily_raw" in loaded_names
        assert "fred_daily_raw" in loaded_names

    def test_source_references_daily_raw_and_fred_daily_raw_literals(self):
        """Grep-style check (T-04-12): the live-load path must reference the
        exact checkpoint names, not a silent all-clear on a missing frame."""
        source = inspect.getsource(monitor.run_tripwire)
        assert '"daily_raw"' in source
        assert '"fred_daily_raw"' in source

    def test_source_reads_thresholds_via_cfg_get_tripwire(self):
        source = inspect.getsource(monitor.run_tripwire)
        assert "cfg.get(\"tripwire\"" in source or "cfg.get('tripwire'" in source


# ── module-level: provisional thresholds documented (A3) ───────────────────


def test_thresholds_marked_provisional():
    source = inspect.getsource(monitor)
    assert "provisional" in source.lower()


# ── CLI — python3 -m trading_crab_lib.platform.tripwire.monitor ────────────


class TestCLI:
    def test_cli_prints_escalation_value_as_last_line_no_network(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        result = subprocess.run(
            [sys.executable, "-m", "trading_crab_lib.platform.tripwire.monitor"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        assert lines, "CLI produced no output"
        assert lines[-1].strip() in {v.value for v in TripwireEscalation}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
