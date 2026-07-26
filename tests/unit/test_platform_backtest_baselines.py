"""Unit tests for trading_crab_lib.platform.backtest.baselines (EVAL-02).

Synthetic-only, no network, no real checkpoints. Mirrors
tests/unit/test_platform_backtest_driver.py's synthetic-frame convention for
the ablation invariant test (self-contained per-file fixtures — no
cross-test-module imports, matching project convention).

Four behaviors under test:

- ``TestFaberNoLookahead``: the Faber SMA position for month t never uses
  month t's own level (1-step decision lag, A4).
- ``TestNoRegimeAblationInvariant``: ``no_regime_ablation`` is a pure
  delegation to ``run_backtest(..., use_regime_tilt=False)`` — never a
  forked allocation implementation (D-02).
- ``TestSpyBuyHold``: the SPY buy-and-hold leg is the equity return series
  unchanged (cost-free by construction).
- ``TestSixtyForty``: the 60/40 leg blends equity/bond returns and applies
  the documented monthly-reconstitution turnover cost when costed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import trading_crab_lib.platform.backtest.driver as driver
from trading_crab_lib.platform.backtest import baselines

N_MONTHS = 60
MIN_TRAIN = 24


def _cfg(*, min_train: int = MIN_TRAIN, skip_l1l2_for_ablation: bool = True) -> dict:
    """A small platform-config-shaped dict — reduced sizes for fast tests.

    Mirrors test_platform_backtest_driver.py::_cfg (self-contained duplicate,
    per project no-cross-test-import convention).
    """
    return {
        "labeling": {"K": 2, "lambda": 5.0, "n_restarts": 2, "embargo_months": 3},
        "allocation": {
            "target_vol_annual": 0.10,
            "ewma_halflife_months": 6,
            "portfolio_vol_min_obs": 3,
            "hysteresis": {"act_threshold": 0.70, "unwind_threshold": 0.40},
        },
        "backtest": {
            "cost_bps": 10,
            "min_train_months": min_train,
            "skip_l1l2_for_ablation": skip_l1l2_for_ablation,
        },
    }


def _make_synthetic_frame(
    n_months: int = N_MONTHS, start: str = "2018-01-31", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (monthly_features, asset_returns, cash_returns) on a month-end index.

    Mirrors test_platform_backtest_driver.py::_make_synthetic_frame.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="ME")
    lean_cols = [
        "curve_10y3m",
        "curve_10y2y",
        "credit_spread_baa_aaa",
        "fred_vix",
        "gold",
        "oil",
        "trailing_return_1m",
        "trailing_return_3m",
        "realized_vol_1m",
        "realized_vol_3m",
        "cape_shiller",
        "div_yield",
        "real_rate_level",
    ]
    monthly_features = pd.DataFrame(
        {col: rng.normal(0, 1, n_months) for col in lean_cols}, index=idx
    )
    asset_returns = pd.DataFrame(
        {
            "SPY": rng.normal(0.006, 0.03, n_months),
            "TLT": rng.normal(0.001, 0.02, n_months),
        },
        index=idx,
    )
    cash_returns = pd.Series(rng.normal(0.001, 0.0005, n_months), index=idx, name="cash")
    return monthly_features, asset_returns, cash_returns


# ── TestFaberNoLookahead (A4) ────────────────────────────────────────────────


class TestFaberNoLookahead:
    def test_position_never_uses_same_month_level(self):
        """A huge perturbation at date t must not change the position AT t
        (decided using data through t-1 only) but MUST change the position
        at t+1 (which is allowed to see t's level, now that month t has
        closed)."""
        idx = pd.date_range("2000-01-31", periods=30, freq="ME")
        # Strictly declining -> raw signal (level > sma) is False everywhere,
        # so the baseline position is uniformly False (out of market).
        levels = pd.Series(np.linspace(200.0, 100.0, 30), index=idx)

        pos_base = baselines._faber_position(levels, window=6)

        perturb_date = idx[15]
        next_date = idx[16]
        levels_perturbed = levels.copy()
        levels_perturbed.loc[perturb_date] = 1e6  # dwarfs the local rolling mean

        pos_perturbed = baselines._faber_position(levels_perturbed, window=6)

        assert pos_base.loc[perturb_date] == pos_perturbed.loc[perturb_date], (
            "position at t must be unaffected by perturbing level at t (no same-month peeking)"
        )
        assert bool(pos_base.loc[next_date]) != bool(pos_perturbed.loc[next_date]), (
            "position at t+1 SHOULD reflect the (now closed) month-t level"
        )

    def test_shift_invariance(self):
        """Position is a pure function of the ordered level VALUES (rolling
        by row-count, not by calendar date) — shifting the whole input
        series forward by one month shifts the position vector along with
        it, values unchanged."""
        idx = pd.date_range("2000-01-31", periods=30, freq="ME")
        rng = np.random.default_rng(7)
        levels = pd.Series(100 + rng.normal(0, 5, 30).cumsum(), index=idx)
        pos = baselines._faber_position(levels, window=6)

        shifted_idx = pd.date_range(idx[0] + pd.offsets.MonthEnd(1), periods=30, freq="ME")
        shifted_levels = pd.Series(levels.values, index=shifted_idx)
        shifted_pos = baselines._faber_position(shifted_levels, window=6)

        np.testing.assert_array_equal(pos.values, shifted_pos.values)

    def test_faber_sma_returns_equity_leg_in_market_cash_leg_out(self):
        """faber_sma switches between the equity return and the cash return
        per the (already-lagged) position vector."""
        idx = pd.date_range("2000-01-31", periods=8, freq="ME")
        levels = pd.Series([100.0] * 4 + [200.0] * 4, index=idx)
        cash_ret = pd.Series([0.001] * 8, index=idx)

        result = baselines.faber_sma(levels, cash_ret, window=3, cost_bps=0.0)

        pos = baselines._faber_position(levels, window=3)
        equity_ret = levels.pct_change()
        expected = pd.Series(
            [float(equity_ret.loc[d]) if bool(pos.loc[d]) else float(cash_ret.loc[d]) for d in idx],
            index=idx,
        )
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_cost_applied_only_on_switches(self):
        """apply_transaction_cost is charged only on the months where the
        position actually switches (turnover 1.0), never on months holding
        steady (turnover 0.0)."""
        idx = pd.date_range("2000-01-31", periods=8, freq="ME")
        levels = pd.Series([100.0] * 4 + [200.0] * 4, index=idx)
        cash_ret = pd.Series([0.0] * 8, index=idx)

        zero_cost = baselines.faber_sma(levels, cash_ret, window=3, cost_bps=0.0)
        costed = baselines.faber_sma(levels, cash_ret, window=3, cost_bps=50)

        pos = baselines._faber_position(levels, window=3)
        switches = pos.astype(float).diff().abs().fillna(0.0)
        expected_delta = switches * 50 / 1e4
        actual_delta = zero_cost - costed
        pd.testing.assert_series_equal(actual_delta, expected_delta, check_names=False)


# ── TestNoRegimeAblationInvariant (F5, D-02) ─────────────────────────────────


class TestNoRegimeAblationInvariant:
    def test_ablation_equals_tilt_off_driver_path(self, tmp_path):
        """no_regime_ablation(...) must be byte-identical to calling
        run_backtest(..., use_regime_tilt=False) directly — the ablation is
        the SAME code path, never a hand-rolled parallel implementation."""
        monthly_features, asset_returns, cash_returns = _make_synthetic_frame()
        cfg = _cfg()  # skip_l1l2_for_ablation defaults True

        equity_ablation, metrics_ablation = baselines.no_regime_ablation(
            monthly_features,
            asset_returns,
            cfg,
            cash_returns=cash_returns,
            registry_path=tmp_path / "trials_ablation.jsonl",
        )
        equity_direct, metrics_direct = driver.run_backtest(
            monthly_features,
            asset_returns,
            cfg,
            cash_returns=cash_returns,
            use_regime_tilt=False,
            registry_path=tmp_path / "trials_direct.jsonl",
        )

        pd.testing.assert_frame_equal(equity_ablation, equity_direct)
        assert metrics_ablation == metrics_direct

    def test_no_forked_allocation_import(self):
        """Grep-gate proxy: baselines.py must not define its own
        vol_targeted_tilt/regime_tilt_weights names (D-02 invariant) —
        exercised at the module level via the actual grep gate in Task 3's
        verify step; this asserts the module doesn't shadow those names."""
        assert not hasattr(baselines, "vol_targeted_tilt")
        assert not hasattr(baselines, "regime_tilt_weights")


# ── TestSpyBuyHold ────────────────────────────────────────────────────────────


class TestSpyBuyHold:
    def test_returns_equity_series_unchanged_cost_free(self):
        idx = pd.date_range("2000-01-31", periods=12, freq="ME")
        equity_ret = pd.Series(np.linspace(0.01, 0.02, 12), index=idx)
        result = baselines.spy_buy_hold(equity_ret)
        pd.testing.assert_series_equal(result, equity_ret, check_names=False)


# ── TestSixtyForty ────────────────────────────────────────────────────────────


class TestSixtyForty:
    def test_blends_60_40_zero_cost(self):
        idx = pd.date_range("2000-01-31", periods=24, freq="ME")
        rng = np.random.default_rng(3)
        equity_ret = pd.Series(rng.normal(0.01, 0.03, 24), index=idx)
        bond_ret = pd.Series(rng.normal(0.002, 0.01, 24), index=idx)

        result = baselines.sixty_forty(equity_ret, bond_ret, cost_bps=0.0)
        expected = 0.6 * equity_ret + 0.4 * bond_ret
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_monthly_reconstitution_turnover_reduces_costed_return(self):
        """With cost_bps > 0, the costed series must be <= the zero-cost
        blend at every step (turnover cost is never negative), and strictly
        less on at least one step (there IS documented turnover to cost)."""
        idx = pd.date_range("2000-01-31", periods=12, freq="ME")
        equity_ret = pd.Series([0.05, -0.03, 0.04, -0.02] * 3, index=idx)
        bond_ret = pd.Series([0.001] * 12, index=idx)

        zero_cost = baselines.sixty_forty(equity_ret, bond_ret, cost_bps=0.0)
        costed = baselines.sixty_forty(equity_ret, bond_ret, cost_bps=10)

        assert (costed <= zero_cost + 1e-12).all()
        assert (costed < zero_cost).any()

    def test_unsupported_rebalance_convention_raises(self):
        idx = pd.date_range("2000-01-31", periods=6, freq="ME")
        equity_ret = pd.Series([0.01] * 6, index=idx)
        bond_ret = pd.Series([0.002] * 6, index=idx)
        with pytest.raises(ValueError):
            baselines.sixty_forty(equity_ret, bond_ret, rebalance="quarterly")


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
