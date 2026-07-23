"""
Tests for the L4-01 vol-targeted tilt (design §6.2/§7, D-03).

Headline invariants: ``vol_target_scale`` never levers above 1.0 and degrades
to 0.0 (all-cash) on non-positive vol; ``portfolio_vol`` falls back to the
conservative linear-sum-of-vols estimate on short joint history — it can only
UNDER-lever, never over-lever.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.allocation.tilt import (
    portfolio_vol,
    regime_tilt_weights,
    vol_target_scale,
    vol_targeted_tilt,
)
from trading_crab_lib.platform.assets.vol import ewma_vol

# ── vol_target_scale: leverage cap + zero-vol degrade ────────────────────────


class TestVolTargetScale:
    def test_scales_down_when_portfolio_vol_exceeds_target(self):
        assert vol_target_scale(0.10, 0.20) == 0.5

    def test_caps_at_one_never_levers(self):
        assert vol_target_scale(0.10, 0.05) == 1.0

    def test_degrades_to_zero_on_zero_vol(self):
        assert vol_target_scale(0.10, 0.0) == 0.0

    def test_degrades_to_zero_on_negative_vol_no_divide_by_zero(self):
        assert vol_target_scale(0.10, -1.0) == 0.0


# ── portfolio_vol: blended EWMA path vs conservative linear-sum fallback ─────


class TestPortfolioVol:
    def _asset_returns(self, n_months: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
        return pd.DataFrame(
            {"SPY": rng.normal(0.01, 0.04, n_months), "TLT": rng.normal(0.0, 0.02, n_months)},
            index=idx,
        )

    def test_uses_blended_ewma_path_with_sufficient_joint_history(self):
        """>= min_obs overlapping non-NaN months -> blended weighted-return
        EWMA path, matching a hand-computed reference exactly."""
        asset_returns = self._asset_returns(24)
        weights = pd.Series({"SPY": 0.6, "TLT": 0.4})

        result = portfolio_vol(weights, asset_returns, halflife=6, min_obs=12)

        blended = (asset_returns * weights).sum(axis=1)
        expected = float(ewma_vol(blended, halflife=6, annualization_factor=12).iloc[-1])
        assert result == pytest.approx(expected)

    def test_falls_back_to_conservative_linear_sum_on_short_joint_history(self):
        """Too few overlapping non-NaN months (< min_obs) -> the linear-sum
        fallback, which over-estimates (ignores diversification) and can
        only under-lever, never over-lever."""
        asset_returns = self._asset_returns(24)
        # TLT has only 5 non-NaN months of joint history with SPY.
        asset_returns.loc[asset_returns.index[5:], "TLT"] = np.nan
        weights = pd.Series({"SPY": 0.6, "TLT": 0.4})

        result = portfolio_vol(weights, asset_returns, halflife=6, min_obs=12)

        spy_vol = float(ewma_vol(asset_returns["SPY"].dropna(), halflife=6, annualization_factor=12).iloc[-1])
        tlt_vol = float(ewma_vol(asset_returns["TLT"].dropna(), halflife=6, annualization_factor=12).iloc[-1])
        expected = 0.6 * spy_vol + 0.4 * tlt_vol
        assert result == pytest.approx(expected)


# ── regime_tilt_weights: clipped, normalized, deterministic ─────────────────


class TestRegimeTiltWeights:
    def _returns_by_regime(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"regime": 0, "asset": "SPY", "sharpe_annualized": 1.2, "n_obs": 30},
                {"regime": 0, "asset": "TLT", "sharpe_annualized": -0.5, "n_obs": 30},
                {"regime": 0, "asset": "GLD", "sharpe_annualized": 0.3, "n_obs": 30},
                {"regime": 1, "asset": "SPY", "sharpe_annualized": -0.8, "n_obs": 30},
                {"regime": 1, "asset": "TLT", "sharpe_annualized": 1.5, "n_obs": 30},
                {"regime": 1, "asset": "GLD", "sharpe_annualized": 0.6, "n_obs": 30},
            ]
        )

    def test_negative_sharpe_clipped_to_zero(self):
        weights = regime_tilt_weights(0, self._returns_by_regime(), {0: 1.0})
        assert weights["TLT"] == 0.0

    def test_weights_sum_to_one(self):
        weights = regime_tilt_weights(0, self._returns_by_regime(), {0: 1.0})
        assert weights.sum() == pytest.approx(1.0)

    def test_blends_across_regimes_by_probs(self):
        weights = regime_tilt_weights(0, self._returns_by_regime(), {0: 0.5, 1: 0.5})
        assert weights.sum() == pytest.approx(1.0)
        assert (weights >= 0).all()

    def test_deterministic(self):
        returns_by_regime = self._returns_by_regime()
        first = regime_tilt_weights(0, returns_by_regime, {0: 0.6, 1: 0.4})
        second = regime_tilt_weights(0, returns_by_regime, {0: 0.6, 1: 0.4})
        pd.testing.assert_series_equal(first.sort_index(), second.sort_index())


# ── vol_targeted_tilt: leverage-capped weights + cash residual ──────────────


class TestVolTargetedTilt:
    def _returns_by_regime(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"regime": 0, "asset": "SPY", "sharpe_annualized": 1.2, "n_obs": 30},
                {"regime": 0, "asset": "TLT", "sharpe_annualized": 0.4, "n_obs": 30},
            ]
        )

    def _asset_returns(self, n_months: int = 24, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
        return pd.DataFrame(
            {"SPY": rng.normal(0.01, 0.04, n_months), "TLT": rng.normal(0.0, 0.02, n_months)},
            index=idx,
        )

    def test_weights_nonnegative_and_sum_to_scale(self):
        result = vol_targeted_tilt(
            {0: 1.0}, self._returns_by_regime(), self._asset_returns(),
            target_vol_annual=0.10, halflife=6, min_obs=12,
        )
        assert (result["weights"] >= 0).all()
        assert result["weights"].sum() == pytest.approx(result["scale"])

    def test_cash_absorbs_residual_and_totals_one(self):
        result = vol_targeted_tilt(
            {0: 1.0}, self._returns_by_regime(), self._asset_returns(),
            target_vol_annual=0.10, halflife=6, min_obs=12,
        )
        assert result["cash"] == pytest.approx(1.0 - result["scale"])
        assert result["weights"].sum() + result["cash"] == pytest.approx(1.0)

    def test_scale_never_exceeds_one(self):
        """Extremely low target_vol still never levers above 1.0."""
        result = vol_targeted_tilt(
            {0: 1.0}, self._returns_by_regime(), self._asset_returns(),
            target_vol_annual=100.0, halflife=6, min_obs=12,
        )
        assert result["scale"] <= 1.0

    def test_accepts_bare_regime_id_as_single_regime_point_bet(self):
        """regime_or_probs may also be a bare active-regime id (not a probs
        mapping) — treated as a 100% point bet on that regime."""
        result = vol_targeted_tilt(
            0, self._returns_by_regime(), self._asset_returns(),
            target_vol_annual=0.10, halflife=6, min_obs=12,
        )
        assert (result["weights"] >= 0).all()
