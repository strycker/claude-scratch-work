"""Tests for src/trading_crab/reporting.py — dashboard + portfolio construction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_crab.reporting import (
    asset_signals,
    print_dashboard,
    save_dashboard_csv,
    simple_regime_portfolio,
    blended_regime_portfolio,
    generate_recommendation,
    build_recommendation_digest,
    save_recommendation_bundle,
    write_weekly_report_md,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def regime_returns():
    """Pivoted DataFrame: index=regime, columns=tickers, values=median return."""
    return pd.DataFrame(
        {
            "SPY": [0.06, 0.02, -0.01],
            "GLD": [0.01, 0.05, 0.03],
            "TLT": [-0.02, 0.03, 0.07],
        },
        index=[0, 1, 2],
    )


@pytest.fixture
def ranked_returns():
    """Flat ranked DataFrame (output of rank_assets_by_regime)."""
    return pd.DataFrame([
        {"regime": 0, "asset": "SPY", "median_quarterly_return": 0.06, "rank": 1},
        {"regime": 0, "asset": "GLD", "median_quarterly_return": 0.01, "rank": 2},
        {"regime": 0, "asset": "TLT", "median_quarterly_return": -0.02, "rank": 3},
        {"regime": 1, "asset": "GLD", "median_quarterly_return": 0.05, "rank": 1},
        {"regime": 1, "asset": "TLT", "median_quarterly_return": 0.03, "rank": 2},
        {"regime": 1, "asset": "SPY", "median_quarterly_return": 0.02, "rank": 3},
    ])


@pytest.fixture
def prediction():
    return {"regime": 0, "probabilities": {0: 0.6, 1: 0.3, 2: 0.1}}


@pytest.fixture
def transition_matrix():
    return pd.DataFrame(
        [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.2, 0.5]],
        index=[0, 1, 2],
        columns=[0, 1, 2],
    )


# ── asset_signals ─────────────────────────────────────────────────────────────

class TestAssetSignals:
    def test_signals_for_regime(self, ranked_returns):
        result = asset_signals(ranked_returns, current_regime=0)
        assert set(result.columns) >= {"asset", "median_quarterly_return", "signal"}
        assert len(result) == 3

    def test_green_signal_above_threshold(self, ranked_returns):
        result = asset_signals(ranked_returns, current_regime=0)
        spy_row = result[result["asset"] == "SPY"].iloc[0]
        assert spy_row["signal"] == "GREEN"  # 0.06 >= 0.05

    def test_yellow_signal_between_thresholds(self, ranked_returns):
        result = asset_signals(ranked_returns, current_regime=0)
        gld_row = result[result["asset"] == "GLD"].iloc[0]
        assert gld_row["signal"] == "YELLOW"  # 0.01 >= 0.0 but < 0.05

    def test_red_signal_below_zero(self, ranked_returns):
        result = asset_signals(ranked_returns, current_regime=0)
        tlt_row = result[result["asset"] == "TLT"].iloc[0]
        assert tlt_row["signal"] == "RED"  # -0.02 < 0.0

    def test_custom_thresholds(self, ranked_returns):
        result = asset_signals(
            ranked_returns, current_regime=0,
            thresholds={"green": 0.10, "yellow": 0.05},
        )
        # With high thresholds, SPY (0.06) should be YELLOW not GREEN
        spy_row = result[result["asset"] == "SPY"].iloc[0]
        assert spy_row["signal"] == "YELLOW"


# ── print_dashboard ──────────────────────────────────────────────────────────

def test_print_dashboard_does_not_crash(
    prediction, ranked_returns, transition_matrix, capsys
):
    sigs = asset_signals(ranked_returns, current_regime=0)
    print_dashboard(prediction, {0: "Growth", 1: "Recession"}, sigs, transition_matrix)
    captured = capsys.readouterr()
    assert "CURRENT REGIME" in captured.out
    assert "Growth" in captured.out


# ── save_dashboard_csv ────────────────────────────────────────────────────────

def test_save_dashboard_csv(ranked_returns, tmp_path):
    sigs = asset_signals(ranked_returns, current_regime=0)
    result_path = save_dashboard_csv(sigs, tmp_path)
    assert result_path.exists()
    loaded = pd.read_csv(result_path)
    assert len(loaded) == 3


# ── simple_regime_portfolio ──────────────────────────────────────────────────

class TestSimpleRegimePortfolio:
    def test_equal_weights_top_n(self, regime_returns):
        weights = simple_regime_portfolio(regime_returns, current_regime=0, top_n=2)
        assert len(weights) == 2
        assert weights.sum() == pytest.approx(1.0)

    def test_top_asset_is_best_performer(self, regime_returns):
        weights = simple_regime_portfolio(regime_returns, current_regime=0, top_n=1)
        assert "SPY" in weights.index  # SPY has 0.06, highest in regime 0

    def test_missing_regime_returns_empty(self, regime_returns):
        weights = simple_regime_portfolio(regime_returns, current_regime=99)
        assert weights.empty


# ── blended_regime_portfolio ─────────────────────────────────────────────────

class TestBlendedRegimePortfolio:
    def test_blended_sums_to_one(self, regime_returns):
        probs = {0: 0.6, 1: 0.3, 2: 0.1}
        weights = blended_regime_portfolio(regime_returns, probs, top_n=2)
        assert weights.sum() == pytest.approx(1.0)

    def test_dominant_regime_dominates_weights(self, regime_returns):
        probs = {0: 0.9, 1: 0.05, 2: 0.05}
        weights = blended_regime_portfolio(regime_returns, probs, top_n=2)
        # SPY should have high weight since regime 0 dominates and SPY is best there
        assert "SPY" in weights.index
        assert weights["SPY"] > 0.3

    def test_zero_probs_returns_empty(self, regime_returns):
        probs = {0: 0.0, 1: 0.0, 2: 0.0}
        weights = blended_regime_portfolio(regime_returns, probs)
        assert weights.empty


# ── generate_recommendation ──────────────────────────────────────────────────

class TestGenerateRecommendation:
    def test_all_cash_generates_buys(self):
        target = pd.Series({"SPY": 0.5, "GLD": 0.3, "TLT": 0.2})
        rec = generate_recommendation(target, current_weights=None)
        assert (rec["signal"] == "BUY").all()

    def test_hold_when_close(self):
        target = pd.Series({"SPY": 0.5, "GLD": 0.5})
        current = pd.Series({"SPY": 0.48, "GLD": 0.52})
        rec = generate_recommendation(target, current_weights=current, threshold=0.05)
        assert (rec["signal"] == "HOLD").all()

    def test_sell_when_reduce(self):
        target = pd.Series({"SPY": 0.0})
        current = pd.Series({"SPY": 0.5})
        rec = generate_recommendation(target, current_weights=current, threshold=0.03)
        assert rec.loc["SPY", "signal"] == "SELL"

    def test_columns_present(self):
        target = pd.Series({"SPY": 0.5})
        rec = generate_recommendation(target)
        assert set(rec.columns) >= {"current_pct", "target_pct", "delta_pct", "signal"}


# ── build_recommendation_digest ──────────────────────────────────────────────

def test_build_recommendation_digest_basic():
    behavior = pd.DataFrame([
        {"regime": 0, "asset": "SPY", "signal_display": "green_strong", "rank": 1,
         "median_return": 0.06, "score_relative": 100.0, "score_absolute": 85.0},
        {"regime": 0, "asset": "GLD", "signal_display": "neutral", "rank": 2,
         "median_return": 0.01, "score_relative": 50.0, "score_absolute": 55.0},
    ])
    current = pd.Series({"SPY": 0.5})
    target = pd.Series({"SPY": 0.6, "GLD": 0.4})
    rec = generate_recommendation(target, current_weights=current)
    digest = build_recommendation_digest(behavior, 0, current, target, rec, top_n=2)
    assert not digest.empty
    assert "in_holdings" in digest.columns
    assert "is_strong_green" in digest.columns


# ── save_recommendation_bundle ───────────────────────────────────────────────

def test_save_recommendation_bundle(tmp_path):
    digest = pd.DataFrame([{"asset": "SPY", "score": 85.0}])
    path = tmp_path / "bundle.parquet"
    save_recommendation_bundle(digest, 0, "Growth", {0: 0.6, 1: 0.4}, path)
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert "current_regime" in loaded.columns
    assert "regime_name" in loaded.columns


# ── write_weekly_report_md ───────────────────────────────────────────────────

def test_write_weekly_report_md(tmp_path):
    rec = pd.DataFrame(
        {"current_pct": [0.0, 50.0], "target_pct": [40.0, 0.0],
         "delta_pct": [40.0, -50.0], "signal": ["BUY", "SELL"]},
        index=pd.Index(["SPY", "GLD"], name="asset"),
    )
    trans = pd.Series({0: 0.5, 1: 0.3, 2: 0.2})
    path = tmp_path / "weekly_report.md"
    write_weekly_report_md(0, "Growth", {0: 0.6, 1: 0.4}, rec, trans, path)
    assert path.exists()
    text = path.read_text()
    assert "Current regime" in text
    assert "BUY" in text
    assert "SELL" in text
    assert "Growth" in text
