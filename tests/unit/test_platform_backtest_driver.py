"""Unit tests for trading_crab_lib.platform.backtest.driver (EVAL-01).

Synthetic-only, no network, no real checkpoints — mirrors
tests/unit/test_platform_walkforward.py's synthetic-frame convention.
tmp_path is used for the registry ledger.

Most tests monkeypatch the driver's private ``_refit_l1``/``_refit_l2``
(and, for the cash-residual test, ``vol_targeted_tilt``) module-level names
so the walk-forward loop's plumbing (holdout boundary, compounding,
registry logging, cash-residual accrual, ablation-skip) can be verified
fast and deterministically, independent of the real jump-model/nowcaster
fit quality on tiny synthetic data (which is exactly Pitfall 2 territory —
covered separately by the real-fit resilience behavior, not re-tested here
with mocked internals).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import trading_crab_lib.platform.backtest.driver as driver
from trading_crab_lib.platform.honesty.registry import read_trials

N_MONTHS = 60
MIN_TRAIN = 24


def _cfg(*, min_train: int = MIN_TRAIN, skip_l1l2_for_ablation: bool = True) -> dict:
    """A small platform-config-shaped dict — reduced sizes for fast tests."""
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

    monthly_features includes the lean_feature_set columns the jump model
    needs (in particular trailing_return_1m, for canonicalize_states).
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


def _fake_refit_l1(train_features: pd.DataFrame, cfg: dict) -> pd.Series:
    """Fast, deterministic stand-in for the real L1 jump-model refit."""
    return pd.Series(0, index=train_features.index)


def _fake_refit_l2(train_features: pd.DataFrame, states: pd.Series, feature_row: pd.DataFrame, cfg: dict) -> pd.Series:
    """Fast, deterministic stand-in for the real L2 nowcaster refit."""
    return pd.Series({0: 0.6, 1: 0.4})


# ── TestHoldoutBoundary ──────────────────────────────────────────────────────


class TestHoldoutBoundary:
    def test_never_visits_a_date_past_the_holdout_cutoff(self, tmp_path, monkeypatch):
        """monthly_features extends into 2022; every visited train/test date
        captured by the L1/L2 refit spies must be <= 2020-12-31."""
        monthly_features, asset_returns, cash_returns = _make_synthetic_frame(n_months=60, start="2018-01-31")

        captured_dates: list[pd.Timestamp] = []

        def spy_refit_l1(train_features, cfg):
            captured_dates.append(train_features.index.max())
            return _fake_refit_l1(train_features, cfg)

        def spy_refit_l2(train_features, states, feature_row, cfg):
            captured_dates.append(train_features.index.max())
            captured_dates.append(feature_row.index[0])
            return _fake_refit_l2(train_features, states, feature_row, cfg)

        monkeypatch.setattr(driver, "_refit_l1", spy_refit_l1)
        monkeypatch.setattr(driver, "_refit_l2", spy_refit_l2)

        driver.run_backtest(
            monthly_features,
            asset_returns,
            _cfg(min_train=24),
            cash_returns=cash_returns,
            registry_path=tmp_path / "trials.jsonl",
        )

        assert captured_dates, "no dates were captured — refit spies never invoked"
        cutoff = pd.Timestamp("2020-12-31")
        for d in captured_dates:
            assert d <= cutoff, f"a fit saw date {d}, which is past the holdout cutoff {cutoff}"


# ── TestEquityCurveCompounding ───────────────────────────────────────────────


class TestEquityCurveCompounding:
    def test_terminal_log_wealth_equals_cumsum_log1p_return(self, tmp_path, monkeypatch):
        """The registry's logged terminal_log_wealth equals an independent
        cumsum(log1p(step_return)) computed from the RETURNED equity curve."""
        monthly_features, asset_returns, cash_returns = _make_synthetic_frame()
        monkeypatch.setattr(driver, "_refit_l1", _fake_refit_l1)
        monkeypatch.setattr(driver, "_refit_l2", _fake_refit_l2)

        registry_path = tmp_path / "trials.jsonl"
        equity_curve, _ = driver.run_backtest(
            monthly_features, asset_returns, _cfg(), cash_returns=cash_returns, registry_path=registry_path
        )

        expected_log_wealth = float(np.log1p(equity_curve["return"]).cumsum().iloc[-1])

        trials = read_trials(path=registry_path)
        assert len(trials) == 1
        logged_log_wealth = trials.iloc[0]["metrics"]["terminal_log_wealth"]
        assert logged_log_wealth == pytest.approx(expected_log_wealth)


# ── TestRegistryLogging ──────────────────────────────────────────────────────


class TestRegistryLogging:
    def test_exactly_one_new_row_per_call(self, tmp_path, monkeypatch):
        monthly_features, asset_returns, cash_returns = _make_synthetic_frame()
        monkeypatch.setattr(driver, "_refit_l1", _fake_refit_l1)
        monkeypatch.setattr(driver, "_refit_l2", _fake_refit_l2)

        registry_path = tmp_path / "trials.jsonl"
        assert not registry_path.exists()

        driver.run_backtest(
            monthly_features, asset_returns, _cfg(), cash_returns=cash_returns, registry_path=registry_path
        )
        assert len(read_trials(path=registry_path)) == 1

        driver.run_backtest(
            monthly_features, asset_returns, _cfg(), cash_returns=cash_returns, registry_path=registry_path
        )
        assert len(read_trials(path=registry_path)) == 2


# ── TestRefitFromTrainWindowOnly ─────────────────────────────────────────────


class TestRefitFromTrainWindowOnly:
    def test_train_window_never_contains_the_decision_date_and_no_checkpoint_load(self, tmp_path, monkeypatch):
        """Every _refit_l2 call's train_features index is strictly before
        that step's decision date t; CheckpointManager.load_model is never
        called (no reuse of the single Phase 3/4 nowcaster checkpoint)."""
        monthly_features, asset_returns, cash_returns = _make_synthetic_frame()

        captured: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        def spy_refit_l2(train_features, states, feature_row, cfg):
            captured.append((train_features.index.max(), feature_row.index[0]))
            return _fake_refit_l2(train_features, states, feature_row, cfg)

        monkeypatch.setattr(driver, "_refit_l1", _fake_refit_l1)
        monkeypatch.setattr(driver, "_refit_l2", spy_refit_l2)

        def _forbidden_load_model(self, name):
            raise AssertionError(f"CheckpointManager.load_model('{name}') must never be called by run_backtest")

        from trading_crab_lib.checkpoints import CheckpointManager

        monkeypatch.setattr(CheckpointManager, "load_model", _forbidden_load_model)

        driver.run_backtest(
            monthly_features,
            asset_returns,
            _cfg(),
            cash_returns=cash_returns,
            registry_path=tmp_path / "trials.jsonl",
        )

        assert captured, "no _refit_l2 calls were captured"
        for train_max, t in captured:
            assert train_max < t, f"train window max {train_max} is not strictly before decision date {t}"


# ── TestCashResidualEarnsCashReturn (review F4) ──────────────────────────────


class TestCashResidualEarnsCashReturn:
    def test_cash_residual_earns_cash_return_series_not_zero(self, tmp_path, monkeypatch):
        """A fixed cash residual weight must earn cash_returns for the test
        month; supplying cash_returns=None instead must reduce every step's
        realized return by exactly cash_weight * cash_return (symmetric with
        the baseline legs' cash convention — the residual is NOT a hard 0%)."""
        monthly_features, asset_returns, cash_returns = _make_synthetic_frame()

        fixed_weights = pd.Series({"SPY": 0.5, "TLT": 0.2})
        fixed_cash = 0.3

        def fake_vol_targeted_tilt(*args, **kwargs):
            return {"weights": fixed_weights, "cash": fixed_cash, "scale": 0.7, "portfolio_vol": 0.1}

        monkeypatch.setattr(driver, "_refit_l1", _fake_refit_l1)
        monkeypatch.setattr(driver, "_refit_l2", _fake_refit_l2)
        monkeypatch.setattr(driver, "vol_targeted_tilt", fake_vol_targeted_tilt)

        cfg = _cfg()

        equity_with_cash, _ = driver.run_backtest(
            monthly_features,
            asset_returns,
            cfg,
            cash_returns=cash_returns,
            registry_path=tmp_path / "trials_with_cash.jsonl",
        )
        equity_without_cash, _ = driver.run_backtest(
            monthly_features,
            asset_returns,
            cfg,
            cash_returns=None,
            registry_path=tmp_path / "trials_without_cash.jsonl",
        )

        assert len(equity_with_cash) == len(equity_without_cash)
        assert len(equity_with_cash) > 0

        for date in equity_with_cash.index:
            expected_cash_leg = fixed_cash * float(cash_returns.loc[date])
            actual_delta = equity_with_cash.loc[date, "return"] - equity_without_cash.loc[date, "return"]
            assert actual_delta == pytest.approx(expected_cash_leg), (
                f"step {date}: delta {actual_delta} != cash_weight*cash_return {expected_cash_leg} "
                "(cash residual must earn cash_returns, not a hard 0%)"
            )


# ── TestAblationSkipInvariant (review F5) ────────────────────────────────────


class TestAblationSkipInvariant:
    def test_skip_true_vs_false_identical_equity_curve_tilt_off(self, tmp_path, monkeypatch):
        """use_regime_tilt=False: skip_l1l2_for_ablation True vs False must
        produce a byte-identical equity curve (the L1/L2 output is discarded
        either way when the tilt is off); with skip=True the L1/L2 refit
        helpers must never be invoked."""
        monthly_features, asset_returns, cash_returns = _make_synthetic_frame()

        call_counts = {"l1": 0, "l2": 0}

        def counting_refit_l1(train_features, cfg):
            call_counts["l1"] += 1
            return _fake_refit_l1(train_features, cfg)

        def counting_refit_l2(train_features, states, feature_row, cfg):
            call_counts["l2"] += 1
            return _fake_refit_l2(train_features, states, feature_row, cfg)

        monkeypatch.setattr(driver, "_refit_l1", counting_refit_l1)
        monkeypatch.setattr(driver, "_refit_l2", counting_refit_l2)

        equity_skip_true, _ = driver.run_backtest(
            monthly_features,
            asset_returns,
            _cfg(skip_l1l2_for_ablation=True),
            cash_returns=cash_returns,
            use_regime_tilt=False,
            registry_path=tmp_path / "trials_skip_true.jsonl",
        )
        assert call_counts["l1"] == 0
        assert call_counts["l2"] == 0

        equity_skip_false, _ = driver.run_backtest(
            monthly_features,
            asset_returns,
            _cfg(skip_l1l2_for_ablation=False),
            cash_returns=cash_returns,
            use_regime_tilt=False,
            registry_path=tmp_path / "trials_skip_false.jsonl",
        )
        assert call_counts["l1"] > 0
        assert call_counts["l2"] > 0

        pd.testing.assert_frame_equal(equity_skip_true, equity_skip_false)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])


class TestWindowActiveFeatures:
    """Per-feature min_history activation (approach ii): a late-starting feature
    (VIX, extra curve tenors, any post-1990 ticker) enters the regime model once
    it has ≥ min_history months of data; windows without it use the features that
    exist then, so recent features are leveraged where available without
    discarding pre-1990 rows where they are not."""

    def test_activation_by_min_history(self):
        idx = pd.date_range("1962-01-31", periods=120, freq="ME")
        df = pd.DataFrame(index=idx)
        df["long_feature"] = np.arange(120, dtype=float)            # 120 obs
        df["mid_feature"] = np.arange(120, dtype=float)
        df.iloc[:30, df.columns.get_loc("mid_feature")] = np.nan     # 90 obs
        df["too_recent"] = np.arange(120, dtype=float)
        df.iloc[:110, df.columns.get_loc("too_recent")] = np.nan     # 10 obs
        df["not_started"] = np.nan                                   # 0 obs

        active = driver._window_active_features(df, list(df.columns), min_history=60)

        assert "long_feature" in active     # 120 ≥ 60
        assert "mid_feature" in active       # 90 ≥ 60 → leveraged
        assert "too_recent" not in active    # 10 < 60 → not yet
        assert "not_started" not in active   # 0 < 60

    def _lean_frame(self, n=180, vix_obs=0, seed=0):
        from trading_crab_lib.platform.config import load_platform_config
        from trading_crab_lib.platform.taxonomy import lean_feature_set

        cfg = load_platform_config()
        cfg = {**cfg, "labeling": {**cfg.get("labeling", {}), "n_restarts": 2, "K": 4}}
        lean = sorted(lean_feature_set(cfg))
        idx = pd.date_range("1962-01-31", periods=n, freq="ME")
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.normal(0, 1, (n, len(lean))), index=idx, columns=lean)
        if "fred_vix" in df.columns:  # VIX-like: only the trailing `vix_obs` months present
            df.iloc[: n - vix_obs, df.columns.get_loc("fred_vix")] = np.nan
        return df, cfg

    def test_too_recent_feature_excluded_long_history_kept(self):
        # VIX present only 24 months (< default 120) → excluded; the long-history
        # features train across the full window (early rows retained).
        df, cfg = self._lean_frame(n=180, vix_obs=24)
        states = driver._refit_l1(df, cfg)
        assert states.index.min() == df.index.min()   # pre-VIX rows retained
        assert len(states) >= 170

    def test_qualified_late_feature_is_leveraged(self):
        # VIX present 150 of 180 months (≥ 120) → ACTIVE; training block becomes
        # the rectangular VIX-era block (its start onward), so it IS used.
        df, cfg = self._lean_frame(n=180, vix_obs=150)
        states = driver._refit_l1(df, cfg)
        # block starts where VIX begins (row 30) → far fewer than the full 180 rows
        assert states.index.min() == df.index[30]
        assert len(states) <= 155


class TestCvSafeActiveFeatures:
    """The L2 degrades that cluster around a late feature's activation date are
    caused by a RARE CLASS in the block that feature truncates the training set
    to — CalibratedClassifierCV needs >= n_splits examples of every class present.
    That is independent of block SIZE, so raising min_history does not reliably
    fix it; admission must be gated on the induced block itself."""

    def _frame(self, n=240, late_start=200, rare_count=2):
        """Long feature over all n rows; `late` starting at `late_start`.

        Class 9 is PLENTIFUL in the early rows but appears only `rare_count`
        times inside the block `late` truncates the training set to — the real
        shape of the problem (a regime that has barely recurred since the late
        feature's start date). So the full block is CV-safe while the late block
        may not be, and dropping `late` is a fix that actually exists.
        """
        idx = pd.date_range("1990-01-31", periods=n, freq="ME")
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {"long": rng.normal(size=n), "late": rng.normal(size=n)}, index=idx
        )
        X.iloc[:late_start, X.columns.get_loc("late")] = np.nan
        # Early rows: 0/1/9 all plentiful. Late block: 0/1 alternating, with only
        # `rare_count` examples of class 9.
        early = [(0, 1, 9)[i % 3] for i in range(late_start)]
        late = [9] * rare_count + [i % 2 for i in range(n - late_start - rare_count)]
        return X, pd.Series(early + late, index=idx)

    def test_drops_late_feature_when_its_block_starves_a_class(self):
        X, y = self._frame(rare_count=2)
        active = driver._cv_safe_active_features(
            X, y, list(X.columns), min_history=30, n_splits=5
        )
        # `late` is old enough (40 obs >= 30) but its block holds only 2 examples
        # of class 9 (< 5) → dropped; the long-history feature carries the step.
        assert active == ["long"]

    def test_admits_late_feature_once_its_block_is_cv_safe(self):
        X, y = self._frame(rare_count=8)
        active = driver._cv_safe_active_features(
            X, y, list(X.columns), min_history=30, n_splits=5
        )
        # 8 examples of the rare class >= 5 folds → safe, so the late feature is
        # leveraged rather than needlessly withheld.
        assert set(active) == {"long", "late"}

    def test_guard_prevents_the_fit_error_it_targets(self):
        """End-to-end: the unguarded selection raises the exact
        CalibratedClassifierCV ValueError; the guarded one fits."""
        from trading_crab_lib.platform.prediction.nowcaster import fit_nowcaster

        X, y = self._frame(rare_count=2)
        unguarded = driver._window_active_features(X, list(X.columns), min_history=30)
        assert set(unguarded) == {"long", "late"}
        with pytest.raises(ValueError, match="less than 5 examples"):
            fit_nowcaster(X[unguarded], y, n_splits=5)

        guarded = driver._cv_safe_active_features(
            X, y, list(X.columns), min_history=30, n_splits=5
        )
        fit_nowcaster(X[guarded], y, n_splits=5)  # must not raise

    def test_returns_empty_when_no_subset_is_safe(self):
        # Every class is rare everywhere → nothing clears the bar; the caller's
        # fit then raises and the step degrades, exactly as before.
        idx = pd.date_range("1990-01-31", periods=40, freq="ME")
        X = pd.DataFrame({"a": np.arange(40, dtype=float)}, index=idx)
        y = pd.Series(range(40), index=idx)  # 40 singleton classes
        active = driver._cv_safe_active_features(
            X, y, list(X.columns), min_history=10, n_splits=5
        )
        assert active == []
