"""Validation gap **G6** — which consumers of ``vol_targeted_tilt`` pool, and which do not.

ADR-0001 § RE-PIN 2026-09-18 recurrence-exemption **condition (iv)** requires that a
sub-floor regime's per-regime Sharpe is *not* used as an unshrunk point estimate —
it must be partially pooled toward the all-history model (design §6.1 mitigation 2).
``allocation/joint_tilt.py::pool_low_n_regime_sharpe`` implements exactly that.

Three call sites consume ``vol_targeted_tilt``'s ``returns_by_regime`` argument, and
they do **not** agree:

===========================================  =========  ==================================
call site                                    pools?     arm that establishes it
===========================================  =========  ==================================
``backtest/driver.py:497``                   **no**     ``TestDriverConsumer``
``report/weekly.py:249``                     **no**     ``TestWeeklyConsumer``
``allocation/joint_tilt.py:319``             **yes**    ``TestJointTiltContrast``
===========================================  =========  ==================================

**This module pins the NON-compliance, deliberately.** Asserting that the driver
receives a condition-(iv)-compliant frame would be a test that *can only pass* once
somebody makes it compliant — the exact evidence shape ``.planning/UAT-AUDIT-2026-09-09.md``
records this project committing six times. Pinning what the code actually does means the
day someone closes the gap, these arms go red and the fix has to be conscious.

**The both-halves rule.** ``pool_low_n_regime_sharpe``'s own docstring: *"Regimes at or
above the floor are returned unchanged, so this is a strict no-op for a labeling that
satisfies §4.4 criterion 1 without the exemption."* So "the observed frame equals the
unpooled table" is *also* satisfied by a pooled implementation whenever the fixture has
no sub-floor regime. Every arm therefore (a) asserts its fixture really does contain a
sub-floor regime before evaluating the pin, and (b) asserts BOTH that the observed frame
equals the unpooled table AND that it differs from the pooled one, naming a differing
cell. Either half alone is a check that can only confirm.

**Out of scope, stated so it is not read as an omission.** Condition (iv)'s **covariance
clause** ("no per-regime covariance ... without partial pooling") is NOT pinned here: no
per-regime covariance exists at L4-01 — ``portfolio_vol`` is a linear-sum/EWMA estimate on
the blended weight vector — and the clause falls to L3 (design §6.2, Ledoit-Wolf within
regime). G6 pins the **Sharpe** pooling seam only. See ``08-G6.md``.

Synthetic-only: no network, no real checkpoints, no real registry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import trading_crab_lib.platform.backtest.driver as driver
from trading_crab_lib.platform.allocation.joint_tilt import (
    OCCUPANCY_FLOOR,
    low_n_regime_flags,
    pool_low_n_regime_sharpe,
)
from trading_crab_lib.platform.assets.returns import returns_by_regime_stats

_SHARPE_COL = "sharpe_annualized"

#: The regime the fixtures push strictly below ``OCCUPANCY_FLOOR``. Every arm asserts
#: this before it evaluates its pin — a fixture without a sub-floor regime makes
#: ``pool_low_n_regime_sharpe`` a strict no-op and the pin vacuous.
SUB_FLOOR_REGIME = 2


# ── shared fixture helpers ───────────────────────────────────────────────────


def _sub_floor_states(index: pd.Index) -> pd.Series:
    """Label the window so regime 2 occupies ~6% of months — below the 8% floor.

    Positions 0, 20, 40, ... get regime ``SUB_FLOOR_REGIME``; the rest alternate
    between regimes 0 and 1, which both land comfortably above the floor. The
    expanding-window driver always trains on ``index[:i]``, so these positions are
    stable across steps and the sub-floor property holds for every window the test
    exercises.
    """
    labels = [SUB_FLOOR_REGIME if i % 20 == 0 else (i % 2) for i in range(len(index))]
    return pd.Series(labels, index=index, dtype=int)


def _keyed_sharpe(table: pd.DataFrame) -> pd.Series:
    """The ``sharpe_annualized`` column keyed on ``(regime, asset)``, sorted."""
    return (
        table.set_index(["regime", "asset"])[_SHARPE_COL]
        .astype(float)
        .sort_index()
        .rename(_SHARPE_COL)
    )


def _assert_sub_floor_precondition(unpooled: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly if the fixture stopped containing a sub-floor regime.

    Without this, every arm below would pass vacuously: pooling would be a strict
    no-op and "equals the unpooled table" would be true of a pooled implementation too.
    """
    flags = low_n_regime_flags(unpooled)
    assert flags["low_n"].any(), (
        "fixture precondition FAILED: no regime sits below the "
        f"{OCCUPANCY_FLOOR:.0%} occupancy floor, so pool_low_n_regime_sharpe is a "
        "strict no-op and the pins below would prove nothing. Occupancies: "
        f"{flags.set_index('regime')['occupancy'].to_dict()}"
    )
    return flags


def _assert_unpooled_not_pooled(observed: pd.DataFrame, unpooled: pd.DataFrame, *, consumer: str) -> None:
    """The both-halves pin: observed == unpooled AND observed != pooled.

    NOTE TO A FUTURE READER: a red result here does **not** mean "add
    ``pool_low_n_regime_sharpe`` to the consumer and move on". Pooling the estimate
    changes the tilt, therefore the allocation weights, therefore ROADMAP criterion 7's
    measured lift — a number Phase 8 re-measures for an unrelated reason. Fixing G6 is
    NOT authorised by this phase. See ``08-G6.md`` § "What a red result means".
    """
    pooled, _ = pool_low_n_regime_sharpe(unpooled)

    obs = _keyed_sharpe(observed)
    raw = _keyed_sharpe(unpooled)
    shrunk = _keyed_sharpe(pooled)

    # Half 1 — the consumer received the raw per-regime estimate.
    pd.testing.assert_series_equal(
        obs,
        raw,
        check_exact=True,
        obj=f"{consumer}: sharpe_annualized handed to vol_targeted_tilt",
    )

    # Half 2 — and that is genuinely distinguishable from the pooled estimate on this
    # fixture. Without this, half 1 is satisfied by a pooled implementation too.
    differing = [
        (key, float(obs.loc[key]), float(shrunk.loc[key]))
        for key in obs.index
        if not np.isclose(obs.loc[key], shrunk.loc[key], equal_nan=True)
    ]
    assert differing, (
        f"{consumer}: the observed frame is indistinguishable from the POOLED table, so "
        "half 1 proves nothing — pooling was a no-op on this fixture. Rebuild the "
        "fixture with a regime strictly below the occupancy floor."
    )
    (regime, asset), got, pooled_value = differing[0]
    assert got != pytest.approx(pooled_value), (
        f"{consumer}: expected the UNPOOLED estimate but cell (regime={regime}, "
        f"asset={asset}) matches the pooled one"
    )


# ── arm 1: backtest/driver.py:497 — unpooled ─────────────────────────────────

N_MONTHS = 52
MIN_TRAIN = 40


def _cfg() -> dict:
    return {
        "labeling": {"K": 3, "lambda": 5.0, "n_restarts": 2, "embargo_months": 3},
        "allocation": {
            "target_vol_annual": 0.10,
            "ewma_halflife_months": 6,
            "portfolio_vol_min_obs": 3,
            "hysteresis": {"act_threshold": 0.70, "unwind_threshold": 0.40},
        },
        "backtest": {"cost_bps": 10, "min_train_months": MIN_TRAIN, "skip_l1l2_for_ablation": True},
    }


def _make_synthetic_frame(
    n_months: int = N_MONTHS, start: str = "2014-01-31", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(monthly_features, asset_returns, cash_returns), sub-floor months shocked.

    The sub-floor regime's months carry a large return offset so its per-regime Sharpe
    is far from the all-history Sharpe — which is what makes pooling visibly move a
    cell rather than nudge it inside float noise.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="ME")
    lean_cols = [
        "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
        "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
        "cape_shiller", "div_yield", "real_rate_level",
    ]
    monthly_features = pd.DataFrame({col: rng.normal(0, 1, n_months) for col in lean_cols}, index=idx)
    spy = rng.normal(0.006, 0.03, n_months)
    tlt = rng.normal(0.001, 0.02, n_months)
    shock = np.array([i % 20 == 0 for i in range(n_months)])
    spy[shock] -= 0.10
    tlt[shock] += 0.06
    asset_returns = pd.DataFrame({"SPY": spy, "TLT": tlt}, index=idx)
    cash_returns = pd.Series(rng.normal(0.001, 0.0005, n_months), index=idx, name="cash")
    return monthly_features, asset_returns, cash_returns


def _fake_refit_l2(train_features: pd.DataFrame, states: pd.Series, feature_row: pd.DataFrame, cfg: dict) -> pd.Series:
    """Probabilities over all three regimes, including the sub-floor one.

    The sub-floor regime carries real probability mass, so whether its Sharpe is pooled
    genuinely changes the weights — the non-compliance is material, not cosmetic.
    """
    return pd.Series({0: 0.5, 1: 0.3, SUB_FLOOR_REGIME: 0.2})


class TestDriverConsumer:
    """``backtest/driver.py:497`` hands ``vol_targeted_tilt`` the UNPOOLED estimate."""

    def test_driver_tilt_call_receives_the_unpooled_returns_by_regime(self, tmp_path, monkeypatch):
        """G6, driver arm: a pin of a KNOWN non-compliance with ADR-0001 condition (iv).

        A red result means somebody made ``driver.py`` pool before tilting — the pin must
        then be moved deliberately (and said so in the commit), not deleted, because the
        change also moves the allocation weights behind ROADMAP criterion 7.
        """
        monthly_features, asset_returns, cash_returns = _make_synthetic_frame()

        captured_states: list[pd.Series] = []
        captured_tables: list[pd.DataFrame] = []
        real_tilt = driver.vol_targeted_tilt

        def spy_refit_l1(train_features, cfg, *, frozen_features=None):
            states = _sub_floor_states(train_features.index)
            captured_states.append(states)
            return states

        def spy_tilt(regime_or_probs, returns_by_regime, asset_returns_arg, **kwargs):
            captured_tables.append(returns_by_regime.copy())
            return real_tilt(regime_or_probs, returns_by_regime, asset_returns_arg, **kwargs)

        monkeypatch.setattr(driver, "_refit_l1", spy_refit_l1)
        monkeypatch.setattr(driver, "_refit_l2", _fake_refit_l2)
        monkeypatch.setattr(driver, "vol_targeted_tilt", spy_tilt)

        driver.run_backtest(
            monthly_features,
            asset_returns,
            _cfg(),
            cash_returns=cash_returns,
            registry_path=tmp_path / "trials.jsonl",
        )

        assert captured_tables, "vol_targeted_tilt was never called — the spy captured nothing"
        observed = captured_tables[-1]
        states = captured_states[-1]

        # Recomputed independently from the same train window, not read back off the spy.
        unpooled = returns_by_regime_stats(asset_returns.loc[states.index], states)

        flags = _assert_sub_floor_precondition(unpooled)
        assert bool(flags.set_index("regime").loc[SUB_FLOOR_REGIME, "low_n"]), (
            f"regime {SUB_FLOOR_REGIME} was expected to be the sub-floor one"
        )

        _assert_unpooled_not_pooled(observed, unpooled, consumer="backtest/driver.py:497")
