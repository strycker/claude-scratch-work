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


# ── arm 2: report/weekly.py:249 — unpooled, and this is the one Glenn reads ──


def _hand_built_sub_floor_table() -> pd.DataFrame:
    """The ``returns_by_regime`` checkpoint shape, with regime 2 at 5.98% occupancy.

    ``regime_occupancy`` estimates a regime's months as the max ``n_obs`` across its
    assets, so 120 / 100 / 14 gives shares 0.5128 / 0.4274 / **0.0598** — only the last
    is below ``OCCUPANCY_FLOOR``. Sharpes are derived from the per-cell moments rather
    than typed independently, so the table is internally coherent and
    ``_all_history_sharpe``'s exact sufficient-statistic reconstruction is exercised.
    """
    #: The sub-floor regime carries TWO positive-Sharpe assets whose ratio pooling
    #: changes. With only one positive asset, ``_per_regime_tilt`` normalizes it to 1.0
    #: either way and pooling moves the Sharpes but NOT the weights — the contrast arm
    #: would then be vacuous for exactly the reason half 1 alone is (issue found while
    #: executing 08-04; the first fixture attempt had it).
    rows = [
        (0, "SPY", 0.0120, 0.030, 120),
        (0, "TLT", 0.0005, 0.020, 120),
        (1, "SPY", 0.0040, 0.045, 100),
        (1, "TLT", 0.0010, 0.018, 100),
        (SUB_FLOOR_REGIME, "SPY", 0.0005, 0.040, 14),
        (SUB_FLOOR_REGIME, "TLT", 0.0300, 0.030, 14),
    ]
    return pd.DataFrame(
        [
            {
                "regime": regime,
                "asset": asset,
                "mean_monthly_return": mean,
                "std_monthly_return": std,
                _SHARPE_COL: (mean / std) * np.sqrt(12),
                "hit_rate": 0.5,
                "max_drawdown": -0.2,
                "n_obs": n_obs,
            }
            for regime, asset, mean, std, n_obs in rows
        ]
    )


def _synthetic_asset_returns(n_months: int = 36, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        {"SPY": rng.normal(0.006, 0.03, n_months), "TLT": rng.normal(0.001, 0.02, n_months)},
        index=idx,
    )


class _FakeNowcaster:
    """Minimal ``predict_proba`` stand-in — classes span all three regimes."""

    classes_ = np.array([0, 1, SUB_FLOOR_REGIME])

    def predict_proba(self, features_row: pd.DataFrame) -> np.ndarray:
        return np.array([[0.5, 0.3, 0.2]])


class _FakeCheckpointManager:
    """Serves exactly the five artifacts ``_build_report_inputs`` loads."""

    def __init__(self, returns_by_regime: pd.DataFrame, asset_returns: pd.DataFrame) -> None:
        self._payload = {
            "regime_labels": pd.DataFrame({"state": [0, 1, SUB_FLOOR_REGIME, 0, 1, 0]}),
            "returns_by_regime": returns_by_regime,
            "asset_returns": asset_returns,
        }

    def load_model(self, name: str) -> _FakeNowcaster:
        assert name == "nowcaster"
        return _FakeNowcaster()

    def load(self, name: str):
        return self._payload[name]


class TestWeeklyConsumer:
    """``report/weekly.py:249`` hands ``vol_targeted_tilt`` the UNPOOLED estimate.

    This is the consumer that reaches the human: the weekly markdown Glenn reads before
    trading in Fidelity is built from the tilt computed here.
    """

    def test_weekly_tilt_call_receives_the_unpooled_returns_by_regime(self, monkeypatch):
        """G6, weekly arm: the same known non-compliance, in the human-facing path.

        A red result means ``report/weekly.py`` started pooling — move this pin
        deliberately (see ``08-G6.md``); it is not a bug in the test.
        """
        from trading_crab_lib.platform.report import weekly

        unpooled = _hand_built_sub_floor_table()
        asset_returns = _synthetic_asset_returns()
        _assert_sub_floor_precondition(unpooled)

        captured_tables: list[pd.DataFrame] = []
        real_tilt = weekly.vol_targeted_tilt

        def spy_tilt(regime_or_probs, returns_by_regime, asset_returns_arg, **kwargs):
            captured_tables.append(returns_by_regime.copy())
            return real_tilt(regime_or_probs, returns_by_regime, asset_returns_arg, **kwargs)

        monkeypatch.setattr(weekly, "vol_targeted_tilt", spy_tilt)
        monkeypatch.setattr(weekly, "load_full_span", lambda name: _synthetic_asset_returns())
        monkeypatch.setattr(weekly, "load_active_regime", lambda cm: None)
        monkeypatch.setattr(weekly, "save_active_regime", lambda regime, cm: None)

        cfg = {"allocation": {"target_vol_annual": 0.10, "ewma_halflife_months": 6, "portfolio_vol_min_obs": 3}}
        weekly._build_report_inputs(cfg, cm=_FakeCheckpointManager(unpooled, asset_returns))

        assert captured_tables, "vol_targeted_tilt was never called — the spy captured nothing"
        _assert_unpooled_not_pooled(captured_tables[-1], unpooled, consumer="report/weekly.py:249")


# ── arm 3: allocation/joint_tilt.py:319 — pools, and the difference is real ──


class TestJointTiltContrast:
    """``joint_tilt`` DOES pool, and that changes the portfolio, not just a label.

    Without this arm G6 could be read as a difference of naming. It is not: on the same
    sub-floor fixture the two paths produce different weights.
    """

    def test_pooled_joint_tilt_produces_different_weights_than_the_unpooled_path(self):
        """The asymmetry is behavioural. A red result means ``joint_tilt`` stopped pooling."""
        from trading_crab_lib.platform.allocation.joint_tilt import blend_regime_tilts
        from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt

        unpooled = _hand_built_sub_floor_table()
        asset_returns = _synthetic_asset_returns()
        _assert_sub_floor_precondition(unpooled)

        probs = pd.Series({0: 0.5, 1: 0.3, SUB_FLOOR_REGIME: 0.2})

        # weight_1 = 1.0 makes the blend the classifier-#1-alone path, so pooling is the
        # ONLY remaining difference between the two calls — a one-parameter contrast.
        pooled_result = blend_regime_tilts(
            probs, unpooled, probs, unpooled, asset_returns,
            weight_1=1.0, target_vol_annual=0.10, halflife=6, min_obs=3,
        )
        unpooled_result = vol_targeted_tilt(
            probs, unpooled, asset_returns, target_vol_annual=0.10, halflife=6, min_obs=3,
        )

        pooled_weights = pooled_result["weights"]
        raw_weights = unpooled_result["weights"]
        assert not pooled_weights.empty and pooled_weights.sum() > 0, "pooled path degenerated to all-cash"
        assert not raw_weights.empty and raw_weights.sum() > 0, "unpooled path degenerated to all-cash"

        deltas = {
            asset: float(pooled_weights.get(asset, 0.0) - raw_weights.get(asset, 0.0))
            for asset in sorted(set(pooled_weights.index) | set(raw_weights.index))
        }
        moved = {asset: d for asset, d in deltas.items() if abs(d) > 1e-9}
        assert moved, (
            "joint_tilt's condition-(iv) pooling changed no weight on a fixture with a "
            f"sub-floor regime — G6 would then be a distinction without a difference. Deltas: {deltas}"
        )


# ── the three-row summary, asserted rather than written ──────────────────────

#: (call site, pools?, class, test method). Asserted by ``TestG6Summary`` — renaming an
#: arm without updating this row fails the suite, so the table cannot drift from reality.
G6_CONSUMER_TABLE = (
    ("backtest/driver.py:497", False, "TestDriverConsumer",
     "test_driver_tilt_call_receives_the_unpooled_returns_by_regime"),
    ("report/weekly.py:249", False, "TestWeeklyConsumer",
     "test_weekly_tilt_call_receives_the_unpooled_returns_by_regime"),
    ("allocation/joint_tilt.py:319", True, "TestJointTiltContrast",
     "test_pooled_joint_tilt_produces_different_weights_than_the_unpooled_path"),
)


class TestG6Summary:
    """The record G6 leaves behind: three consumers, two unpooled, one pooled."""

    def test_every_row_of_the_summary_table_resolves_to_a_real_arm(self):
        for call_site, _pools, class_name, method_name in G6_CONSUMER_TABLE:
            cls = globals().get(class_name)
            assert cls is not None, f"{call_site}: summary names {class_name}, which does not exist here"
            assert hasattr(cls, method_name), f"{call_site}: {class_name} has no test {method_name}"

    def test_exactly_two_consumers_are_unpooled_and_one_pools(self):
        pooling = [pools for _site, pools, _cls, _method in G6_CONSUMER_TABLE]
        assert len(G6_CONSUMER_TABLE) == 3
        assert len({site for site, _p, _c, _m in G6_CONSUMER_TABLE}) == 3, "duplicate call site in the summary"
        assert pooling.count(False) == 2, "G6 records two unpooled consumers"
        assert pooling.count(True) == 1, "G6 records exactly one pooling consumer"
