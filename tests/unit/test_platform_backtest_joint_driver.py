"""Unit tests for trading_crab_lib.platform.backtest.joint_driver (criterion 7).

Synthetic-only, no network, no real checkpoints — mirrors
``tests/unit/test_platform_backtest_driver.py``'s convention. ``tmp_path`` is
used for every registry ledger, so no test can touch the live ledger and
inflate D-16's denominator.

The load-bearing assertions here are the ones that would catch a *silently
different window* between the joint leg and the #1-alone baseline — the
failure ``07-11-PLAN.md``'s threat T-07-26 names and the failure
``.planning/UAT-AUDIT-2026-09-09.md`` documents in a different costume:

- the two legs' equity-curve indexes are compared ELEMENT-WISE, never by
  length (two legs can share a length and cover different months);
- ``joint_lift_table`` returns its window in the SAME mapping as the deltas,
  so a caller physically cannot read a delta without its step count and
  endpoints;
- a deliberately mismatched pair of curves produces a WARNING naming BOTH
  index sizes and an ``indexes_identical=False`` flag, rather than a silently
  truncated comparison.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

import trading_crab_lib.platform.backtest.joint_driver as jd
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF
from trading_crab_lib.platform.honesty.registry import NO_REGISTRY, read_trials

N_MONTHS = 72
MIN_TRAIN = 24

C1_COLS = [
    "curve_10y3m",
    "credit_spread_baa_aaa",
    "cape_shiller",
    "div_yield",
    "oil",
    "real_rate_level",
    "realized_vol_1m",
    "realized_vol_3m",
    "trailing_return_1m",
    "trailing_return_3m",
]
C2_COLS = [
    "rs_equities_bonds",
    "rs_oil_equities",
    "equities_tr_mom_12m",
    "long_duration_tr_mom_12m",
    "oil_mom_12m",
]


def _cfg(*, min_train: int = MIN_TRAIN) -> dict:
    """A small platform-config-shaped dict — reduced sizes for fast tests."""
    return {
        "labeling": {"K": 2, "lambda": 5.0, "n_restarts": 2, "embargo_months": 3},
        "labeling_2": {
            "K": 2,
            "lambda": 2.0 * len(C2_COLS),
            "n_restarts": 2,
            "sort_column": "rs_equities_bonds",
            "features": list(C2_COLS),
        },
        "allocation": {
            "target_vol_annual": 0.10,
            "ewma_halflife_months": 6,
            "portfolio_vol_min_obs": 3,
            "blend_weight_1": 0.50,
            "hysteresis": {"act_threshold": 0.70, "unwind_threshold": 0.40},
        },
        "backtest": {
            "cost_bps": 10,
            "min_train_months": min_train,
            "nowcaster_cv_splits": 2,
            "feature_min_history": 6,
        },
    }


def _frames(
    n_months: int = N_MONTHS, start: str = "2010-01-31", seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """(features_1, features_2, asset_returns, cash_returns) on a month-end index."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="ME")
    # A slow drifting regime signal so the jump model finds two real states
    # rather than fitting noise (which would make the tests flaky).
    drift = np.sin(np.linspace(0, 3 * np.pi, n_months))
    features_1 = pd.DataFrame(
        {col: drift + rng.normal(0, 0.3, n_months) for col in C1_COLS}, index=idx
    )
    features_2 = pd.DataFrame(
        {col: -drift + rng.normal(0, 0.3, n_months) for col in C2_COLS}, index=idx
    )
    asset_returns = pd.DataFrame(
        {
            "SPY": rng.normal(0.006, 0.03, n_months),
            "TLT": rng.normal(0.001, 0.02, n_months),
            "GLD": rng.normal(0.002, 0.04, n_months),
        },
        index=idx,
    )
    cash_returns = pd.Series(rng.normal(0.002, 0.0005, n_months), index=idx)
    return features_1, features_2, asset_returns, cash_returns


def _run(blend_weight_1: float, *, tmp_path, tag="t", **kwargs):
    f1, f2, ar, cash = _frames()
    return jd.run_joint_backtest(
        f1,
        ar,
        _cfg(),
        blend_weight_1=blend_weight_1,
        features_2=f2,
        frozen_features_1=C1_COLS,
        frozen_features_2=C2_COLS,
        cash_returns=cash,
        registry_path=(tmp_path / "trials.jsonl") if tmp_path is not None else NO_REGISTRY,
        trial_tag=tag,
        **kwargs,
    )


# ── the two legs share ONE window ───────────────────────────────────────────


class TestOneHarnessTwoLegs:
    def test_both_legs_run_and_produce_records(self, tmp_path):
        joint, _ = _run(0.5, tmp_path=tmp_path, tag="joint")
        base, _ = _run(1.0, tmp_path=tmp_path, tag="base")
        assert not joint.empty
        assert not base.empty

    def test_leg_indexes_are_element_wise_equal(self, tmp_path):
        """Element-wise, NOT by length — the whole point of threat T-07-26.

        Two legs can share a length and still cover different months; a
        length-only assertion would pass on exactly the failure this plan
        exists to prevent.
        """
        joint, _ = _run(0.5, tmp_path=tmp_path, tag="joint")
        base, _ = _run(1.0, tmp_path=tmp_path, tag="base")
        assert joint.index.equals(base.index)
        assert list(joint.index) == list(base.index)

    def test_legs_differ_in_returns_so_the_ablation_is_not_a_no_op(self, tmp_path):
        """A blend weight that never took effect would make criterion 7 vacuous."""
        joint, _ = _run(0.5, tmp_path=tmp_path, tag="joint")
        base, _ = _run(1.0, tmp_path=tmp_path, tag="base")
        assert not np.allclose(joint["return"].to_numpy(), base["return"].to_numpy())

    def test_weight_one_equals_classifier_one_alone(self, tmp_path):
        """blend_weight_1 = 1.0 IS the #1-alone leg: classifier #2 contributes nothing.

        Proven by re-running with a DIFFERENT classifier #2 feature signal and
        asserting the equity curve is unchanged.
        """
        f1, f2, ar, cash = _frames()
        f2_alt = f2 * -3.0 + 1.0
        common = dict(
            cfg=_cfg(),
            blend_weight_1=1.0,
            frozen_features_1=C1_COLS,
            frozen_features_2=C2_COLS,
            cash_returns=cash,
            registry_path=NO_REGISTRY,
            trial_tag="t",
        )
        a, _ = jd.run_joint_backtest(f1, ar, features_2=f2, **common)
        b, _ = jd.run_joint_backtest(f1, ar, features_2=f2_alt, **common)
        pd.testing.assert_series_equal(a["return"], b["return"])

    def test_metadata_records_the_routing(self, tmp_path):
        _, meta = _run(0.5, tmp_path=None, tag="t")
        assert meta["routing"] == jd.ROUTING_L1_ONLY
        assert meta["blend_weight_1"] == 0.5

    def test_l2_routing_is_selectable_and_recorded(self, tmp_path):
        _, meta = _run(0.5, tmp_path=None, tag="t", routing=jd.ROUTING_L2_NOWCAST)
        assert meta["routing"] == jd.ROUTING_L2_NOWCAST

    def test_unknown_routing_raises(self, tmp_path):
        with pytest.raises(ValueError, match="routing"):
            _run(0.5, tmp_path=None, tag="t", routing="telepathy")


# ── holdout ────────────────────────────────────────────────────────────────


class TestHoldout:
    def test_max_visited_date_respects_the_cutoff_constant(self):
        """Cutoff read from honesty/holdout.py, never hardcoded here."""
        f1, f2, ar, cash = _frames(n_months=200, start="2012-01-31")
        assert f1.index.max() > pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF), (
            "fixture must physically extend past the cutoff or this test cannot fail"
        )
        curve, meta = jd.run_joint_backtest(
            f1, ar, _cfg(), blend_weight_1=0.5, features_2=f2,
            frozen_features_1=C1_COLS, frozen_features_2=C2_COLS,
            cash_returns=cash, registry_path=NO_REGISTRY, trial_tag="t",
        )
        assert curve.index.max() <= pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)
        assert pd.Timestamp(meta["last_date"]) <= pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)


# ── registry ───────────────────────────────────────────────────────────────


class TestRegistry:
    def test_exactly_one_row_per_tagged_call(self, tmp_path):
        ledger = tmp_path / "trials.jsonl"
        _run(0.5, tmp_path=tmp_path, tag="joint")
        assert len(read_trials(ledger)) == 1
        _run(1.0, tmp_path=tmp_path, tag="base")
        assert len(read_trials(ledger)) == 2

    def test_exactly_zero_rows_with_the_sentinel(self, tmp_path):
        ledger = tmp_path / "trials.jsonl"
        f1, f2, ar, cash = _frames()
        jd.run_joint_backtest(
            f1, ar, _cfg(), blend_weight_1=0.5, features_2=f2,
            frozen_features_1=C1_COLS, frozen_features_2=C2_COLS,
            cash_returns=cash, registry_path=NO_REGISTRY, trial_tag="t",
        )
        assert len(read_trials(ledger)) == 0
        assert not ledger.exists()

    def test_untagged_call_surfaces_the_underlying_refusal(self, tmp_path):
        f1, f2, ar, cash = _frames()
        with pytest.raises(ValueError, match="trial_tag"):
            jd.run_joint_backtest(
                f1, ar, _cfg(), blend_weight_1=0.5, features_2=f2,
                frozen_features_1=C1_COLS, frozen_features_2=C2_COLS,
                cash_returns=cash, registry_path=tmp_path / "trials.jsonl",
                trial_tag=None,
            )

    def test_registry_row_carries_blend_weight_routing_and_both_feature_lists(self, tmp_path):
        ledger = tmp_path / "trials.jsonl"
        _run(0.5, tmp_path=tmp_path, tag="joint")
        row = read_trials(ledger).iloc[0]
        assert row["config"]["blend_weight_1"] == 0.5
        assert row["config"]["routing"] == jd.ROUTING_L1_ONLY
        assert row["config"]["trial_tag"] == "joint"
        assert row["config"]["features_1"] == C1_COLS
        assert row["config"]["features_2"] == C2_COLS
        assert row["config"]["K_1"] == 2
        assert row["config"]["K_2"] == 2
        assert "n_steps" in row["metrics"]
        assert "terminal_log_wealth" in row["metrics"]


# ── degradation ────────────────────────────────────────────────────────────


class TestDegradation:
    def test_a_failing_refit_degrades_rather_than_crashing(self, tmp_path, monkeypatch):
        calls = {"n": 0}
        real = jd._refit_classifier2

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise ValueError("synthetic degenerate fit")
            return real(*args, **kwargs)

        monkeypatch.setattr(jd, "_refit_classifier2", flaky)
        curve, meta = _run(0.5, tmp_path=None, tag="t")
        assert meta["n_degraded"] > 0
        assert meta["n_degraded_classifier_2"] > 0
        assert bool(curve["degraded"].any())
        # Degraded steps are still RECORDED (weights held) — the index must not
        # silently shrink, or the two legs would cover different months.
        assert len(curve) == meta["n_steps"]

    def test_degraded_steps_are_excluded_from_per_step_metrics(self, tmp_path, monkeypatch):
        real = jd._refit_classifier2
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] % 3 == 0:
                raise ValueError("synthetic degenerate fit")
            return real(*args, **kwargs)

        monkeypatch.setattr(jd, "_refit_classifier2", flaky)
        curve, meta = _run(0.5, tmp_path=None, tag="t")
        assert len(meta["per_step_metrics_1"]["dates"]) == meta["n_steps"] - meta["n_degraded"]


# ── joint_lift_table ───────────────────────────────────────────────────────


class TestJointLiftTable:
    @staticmethod
    def _curve(values, start="2000-01-31"):
        idx = pd.date_range(start, periods=len(values), freq="ME")
        return pd.DataFrame({"return": values}, index=idx)

    def test_returns_both_axes_with_the_window_in_the_same_mapping(self):
        joint = self._curve([0.02] * 10)
        base = self._curve([0.01] * 10)
        out = jd.joint_lift_table(joint, base)
        for key in (
            "wealth_delta",
            "dd_delta",
            "n_steps",
            "first_date",
            "last_date",
            "indexes_identical",
        ):
            assert key in out, f"missing {key} — a delta without its window is REJECTED"
        assert out["n_steps"] == 10
        assert pd.Timestamp(out["first_date"]) == joint.index[0]
        assert pd.Timestamp(out["last_date"]) == joint.index[-1]

    def test_wealth_delta_is_joint_minus_baseline_in_nats(self):
        joint = self._curve([0.10] * 4)
        base = self._curve([0.0] * 4)
        out = jd.joint_lift_table(joint, base)
        assert out["wealth_delta"] == pytest.approx(4 * np.log1p(0.10))
        assert out["baseline_terminal_log_wealth"] == pytest.approx(0.0)

    def test_dd_delta_is_joint_minus_baseline_drawdown(self):
        # A peak must exist BEFORE the drop: max_drawdown_and_duration measures
        # against the running peak, and the first observation IS its own peak.
        joint = self._curve([0.0, -0.20, 0.0, 0.0])
        base = self._curve([0.0, -0.10, 0.0, 0.0])
        out = jd.joint_lift_table(joint, base)
        assert out["dd_delta"] == pytest.approx(-0.20 - (-0.10))

    def test_mismatched_indexes_warn_with_both_sizes_and_flag_it(self, caplog):
        joint = self._curve([0.01] * 10)
        base = self._curve([0.01] * 7)
        with caplog.at_level(logging.WARNING, logger=jd.log.name):
            out = jd.joint_lift_table(joint, base)
        assert out["indexes_identical"] is False
        text = caplog.text
        assert "10" in text and "7" in text, (
            "the WARNING must name BOTH index sizes — a silently truncated "
            "comparison is REJECTED"
        )
        # The comparison is made over the intersection and says so.
        assert out["n_steps"] == 7
        assert out["n_steps_joint"] == 10
        assert out["n_steps_baseline"] == 7

    def test_identical_indexes_do_not_warn(self, caplog):
        joint = self._curve([0.01] * 10)
        base = self._curve([0.02] * 10)
        with caplog.at_level(logging.WARNING, logger=jd.log.name):
            out = jd.joint_lift_table(joint, base)
        assert out["indexes_identical"] is True
        assert "index" not in caplog.text.lower()

    def test_empty_intersection_reports_zero_steps_and_no_bare_delta(self):
        joint = self._curve([0.01] * 5, start="2000-01-31")
        base = self._curve([0.01] * 5, start="2010-01-31")
        out = jd.joint_lift_table(joint, base)
        assert out["n_steps"] == 0
        assert out["first_date"] is None
        assert out["last_date"] is None

    def test_band_verdicts_are_reported_against_the_governing_tier(self):
        """07-BANDS.md §8: universal tier governs; domain tier records a note."""
        joint = self._curve([0.02] * 10)
        base = self._curve([0.01] * 10)
        out = jd.joint_lift_table(joint, base)
        assert out["wealth_delta_universal_ok"] is True
        assert out["dd_delta_universal_ok"] is True
        assert out["wealth_delta_domain_note"] is False
        assert out["dd_delta_domain_note"] is False

    def test_universal_band_breach_is_detected(self):
        """A broken measurement must be flagged, not reported as a lift."""
        joint = self._curve([5.0] * 20)  # e^20-ish terminal log wealth: impossible
        base = self._curve([0.0] * 20)
        out = jd.joint_lift_table(joint, base)
        assert out["wealth_delta_universal_ok"] is False
        assert out["wealth_delta_domain_note"] is True

    def test_dd_delta_universal_band_is_the_revised_minus_one_to_one(self):
        assert jd.DD_DELTA_UNIVERSAL == (-1.0, 1.0), (
            "07-BANDS.md §8 band 2 revised [-2, 2] -> [-1, 1]; the old bound was "
            "wider than the quantity's own arithmetic range and could only confirm"
        )
        assert jd.WEALTH_DELTA_UNIVERSAL == 15.0
        assert jd.WEALTH_DELTA_DOMAIN == 5.0
        assert jd.DD_DELTA_DOMAIN == 0.5


# ── the classifier-#2 refit has not forked driver.py's frozen-features policy ──


class TestRefitParity:
    def test_refit_classifier2_reproduces_refit_l1_on_classifier_ones_inputs(self):
        """joint_driver's local refit is _refit_l1 plus a sort_column seam.

        ``driver.py::_refit_l1`` hardcodes ``canonicalize_states``' default
        ``sort_column='trailing_return_1m'``, which classifier #2's disjoint
        feature set does not contain, so a second refit entry point is
        unavoidable. This test pins that it is not a FORK: given classifier
        #1's own frozen columns, config and sort column, it returns exactly
        what ``_refit_l1`` returns.
        """
        f1, _f2, _ar, _cash = _frames()
        train = f1.iloc[:40]
        cfg = _cfg()
        expected = jd._refit_l1(train, cfg, frozen_features=C1_COLS)
        got = jd._refit_classifier2(
            train,
            frozen_features=C1_COLS,
            K=cfg["labeling"]["K"],
            lam=cfg["labeling"]["lambda"],
            n_restarts=cfg["labeling"]["n_restarts"],
            sort_column="trailing_return_1m",
        )
        pd.testing.assert_series_equal(expected, got)

    def test_refit_classifier2_raises_on_an_absent_sort_column(self):
        f1, f2, _ar, _cash = _frames()
        with pytest.raises(ValueError):
            jd._refit_classifier2(
                f2.iloc[:40],
                frozen_features=C2_COLS,
                K=2,
                lam=10.0,
                n_restarts=2,
                sort_column="not_a_column",
            )

    def test_refit_classifier2_raises_on_an_empty_frozen_list(self):
        f1, _f2, _ar, _cash = _frames()
        with pytest.raises(ValueError, match="0 usable"):
            jd._refit_classifier2(
                f1.iloc[:40],
                frozen_features=["nope"],
                K=2,
                lam=10.0,
                n_restarts=2,
                sort_column="nope",
            )


# ── input validation ───────────────────────────────────────────────────────


class TestInputValidation:
    def test_misaligned_feature_frames_raise(self, tmp_path):
        f1, f2, ar, cash = _frames()
        with pytest.raises(ValueError, match="index"):
            jd.run_joint_backtest(
                f1, ar, _cfg(), blend_weight_1=0.5, features_2=f2.iloc[3:],
                frozen_features_1=C1_COLS, frozen_features_2=C2_COLS,
                cash_returns=cash, registry_path=NO_REGISTRY, trial_tag="t",
            )

    def test_blend_weight_outside_the_unit_interval_raises(self, tmp_path):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            _run(1.5, tmp_path=None, tag="t")


# ── the Bayes filter (plan 08-08): l2 only, belief carried, l1only pinned bit-for-bit ──
#
# The default _frames() fixture degrades EVERY l2 step (a single class survives the
# embargo), so it cannot exercise the filter. This fixture is a square wave with six
# cycles over 96 months: under min_train=36 it yields 60 steps, 8 early degraded, and
# the rest carry real two-class posteriors.

L2_N_MONTHS = 96
L2_MIN_TRAIN = 36


def _l2_frames(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = L2_N_MONTHS
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    drift = np.sign(np.sin(np.linspace(0, 6 * 2 * np.pi, n)))
    features_1 = pd.DataFrame({c: drift + rng.normal(0, 0.5, n) for c in C1_COLS}, index=idx)
    features_2 = pd.DataFrame({c: -drift + rng.normal(0, 0.5, n) for c in C2_COLS}, index=idx)
    asset_returns = pd.DataFrame(
        {
            "SPY": rng.normal(0.006, 0.03, n) + 0.01 * drift,
            "TLT": rng.normal(0.001, 0.02, n) - 0.005 * drift,
            "GLD": rng.normal(0.002, 0.04, n),
        },
        index=idx,
    )
    cash_returns = pd.Series(rng.normal(0.002, 0.0005, n), index=idx)
    return features_1, features_2, asset_returns, cash_returns


def _l2_run(*, routing: str, use_regime_filter: bool):
    f1, f2, ar, cash = _l2_frames()
    return jd.run_joint_backtest(
        f1, ar, _cfg(min_train=L2_MIN_TRAIN), blend_weight_1=0.5, features_2=f2,
        frozen_features_1=C1_COLS, frozen_features_2=C2_COLS, cash_returns=cash,
        registry_path=NO_REGISTRY, trial_tag="t", routing=routing, use_regime_filter=use_regime_filter,
    )


class _Spy:
    """Wrap a module attribute, record every call's args and return value."""

    def __init__(self, monkeypatch, module, name):
        self.calls: list[tuple[tuple, dict, object]] = []
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            out = real(*args, **kwargs)
            self.calls.append((args, kwargs, out))
            return out

        monkeypatch.setattr(module, name, wrapper)


def _first_diff_date(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]):
    for date in a.index:
        if not a.loc[date, cols].equals(b.loc[date, cols]):
            return date
    return None


@pytest.fixture(scope="module")
def l1only_runs():
    on = _l2_run(routing=jd.ROUTING_L1_ONLY, use_regime_filter=True)
    off = _l2_run(routing=jd.ROUTING_L1_ONLY, use_regime_filter=False)
    return on, off


@pytest.fixture(scope="module")
def l2_off():
    return _l2_run(routing=jd.ROUTING_L2_NOWCAST, use_regime_filter=False)


@pytest.fixture(scope="module")
def l2_on_spied():
    """One filter-on l2 run with spies on the tilt, the hysteresis, the filter and the cold start."""
    mp = pytest.MonkeyPatch()
    try:
        tilt = _Spy(mp, jd, "blend_regime_tilts")
        hyst = _Spy(mp, jd, "update_active_regime")
        filt = _Spy(mp, jd, "filter_step")
        cold = _Spy(mp, jd, "unconditional_belief")
        curve, meta = _l2_run(routing=jd.ROUTING_L2_NOWCAST, use_regime_filter=True)
    finally:
        mp.undo()
    return curve, meta, tilt, hyst, filt, cold


class TestRegimeFilterWiring:
    def test_l1only_curve_is_bit_identical_with_the_filter_on_and_off(self, l1only_runs):
        """The decision-bearing pin. Fails the moment the filter leaks into ROUTING_L1_ONLY."""
        (on, meta_on), (off, _meta_off) = l1only_runs
        pd.testing.assert_frame_equal(on, off, check_exact=True)
        for col in ("active_regime", "scale", "turnover", "return"):
            assert on[col].equals(off[col]), col
        # ...and the filter never ran on that routing.
        assert meta_on["use_regime_filter"] is False
        assert meta_on["per_step_belief_1"]["dates"] == [] and meta_on["per_step_belief_2"]["dates"] == []

    def test_l2_curve_differs_with_the_filter_on(self, l2_on_spied, l2_off):
        """Without this, a filter that silently no-ops would pass the l1only pin above."""
        on, meta_on = l2_on_spied[0], l2_on_spied[1]
        off, _ = l2_off
        assert on.index.equals(off.index)
        cols = ["return", "turnover", "scale", "active_regime"]
        first = _first_diff_date(on, off, cols)
        assert first is not None, "the l2 curve is identical with the filter on and off — the filter no-ops"
        assert not on["return"].equals(off["return"]), f"returns identical; first differing date {first}"
        assert meta_on["use_regime_filter"] is True
        # The degraded set is unchanged by the filter: no feature column was added.
        assert on["degraded"].equals(off["degraded"])

    def test_the_tilt_and_the_hysteresis_receive_the_belief_not_the_raw_posterior(self, l2_on_spied):
        _curve, meta, tilt, hyst, _filt, _cold = l2_on_spied
        beliefs_1 = meta["per_step_belief_1"]["proba"]
        beliefs_2 = meta["per_step_belief_2"]["proba"]
        raws_1 = meta["per_step_metrics_1"]["proba"]
        assert len(tilt.calls) == len(beliefs_1) == len(raws_1) == len(hyst.calls) > 0
        n_differ = 0
        for (args, _kw, _out), (h_args, _hkw, _hout), b1, b2, r1 in zip(
            tilt.calls, hyst.calls, beliefs_1, beliefs_2, raws_1
        ):
            np.testing.assert_array_equal(args[0].to_numpy(), b1)   # tilt's probs_1 IS the belief
            np.testing.assert_array_equal(args[2].to_numpy(), b2)   # tilt's probs_2 IS the belief
            np.testing.assert_array_equal(h_args[0].to_numpy(), b1)  # hysteresis sees the belief too
            raw = pd.Series(r1, index=meta["per_step_metrics_1"]["classes"][0]).reindex(args[0].index, fill_value=0.0)
            if float(np.max(np.abs(raw.to_numpy() - b1))) > 1e-6:
                n_differ += 1
        assert n_differ >= 1, "belief never differed from the raw posterior by more than 1e-6"

    def test_cold_start_is_unconditional_belief_on_the_steps_own_labels(self, l2_on_spied):
        """The first non-degraded step's prior is unconditional_belief(states_1) exactly —
        not uniform 1/K, not a dropped row."""
        _curve, _meta, _tilt, _hyst, filt, cold = l2_on_spied
        (f_args, _fkw, _fout) = filt.calls[0]
        start, _a, _posterior, class_prior = f_args
        (c_args, c_kw, c_out) = cold.calls[0]
        pd.testing.assert_series_equal(start, class_prior)
        pd.testing.assert_series_equal(start, c_out)
        from trading_crab_lib.platform.prediction.regime_filter import unconditional_belief as ub
        pd.testing.assert_series_equal(start, ub(c_args[0], **c_kw))
        k = len(start)
        assert not np.allclose(start.to_numpy(), 1.0 / k), "cold start is uniform; the fixture cannot discriminate"
        # The next classifier-#1 filter call starts from the carried belief, not the prior again.
        (f2_args, _k2, _o2) = filt.calls[2]
        pd.testing.assert_series_equal(f2_args[0], filt.calls[0][2])

    def test_a_degraded_step_advances_the_belief_by_predict_only_not_a_hold(self, monkeypatch):
        """Force classifier #1's L2 to degrade at one mid-run step k. The belief that
        enters step k+1 must be predict_only_step(belief_{k-1}, A_k) and must differ from
        belief_{k-1}. Fails if the missing-observation rule was implemented as a hold."""
        real_l2 = jd._refit_l2
        n = {"calls": 0, "forced": None}
        # Classifier #1 is the odd call on each non-degraded step; the fixture's first
        # 8 steps degrade in classifier #1's L2 already, so count successes.
        forced_after = 6

        def l2(train, states, row, cfg):
            n["calls"] += 1
            out = real_l2(train, states, row, cfg)  # raises on the fixture's own early degrades
            if n["forced"] is None and n["calls"] > 8 and (n["calls"] - 8) == 2 * forced_after + 1:
                n["forced"] = row.index[0]
                raise ValueError("forced L2 degrade for the missing-observation test")
            return out

        monkeypatch.setattr(jd, "_refit_l2", l2)
        filt = _Spy(monkeypatch, jd, "filter_step")
        pred = _Spy(monkeypatch, jd, "predict_only_step")
        tm = _Spy(monkeypatch, jd, "transition_matrix_for")
        curve, meta = _l2_run(routing=jd.ROUTING_L2_NOWCAST, use_regime_filter=True)

        forced = n["forced"]
        assert forced is not None and bool(curve.loc[forced, "degraded"]) is True
        assert len(pred.calls) >= 2, "no predict-only step on the forced degrade"
        (p_args, _pkw, p_out) = pred.calls[0]  # classifier #1, forced step
        belief_before, a_k = p_args
        # belief_{k-1}: the last classifier-#1 filter output before the forced step.
        dates = meta["per_step_belief_1"]["dates"]
        last_before = max(i for i, d in enumerate(dates) if d < forced)
        np.testing.assert_array_equal(belief_before.to_numpy(), meta["per_step_belief_1"]["proba"][last_before])
        # A_k is this step's own transition matrix (the last one built before predict-only).
        assert any(out is a_k for _a, _k, out in tm.calls)
        from trading_crab_lib.platform.prediction.regime_filter import predict_only_step as pos
        pd.testing.assert_series_equal(p_out, pos(belief_before, a_k))
        assert not np.allclose(p_out.to_numpy(), belief_before.to_numpy()), "predict-only equals a hold here"
        # ...and it is what the NEXT classifier-#1 filter step starts from.
        next_c1 = [args for args, _kw, _o in filt.calls if args[0] is p_out]
        assert len(next_c1) == 1


def test_filter_is_gated_on_the_l2_routing_in_code_not_by_data():
    import re
    from pathlib import Path

    src = Path(jd.__file__).read_text()
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert re.search(r"if\s+routing\s*==\s*ROUTING_L2_NOWCAST[^:]*use_regime_filter", code)
