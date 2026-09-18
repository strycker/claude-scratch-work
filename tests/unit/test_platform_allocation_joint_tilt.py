"""
Tests for the joint (classifier #1 + classifier #2) allocation blend (D-14, plan 07-10).

Two groups of invariants:

1. **Contract preservation.** ``blend_regime_tilts`` returns exactly the four
   keys ``vol_targeted_tilt`` returns, its weights are long-only and sum to
   ``scale``, and at ``weight_1`` of 1.0 / 0.0 it reduces *element-wise* to the
   single-classifier path — so the new code composes with the tested one
   rather than diverging from it.

2. **ADR-0001 recurrence-exemption condition (iv).** A regime below design
   §4.4 criterion 1's ~8% occupancy floor must be flagged, and its per-regime
   Sharpe must be partially pooled toward the all-history estimate (design §6.1
   mitigation 2) before it can move a single portfolio weight. The pooling
   tests carry hand-derived oracles computed from raw returns, so deleting the
   shrinkage makes them fail rather than merely un-flag something.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.allocation.joint_tilt import (
    DEFAULT_BLEND_WEIGHT_1,
    OCCUPANCY_FLOOR,
    blend_regime_tilts,
    blend_weight_from_config,
    low_n_regime_flags,
    pool_low_n_regime_sharpe,
    regime_occupancy,
)
from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt
from trading_crab_lib.platform.assets.returns import returns_by_regime_stats

# ── shared fixtures ─────────────────────────────────────────────────────────


def _asset_returns(n_months: int = 36, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    return pd.DataFrame(
        {
            "SPY": rng.normal(0.01, 0.04, n_months),
            "TLT": rng.normal(0.00, 0.02, n_months),
            "GLD": rng.normal(0.005, 0.03, n_months),
        },
        index=idx,
    )


def _rbr_1() -> pd.DataFrame:
    """Classifier #1's table — two balanced regimes, both above the floor."""
    return pd.DataFrame(
        [
            {"regime": 0, "asset": "SPY", "sharpe_annualized": 1.5, "n_obs": 60},
            {"regime": 0, "asset": "TLT", "sharpe_annualized": 0.5, "n_obs": 60},
            {"regime": 1, "asset": "SPY", "sharpe_annualized": -0.8, "n_obs": 60},
            {"regime": 1, "asset": "TLT", "sharpe_annualized": 1.2, "n_obs": 60},
        ]
    )


def _rbr_2() -> pd.DataFrame:
    """Classifier #2's table — a different asset set, both regimes above the floor."""
    return pd.DataFrame(
        [
            {"regime": 0, "asset": "TLT", "sharpe_annualized": 1.0, "n_obs": 60},
            {"regime": 0, "asset": "GLD", "sharpe_annualized": 3.0, "n_obs": 60},
            {"regime": 1, "asset": "TLT", "sharpe_annualized": 2.0, "n_obs": 60},
            {"regime": 1, "asset": "GLD", "sharpe_annualized": 0.5, "n_obs": 60},
        ]
    )


def _blend(probs_1, probs_2, *, weight_1: float, **kwargs) -> dict:
    return blend_regime_tilts(
        probs_1,
        _rbr_1(),
        probs_2,
        _rbr_2(),
        _asset_returns(),
        weight_1=weight_1,
        target_vol_annual=0.10,
        halflife=6,
        min_obs=12,
        **kwargs,
    )


# ── output contract: the same four keys, long-only, summing to scale ────────


class TestOutputContract:
    def test_key_set_equals_vol_targeted_tilts_key_set(self):
        """Compared against a LIVE vol_targeted_tilt call, never a literal list —
        a hardcoded four-element list cannot detect drift in the function it
        is supposed to track."""
        single = vol_targeted_tilt(
            {0: 1.0}, _rbr_1(), _asset_returns(), target_vol_annual=0.10, halflife=6, min_obs=12
        )
        joint = _blend({0: 1.0}, {0: 1.0}, weight_1=0.5)

        assert set(joint.keys()) == set(single.keys())

    def test_weights_nonnegative_and_sum_exactly_to_scale(self):
        result = _blend({0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7}, weight_1=0.5)

        assert result["weights"].min() >= 0.0
        assert abs(result["weights"].sum() - result["scale"]) < 1e-12
        assert abs(result["cash"] - (1.0 - result["scale"])) < 1e-12

    def test_long_only_holds_when_an_input_tilt_would_be_negative_before_clipping(self):
        """Classifier #1's regime 1 carries a -0.8 SPY Sharpe. The blend must
        never let that become a short position."""
        result = _blend({1: 1.0}, {1: 1.0}, weight_1=0.5)

        assert result["weights"].min() >= 0.0
        assert result["weights"].get("SPY", 0.0) == 0.0

    def test_scale_never_exceeds_one(self):
        result = _blend({0: 1.0}, {0: 1.0}, weight_1=0.5, )

        assert result["scale"] <= 1.0

    def test_deterministic(self):
        first = _blend({0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7}, weight_1=0.5)
        second = _blend({0: 0.6, 1: 0.4}, {0: 0.3, 1: 0.7}, weight_1=0.5)

        pd.testing.assert_series_equal(first["weights"], second["weights"])
        assert first["scale"] == second["scale"]


# ── endpoint reduction: the blend composes with the tested single path ──────


class TestEndpointReduction:
    def test_weight_one_reduces_exactly_to_classifier_1_alone(self):
        probs_1 = {0: 0.6, 1: 0.4}
        single = vol_targeted_tilt(
            probs_1, _rbr_1(), _asset_returns(), target_vol_annual=0.10, halflife=6, min_obs=12
        )
        joint = _blend(probs_1, {0: 0.3, 1: 0.7}, weight_1=1.0)

        pd.testing.assert_series_equal(joint["weights"], single["weights"])
        assert joint["scale"] == single["scale"]
        assert joint["cash"] == single["cash"]
        assert joint["portfolio_vol"] == single["portfolio_vol"]

    def test_weight_zero_reduces_exactly_to_classifier_2_alone(self):
        probs_2 = {0: 0.3, 1: 0.7}
        single = vol_targeted_tilt(
            probs_2, _rbr_2(), _asset_returns(), target_vol_annual=0.10, halflife=6, min_obs=12
        )
        joint = _blend({0: 0.6, 1: 0.4}, probs_2, weight_1=0.0)

        pd.testing.assert_series_equal(joint["weights"], single["weights"])
        assert joint["scale"] == single["scale"]
        assert joint["cash"] == single["cash"]
        assert joint["portfolio_vol"] == single["portfolio_vol"]


# ── the blend itself, against a hand-computed value ─────────────────────────


class TestHandComputedBlend:
    def test_half_blend_matches_hand_computed_pre_scaling_weight(self):
        """Classifier #1's regime 0 tilt: SPY 1.5, TLT 0.5 -> SPY 0.75, TLT 0.25.
        Classifier #2's regime 0 tilt: TLT 1.0, GLD 3.0 -> TLT 0.25, GLD 0.75.

        SPY is favoured by #1 only, so its blended pre-scaling weight is
        0.5 * 0.75 / 1.0 = 0.375; TLT is 0.5*0.25 + 0.5*0.25 = 0.25; GLD is
        0.5 * 0.75 = 0.375.
        """
        result = _blend({0: 1.0}, {0: 1.0}, weight_1=0.5)
        scale = result["scale"]

        assert result["weights"]["SPY"] == pytest.approx(0.375 * scale)
        assert result["weights"]["TLT"] == pytest.approx(0.250 * scale)
        assert result["weights"]["GLD"] == pytest.approx(0.375 * scale)

    def test_asymmetric_weight_shifts_the_blend_by_exactly_that_weight(self):
        """weight_1 = 0.8: SPY -> 0.8*0.75 = 0.60, GLD -> 0.2*0.75 = 0.15,
        TLT -> 0.8*0.25 + 0.2*0.25 = 0.25."""
        result = _blend({0: 1.0}, {0: 1.0}, weight_1=0.8)
        scale = result["scale"]

        assert result["weights"]["SPY"] == pytest.approx(0.60 * scale)
        assert result["weights"]["TLT"] == pytest.approx(0.25 * scale)
        assert result["weights"]["GLD"] == pytest.approx(0.15 * scale)

    def test_never_forms_a_product_state_space(self):
        """D-14: the two labelings' states are never crossed. Every returned
        index entry is an asset name from the input universe — no (s1, s2)
        pair, tuple key, or crossed label appears anywhere."""
        result = _blend({0: 0.5, 1: 0.5}, {0: 0.5, 1: 0.5}, weight_1=0.5)

        assert set(result["weights"].index) <= {"SPY", "TLT", "GLD"}
        assert all(isinstance(name, str) for name in result["weights"].index)


# ── degenerate branches, matching vol_targeted_tilt's own edges ─────────────


class TestDegenerateBranches:
    def test_both_inputs_empty_gives_all_cash_with_nan_portfolio_vol(self):
        result = _blend({}, {}, weight_1=0.5)

        assert result["weights"].empty
        assert result["cash"] == 1.0
        assert result["scale"] == 0.0
        assert np.isnan(result["portfolio_vol"])

    def test_empty_matches_vol_targeted_tilts_own_degenerate_branch(self):
        single = vol_targeted_tilt(
            {}, _rbr_1(), _asset_returns(), target_vol_annual=0.10, halflife=6, min_obs=12
        )
        joint = _blend({}, {}, weight_1=0.5)

        assert set(joint) == set(single)
        assert joint["cash"] == single["cash"]
        assert joint["scale"] == single["scale"]
        assert np.isnan(joint["portfolio_vol"]) and np.isnan(single["portfolio_vol"])

    def test_one_input_empty_degrades_to_the_other_classifiers_tilt_not_to_cash(self):
        probs_2 = {0: 1.0}
        single = vol_targeted_tilt(
            probs_2, _rbr_2(), _asset_returns(), target_vol_annual=0.10, halflife=6, min_obs=12
        )
        joint = _blend({}, probs_2, weight_1=0.5)

        assert joint["cash"] < 1.0
        pd.testing.assert_series_equal(joint["weights"], single["weights"])

    def test_missing_leg_still_degrades_when_its_blend_weight_is_the_whole_one(self):
        """weight_1 = 1.0 with classifier #1 absent must still use classifier
        #2 rather than silently returning all-cash."""
        joint = _blend({}, {0: 1.0}, weight_1=1.0)

        assert not joint["weights"].empty
        assert joint["cash"] < 1.0


# ── weight_1 validation: outside [0, 1] is a long-only violation ────────────


class TestBlendWeightValidation:
    @pytest.mark.parametrize("bad", [1.5, -0.5, 1.0000001])
    def test_weight_outside_the_unit_interval_raises_value_error(self, bad):
        with pytest.raises(ValueError, match="weight_1"):
            _blend({0: 1.0}, {0: 1.0}, weight_1=bad)

    def test_nan_weight_raises_rather_than_propagating(self):
        with pytest.raises(ValueError, match="weight_1"):
            _blend({0: 1.0}, {0: 1.0}, weight_1=float("nan"))

    @pytest.mark.parametrize("ok", [0.0, 0.5, 1.0])
    def test_endpoints_of_the_unit_interval_are_accepted(self, ok):
        assert _blend({0: 1.0}, {0: 1.0}, weight_1=ok)["scale"] >= 0.0


# ── blend_weight_from_config: pre-declared, defensive, never swept ──────────


class TestBlendWeightFromConfig:
    def test_reads_the_configured_value(self):
        assert blend_weight_from_config({"allocation": {"blend_weight_1": 0.3}}) == 0.3

    def test_missing_key_falls_back_to_the_adr_pinned_value_with_a_warning(self, caplog):
        with caplog.at_level("WARNING"):
            value = blend_weight_from_config({})

        assert value == DEFAULT_BLEND_WEIGHT_1
        assert "blend_weight_1" in caplog.text

    def test_adr_pinned_default_is_the_equal_weight_no_information_prior(self):
        assert DEFAULT_BLEND_WEIGHT_1 == 0.50

    def test_out_of_range_configured_value_raises(self):
        with pytest.raises(ValueError, match="blend_weight_1"):
            blend_weight_from_config({"allocation": {"blend_weight_1": 2.0}})


# ── ADR-0001 condition (iv): low-n flagging ────────────────────────────────


def _crisis_table() -> pd.DataFrame:
    """Regime 0 occupies 40 of 695 months (5.7554%) — below §4.4's ~8% floor,
    matching classifier #1's exempted crisis state. Regime 1 is above it."""
    return pd.DataFrame(
        [
            {"regime": 0, "asset": "SPY", "mean_monthly_return": -0.02, "std_monthly_return": 0.08,
             "sharpe_annualized": -0.8660254037844387, "n_obs": 40},
            {"regime": 0, "asset": "TLT", "mean_monthly_return": 0.01, "std_monthly_return": 0.02,
             "sharpe_annualized": 1.7320508075688772, "n_obs": 40},
            {"regime": 1, "asset": "SPY", "mean_monthly_return": 0.01, "std_monthly_return": 0.04,
             "sharpe_annualized": 0.8660254037844386, "n_obs": 655},
            {"regime": 1, "asset": "TLT", "mean_monthly_return": 0.002, "std_monthly_return": 0.02,
             "sharpe_annualized": 0.34641016151377546, "n_obs": 655},
        ]
    )


class TestLowNFlagging:
    def test_occupancy_recovers_the_sub_floor_share(self):
        occ = regime_occupancy(_crisis_table())

        assert occ[0] == pytest.approx(40 / 695)
        assert occ[1] == pytest.approx(655 / 695)
        assert occ.sum() == pytest.approx(1.0)

    def test_sub_floor_regime_is_flagged_and_above_floor_regime_is_not(self):
        flags = low_n_regime_flags(_crisis_table())

        assert bool(flags.loc[flags["regime"] == 0, "low_n"].iloc[0]) is True
        assert bool(flags.loc[flags["regime"] == 1, "low_n"].iloc[0]) is False

    def test_flag_is_visible_to_a_caller_who_never_read_the_adr(self, caplog):
        """Condition (iv) requires an EXPLICIT low-n flag on every downstream
        statistic. A blend over a sub-floor regime logs at WARNING naming the
        regime, its occupancy, the floor and the credibility applied."""
        with caplog.at_level("WARNING"):
            blend_regime_tilts(
                {0: 1.0}, _crisis_table(), {0: 1.0}, _crisis_table(), _asset_returns(),
                weight_1=0.5, target_vol_annual=0.10, halflife=6, min_obs=12,
            )

        assert "low-n" in caplog.text.lower()
        assert "4.4" in caplog.text
        assert "0.0575" in caplog.text or "5.75" in caplog.text

    def test_credibility_is_the_occupancy_share_of_the_floor_capped_at_one(self):
        flags = low_n_regime_flags(_crisis_table()).set_index("regime")

        assert flags.loc[0, "credibility"] == pytest.approx((40 / 695) / OCCUPANCY_FLOOR)
        assert flags.loc[1, "credibility"] == 1.0

    def test_floor_matches_design_4_4_criterion_1(self):
        assert OCCUPANCY_FLOOR == 0.08


# ── ADR-0001 condition (iv): partial pooling toward the all-history model ───


class TestPartialPooling:
    def _labelled_returns(self, seed: int = 7):
        """A real synthetic monthly series with a thin crisis state, so the
        all-history oracle can be computed from the RAW returns rather than
        from the function under test."""
        rng = np.random.default_rng(seed)
        n = 400
        idx = pd.date_range("1980-01-31", periods=n, freq="ME")
        returns = pd.DataFrame(
            {"SPY": rng.normal(0.01, 0.04, n), "TLT": rng.normal(0.002, 0.02, n)},
            index=idx,
        )
        states = pd.Series(1, index=idx, name="state")
        # 20 crisis months (5% — below the ~8% floor), in four separated episodes.
        for start in (30, 130, 250, 340):
            states.iloc[start:start + 5] = 0
        returns.loc[states == 0, "SPY"] -= 0.12
        return returns, states

    def test_pooled_target_equals_the_actual_all_history_sharpe(self):
        """The pooled estimate is reconstructed from the per-regime sufficient
        statistics. It must equal the Sharpe computed on the FULL history —
        an oracle derived from raw returns, independent of the implementation."""
        returns, states = self._labelled_returns()
        table = returns_by_regime_stats(returns, states)

        pooled, _ = pool_low_n_regime_sharpe(table, return_pooled=True)

        for asset in ("SPY", "TLT"):
            col = returns[asset].dropna()
            expected = (col.mean() / col.std()) * np.sqrt(12)
            assert pooled[asset] == pytest.approx(expected, rel=1e-9)

    def test_sub_floor_sharpe_is_shrunk_toward_the_all_history_estimate(self):
        """The hand-derived oracle: shrunk = c * raw + (1 - c) * all-history,
        with c = occupancy / floor. Delete the shrinkage and `shrunk` equals
        `raw`, which this assertion rejects."""
        returns, states = self._labelled_returns()
        table = returns_by_regime_stats(returns, states)
        occupancy = float((states == 0).mean())
        credibility = occupancy / OCCUPANCY_FLOOR

        shrunk, flags = pool_low_n_regime_sharpe(table)

        for asset in ("SPY", "TLT"):
            raw = float(table[(table["regime"] == 0) & (table["asset"] == asset)]["sharpe_annualized"].iloc[0])
            col = returns[asset].dropna()
            all_history = (col.mean() / col.std()) * np.sqrt(12)
            expected = credibility * raw + (1 - credibility) * all_history
            got = float(shrunk[(shrunk["regime"] == 0) & (shrunk["asset"] == asset)]["sharpe_annualized"].iloc[0])

            assert got == pytest.approx(expected, rel=1e-9)
            assert got != pytest.approx(raw, rel=1e-6)
        assert bool(flags.loc[flags["regime"] == 0, "low_n"].iloc[0]) is True

    def test_above_floor_regime_is_left_untouched(self):
        """Condition (iv) binds on the exempted sub-floor state only. Pooling
        every regime would be a silent change to the tested single-classifier
        path, not an implementation of the exemption."""
        returns, states = self._labelled_returns()
        table = returns_by_regime_stats(returns, states)

        shrunk, _ = pool_low_n_regime_sharpe(table)

        above = shrunk[shrunk["regime"] == 1].set_index("asset")["sharpe_annualized"]
        original = table[table["regime"] == 1].set_index("asset")["sharpe_annualized"]
        pd.testing.assert_series_equal(above, original)

    def test_pooling_changes_the_portfolio_weights_not_just_a_flag_field(self):
        """The load-bearing assertion: with a sub-floor regime carrying
        probability mass, the blended weights differ from the unshrunk
        vol_targeted_tilt path. Remove the shrinkage and they coincide."""
        returns, states = self._labelled_returns()
        table = returns_by_regime_stats(returns, states)
        probs = {0: 0.7, 1: 0.3}

        unshrunk = vol_targeted_tilt(
            probs, table, returns, target_vol_annual=0.10, halflife=6, min_obs=12
        )
        shrunk = blend_regime_tilts(
            probs, table, probs, table, returns,
            weight_1=1.0, target_vol_annual=0.10, halflife=6, min_obs=12,
        )

        assert not np.allclose(
            shrunk["weights"].reindex(unshrunk["weights"].index).to_numpy(),
            unshrunk["weights"].to_numpy(),
        )

    def test_a_compliant_labeling_leaves_the_single_path_identical(self):
        """No regime below the floor -> credibility 1.0 everywhere -> pooling
        is a no-op and the weight_1 = 1.0 reduction stays exact."""
        probs_1 = {0: 0.6, 1: 0.4}
        single = vol_targeted_tilt(
            probs_1, _rbr_1(), _asset_returns(), target_vol_annual=0.10, halflife=6, min_obs=12
        )
        joint = _blend(probs_1, {0: 0.3, 1: 0.7}, weight_1=1.0)

        pd.testing.assert_series_equal(joint["weights"], single["weights"])

    def test_table_without_moment_columns_falls_back_to_an_obs_weighted_pooled_sharpe(self):
        """A hand-built table carrying only regime/asset/sharpe/n_obs still
        gets pooled — the fallback target is the n_obs-weighted mean Sharpe."""
        table = pd.DataFrame(
            [
                {"regime": 0, "asset": "SPY", "sharpe_annualized": -2.0, "n_obs": 40},
                {"regime": 1, "asset": "SPY", "sharpe_annualized": 1.0, "n_obs": 660},
            ]
        )

        pooled, _ = pool_low_n_regime_sharpe(table, return_pooled=True)

        assert pooled["SPY"] == pytest.approx((40 * -2.0 + 660 * 1.0) / 700)

    def test_explicit_occupancy_override_is_honoured(self):
        """The driver knows the true label counts; the table's max-n_obs
        estimate is only a fallback."""
        table = _crisis_table()

        _, flags = pool_low_n_regime_sharpe(table, occupancy=pd.Series({0: 0.20, 1: 0.80}))

        assert bool(flags.loc[flags["regime"] == 0, "low_n"].iloc[0]) is False
