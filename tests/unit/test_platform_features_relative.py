"""Unit tests for trading_crab_lib.platform.features.relative (D-10/D-11/INV-01).

Synthetic monthly DataFrames, no network — mirrors
tests/unit/test_platform_macro_ingest.py's fixture-construction convention
and tests/unit/test_platform_labeling.py's synthetic-frame shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.features.relative import (
    DEFAULT_RELATIVE_PAIRS,
    MONTHLY_CORRELATION_WINDOW,
    MONTHLY_MOMENTUM_WINDOWS,
    add_relative_features,
    compute_inflation_acceleration,
    compute_invariant_ratios,
    compute_relative_strength,
    compute_rolling_cross_correlation,
    compute_trailing_momentum,
)
from trading_crab_lib.platform.labeling.classifier2 import freeze_classifier2_columns
from trading_crab_lib.platform.taxonomy import lean_feature_set

N_MONTHS = 60


def _monthly_index(n_months: int = N_MONTHS) -> pd.DatetimeIndex:
    return pd.date_range("2015-01-31", periods=n_months, freq="ME")


def _make_synthetic_monthly_raw(n_months: int = N_MONTHS, seed: int = 11) -> pd.DataFrame:
    """A ``monthly_raw``-shaped synthetic frame with every source column this
    module's functions consume: equities_tr, long_duration_tr, oil,
    fred_cpi, fred_m2sl, fred_totalsl, fred_gdp."""
    rng = np.random.default_rng(seed)
    idx = _monthly_index(n_months)
    equity_returns = rng.normal(0.01, 0.03, n_months)
    bond_returns = rng.normal(0.003, 0.01, n_months)
    oil_returns = rng.normal(0.0, 0.05, n_months)
    equities_tr = 100.0 * np.cumprod(1.0 + equity_returns)
    long_duration_tr = 100.0 * np.cumprod(1.0 + bond_returns)
    oil = 50.0 * np.cumprod(1.0 + oil_returns)
    fred_cpi = 200.0 + np.cumsum(rng.normal(0.2, 0.05, n_months))
    fred_m2sl = 1000.0 + np.cumsum(rng.normal(2.0, 0.5, n_months))
    fred_totalsl = 2000.0 + np.cumsum(rng.normal(3.0, 0.5, n_months))
    fred_gdp = 5000.0 + np.cumsum(rng.normal(10.0, 1.0, n_months))
    return pd.DataFrame(
        {
            "equities_tr": equities_tr,
            "long_duration_tr": long_duration_tr,
            "oil": oil,
            "fred_cpi": fred_cpi,
            "fred_m2sl": fred_m2sl,
            "fred_totalsl": fred_totalsl,
            "fred_gdp": fred_gdp,
        },
        index=idx,
    )


# ── Module constants: re-derived from legacy quarterly [2, 4, 8] and 8 ──────


class TestWindowConstants:
    def test_monthly_momentum_windows_is_exact_re_derived_list(self):
        """Legacy quarterly default [2, 4, 8] * 3 months/quarter -> [6, 12, 24].
        A list of the wrong length or wrong values is a failed port, not a
        len()==3 coincidence."""
        assert MONTHLY_MOMENTUM_WINDOWS == [6, 12, 24]

    def test_monthly_correlation_window_is_exact_re_derived_value(self):
        """Legacy quarterly default 8 * 3 months/quarter -> 24."""
        assert MONTHLY_CORRELATION_WINDOW == 24


# ── compute_trailing_momentum: monthly suffixes, not quarterly ones ─────────


class TestComputeTrailingMomentum:
    def test_default_windows_produce_month_suffixed_columns(self):
        df = _make_synthetic_monthly_raw()
        result = compute_trailing_momentum(df, ["equities_tr"])
        for w in [6, 12, 24]:
            assert f"equities_tr_mom_{w}m" in result.columns
        # A failed port (quarterly suffixes) must be absent.
        for w in [2, 4, 8]:
            assert f"equities_tr_mom_{w}" not in result.columns
            assert f"equities_tr_mom_{w}q" not in result.columns

    def test_value_matches_pct_change_oracle(self):
        df = _make_synthetic_monthly_raw()
        result = compute_trailing_momentum(df, ["equities_tr"], windows=[12])
        expected = df["equities_tr"].pct_change(periods=12)
        pd.testing.assert_series_equal(
            result["equities_tr_mom_12m"], expected, check_names=False
        )

    def test_missing_column_skipped_gracefully(self):
        df = _make_synthetic_monthly_raw()
        result = compute_trailing_momentum(df, ["does_not_exist"], windows=[6])
        assert "does_not_exist_mom_6m" not in result.columns


# ── compute_relative_strength: hand-computed oracle ─────────────────────────


class TestComputeRelativeStrength:
    def test_ratio_matches_hand_computed_two_row_example(self):
        idx = pd.date_range("2020-01-31", periods=2, freq="ME")
        df = pd.DataFrame(
            {"equities_tr": [200.0, 300.0], "long_duration_tr": [100.0, 120.0]}, index=idx
        )
        result = compute_relative_strength(
            df, pairs=[("equities_tr", "long_duration_tr", "rs_equities_bonds")]
        )
        expected = pd.Series([2.0, 2.5], index=idx, name="rs_equities_bonds")
        pd.testing.assert_series_equal(
            result["rs_equities_bonds"], expected, check_exact=False, atol=1e-12
        )

    def test_default_pairs_produce_equities_bonds_and_oil_equities(self):
        df = _make_synthetic_monthly_raw()
        result = compute_relative_strength(df)
        assert "rs_equities_bonds" in result.columns
        assert "rs_oil_equities" in result.columns
        np.testing.assert_allclose(
            result["rs_equities_bonds"].to_numpy(),
            (df["equities_tr"] / df["long_duration_tr"]).to_numpy(),
            atol=1e-12,
        )

    def test_missing_pair_column_skipped_gracefully(self):
        df = _make_synthetic_monthly_raw().drop(columns=["long_duration_tr"])
        result = compute_relative_strength(df)
        assert "rs_equities_bonds" not in result.columns


# ── compute_rolling_cross_correlation: +1.0 / -1.0 oracle, not a range check ─


class TestComputeRollingCrossCorrelation:
    def _perfectly_correlated_frame(self, k: float, n_months: int = 40) -> pd.DataFrame:
        rng = np.random.default_rng(3)
        idx = _monthly_index(n_months)
        returns_a = rng.normal(0.0, 0.02, n_months)
        returns_a[0] = 0.0  # first return unused (level base), keep finite
        level_a = 100.0 * np.cumprod(1.0 + returns_a)
        returns_b = k * returns_a
        level_b = 100.0 * np.cumprod(1.0 + returns_b)
        return pd.DataFrame({"a": level_a, "b": level_b}, index=idx)

    def test_perfectly_positively_correlated_pair_is_one(self):
        df = self._perfectly_correlated_frame(k=2.0)
        result = compute_rolling_cross_correlation(df, pairs=[("a", "b")])
        col = f"corr_a_b_{MONTHLY_CORRELATION_WINDOW}m"
        assert col in result.columns
        last_value = result[col].dropna().iloc[-1]
        assert last_value == pytest.approx(1.0, abs=1e-9)

    def test_perfectly_anti_correlated_pair_is_negative_one(self):
        df = self._perfectly_correlated_frame(k=-1.5)
        result = compute_rolling_cross_correlation(df, pairs=[("a", "b")])
        col = f"corr_a_b_{MONTHLY_CORRELATION_WINDOW}m"
        last_value = result[col].dropna().iloc[-1]
        assert last_value == pytest.approx(-1.0, abs=1e-9)

    def test_default_pairs_use_default_relative_pairs_columns(self):
        df = _make_synthetic_monthly_raw()
        result = compute_rolling_cross_correlation(df)
        for num, denom, _name in DEFAULT_RELATIVE_PAIRS:
            assert f"corr_{num}_{denom}_{MONTHLY_CORRELATION_WINDOW}m" in result.columns


# ── compute_inflation_acceleration: linear vs quadratic, not both-zero ──────


class TestComputeInflationAcceleration:
    def test_linear_series_is_near_zero(self):
        idx = _monthly_index(24)
        df = pd.DataFrame({"fred_cpi": 100.0 + np.arange(24)}, index=idx)
        result = compute_inflation_acceleration(df)
        np.testing.assert_allclose(result["cpi_acceleration"].dropna().to_numpy(), 0.0, atol=1e-9)

    def test_quadratic_series_is_nonzero(self):
        idx = _monthly_index(24)
        df = pd.DataFrame({"fred_cpi": 100.0 + np.arange(24) ** 2.0}, index=idx)
        result = compute_inflation_acceleration(df)
        values = result["cpi_acceleration"].dropna().to_numpy()
        assert len(values) > 0
        assert (np.abs(values) > 1e-6).all()

    def test_missing_column_no_op(self):
        idx = _monthly_index(5)
        df = pd.DataFrame({"other": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
        result = compute_inflation_acceleration(df)
        assert "cpi_acceleration" not in result.columns


# ── compute_invariant_ratios: guarded, named (never anonymous PCs, R4) ──────


class TestComputeInvariantRatios:
    def test_both_ratios_computed_when_sources_present(self):
        df = _make_synthetic_monthly_raw()
        result = compute_invariant_ratios(df)
        assert "m2_gdp" in result.columns
        assert "credit_gdp" in result.columns
        np.testing.assert_allclose(
            result["m2_gdp"].to_numpy(), (df["fred_m2sl"] / df["fred_gdp"]).to_numpy(), atol=1e-12
        )
        np.testing.assert_allclose(
            result["credit_gdp"].to_numpy(),
            (df["fred_totalsl"] / df["fred_gdp"]).to_numpy(),
            atol=1e-12,
        )

    def test_skipped_when_a_source_column_absent(self):
        df = _make_synthetic_monthly_raw().drop(columns=["fred_m2sl"])
        result = compute_invariant_ratios(df)
        assert "m2_gdp" not in result.columns
        assert "credit_gdp" in result.columns  # unaffected by the missing m2 source


# ── add_relative_features: master wrapper, disjointness, no mutation ───────


class TestAddRelativeFeatures:
    def test_does_not_mutate_input(self):
        df = _make_synthetic_monthly_raw()
        original_cols = list(df.columns)
        add_relative_features(df, {})
        assert list(df.columns) == original_cols

    def test_output_index_matches_input(self):
        df = _make_synthetic_monthly_raw()
        result = add_relative_features(df, {})
        pd.testing.assert_index_equal(result.index, df.index)

    def test_disjoint_from_lean_feature_set(self):
        """Criterion 5: no raw column classifier #2's candidate set produces
        may collide with classifier #1's 13 lean raw columns."""
        df = _make_synthetic_monthly_raw()
        result = add_relative_features(df, {})
        lean = lean_feature_set(load_platform_config())
        overlap = set(result.columns) & lean
        assert overlap == set(), f"non-empty intersection with lean_feature_set: {overlap}"

    def test_gold_appears_in_no_produced_column_name(self):
        """D-11: gold's 1985-02 start fails the 1972+ common-support freeze —
        no default candidate here may reference it."""
        df = _make_synthetic_monthly_raw()
        result = add_relative_features(df, {})
        assert not any("gold" in col for col in result.columns)

    def test_produces_columns_from_every_sub_function(self):
        df = _make_synthetic_monthly_raw()
        result = add_relative_features(df, {})
        assert "rs_equities_bonds" in result.columns
        assert "rs_oil_equities" in result.columns
        assert "cpi_acceleration" in result.columns
        assert "m2_gdp" in result.columns
        assert "credit_gdp" in result.columns
        assert f"equities_tr_mom_{MONTHLY_MOMENTUM_WINDOWS[0]}m" in result.columns



class TestClassifier2Disjointness:
    """Criterion 5's disjointness test (D-10, ADR-0002 decision (a)).

    Classifier #2's frozen feature set must share NO raw column with classifier
    #1's 13 lean columns. Asserted twice: on the configured list, and on the
    list ``freeze_classifier2_columns`` actually resolves against a frame built
    by the real ``add_relative_features`` code path — a configured list can be
    correct while the resolved one is not, if the freeze rule ever admits a
    column the config never named.
    """

    def _feature_frame(self) -> pd.DataFrame:
        """A candidate frame produced by the real code path, not hand-listed."""
        raw = _make_synthetic_monthly_raw(n_months=120)
        return add_relative_features(raw, {})

    def test_configured_feature_list_is_disjoint_from_the_lean_set(self):
        cfg = load_platform_config()
        configured = set(cfg["labeling_2"]["features"])
        lean = lean_feature_set(cfg)
        overlap = configured & lean
        assert overlap == set(), (
            f"classifier #2's configured feature list collides with classifier #1's "
            f"lean set on: {sorted(overlap)}"
        )

    def test_resolved_frozen_list_is_disjoint_from_the_lean_set(self):
        cfg = load_platform_config()
        features = self._feature_frame()
        frozen = freeze_classifier2_columns(features, cfg, features.index[36])
        overlap = set(frozen) & lean_feature_set(cfg)
        assert overlap == set(), (
            f"classifier #2's RESOLVED frozen list collides with classifier #1's "
            f"lean set on: {sorted(overlap)}"
        )

    def test_the_lean_set_is_exactly_thirteen_members(self):
        """Without this, disjointness could be satisfied by shrinking the
        denominator — removing a column from classifier #1 instead of keeping
        classifier #2 off it."""
        lean = lean_feature_set(load_platform_config())
        assert len(lean) == 13, f"lean set is {len(lean)} members, not 13: {sorted(lean)}"

    def test_the_resolved_frozen_list_is_the_full_pinned_eight(self):
        """A frozen list that silently shrank would make disjointness trivially
        true; pin the count the ADR pinned."""
        cfg = load_platform_config()
        features = self._feature_frame()
        frozen = freeze_classifier2_columns(features, cfg, features.index[36])
        assert frozen == list(cfg["labeling_2"]["features"])
        assert len(frozen) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
