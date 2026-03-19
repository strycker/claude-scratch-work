import pandas as pd
import numpy as np
from pathlib import Path

from trading_crab_lib.regime import (
    build_profiles,
    suggest_names,
    build_transition_matrix,
    load_name_overrides,
)


def _make_synthetic_features_and_labels():
    """Small quarterly toy dataset with clearly separated regimes."""
    idx = pd.period_range("2000Q1", periods=8, freq="Q")

    # Two simple features: "inflation" and "growth"
    inflation = pd.Series(
        [1.0, 1.1, 1.2, 1.3, 3.0, 3.1, 3.2, 3.3], index=idx, name="us_infl"
    )
    growth = pd.Series(
        [0.5, 0.4, 0.6, 0.5, 2.0, 2.1, 1.9, 2.2], index=idx, name="gdp_growth"
    )

    features = pd.concat([inflation, growth], axis=1)

    # Regime 0 for low inflation / low growth, regime 1 for high inflation / high growth
    labels = pd.Series(
        [0, 0, 0, 0, 1, 1, 1, 1],
        index=idx,
        name="balanced_cluster",
    )

    return features, labels


def test_build_profiles_basic_stats():
    features, labels = _make_synthetic_features_and_labels()

    profile = build_profiles(features, labels)

    # Index should contain both regimes
    assert set(profile.index.tolist()) == {0, 1}

    # Columns are a MultiIndex of stats x features (order may vary by pandas version)
    assert isinstance(profile.columns, pd.MultiIndex)

    # Support both (stat, feature) and (feature, stat) layouts
    if ("mean", "us_infl") in profile.columns:
        mean_infl_col = ("mean", "us_infl")
        mean_growth_col = ("mean", "gdp_growth")
    elif ("us_infl", "mean") in profile.columns:
        mean_infl_col = ("us_infl", "mean")
        mean_growth_col = ("gdp_growth", "mean")
    else:
        raise AssertionError("Could not find mean columns for us_infl / gdp_growth in profile.")

    # Manual means per regime should match profile output
    for regime in (0, 1):
        mask = labels == regime
        expected_infl_mean = features.loc[mask, "us_infl"].mean()
        expected_growth_mean = features.loc[mask, "gdp_growth"].mean()

        assert np.isclose(profile.loc[regime, mean_infl_col], expected_infl_mean)
        assert np.isclose(profile.loc[regime, mean_growth_col], expected_growth_mean)


def test_build_profiles_aligns_on_intersection():
    features, labels = _make_synthetic_features_and_labels()

    # Add an extra row to features only
    extra_idx = pd.period_range("2002Q1", periods=1, freq="Q")
    extra_row = pd.DataFrame(
        {"us_infl": [10.0], "gdp_growth": [10.0]}, index=extra_idx
    )
    features_extra = pd.concat([features, extra_row])

    profile = build_profiles(features_extra, labels)

    # The extra row should not affect means because labels do not cover it
    base_profile = build_profiles(features, labels)
    # We care that the *values* match and that the extra row does not leak in;
    # index dtype differences are not important here.
    pd.testing.assert_frame_equal(profile, base_profile, check_index_type=False)


def test_suggest_names_deterministic_and_complete():
    features, labels = _make_synthetic_features_and_labels()

    names1 = suggest_names(features, labels)
    names2 = suggest_names(features, labels)

    # Deterministic mapping given same input
    assert names1 == names2

    # One non-empty name per regime
    assert set(names1.keys()) == {0, 1}
    assert all(isinstance(v, str) and v for v in names1.values())


def test_load_name_overrides_applied(tmp_path):
    features, labels = _make_synthetic_features_and_labels()

    # Auto names first
    auto_names = suggest_names(features, labels)

    # Create a temporary config directory with an override file
    config_dir = tmp_path
    overrides_path = config_dir / "regime_labels.yaml"
    overrides_path.write_text("0: \"Custom Regime A\"\n")

    overrides = load_name_overrides(config_dir)
    assert overrides == {0: "Custom Regime A"}

    # When merged, overrides should take precedence over auto-suggestions
    merged = {**auto_names, **overrides}
    assert merged[0] == "Custom Regime A"
    # Non-overridden regimes fall back to auto names
    assert merged[1] == auto_names[1]


def test_build_transition_matrix_probabilities():
    # Simple known sequence with transitions 0->0, 0->1, 1->0, 0->1
    labels = pd.Series([0, 0, 1, 0, 1], name="balanced_cluster")

    tm = build_transition_matrix(labels)

    # Matrix should be square with regimes {0, 1}
    assert list(tm.index) == [0, 1]
    assert list(tm.columns) == [0, 1]

    # Rows should sum to 1 within numerical tolerance
    row_sums = tm.sum(axis=1)
    assert np.allclose(row_sums.values, np.ones_like(row_sums.values))

    # Hand-computed transitions:
    # From 0: sequence positions (0->0), (0->1), (0->1) => counts[0,0]=1, counts[0,1]=2
    # From 1: sequence positions (1->0); the last 1 has no outgoing transition
    # So:
    #   P(next=0 | current=0) = 1/3
    #   P(next=1 | current=0) = 2/3
    # From 1, only one outgoing transition (1->0):
    #   P(next=0 | current=1) = 1.0
    assert np.isclose(tm.loc[0, 0], 1.0 / 3.0)
    assert np.isclose(tm.loc[0, 1], 2.0 / 3.0)
    assert np.isclose(tm.loc[1, 0], 1.0)
