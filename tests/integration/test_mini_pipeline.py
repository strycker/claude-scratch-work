"""Mini end-to-end integration tests using synthetic data.

Tests the core feature-engineering → clustering chain without any
network access or file I/O.  All heavy steps (ingestion, prediction,
asset returns) are bypassed; these tests focus on the correctness and
determinism of the analytical kernel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _make_synthetic_macro(n_quarters: int = 80, seed: int = 42) -> pd.DataFrame:
    """Return a minimal synthetic macro DataFrame that can feed engineer_all().

    Columns mirror the real pipeline output after ingestion: a mix of
    multpl.com series, FRED series, and the derived columns added by
    add_cross_ratios().  Values are positive and within plausible ranges so
    log-transforms and ratio calculations work without NaNs.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-03-31", periods=n_quarters, freq="QE")

    sp500  = rng.uniform(300, 5000, n_quarters).cumsum() / n_quarters + 500
    sp500  = np.abs(sp500) + 200          # ensure positive

    data: dict[str, np.ndarray] = {
        # multpl.com
        "sp500":           sp500,
        "sp500_adj":       sp500 * rng.uniform(0.95, 1.05, n_quarters),
        "dividend":        rng.uniform(10, 80, n_quarters),
        "div_yield":       rng.uniform(0.01, 0.06, n_quarters),
        "earnings":        rng.uniform(20, 200, n_quarters),
        "cape_shiller":    rng.uniform(8, 40, n_quarters),
        "gdp":             rng.uniform(5000, 25000, n_quarters),
        "cpi":             rng.uniform(100, 320, n_quarters),
        "10yr_ustreas":    rng.uniform(1.0, 16.0, n_quarters),
        "us_cpi":          rng.uniform(100, 320, n_quarters),
        "book_value":      rng.uniform(100, 800, n_quarters),
        "price_sales":     rng.uniform(0.5, 5.0, n_quarters),
        "price_book":      rng.uniform(1.0, 6.0, n_quarters),
        # FRED
        "fred_gdp":        rng.uniform(5000, 25000, n_quarters),
        "fred_gnp":        rng.uniform(4800, 24000, n_quarters),
        "fred_baa":        rng.uniform(3.0, 12.0, n_quarters),
        "fred_aaa":        rng.uniform(2.5, 10.0, n_quarters),
        "fred_cpi":        rng.uniform(100, 320, n_quarters),
        "fred_gs10":       rng.uniform(1.0, 16.0, n_quarters),
        "fred_tb3ms":      rng.uniform(0.01, 10.0, n_quarters),
        "fred_vix":        rng.uniform(10, 80, n_quarters),
        "fred_unrate":     rng.uniform(3.0, 15.0, n_quarters),
        "fred_m2sl":       rng.uniform(3000, 25000, n_quarters),
        "fred_gs2":        rng.uniform(0.01, 10.0, n_quarters),
        "market_code":     np.zeros(n_quarters, dtype=float),  # placeholder
    }
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# Feature engineering integration tests
# ---------------------------------------------------------------------------

class TestEngineerAllIntegration:
    """engineer_all() produces well-formed output on synthetic data."""

    @pytest.fixture(scope="class")
    def cfg(self):
        from trading_crab_lib.config import load
        return load()

    @pytest.fixture(scope="class")
    def raw_df(self):
        return _make_synthetic_macro(n_quarters=80)

    @pytest.fixture(scope="class")
    def features_df(self, raw_df, cfg):
        from trading_crab_lib.transforms import engineer_all
        return engineer_all(raw_df, cfg, causal=False)

    def test_returns_dataframe(self, features_df):
        assert isinstance(features_df, pd.DataFrame)

    def test_has_rows(self, features_df):
        assert len(features_df) > 0

    def test_derivative_columns_present(self, features_df):
        d1_cols = [c for c in features_df.columns if c.endswith("_d1")]
        assert len(d1_cols) > 0, "Expected at least one _d1 derivative column"

    def test_no_all_nan_columns(self, features_df):
        all_nan = [c for c in features_df.columns if features_df[c].isna().all()]
        assert all_nan == [], f"Columns that are all-NaN: {all_nan}"

    def test_index_is_datetime(self, features_df):
        assert isinstance(features_df.index, pd.DatetimeIndex)


# ---------------------------------------------------------------------------
# Determinism regression tests
# ---------------------------------------------------------------------------

class TestEngineerAllDeterminism:
    """Running engineer_all() twice on identical input produces identical output."""

    def test_identical_output_same_seed(self):
        from trading_crab_lib.config import load
        from trading_crab_lib.transforms import engineer_all

        cfg = load()
        df = _make_synthetic_macro(n_quarters=60, seed=0)

        out1 = engineer_all(df.copy(), cfg, causal=False)
        out2 = engineer_all(df.copy(), cfg, causal=False)

        pd.testing.assert_frame_equal(out1, out2)

    def test_independent_of_market_code_values(self):
        """Changing market_code values must not alter feature outputs."""
        from trading_crab_lib.config import load
        from trading_crab_lib.transforms import engineer_all

        cfg = load()
        df_a = _make_synthetic_macro(n_quarters=60, seed=1)
        df_b = df_a.copy()
        # Set market_code to different values in df_b
        df_b["market_code"] = df_b["market_code"] + 3

        out_a = engineer_all(df_a, cfg, causal=False)
        out_b = engineer_all(df_b, cfg, causal=False)

        # Feature columns (excluding market_code) must be identical
        feature_cols = [c for c in out_a.columns if c != "market_code"]
        pd.testing.assert_frame_equal(
            out_a[feature_cols],
            out_b[feature_cols],
            check_names=True,
        )

    def test_causal_and_centered_produce_different_results(self):
        """Causal smoothing and centered smoothing should differ (look-ahead test)."""
        from trading_crab_lib.config import load
        from trading_crab_lib.transforms import engineer_all

        cfg = load()
        df = _make_synthetic_macro(n_quarters=60, seed=2)

        centered = engineer_all(df.copy(), cfg, causal=False)
        causal    = engineer_all(df.copy(), cfg, causal=True)

        # At least some derivative columns should differ between the two
        d1_cols = [c for c in centered.columns if c.endswith("_d1") and c in causal.columns]
        assert d1_cols, "No _d1 columns found to compare"

        any_diff = any(
            not centered[c].equals(causal[c])
            for c in d1_cols
        )
        assert any_diff, (
            "Centered and causal features are identical — look-ahead bias fix may be broken"
        )


# ---------------------------------------------------------------------------
# PCA + clustering integration tests
# ---------------------------------------------------------------------------

class TestClusteringIntegration:
    """reduce_pca() → fit_clusters() chain works on synthetic feature data."""

    @pytest.fixture(scope="class")
    def pca_df(self):
        from trading_crab_lib.config import load
        from trading_crab_lib.clustering import reduce_pca
        from trading_crab_lib.transforms import engineer_all

        cfg = load()
        raw = _make_synthetic_macro(n_quarters=80)
        features = engineer_all(raw, cfg, causal=False)

        # Drop non-numeric / market_code; keep only numeric clustering features
        numeric = features.select_dtypes(include="number").drop(
            columns=["market_code"], errors="ignore"
        ).dropna()
        assert len(numeric) >= 10, "Not enough rows after dropna for PCA"

        pca_df, _, _ = reduce_pca(numeric, n_components=5)
        return pca_df

    def test_pca_output_shape(self, pca_df):
        assert pca_df.shape[1] == 5

    def test_pca_output_has_no_nans(self, pca_df):
        assert not pca_df.isna().any().any()

    def test_fit_clusters_returns_dataframe(self, pca_df):
        from trading_crab_lib.clustering import fit_clusters
        result = fit_clusters(pca_df, best_k=5, balanced_k=5,
                              use_constrained=False, random_state=42)
        assert isinstance(result, pd.DataFrame)

    def test_fit_clusters_column_names(self, pca_df):
        from trading_crab_lib.clustering import fit_clusters
        result = fit_clusters(pca_df, best_k=5, balanced_k=5,
                              use_constrained=False, random_state=42)
        assert "cluster" in result.columns
        assert "balanced_cluster" in result.columns

    def test_fit_clusters_label_range(self, pca_df):
        from trading_crab_lib.clustering import fit_clusters
        result = fit_clusters(pca_df, best_k=5, balanced_k=5,
                              use_constrained=False, random_state=42)
        labels = result["balanced_cluster"]
        assert labels.min() >= 0
        assert labels.max() <= 4
        assert labels.nunique() <= 5

    def test_clustering_is_deterministic(self, pca_df):
        from trading_crab_lib.clustering import fit_clusters
        r1 = fit_clusters(pca_df, best_k=5, balanced_k=5,
                          use_constrained=False, random_state=42)
        r2 = fit_clusters(pca_df, best_k=5, balanced_k=5,
                          use_constrained=False, random_state=42)
        pd.testing.assert_frame_equal(r1, r2)
