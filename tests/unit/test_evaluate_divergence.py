"""Tests for scripts/evaluate_divergence.py helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluate_divergence import _strip_divergence_from_config, _cluster_metrics


@pytest.fixture
def sample_cfg() -> dict:
    """Minimal config with divergence features in feature lists."""
    return {
        "features": {
            "initial_features": [
                "sp500",
                "10yr_ustreas",
                "credit_spread",
                "fred_vix",
                "log_sp500",
                "div_spy_tlt_z_4q",
                "div_spy_tlt_trigger",
                "div_cred_vix_z_4q",
                "div_cred_vix_trigger",
            ],
            "clustering_features": [
                "10yr_ustreas_d1",
                "credit_spread_d1",
                "log_sp500_d1",
                "div_spy_tlt_z_4q",
                "div_spy_tlt_z_4q_d1",
                "div_spy_tlt_trigger",
                "div_cred_vix_z_4q",
                "div_cred_vix_z_4q_d1",
                "div_cred_vix_trigger",
                "div_d1_spy_tlt_z_4q",
                "div_d1_spy_tlt_trigger",
                "div_d1_cred_vix_z_4q",
                "div_d1_cred_vix_trigger",
            ],
        }
    }


class TestStripDivergenceFromConfig:
    def test_removes_div_spy_from_initial(self, sample_cfg):
        cfg2 = _strip_divergence_from_config(sample_cfg)
        init = cfg2["features"]["initial_features"]
        assert "div_spy_tlt_z_4q" not in init
        assert "div_spy_tlt_trigger" not in init

    def test_removes_div_cred_from_initial(self, sample_cfg):
        cfg2 = _strip_divergence_from_config(sample_cfg)
        init = cfg2["features"]["initial_features"]
        assert "div_cred_vix_z_4q" not in init

    def test_removes_div_d1_from_clustering(self, sample_cfg):
        cfg2 = _strip_divergence_from_config(sample_cfg)
        clust = cfg2["features"]["clustering_features"]
        assert "div_d1_spy_tlt_z_4q" not in clust
        assert "div_d1_cred_vix_trigger" not in clust

    def test_removes_sp500_raw(self, sample_cfg):
        cfg2 = _strip_divergence_from_config(sample_cfg)
        assert "sp500" not in cfg2["features"]["initial_features"]

    def test_preserves_non_divergence_features(self, sample_cfg):
        cfg2 = _strip_divergence_from_config(sample_cfg)
        init = cfg2["features"]["initial_features"]
        assert "10yr_ustreas" in init
        assert "credit_spread" in init
        assert "fred_vix" in init
        assert "log_sp500" in init

    def test_preserves_non_divergence_clustering(self, sample_cfg):
        cfg2 = _strip_divergence_from_config(sample_cfg)
        clust = cfg2["features"]["clustering_features"]
        assert "10yr_ustreas_d1" in clust
        assert "credit_spread_d1" in clust
        assert "log_sp500_d1" in clust

    def test_does_not_mutate_original(self, sample_cfg):
        original_init = list(sample_cfg["features"]["initial_features"])
        _strip_divergence_from_config(sample_cfg)
        assert sample_cfg["features"]["initial_features"] == original_init


class TestClusterMetrics:
    def test_returns_expected_keys(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(100, 10), columns=[f"f{i}" for i in range(10)])
        result = _cluster_metrics(df, n_pca=3, k=3)
        assert "silhouette" in result
        assert "calinski_harabasz" in result
        assert "davies_bouldin" in result
        assert "pca_variance_explained" in result
        assert "n_samples" in result
        assert "n_features" in result

    def test_silhouette_in_valid_range(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(100, 10), columns=[f"f{i}" for i in range(10)])
        result = _cluster_metrics(df, n_pca=3, k=3)
        assert -1 <= result["silhouette"] <= 1

    def test_n_samples_matches_input(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame(rng.randn(80, 5), columns=[f"f{i}" for i in range(5)])
        result = _cluster_metrics(df, n_pca=3, k=2)
        assert result["n_samples"] == 80
        assert result["n_features"] == 5
