"""Tests for scripts/evaluate_momentum.py helper functions."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

# Import the helpers from the script
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from evaluate_momentum import _strip_momentum_from_config, _cluster_metrics, _is_momentum_feature


@pytest.fixture
def sample_cfg() -> dict:
    """Minimal config with momentum features in feature lists."""
    return {
        "features": {
            "initial_features": [
                "10yr_ustreas",
                "credit_spread",
                "log_sp500",
                "fred_vix",
                "sp500_mom_4q",
                "sp500_mom_8q",
                "10yr_ustreas_mom_4q",
                "credit_spread_mom_4q",
                "corr_sp500_10yr_ustreas_8q",
                "cpi_acceleration",
            ],
            "clustering_features": [
                "10yr_ustreas_d1",
                "credit_spread_d1",
                "log_sp500_d1",
                "sp500_mom_4q",
                "sp500_mom_4q_d1",
                "sp500_mom_8q",
                "10yr_ustreas_mom_4q",
                "10yr_ustreas_mom_4q_d1",
                "credit_spread_mom_4q",
                "credit_spread_mom_4q_d1",
                "corr_sp500_10yr_ustreas_8q",
                "corr_sp500_10yr_ustreas_8q_d1",
                "cpi_acceleration",
                "cpi_acceleration_d1",
            ],
        }
    }


class TestIsMomentumFeature:
    def test_sp500_mom_is_momentum(self):
        assert _is_momentum_feature("sp500_mom_4q") is True

    def test_sp500_adj_mom_is_momentum(self):
        assert _is_momentum_feature("sp500_adj_mom_8q") is True

    def test_10yr_mom_is_momentum(self):
        assert _is_momentum_feature("10yr_ustreas_mom_4q") is True

    def test_credit_spread_mom_is_momentum(self):
        assert _is_momentum_feature("credit_spread_mom_4q") is True

    def test_corr_is_momentum(self):
        assert _is_momentum_feature("corr_sp500_10yr_ustreas_8q") is True

    def test_cpi_acceleration_is_momentum(self):
        assert _is_momentum_feature("cpi_acceleration") is True

    def test_cpi_acceleration_d1_is_momentum(self):
        assert _is_momentum_feature("cpi_acceleration_d1") is True

    def test_log_sp500_is_not_momentum(self):
        assert _is_momentum_feature("log_sp500") is False

    def test_divergence_is_not_momentum(self):
        assert _is_momentum_feature("div_spy_tlt_z_4q") is False

    def test_credit_spread_raw_is_not_momentum(self):
        assert _is_momentum_feature("credit_spread") is False


class TestStripMomentumFromConfig:
    def test_removes_sp500_mom_from_initial(self, sample_cfg):
        cfg2 = _strip_momentum_from_config(sample_cfg)
        init = cfg2["features"]["initial_features"]
        assert "sp500_mom_4q" not in init
        assert "sp500_mom_8q" not in init

    def test_removes_corr_from_initial(self, sample_cfg):
        cfg2 = _strip_momentum_from_config(sample_cfg)
        init = cfg2["features"]["initial_features"]
        assert "corr_sp500_10yr_ustreas_8q" not in init

    def test_removes_cpi_acceleration_from_initial(self, sample_cfg):
        cfg2 = _strip_momentum_from_config(sample_cfg)
        init = cfg2["features"]["initial_features"]
        assert "cpi_acceleration" not in init

    def test_removes_momentum_d1_from_clustering(self, sample_cfg):
        cfg2 = _strip_momentum_from_config(sample_cfg)
        clust = cfg2["features"]["clustering_features"]
        assert "sp500_mom_4q_d1" not in clust
        assert "10yr_ustreas_mom_4q_d1" not in clust
        assert "cpi_acceleration_d1" not in clust

    def test_preserves_non_momentum_features(self, sample_cfg):
        cfg2 = _strip_momentum_from_config(sample_cfg)
        init = cfg2["features"]["initial_features"]
        assert "10yr_ustreas" in init
        assert "credit_spread" in init
        assert "log_sp500" in init
        assert "fred_vix" in init

    def test_preserves_non_momentum_clustering(self, sample_cfg):
        cfg2 = _strip_momentum_from_config(sample_cfg)
        clust = cfg2["features"]["clustering_features"]
        assert "10yr_ustreas_d1" in clust
        assert "credit_spread_d1" in clust
        assert "log_sp500_d1" in clust

    def test_does_not_mutate_original(self, sample_cfg):
        original_init = list(sample_cfg["features"]["initial_features"])
        _strip_momentum_from_config(sample_cfg)
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
