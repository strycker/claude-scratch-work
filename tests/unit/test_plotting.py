"""Tests for src/trading_crab_lib/plotting.py — visualization helpers.

Tests verify that plot functions run without crashing and produce matplotlib
figures.  We do NOT assert on pixel-level content — only that figures are
created and saved correctly when run_cfg.save_plots is True.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.runtime import RunConfig
from trading_crab_lib import plotting


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def run_cfg(tmp_path, monkeypatch):
    """RunConfig that saves plots to tmp_path, never calls plt.show()."""
    monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
    return RunConfig(generate_plots=True, save_plots=True, show_plots=False)


@pytest.fixture
def raw_df():
    idx = pd.date_range("2000-03-31", periods=40, freq="QE")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((40, 5)),
        index=idx,
        columns=["sp500", "gdp", "cpi", "gold", "oil"],
    )


@pytest.fixture
def pca_df():
    idx = pd.date_range("2000-03-31", periods=40, freq="QE")
    rng = np.random.default_rng(42)
    data = rng.standard_normal((40, 5))
    return pd.DataFrame(data, index=idx, columns=[f"PC{i+1}" for i in range(5)])


@pytest.fixture
def labels(pca_df):
    vals = [i % 3 for i in range(len(pca_df))]
    return pd.Series(vals, index=pca_df.index, dtype=int, name="cluster")


@pytest.fixture
def regime_names():
    return {0: "Growth", 1: "Recession", 2: "Stagflation"}


# ── _save_or_show ────────────────────────────────────────────────────────────

class TestSaveOrShow:
    def test_saves_file(self, run_cfg, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        plotting._save_or_show(fig, "test_plot.png", run_cfg)
        assert (tmp_path / "test_plot.png").exists()

    def test_no_save_when_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        cfg = RunConfig(save_plots=False, show_plots=False)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        plotting._save_or_show(fig, "should_not_exist.png", cfg)
        assert not (tmp_path / "should_not_exist.png").exists()


# ── _regime_color ────────────────────────────────────────────────────────────

class TestRegimeColor:
    def test_returns_string(self):
        c = plotting._regime_color(0)
        assert isinstance(c, str)
        assert c.startswith("#")

    def test_wraps_around(self):
        n = len(plotting.CUSTOM_COLORS)
        assert plotting._regime_color(0) == plotting._regime_color(n)


# ── Step 01 plots ────────────────────────────────────────────────────────────

class TestPlotRawSeriesCoverage:
    def test_does_not_crash(self, raw_df, run_cfg, tmp_path):
        plotting.plot_raw_series_coverage(raw_df, run_cfg)
        assert (tmp_path / "01_raw_coverage.png").exists()


class TestPlotRawSeriesSample:
    def test_does_not_crash(self, raw_df, run_cfg, tmp_path):
        plotting.plot_raw_series_sample(raw_df, ["sp500", "gdp"], run_cfg)
        assert (tmp_path / "01_raw_series_sample.png").exists()

    def test_skips_missing_series(self, raw_df, run_cfg, tmp_path):
        # Should not crash if series don't exist
        plotting.plot_raw_series_sample(raw_df, ["nonexistent"], run_cfg)
        # No file saved since no valid series
        assert not (tmp_path / "01_raw_series_sample.png").exists()


# ── Step 02 plots ────────────────────────────────────────────────────────────

class TestPlotFeatureDistributions:
    def test_does_not_crash(self, raw_df, run_cfg, tmp_path):
        plotting.plot_feature_distributions(raw_df, run_cfg, cols=["sp500", "gdp"])
        assert (tmp_path / "02_feature_distributions.png").exists()


# ── Step 03 plots ────────────────────────────────────────────────────────────

class TestPlotElbowCurve:
    def test_does_not_crash(self, run_cfg, tmp_path):
        scores = pd.DataFrame({
            "k": [2, 3, 4, 5],
            "silhouette": [0.6, 0.55, 0.5, 0.45],
            "calinski": [100, 120, 110, 105],
            "davies_bouldin": [0.8, 0.9, 1.0, 1.1],
        })
        plotting.plot_elbow_curve(scores, chosen_k=3, run_cfg=run_cfg)
        assert (tmp_path / "03_elbow_curves.png").exists()


class TestPlotPcaScatter:
    def test_does_not_crash(self, pca_df, labels, regime_names, run_cfg, tmp_path):
        plotting.plot_pca_scatter(pca_df, labels, regime_names, run_cfg)
        assert (tmp_path / "03_pca_scatter.png").exists()


class TestPlotClusterSizes:
    def test_does_not_crash(self, labels, regime_names, run_cfg, tmp_path):
        plotting.plot_cluster_sizes(labels, regime_names, run_cfg)
        assert (tmp_path / "03_cluster_sizes.png").exists()


# ── Step 04 plots ────────────────────────────────────────────────────────────

class TestPlotRegimeTimeline:
    def test_does_not_crash(self, labels, regime_names, run_cfg, tmp_path):
        plotting.plot_regime_timeline(labels, regime_names, run_cfg)
        assert (tmp_path / "04_regime_timeline.png").exists()


class TestPlotTransitionMatrix:
    def test_does_not_crash(self, regime_names, run_cfg, tmp_path):
        tm = pd.DataFrame(
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]],
            index=[0, 1, 2], columns=[0, 1, 2],
        )
        plotting.plot_transition_matrix(tm, regime_names, run_cfg)
        assert (tmp_path / "04_transition_matrix.png").exists()


class TestPlotRegimeProfiles:
    def test_does_not_crash(self, raw_df, labels, regime_names, run_cfg, tmp_path):
        plotting.plot_regime_profiles(
            raw_df, labels, regime_names, ["sp500", "gdp", "cpi"], run_cfg
        )
        assert (tmp_path / "04_regime_profiles.png").exists()


# ── Step 05 plots ────────────────────────────────────────────────────────────

class TestPlotFeatureImportance:
    def test_does_not_crash(self, run_cfg, tmp_path):
        class FakeModel:
            feature_importances_ = np.array([0.3, 0.2, 0.15, 0.1, 0.25])

        plotting.plot_feature_importance(
            FakeModel(), ["a", "b", "c", "d", "e"], run_cfg, top_n=3
        )
        assert (tmp_path / "05_feature_importance.png").exists()


class TestPlotForwardProbabilities:
    def test_does_not_crash(self, regime_names, run_cfg, tmp_path):
        prediction = {"regime": 0, "probabilities": {0: 0.6, 1: 0.3, 2: 0.1}}
        plotting.plot_forward_probabilities(prediction, regime_names, run_cfg)
        assert (tmp_path / "05_current_regime_probs.png").exists()

    def test_empty_probs_no_crash(self, regime_names, run_cfg, tmp_path):
        prediction = {"regime": 0, "probabilities": {}}
        plotting.plot_forward_probabilities(prediction, regime_names, run_cfg)
        # No file written since no probs
        assert not (tmp_path / "05_current_regime_probs.png").exists()


# ── Step 06 plots ────────────────────────────────────────────────────────────

class TestPlotAssetReturnsByRegime:
    def test_does_not_crash(self, regime_names, run_cfg, tmp_path):
        profile = pd.DataFrame(
            {"SPY": [0.06, 0.02, -0.01], "GLD": [0.01, 0.05, 0.03]},
            index=[0, 1, 2],
        )
        plotting.plot_asset_returns_by_regime(profile, regime_names, run_cfg)
        assert (tmp_path / "06_asset_returns_by_regime.png").exists()

    def test_empty_profile_no_crash(self, regime_names, run_cfg, tmp_path):
        plotting.plot_asset_returns_by_regime(pd.DataFrame(), regime_names, run_cfg)


class TestPlotAssetHeatmap:
    def test_does_not_crash(self, regime_names, run_cfg, tmp_path):
        profile = pd.DataFrame(
            {"SPY": [0.06, 0.02], "GLD": [0.01, 0.05]},
            index=[0, 1],
        )
        plotting.plot_asset_heatmap(profile, regime_names, run_cfg)
        assert (tmp_path / "06_asset_heatmap.png").exists()


# ── Constants ────────────────────────────────────────────────────────────────

# ── Phase A3: Time-Series & Regime Plots ─────────────────────────────────────

class TestPlotSoftProbabilities:
    def test_does_not_crash(self, regime_names, run_cfg, tmp_path):
        idx = pd.date_range("2000-03-31", periods=20, freq="QE")
        rng = np.random.default_rng(42)
        probs = rng.dirichlet([1, 1, 1], size=20)
        probs_df = pd.DataFrame(probs, index=idx, columns=["prob_0", "prob_1", "prob_2"])
        plotting.plot_soft_probabilities(probs_df, regime_names, run_cfg)
        assert (tmp_path / "03_soft_probabilities.png").exists()

    def test_empty_no_crash(self, regime_names, run_cfg):
        plotting.plot_soft_probabilities(pd.DataFrame(), regime_names, run_cfg)


class TestPlotFeatureRegimeOverlay:
    def test_does_not_crash(self, labels, regime_names, run_cfg, tmp_path):
        idx = labels.index
        feat = pd.Series(np.random.default_rng(42).standard_normal(len(idx)),
                         index=idx, name="sp500")
        plotting.plot_feature_regime_overlay(feat, labels, regime_names, run_cfg)
        assert (tmp_path / "04_feature_overlay_sp500.png").exists()


class TestPlotForwardProbEvolution:
    def test_does_not_crash(self, regime_names, run_cfg, tmp_path):
        mat = pd.DataFrame(
            [[0.6, 0.2, 0.2], [0.1, 0.7, 0.2], [0.3, 0.2, 0.5]],
            index=[0, 1, 2], columns=[0, 1, 2],
        )
        plotting.plot_forward_prob_evolution({1: mat, 4: mat}, regime_names, run_cfg)
        assert (tmp_path / "04_forward_prob_evolution.png").exists()

    def test_empty_no_crash(self, regime_names, run_cfg):
        plotting.plot_forward_prob_evolution({}, regime_names, run_cfg)


class TestPlotGapFillBeforeAfter:
    def test_does_not_crash(self, run_cfg, tmp_path):
        idx = pd.date_range("2000-03-31", periods=20, freq="QE")
        raw = pd.Series([1, 2, np.nan, np.nan, 5, 6, 7, np.nan, 9, 10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        index=idx, name="test_col")
        filled = raw.interpolate()
        plotting.plot_gap_fill_before_after(raw, filled, run_cfg)
        assert (tmp_path / "02_gap_fill_test_col.png").exists()


class TestPlotRegimeColoredPca3d:
    def test_does_not_crash(self, pca_df, labels, regime_names, run_cfg, tmp_path):
        plotting.plot_regime_colored_pca_3d(pca_df, labels, regime_names, run_cfg)
        assert (tmp_path / "03_pca_3d.png").exists()

    def test_too_few_components_no_crash(self, labels, regime_names, run_cfg):
        idx = labels.index
        df = pd.DataFrame({"PC1": range(len(idx)), "PC2": range(len(idx))}, index=idx)
        plotting.plot_regime_colored_pca_3d(df, labels, regime_names, run_cfg)


# ── Phase A2: PCA & Clustering Plots ─────────────────────────────────────────

class TestPlotScree:
    def test_does_not_crash(self, run_cfg, tmp_path):
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 10))
        pca = PCA(n_components=5).fit(X)
        plotting.plot_scree(pca, run_cfg)
        assert (tmp_path / "03_scree.png").exists()


class TestPlotPcaLoadings:
    def test_does_not_crash(self, run_cfg, tmp_path):
        from sklearn.decomposition import PCA
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 10))
        pca = PCA(n_components=3).fit(X)
        names = [f"feat_{i}" for i in range(10)]
        plotting.plot_pca_loadings(pca, names, run_cfg, top_n=5)
        assert (tmp_path / "03_pca_loadings.png").exists()


class TestPlotSilhouetteSamples:
    def test_does_not_crash(self, run_cfg, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        labels = np.array([i % 3 for i in range(40)])
        plotting.plot_silhouette_samples(X, labels, run_cfg)
        assert (tmp_path / "03_silhouette_samples.png").exists()


class TestPlotGmmBicSurface:
    def test_does_not_crash(self, run_cfg, tmp_path):
        bic_df = pd.DataFrame({
            "k": [2, 2, 3, 3],
            "covariance_type": ["diag", "full", "diag", "full"],
            "bic": [1200, 1150, 1100, 1050],
        })
        plotting.plot_gmm_bic_surface(bic_df, run_cfg)
        assert (tmp_path / "03_gmm_bic_surface.png").exists()

    def test_missing_columns_no_crash(self, run_cfg):
        plotting.plot_gmm_bic_surface(pd.DataFrame({"x": [1]}), run_cfg)


class TestPlotMethodComparisonTable:
    def test_does_not_crash(self, run_cfg, tmp_path):
        df = pd.DataFrame({
            "method": ["KMeans", "GMM", "Spectral"],
            "n_clusters": [5, 4, 5],
            "silhouette": [0.25, 0.22, 0.19],
            "davies_bouldin": [1.1, 1.3, 1.5],
            "calinski": [45.0, 40.0, 38.0],
        })
        plotting.plot_method_comparison_table(df, run_cfg)
        assert (tmp_path / "03_method_comparison.png").exists()

    def test_empty_no_crash(self, run_cfg):
        plotting.plot_method_comparison_table(pd.DataFrame(), run_cfg)


# ── Phase A1: Model Evaluation Plots ─────────────────────────────────────────

class TestPlotDecisionTree:
    def test_does_not_crash(self, regime_names, run_cfg, tmp_path):
        from sklearn.tree import DecisionTreeClassifier
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        y = np.array([i % 3 for i in range(40)])
        tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
        features = [f"feat_{i}" for i in range(5)]
        plotting.plot_decision_tree(tree, features, regime_names, run_cfg)
        assert (tmp_path / "05_decision_tree.png").exists()


class TestPlotCvFoldAccuracy:
    def test_does_not_crash(self, run_cfg, tmp_path):
        plotting.plot_cv_fold_accuracy([0.6, 0.7, 0.55, 0.65, 0.72], run_cfg)
        assert (tmp_path / "05_cv_fold_accuracy.png").exists()

    def test_empty_no_crash(self, run_cfg):
        plotting.plot_cv_fold_accuracy([], run_cfg)


class TestPlotModelComparisonBar:
    def test_does_not_crash(self, run_cfg, tmp_path):
        metrics = {
            "RF": {"accuracy": 0.72, "f1": 0.68},
            "DT": {"accuracy": 0.58, "f1": 0.55},
        }
        plotting.plot_model_comparison_bar(metrics, run_cfg)
        assert (tmp_path / "05_model_comparison.png").exists()

    def test_empty_no_crash(self, run_cfg):
        plotting.plot_model_comparison_bar({}, run_cfg)


class TestPlotCalibrationCurve:
    def test_does_not_crash(self, regime_names, run_cfg, tmp_path):
        rng = np.random.default_rng(42)
        y_true = np.array([i % 3 for i in range(60)])
        y_proba = rng.dirichlet([1, 1, 1], size=60)
        plotting.plot_calibration_curve(y_true, y_proba, regime_names, run_cfg)
        assert (tmp_path / "05_calibration_curve.png").exists()


class TestPlotLearningCurve:
    def test_does_not_crash(self, run_cfg, tmp_path):
        from sklearn.tree import DecisionTreeClassifier
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 5))
        y = np.array([i % 3 for i in range(60)])
        model = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
        plotting.plot_learning_curve(model, X, y, run_cfg, cv=3, n_points=4)
        assert (tmp_path / "05_learning_curve.png").exists()


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_custom_colors_count(self):
        assert len(plotting.CUSTOM_COLORS) == 5

    def test_regime_cmap_is_colormap(self):
        assert isinstance(plotting.REGIME_CMAP, matplotlib.colors.ListedColormap)
