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


# ── Phase A4: Specialty Diagnostic Plots ─────────────────────────────────────

class TestPlotRrgScatter:
    def test_does_not_crash(self, run_cfg, tmp_path):
        rrg_df = pd.DataFrame({
            "rs": [102, 98, 95, 105],
            "rm": [101, 103, 97, 96],
            "quadrant": ["LEADING", "IMPROVING", "LAGGING", "WEAKENING"],
        }, index=["SPY", "GLD", "TLT", "USO"])
        plotting.plot_rrg_scatter(rrg_df, run_cfg)
        assert (tmp_path / "08_rrg_scatter.png").exists()

    def test_empty_no_crash(self, run_cfg):
        plotting.plot_rrg_scatter(pd.DataFrame(), run_cfg)


class TestPlotFeatureImportanceComparison:
    def test_does_not_crash(self, run_cfg, tmp_path):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 10))
        y = np.array([i % 3 for i in range(40)])
        rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
        names = [f"feat_{i}" for i in range(10)]
        plotting.plot_feature_importance_comparison({"RF": rf, "DT": dt}, names, run_cfg, top_n=5)
        assert (tmp_path / "05_feature_importance_comparison.png").exists()

    def test_empty_no_crash(self, run_cfg):
        plotting.plot_feature_importance_comparison({}, [], run_cfg)


class TestPlotRegimeDurationHistogram:
    def test_does_not_crash(self, labels, regime_names, run_cfg, tmp_path):
        plotting.plot_regime_duration_histogram(labels, regime_names, run_cfg)
        assert (tmp_path / "04_regime_duration.png").exists()

    def test_empty_no_crash(self, regime_names, run_cfg):
        plotting.plot_regime_duration_histogram(pd.Series(dtype=int), regime_names, run_cfg)


class TestPlotCorrelationChangeHeatmap:
    def test_does_not_crash(self, labels, regime_names, run_cfg, tmp_path):
        rng = np.random.default_rng(42)
        idx = labels.index
        features = pd.DataFrame(rng.standard_normal((len(idx), 8)),
                                index=idx, columns=[f"f{i}" for i in range(8)])
        plotting.plot_correlation_change_heatmap(features, labels, regime_names, run_cfg, top_n=5)
        assert (tmp_path / "04_correlation_change.png").exists()


class TestPlotFeatureVarianceRanking:
    def test_does_not_crash(self, run_cfg, tmp_path):
        rng = np.random.default_rng(42)
        features = pd.DataFrame(rng.standard_normal((40, 10)),
                                columns=[f"feat_{i}" for i in range(10)])
        plotting.plot_feature_variance_ranking(features, run_cfg, top_n=8)
        assert (tmp_path / "02_feature_variance_ranking.png").exists()

    def test_empty_no_crash(self, run_cfg):
        plotting.plot_feature_variance_ranking(pd.DataFrame(), run_cfg)


# ── Phase A5: Feature Engineering & Selection Plots ──────────────────────────

class TestPlotFeatureSelectionCurve:
    def test_does_not_crash(self, run_cfg, tmp_path):
        importances = [("f1", 0.3), ("f2", 0.2), ("f3", 0.15), ("f4", 0.1),
                       ("f5", 0.08), ("f6", 0.07), ("f7", 0.05), ("f8", 0.05)]
        plotting.plot_feature_selection_curve(importances, run_cfg)
        assert (tmp_path / "05_feature_selection_curve.png").exists()

    def test_series_input(self, run_cfg, tmp_path):
        s = pd.Series({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1})
        plotting.plot_feature_selection_curve(s, run_cfg)
        assert (tmp_path / "05_feature_selection_curve.png").exists()

    def test_empty_no_crash(self, run_cfg):
        plotting.plot_feature_selection_curve([], run_cfg)


class TestPlotDivergenceTimeseries:
    def test_does_not_crash(self, labels, run_cfg, tmp_path):
        idx = labels.index
        rng = np.random.default_rng(42)
        div = pd.DataFrame({
            "div_spy_tlt_z_4q": rng.standard_normal(len(idx)),
            "div_cred_vix_z_4q": rng.standard_normal(len(idx)),
        }, index=idx)
        plotting.plot_divergence_timeseries(div, labels, run_cfg)
        assert (tmp_path / "02_divergence_timeseries.png").exists()

    def test_no_z_cols_no_crash(self, labels, run_cfg):
        idx = labels.index
        div = pd.DataFrame({"other": range(len(idx))}, index=idx)
        plotting.plot_divergence_timeseries(div, labels, run_cfg)


class TestPlotMomentumDashboard:
    def test_does_not_crash(self, labels, run_cfg, tmp_path):
        idx = labels.index
        rng = np.random.default_rng(42)
        mom = pd.DataFrame({
            "sp500_mom_4q": rng.standard_normal(len(idx)),
            "sp500_mom_8q": rng.standard_normal(len(idx)),
        }, index=idx)
        plotting.plot_momentum_dashboard(mom, labels, run_cfg)
        assert (tmp_path / "02_momentum_dashboard.png").exists()


class TestPlotNanHeatmap:
    def test_does_not_crash(self, run_cfg, tmp_path):
        idx = pd.date_range("2000-03-31", periods=20, freq="QE")
        rng = np.random.default_rng(42)
        data = rng.standard_normal((20, 5))
        data[3:5, 1] = np.nan
        data[10:14, 3] = np.nan
        df = pd.DataFrame(data, index=idx, columns=[f"f{i}" for i in range(5)])
        plotting.plot_nan_heatmap(df, run_cfg)
        assert (tmp_path / "01_nan_heatmap.png").exists()

    def test_empty_no_crash(self, run_cfg):
        plotting.plot_nan_heatmap(pd.DataFrame(), run_cfg)


class TestPlotCenteredVsCausalComparison:
    def test_does_not_crash(self, run_cfg, tmp_path):
        idx = pd.date_range("2000-03-31", periods=20, freq="QE")
        rng = np.random.default_rng(42)
        centered = pd.DataFrame({"f1": rng.standard_normal(20),
                                  "f2": rng.standard_normal(20)}, index=idx)
        causal = pd.DataFrame({"f1": rng.standard_normal(20),
                                "f2": rng.standard_normal(20)}, index=idx)
        plotting.plot_centered_vs_causal_comparison(centered, causal, ["f1", "f2"], run_cfg)
        assert (tmp_path / "02_centered_vs_causal.png").exists()

    def test_no_matching_cols_no_crash(self, run_cfg):
        df = pd.DataFrame({"a": [1]})
        plotting.plot_centered_vs_causal_comparison(df, df, ["nonexistent"], run_cfg)


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


# ── Plot Reuse Infrastructure ────────────────────────────────────────────────


class TestPlotIsFresh:
    """Tests for _plot_is_fresh() freshness check."""

    def test_missing_png_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        assert plotting._plot_is_fresh("nonexistent.png") is False

    def test_existing_png_no_checkpoint_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        (tmp_path / "test.png").write_bytes(b"PNG")
        assert plotting._plot_is_fresh("test.png") is True

    def test_existing_png_no_checkpoint_name_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        (tmp_path / "test.png").write_bytes(b"PNG")
        assert plotting._plot_is_fresh("test.png", checkpoint_name=None) is True

    def test_png_newer_than_checkpoint_is_fresh(self, tmp_path, monkeypatch):
        import json, time
        from datetime import datetime, timedelta

        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        monkeypatch.setattr("trading_crab_lib.checkpoints.CHECKPOINT_DIR", ckpt_dir)

        # Create checkpoint meta with a timestamp in the past
        meta = {"created": (datetime.now() - timedelta(hours=2)).isoformat()}
        (ckpt_dir / "features.meta.json").write_text(json.dumps(meta))

        # Create PNG (mtime = now, which is after checkpoint)
        (tmp_path / "test.png").write_bytes(b"PNG")

        assert plotting._plot_is_fresh("test.png", checkpoint_name="features") is True

    def test_png_older_than_checkpoint_is_stale(self, tmp_path, monkeypatch):
        import json, os, time
        from datetime import datetime, timedelta

        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        monkeypatch.setattr("trading_crab_lib.checkpoints.CHECKPOINT_DIR", ckpt_dir)

        # Create PNG with old mtime
        png = tmp_path / "test.png"
        png.write_bytes(b"PNG")
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(png, (old_time, old_time))

        # Create checkpoint meta with a recent timestamp
        meta = {"created": datetime.now().isoformat()}
        (ckpt_dir / "features.meta.json").write_text(json.dumps(meta))

        assert plotting._plot_is_fresh("test.png", checkpoint_name="features") is False

    def test_missing_checkpoint_meta_treated_as_fresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        monkeypatch.setattr("trading_crab_lib.checkpoints.CHECKPOINT_DIR", ckpt_dir)

        (tmp_path / "test.png").write_bytes(b"PNG")
        # No meta.json file exists
        assert plotting._plot_is_fresh("test.png", checkpoint_name="features") is True

    def test_corrupt_checkpoint_meta_treated_as_fresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        monkeypatch.setattr("trading_crab_lib.checkpoints.CHECKPOINT_DIR", ckpt_dir)

        (tmp_path / "test.png").write_bytes(b"PNG")
        (ckpt_dir / "features.meta.json").write_text("not valid json{{{")

        assert plotting._plot_is_fresh("test.png", checkpoint_name="features") is True


class TestLoadOrGenerate:
    """Tests for load_or_generate() plot caching."""

    def _dummy_plot(self, run_cfg, *, filename="dummy.png"):
        """Minimal plot function matching the standard signature."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        plotting._save_or_show(fig, filename, run_cfg)

    def test_regenerates_when_no_cached_png(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        cfg = RunConfig(save_plots=True, show_plots=False)

        plotting.load_or_generate(
            self._dummy_plot,
            filename="dummy.png",
            run_cfg=cfg,
        )
        assert (tmp_path / "dummy.png").exists()

    def test_uses_cache_when_fresh(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        # Pre-create a cached PNG
        (tmp_path / "dummy.png").write_bytes(b"CACHED_PNG")
        cfg = RunConfig(save_plots=True, show_plots=False)

        call_count = 0
        original = self._dummy_plot

        def counting_plot(*a, **kw):
            nonlocal call_count
            call_count += 1
            return original(*a, **kw)

        # _in_jupyter() is False in test env, so it should log and return
        plotting.load_or_generate(
            counting_plot,
            filename="dummy.png",
            run_cfg=cfg,
        )
        # Plot function should NOT have been called
        assert call_count == 0
        # Original cached content should be unchanged
        assert (tmp_path / "dummy.png").read_bytes() == b"CACHED_PNG"

    def test_regenerates_when_stale(self, tmp_path, monkeypatch):
        import json, os, time
        from datetime import datetime

        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        monkeypatch.setattr("trading_crab_lib.checkpoints.CHECKPOINT_DIR", ckpt_dir)

        # Create old PNG
        png = tmp_path / "dummy.png"
        png.write_bytes(b"OLD")
        old_time = time.time() - 7200
        os.utime(png, (old_time, old_time))

        # Create newer checkpoint
        meta = {"created": datetime.now().isoformat()}
        (ckpt_dir / "features.meta.json").write_text(json.dumps(meta))

        cfg = RunConfig(save_plots=True, show_plots=False)
        plotting.load_or_generate(
            self._dummy_plot,
            filename="dummy.png",
            run_cfg=cfg,
            checkpoint_name="features",
        )
        # PNG should be regenerated (larger than "OLD")
        assert (tmp_path / "dummy.png").stat().st_size > 3

    def test_forwards_extra_kwargs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        cfg = RunConfig(save_plots=True, show_plots=False)

        received = {}

        def capture_plot(*args, run_cfg, filename="x.png", title="default"):
            received["title"] = title

        plotting.load_or_generate(
            capture_plot,
            filename="capture.png",
            run_cfg=cfg,
            title="Custom Title",
        )
        assert received["title"] == "Custom Title"


class TestListAvailablePlots:
    """Tests for list_available_plots() utility."""

    def test_empty_directory(self, tmp_path):
        result = plotting.list_available_plots(tmp_path)
        assert "No plots found" in result

    def test_nonexistent_directory(self, tmp_path):
        result = plotting.list_available_plots(tmp_path / "nonexistent")
        assert "No plots directory found" in result

    def test_lists_png_files(self, tmp_path):
        (tmp_path / "01_raw_coverage.png").write_bytes(b"x" * 2048)
        (tmp_path / "03_pca_scatter.png").write_bytes(b"x" * 1024)
        (tmp_path / "not_a_png.txt").write_text("ignore me")

        result = plotting.list_available_plots(tmp_path)
        assert "01_raw_coverage.png" in result
        assert "03_pca_scatter.png" in result
        assert "not_a_png.txt" not in result
        assert "Total: 2 plots" in result

    def test_shows_file_sizes(self, tmp_path):
        (tmp_path / "small.png").write_bytes(b"x" * 512)
        result = plotting.list_available_plots(tmp_path)
        assert "KB" in result

    def test_shows_modification_times(self, tmp_path):
        (tmp_path / "test.png").write_bytes(b"x" * 100)
        result = plotting.list_available_plots(tmp_path)
        # Should contain a date-like string
        assert "202" in result  # year prefix

    def test_default_plot_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(plotting, "PLOT_DIR", tmp_path)
        (tmp_path / "test.png").write_bytes(b"PNG")
        result = plotting.list_available_plots()
        assert "test.png" in result
        assert "Total: 1 plots" in result

    def test_large_file_shows_mb(self, tmp_path):
        (tmp_path / "big.png").write_bytes(b"x" * (2 * 1024 * 1024))
        result = plotting.list_available_plots(tmp_path)
        assert "MB" in result
