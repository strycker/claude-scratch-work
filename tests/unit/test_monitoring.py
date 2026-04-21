"""Tests for pipeline monitoring helpers (Phases C1 + C2)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.monitoring import (
    format_completeness_table,
    validate_date_range,
    count_source_columns,
    compute_feature_quality,
    format_method_comparison,
    compute_regime_stability,
    RegimeStabilityReport,
    compute_cv_fold_scores,
    CVFoldReport,
    check_regime_probabilities,
    format_tactics_summary,
    validate_step_output,
    PipelineHealthSummary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def quarterly_df():
    """20-quarter DataFrame with 3 columns."""
    idx = pd.date_range("2000-03-31", periods=20, freq="QE")
    return pd.DataFrame(
        {"col_a": range(20), "col_b": np.random.default_rng(42).random(20), "col_c": range(20)},
        index=idx,
    )


@pytest.fixture
def sample_cfg():
    """Minimal config with FRED + multpl + macrotrends + ETF keys."""
    return {
        "fred": {
            "series": {
                "GDP": {"name": "fred_gdp", "shift": True},
                "CPIAUCSL": {"name": "fred_cpi", "shift": False},
            }
        },
        "multpl": {
            "datasets": [
                ["sp500", "S&P 500", "https://example.com", "number"],
                ["us_cpi", "US CPI", "https://example.com", "number"],
            ]
        },
        "macrotrends": {
            "series": [
                {"name": "gold_spot", "url": "https://example.com"},
            ]
        },
        "features": {
            "asset_price_columns": ["SPY", "TLT"],
        },
    }


# ── C1.1: format_completeness_table ──────────────────────────────────────


class TestFormatCompletenessTable:
    def test_pass_report_formatting(self, quarterly_df):
        from trading_crab_lib.ingestion import ingestion_completeness_report

        report = ingestion_completeness_report(
            quarterly_df, expected_columns=["col_a", "col_b", "col_c"]
        )
        text = format_completeness_table(report)
        assert "PASS" in text
        assert "3/3" in text

    def test_fail_report_shows_nan_table(self):
        from trading_crab_lib.ingestion import ingestion_completeness_report

        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame(
            {
                "good": range(10),
                "bad": [np.nan] * 6 + [1.0, 2.0, 3.0, 4.0],
            },
            index=idx,
        )
        report = ingestion_completeness_report(df, nan_threshold=0.5)
        text = format_completeness_table(report)
        assert "FAIL" in text
        assert "Top NaN columns" in text
        assert "bad" in text

    def test_no_nan_columns_no_table(self, quarterly_df):
        from trading_crab_lib.ingestion import ingestion_completeness_report

        report = ingestion_completeness_report(quarterly_df)
        text = format_completeness_table(report)
        assert "Top NaN columns" not in text


# ── C1.2: validate_date_range ─────────────────────────────────────────────


class TestValidateDateRange:
    def test_current_data_passes(self):
        idx = pd.date_range("2020-03-31", periods=25, freq="QE")
        df = pd.DataFrame({"x": range(25)}, index=idx)
        # Use a reference date that's within the data range
        ref = idx[-1] + pd.DateOffset(days=30)
        report = validate_date_range(df, reference_date=ref)
        assert report.passed
        assert report.quarters_behind <= 1

    def test_stale_data_warns(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        # Reference 10 years after data ends
        ref = datetime(2012, 6, 30)
        report = validate_date_range(df, reference_date=ref)
        assert not report.passed
        assert report.quarters_behind > 1

    def test_empty_df_fails(self):
        df = pd.DataFrame()
        report = validate_date_range(df)
        assert not report.passed

    def test_stale_series_detected(self):
        idx = pd.date_range("2020-03-31", periods=20, freq="QE")
        df = pd.DataFrame(
            {
                "fresh": range(20),
                "stale": [1.0] * 5 + [np.nan] * 15,
            },
            index=idx,
        )
        report = validate_date_range(
            df,
            reference_date=idx[-1],
            stale_threshold_quarters=2,
        )
        assert any("stale" == col for col, _ in report.stale_series)

    def test_summary_format(self):
        idx = pd.date_range("2020-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        report = validate_date_range(df, reference_date=idx[-1])
        text = report.summary()
        assert "Date range" in text

    def test_quarters_behind_calculation(self):
        idx = pd.date_range("2020-03-31", periods=4, freq="QE")
        df = pd.DataFrame({"x": range(4)}, index=idx)
        # Reference 2 years after last data point
        ref = datetime(2023, 3, 31)
        report = validate_date_range(df, reference_date=ref)
        assert report.quarters_behind >= 2


# ── C1.3: count_source_columns ────────────────────────────────────────────


class TestCountSourceColumns:
    def test_identifies_fred_columns(self, sample_cfg):
        df = pd.DataFrame(
            {"fred_gdp": [1], "fred_cpi": [2], "sp500": [3]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        assert counts.fred == 2
        assert counts.multpl == 1
        assert counts.total_columns == 3

    def test_identifies_etf_columns(self, sample_cfg):
        df = pd.DataFrame(
            {"etf_spy": [1], "etf_tlt": [2], "fred_gdp": [3]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        assert counts.etf == 2
        assert counts.fred == 1

    def test_identifies_macrotrends_columns(self, sample_cfg):
        df = pd.DataFrame(
            {"gold_spot": [1], "sp500": [2]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        assert counts.macrotrends == 1
        assert counts.multpl == 1

    def test_other_columns_counted(self, sample_cfg):
        df = pd.DataFrame(
            {"market_code": [1], "unknown_col": [2], "fred_gdp": [3]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        assert counts.other == 2  # market_code + unknown_col
        assert counts.fred == 1

    def test_summary_format(self, sample_cfg):
        df = pd.DataFrame(
            {"fred_gdp": [1], "sp500": [2]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, sample_cfg)
        text = counts.summary()
        assert "FRED API" in text
        assert "multpl.com" in text
        assert "TOTAL" in text

    def test_empty_config(self):
        df = pd.DataFrame(
            {"a": [1], "b": [2]},
            index=pd.date_range("2020-03-31", periods=1, freq="QE"),
        )
        counts = count_source_columns(df, {})
        assert counts.other == 2
        assert counts.fred == 0


# ── C1.4: compute_feature_quality ─────────────────────────────────────────


class TestComputeFeatureQuality:
    def test_basic_quality_report(self):
        idx = pd.date_range("2000-03-31", periods=20, freq="QE")
        df = pd.DataFrame(
            {
                "feat_a": np.random.default_rng(0).random(20),
                "feat_b": np.random.default_rng(1).random(20) * 100,
                "feat_c": np.random.default_rng(2).random(20),
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        assert report.n_rows == 20
        assert report.n_cols == 3
        assert len(report.top_variance_columns) == 3
        assert len(report.top_correlation_pairs) <= 3

    def test_nan_detection(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame(
            {
                "clean": range(10),
                "dirty": [np.nan, np.nan, np.nan] + list(range(7)),
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        assert report.nan_counts["dirty"] == 3
        assert report.nan_counts["clean"] == 0
        assert report.top_nan_columns[0][0] == "dirty"

    def test_market_code_excluded(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame(
            {
                "feat": range(10),
                "market_code": [0] * 10,
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        assert report.n_cols == 1
        assert "market_code" not in report.nan_counts

    def test_variance_ranking(self):
        idx = pd.date_range("2000-03-31", periods=50, freq="QE")
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "low_var": rng.random(50) * 0.01,
                "high_var": rng.random(50) * 1000,
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        assert report.top_variance_columns[0][0] == "high_var"

    def test_correlation_pairs(self):
        idx = pd.date_range("2000-03-31", periods=50, freq="QE")
        x = np.arange(50, dtype=float)
        df = pd.DataFrame(
            {
                "a": x,
                "b": x * 2 + 1,  # perfectly correlated with a
                "c": np.random.default_rng(0).random(50),  # uncorrelated
            },
            index=idx,
        )
        report = compute_feature_quality(df)
        # a and b should be the most correlated pair
        top_pair = report.top_correlation_pairs[0]
        assert {top_pair[0], top_pair[1]} == {"a", "b"}
        assert abs(top_pair[2]) > 0.99

    def test_summary_string(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"x": range(10), "y": range(10)}, index=idx)
        report = compute_feature_quality(df)
        text = report.summary()
        assert "Feature quality" in text
        assert "NaN cells" in text
        assert "Top-5 highest-variance" in text

    def test_single_column(self):
        idx = pd.date_range("2000-03-31", periods=10, freq="QE")
        df = pd.DataFrame({"x": range(10)}, index=idx)
        report = compute_feature_quality(df)
        assert report.n_cols == 1
        assert not report.top_correlation_pairs

    def test_empty_df(self):
        df = pd.DataFrame()
        report = compute_feature_quality(df)
        assert report.n_rows == 0
        assert report.n_cols == 0


# ── C2.3: format_method_comparison ───────────────────────────────────────


class TestFormatMethodComparison:
    def test_basic_formatting(self):
        df = pd.DataFrame({
            "method": ["KMeans", "GMM"],
            "n_clusters": [5, 4],
            "silhouette": [0.35, 0.28],
            "davies_bouldin": [1.2, 1.5],
            "calinski": [120.5, 98.3],
        })
        text = format_method_comparison(df)
        assert "Clustering method comparison" in text
        assert "KMeans" in text
        assert "GMM" in text
        assert "Best by silhouette: KMeans" in text

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        text = format_method_comparison(df)
        assert "no methods to compare" in text

    def test_single_method(self):
        df = pd.DataFrame({
            "method": ["KMeans"],
            "n_clusters": [5],
            "silhouette": [0.30],
            "davies_bouldin": [1.1],
            "calinski": [100.0],
        })
        text = format_method_comparison(df)
        assert "Best by silhouette: KMeans" in text

    def test_best_method_identified(self):
        df = pd.DataFrame({
            "method": ["Bad", "Good", "Medium"],
            "n_clusters": [3, 4, 5],
            "silhouette": [0.10, 0.50, 0.30],
            "davies_bouldin": [2.0, 1.0, 1.5],
            "calinski": [50.0, 200.0, 100.0],
        })
        text = format_method_comparison(df)
        assert "Best by silhouette: Good" in text


# ── C2.4: compute_regime_stability ───────────────────────────────────────


class TestComputeRegimeStability:
    @pytest.fixture
    def transition_matrix(self):
        """3-regime transition matrix with known persistence."""
        return pd.DataFrame(
            [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.15, 0.15, 0.7]],
            index=[0, 1, 2],
            columns=[0, 1, 2],
        )

    @pytest.fixture
    def regime_labels(self):
        """Labels: 0,0,0,1,1,2,2,2,2,0,0"""
        return pd.Series(
            [0, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0],
            index=pd.date_range("2000-03-31", periods=11, freq="QE"),
        )

    def test_persistence_extracted(self, transition_matrix, regime_labels):
        report = compute_regime_stability(transition_matrix, regime_labels)
        assert report.persistence[0] == pytest.approx(0.8)
        assert report.persistence[1] == pytest.approx(0.6)
        assert report.persistence[2] == pytest.approx(0.7)

    def test_most_least_stable(self, transition_matrix, regime_labels):
        report = compute_regime_stability(transition_matrix, regime_labels)
        assert report.most_stable[0] == 0
        assert report.most_stable[1] == pytest.approx(0.8)
        assert report.least_stable[0] == 1
        assert report.least_stable[1] == pytest.approx(0.6)

    def test_avg_duration(self, transition_matrix, regime_labels):
        report = compute_regime_stability(transition_matrix, regime_labels)
        # Regime 0: runs of 3 and 2 → avg 2.5
        assert report.avg_duration[0] == pytest.approx(2.5)
        # Regime 1: run of 2 → avg 2.0
        assert report.avg_duration[1] == pytest.approx(2.0)
        # Regime 2: run of 4 → avg 4.0
        assert report.avg_duration[2] == pytest.approx(4.0)

    def test_summary_format(self, transition_matrix, regime_labels):
        report = compute_regime_stability(transition_matrix, regime_labels)
        text = report.summary()
        assert "Regime stability" in text
        assert "persist=" in text
        assert "avg_run=" in text
        assert "Most stable" in text
        assert "Least stable" in text

    def test_empty_labels(self, transition_matrix):
        labels = pd.Series([], dtype=int)
        report = compute_regime_stability(transition_matrix, labels)
        assert report.persistence[0] == pytest.approx(0.8)
        assert not report.avg_duration

    def test_empty_persistence_summary(self):
        report = RegimeStabilityReport()
        text = report.summary()
        assert "no persistence data" in text


# ── C3.1: compute_cv_fold_scores + CVFoldReport ─────────────────────────


_has_sklearn = True
try:
    import sklearn  # noqa: F401  # pylint: disable=unused-import
except ImportError:
    _has_sklearn = False


@pytest.mark.skipif(not _has_sklearn, reason="sklearn not installed")
class TestComputeCVFoldScores:
    def test_returns_correct_number_of_folds(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(42)
        feat = pd.DataFrame(rng.random((50, 5)), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(rng.integers(0, 3, 50))
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(feat, y)
        scores = compute_cv_fold_scores(model, feat, y, n_splits=3)
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_does_not_mutate_original_model(self):
        from sklearn.tree import DecisionTreeClassifier
        rng = np.random.default_rng(0)
        feat = pd.DataFrame(rng.random((30, 3)), columns=["a", "b", "c"])
        y = pd.Series(rng.integers(0, 2, 30))
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        model.fit(feat, y)
        orig_pred = model.predict(feat).copy()
        compute_cv_fold_scores(model, feat, y, n_splits=3)
        np.testing.assert_array_equal(model.predict(feat), orig_pred)


class TestCVFoldReport:
    def test_add_and_summary(self):
        report = CVFoldReport()
        report.add("RF", [0.5, 0.6, 0.7])
        report.add("DT", [0.4, 0.5, 0.6])
        text = report.summary()
        assert "RF" in text
        assert "DT" in text
        assert "mean=" in text

    def test_empty_report(self):
        report = CVFoldReport()
        text = report.summary()
        assert "Cross-validation" in text


# ── C3.5: check_regime_probabilities ─────────────────────────────────────


class TestCheckRegimeProbabilities:
    def test_all_ok(self):
        probs = {0: 0.3, 1: 0.4, 2: 0.3}
        warnings = check_regime_probabilities(probs)
        assert not warnings

    def test_low_probability_detected(self):
        probs = {0: 0.9, 1: 0.08, 2: 0.02}
        warnings = check_regime_probabilities(probs)
        assert len(warnings) == 1
        assert "Regime 2" in warnings[0]

    def test_custom_threshold(self):
        probs = {0: 0.5, 1: 0.4, 2: 0.1}
        warnings = check_regime_probabilities(probs, min_threshold=0.15)
        assert len(warnings) == 1
        assert "Regime 2" in warnings[0]

    def test_multiple_warnings(self):
        probs = {0: 0.9, 1: 0.04, 2: 0.03, 3: 0.03}
        warnings = check_regime_probabilities(probs)
        assert len(warnings) == 3

    def test_exact_threshold_passes(self):
        probs = {0: 0.5, 1: 0.45, 2: 0.05}
        warnings = check_regime_probabilities(probs)
        assert not warnings


# ── C4.2: format_tactics_summary ─────────────────────────────────────────


class TestFormatTacticsSummary:
    def test_basic_summary(self):
        df = pd.DataFrame({
            "ticker": ["SPY", "TLT", "GLD", "VNQ"],
            "classification": ["buy_hold", "swing", "swing", "stand_aside"],
        })
        text = format_tactics_summary(df)
        assert "buy_hold" in text
        assert "swing" in text
        assert "stand_aside" in text
        assert "TOTAL" in text
        assert "4 assets" in text

    def test_empty_df(self):
        df = pd.DataFrame()
        text = format_tactics_summary(df)
        assert "no tactics data" in text

    def test_missing_classification_column(self):
        df = pd.DataFrame({"ticker": ["SPY"], "signal": ["green"]})
        text = format_tactics_summary(df)
        assert "no tactics data" in text

    def test_all_one_category(self):
        df = pd.DataFrame({
            "classification": ["buy_hold"] * 5,
        })
        text = format_tactics_summary(df)
        assert "5 assets" in text


# ── C4.3: validate_step_output ───────────────────────────────────────────


class TestValidateStepOutput:
    def test_valid_output(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        result = validate_step_output(1, {"features": df})
        assert result.passed
        assert result.step_num == 1

    def test_missing_output(self):
        result = validate_step_output(2, {"missing": None})
        assert not result.passed

    def test_empty_output(self):
        result = validate_step_output(3, {"empty": pd.DataFrame()})
        assert not result.passed

    def test_high_nan_detected(self):
        df = pd.DataFrame({
            "clean": [1, 2, 3, 4, 5],
            "dirty": [np.nan, np.nan, np.nan, np.nan, 5],
        })
        result = validate_step_output(1, {"data": df}, max_nan_pct=0.5)
        # dirty is 80% NaN, should fail
        nan_checks = [c for c in result.checks if "NaN" in c[0]]
        assert any(not ok for _, ok, _ in nan_checks)

    def test_summary_format(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = validate_step_output(5, {"output": df})
        text = result.summary()
        assert "Step 5" in text
        assert "OK" in text

    def test_multiple_outputs(self):
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3.0, 4.0]})
        result = validate_step_output(1, {"first": df1, "second": df2})
        assert result.passed
        assert len(result.checks) >= 4  # shape + NaN for each


# ── C4.4 + C4.5: PipelineHealthSummary ───────────────────────────────────


class TestPipelineHealthSummary:
    def test_record_and_summary(self):
        health = PipelineHealthSummary()
        health.record_step(1, 2.5)
        health.record_step(2, 5.0)
        health.record_step(3, 1.0, failed=True)
        text = health.summary()
        assert "Step 1" in text
        assert "2.5s" in text
        assert "FAIL" in text
        assert "TOTAL" in text

    def test_steps_tracking(self):
        health = PipelineHealthSummary()
        health.record_step(1, 1.0)
        health.record_step(2, 2.0)
        assert health.steps_run == [1, 2]
        assert not health.steps_failed

    def test_failed_steps(self):
        health = PipelineHealthSummary()
        health.record_step(1, 1.0)
        health.record_step(2, 0.5, failed=True)
        assert 2 in health.steps_failed
        assert 1 not in health.steps_failed

    def test_empty_summary(self):
        health = PipelineHealthSummary()
        text = health.summary()
        assert "Pipeline Health Summary" in text
        assert "TOTAL" in text
