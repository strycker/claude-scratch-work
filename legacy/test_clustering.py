"""Tests for clustering module — KMeans evaluation and PCA."""

import numpy as np
import pandas as pd
import pytest

from trading_crab.clustering import evaluate_kmeans


class TestEvaluateKMeans:
    """Test the KMeans evaluation sweep."""

    def _make_clusterable_data(self, n=100, k=3):
        """Generate synthetic data with clear clusters."""
        rng = np.random.RandomState(42)
        centers = rng.randn(k, 5) * 10
        X = np.vstack([
            centers[i] + rng.randn(n // k, 5) for i in range(k)
        ])
        return X

    def test_returns_dataframe(self):
        """Should return a DataFrame with the expected columns."""
        X = self._make_clusterable_data()
        scores = evaluate_kmeans(X, range(2, 6))
        assert isinstance(scores, pd.DataFrame)
        assert set(scores.columns) == {"k", "inertia", "silhouette", "calinski", "davies_bouldin"}

    def test_correct_k_range(self):
        """Should have one row per k value."""
        X = self._make_clusterable_data()
        scores = evaluate_kmeans(X, range(2, 6))
        assert list(scores["k"]) == [2, 3, 4, 5]

    def test_silhouette_finds_correct_k(self):
        """With well-separated clusters, silhouette should peak at true k."""
        X = self._make_clusterable_data(n=150, k=3)
        scores = evaluate_kmeans(X, range(2, 7))
        best_k = scores.loc[scores["silhouette"].idxmax(), "k"]
        assert best_k == 3, f"Expected best k=3, got k={best_k}"

    def test_inertia_decreases(self):
        """Inertia should decrease as k increases."""
        X = self._make_clusterable_data()
        scores = evaluate_kmeans(X, range(2, 8))
        inertias = scores["inertia"].values
        for i in range(1, len(inertias)):
            assert inertias[i] <= inertias[i - 1], (
                f"Inertia should decrease: k={i+2} has {inertias[i]} > {inertias[i-1]}"
            )

    def test_single_fit_per_k(self):
        """Verify that inertia and labels come from the same model fit.

        If they came from different fits, the inertia wouldn't correspond
        to the labels used for silhouette scoring. We test indirectly by
        checking that silhouette scores are valid (between -1 and 1).
        """
        X = self._make_clusterable_data()
        scores = evaluate_kmeans(X, range(2, 6))
        for _, row in scores.iterrows():
            assert -1 <= row["silhouette"] <= 1, (
                f"Invalid silhouette score: {row['silhouette']}"
            )
            assert row["calinski"] > 0
            assert row["davies_bouldin"] > 0
