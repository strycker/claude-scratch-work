"""Monitoring helpers for feature engineering quality (C1.4)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FeatureQualityReport:
    """Summary of feature DataFrame quality metrics."""

    n_rows: int = 0
    n_cols: int = 0
    nan_counts: dict[str, int] = field(default_factory=dict)
    top_nan_columns: list[tuple[str, int]] = field(default_factory=list)
    top_variance_columns: list[tuple[str, float]] = field(default_factory=list)
    top_correlation_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted summary of NaN counts, variance, and correlations."""
        lines = [
            f"  Feature quality: {self.n_rows} rows × {self.n_cols} columns",
        ]

        # NaN summary
        total_nans = sum(self.nan_counts.values())
        cols_with_nans = sum(1 for v in self.nan_counts.values() if v > 0)
        lines.append(f"  NaN cells: {total_nans} across {cols_with_nans} columns")
        if self.top_nan_columns:
            lines.append("  Top-5 NaN columns:")
            max_name = max(len(c) for c, _ in self.top_nan_columns)
            for col, cnt in self.top_nan_columns[:5]:
                pct = cnt / self.n_rows * 100 if self.n_rows else 0
                lines.append(f"    {col:<{max_name}}  {cnt:4d} ({pct:5.1f}%)")

        # Variance summary
        if self.top_variance_columns:
            lines.append("  Top-5 highest-variance features:")
            max_name = max(len(c) for c, _ in self.top_variance_columns)
            for col, var in self.top_variance_columns[:5]:
                lines.append(f"    {col:<{max_name}}  var={var:.4f}")

        # Correlation summary
        if self.top_correlation_pairs:
            lines.append("  Top-5 highest-correlation pairs:")
            for col_a, col_b, corr in self.top_correlation_pairs[:5]:
                lines.append(f"    {col_a} × {col_b}: {corr:.3f}")

        return "\n".join(lines)


def compute_feature_quality(df: pd.DataFrame) -> FeatureQualityReport:
    """Compute feature quality metrics for a feature DataFrame.

    Parameters
    ----------
    df :
        Feature DataFrame (output of engineer_all). The ``market_code``
        column is excluded from analysis if present.

    Returns
    -------
    FeatureQualityReport with NaN counts, variance ranking, and correlation pairs.
    """
    feat = df.drop(columns=["market_code"], errors="ignore")
    # Only numeric columns
    feat = feat.select_dtypes(include=[np.number])

    report = FeatureQualityReport(n_rows=len(feat), n_cols=len(feat.columns))

    # NaN counts per column
    nan_counts = feat.isna().sum()
    report.nan_counts = nan_counts.to_dict()
    top_nan = nan_counts[nan_counts > 0].sort_values(ascending=False)
    report.top_nan_columns = [(col, int(cnt)) for col, cnt in top_nan.head(5).items()]

    # Variance ranking (on non-NaN data)
    variances = feat.var(skipna=True).sort_values(ascending=False)
    report.top_variance_columns = [
        (col, float(v)) for col, v in variances.head(5).items()
    ]

    # Top correlation pairs (absolute value, excluding self-correlations)
    if len(feat.columns) > 1 and len(feat) > 2:
        corr_matrix = feat.corr(min_periods=3)
        pairs: list[tuple[str, str, float]] = []
        cols = list(corr_matrix.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if pd.notna(val):
                    pairs.append((cols[i], cols[j], float(val)))
        pairs.sort(key=lambda t: abs(t[2]), reverse=True)
        report.top_correlation_pairs = pairs[:5]

    log.info("Feature quality report:\n%s", report.summary())
    return report
