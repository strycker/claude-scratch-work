"""
Pipeline monitoring helpers — validation summaries, diagnostic reports, QA gates.

Phase C1: Steps 1-2 monitoring (ingestion + feature engineering).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── C1.1: Enhanced completeness report formatting ─────────────────────────


def format_completeness_table(report) -> str:
    """Format a CompletenessReport as a readable table for logging.

    Enhances the existing report.summary() with a per-column NaN table
    showing the worst offenders.
    """
    lines: list[str] = [report.summary()]

    # Show top-10 highest NaN-fraction columns (even if below threshold)
    if report.nan_fractions:
        sorted_nans = sorted(
            report.nan_fractions.items(), key=lambda x: x[1], reverse=True
        )
        top_nan = [(col, frac) for col, frac in sorted_nans if frac > 0][:10]
        if top_nan:
            lines.append("  Top NaN columns:")
            max_name = max(len(col) for col, _ in top_nan)
            for col, frac in top_nan:
                bar = "#" * int(frac * 20)
                lines.append(f"    {col:<{max_name}}  {frac:5.1%}  {bar}")

    return "\n".join(lines)


# ── C1.2: Date-range validation ───────────────────────────────────────────


@dataclass
class DateRangeReport:
    """Result of a date-range validation check."""

    last_date: pd.Timestamp | None = None
    current_quarter_end: pd.Timestamp | None = None
    quarters_behind: int = 0
    stale_series: list[tuple[str, pd.Timestamp]] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        lines: list[str] = []
        if self.last_date is not None:
            lines.append(
                f"  Date range: last data = {self.last_date.strftime('%Y-%m-%d')}, "
                f"current quarter end = {self.current_quarter_end.strftime('%Y-%m-%d')}"
            )
        if self.quarters_behind > 0:
            lines.append(
                f"  WARNING: data is {self.quarters_behind} quarter(s) behind current date"
            )
        if self.stale_series:
            lines.append(f"  Stale series ({len(self.stale_series)}):")
            for col, last in self.stale_series[:10]:
                lines.append(f"    {col}: last value at {last.strftime('%Y-%m-%d')}")
        if self.passed:
            lines.append("  Date range: OK")
        else:
            lines.append("  Date range: STALE")
        return "\n".join(lines)


def validate_date_range(
    df: pd.DataFrame,
    *,
    reference_date: datetime | None = None,
    stale_threshold_quarters: int = 2,
) -> DateRangeReport:
    """Check whether the DataFrame extends to the current quarter.

    Parameters
    ----------
    df :
        Time-indexed DataFrame (e.g. macro_raw).
    reference_date :
        Date to compare against. Defaults to today.
    stale_threshold_quarters :
        Number of quarters a per-column last-valid date can lag behind
        the DataFrame's last index date before being flagged as stale.

    Returns
    -------
    DateRangeReport with details on data freshness.
    """
    report = DateRangeReport()

    if df.empty or not hasattr(df.index, "max"):
        report.passed = False
        return report

    ref = pd.Timestamp(reference_date or datetime.now())
    # Current quarter end
    current_qe = pd.Timestamp(ref).to_period("Q").end_time.normalize()
    report.current_quarter_end = current_qe

    last_date = df.index.max()
    report.last_date = last_date

    # How many quarters behind?
    if last_date < current_qe:
        diff_days = (current_qe - last_date).days
        report.quarters_behind = max(1, int(diff_days / 91))

    # Per-column staleness: find columns whose last non-NaN value is
    # significantly older than the DataFrame's last row
    last_idx = df.index.max()
    threshold_date = last_idx - pd.DateOffset(months=3 * stale_threshold_quarters)
    feature_cols = [c for c in df.columns if c != "market_code"]
    for col in feature_cols:
        valid = df[col].dropna()
        if valid.empty:
            report.stale_series.append((col, pd.NaT))
        elif valid.index.max() < threshold_date:
            report.stale_series.append((col, valid.index.max()))

    # Pass if data reaches within 1 quarter of current date and no stale series
    report.passed = report.quarters_behind <= 1 and len(report.stale_series) == 0

    if report.passed:
        log.info("Date-range validation: OK (last=%s)", last_date.strftime("%Y-%m-%d"))
    else:
        log.warning(
            "Date-range validation: data may be stale (%d quarters behind, %d stale series)",
            report.quarters_behind,
            len(report.stale_series),
        )

    return report


# ── C1.3: Per-source row count summary ────────────────────────────────────


@dataclass
class SourceRowCounts:
    """Row count breakdown by data source."""

    multpl: int = 0
    fred: int = 0
    macrotrends: int = 0
    etf: int = 0
    other: int = 0
    total_columns: int = 0

    def summary(self) -> str:
        lines = [
            "  Per-source column counts:",
            f"    multpl.com:    {self.multpl:3d} columns",
            f"    FRED API:      {self.fred:3d} columns",
            f"    macrotrends:   {self.macrotrends:3d} columns",
            f"    ETF prices:    {self.etf:3d} columns",
            f"    other/derived: {self.other:3d} columns",
            f"    TOTAL:         {self.total_columns:3d} columns",
        ]
        return "\n".join(lines)


def count_source_columns(df: pd.DataFrame, cfg: dict) -> SourceRowCounts:
    """Count columns in the DataFrame grouped by data source.

    Uses config to identify which columns come from which source.
    """
    counts = SourceRowCounts(total_columns=len(df.columns))

    # FRED columns
    fred_names = set()
    for series_id, meta in cfg.get("fred", {}).get("series", {}).items():
        fred_names.add(meta.get("name", series_id.lower()))
    counts.fred = sum(1 for c in df.columns if c in fred_names)

    # multpl columns
    multpl_names = set()
    for ds in cfg.get("multpl", {}).get("datasets", []):
        name = ds[0] if isinstance(ds, list) else ds["name"]
        multpl_names.add(name)
    counts.multpl = sum(1 for c in df.columns if c in multpl_names)

    # macrotrends columns
    mt_names = set()
    for ds in cfg.get("macrotrends", {}).get("series", []):
        name = ds["name"] if isinstance(ds, dict) else ds[0]
        mt_names.add(name)
    counts.macrotrends = sum(1 for c in df.columns if c in mt_names)

    # ETF columns
    counts.etf = sum(1 for c in df.columns if c.startswith("etf_"))

    # Other (market_code, derived, etc.)
    known = fred_names | multpl_names | mt_names | {c for c in df.columns if c.startswith("etf_")}
    counts.other = sum(1 for c in df.columns if c not in known)

    return counts


# ── C1.4: Feature quality report ──────────────────────────────────────────


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
