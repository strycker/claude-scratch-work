"""
Data ingestion package — FRED, multpl.com, macrotrends.net, yfinance.

Provides ingestion_completeness_report() to validate that a combined
DataFrame has the expected number of columns and no excessive NaN gaps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class CompletenessReport:
    """Result of an ingestion completeness check."""
    total_columns: int = 0
    expected_columns: int = 0
    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    high_nan_columns: list[str] = field(default_factory=list)
    nan_fractions: dict[str, float] = field(default_factory=dict)
    passed: bool = True

    def summary(self) -> str:
        lines = [
            f"Ingestion completeness: {self.total_columns}/{self.expected_columns} columns",
        ]
        if self.missing_columns:
            lines.append(f"  MISSING ({len(self.missing_columns)}): {self.missing_columns[:10]}")
        if self.high_nan_columns:
            lines.append(f"  HIGH NaN ({len(self.high_nan_columns)}): {self.high_nan_columns[:10]}")
        if self.passed:
            lines.append("  STATUS: PASS")
        else:
            lines.append("  STATUS: FAIL")
        return "\n".join(lines)


def ingestion_completeness_report(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
    nan_threshold: float = 0.5,
) -> CompletenessReport:
    """
    Check ingestion output for missing columns and excessive NaN gaps.

    Args:
        df:               Combined ingested DataFrame (macro_raw).
        expected_columns: List of column names that should be present.
                          If None, only NaN checks are performed.
        nan_threshold:    Fraction of NaN above which a column is flagged
                          (default 0.5 = 50%).

    Returns:
        CompletenessReport with details on missing/extra/high-NaN columns.
    """
    report = CompletenessReport()
    report.total_columns = len(df.columns)

    if expected_columns is not None:
        report.expected_columns = len(expected_columns)
        expected_set = set(expected_columns)
        actual_set = set(df.columns)
        report.missing_columns = sorted(expected_set - actual_set)
        report.extra_columns = sorted(actual_set - expected_set)
    else:
        report.expected_columns = report.total_columns

    # NaN fraction per column
    if len(df) > 0:
        nan_frac = df.isna().mean()
        report.nan_fractions = nan_frac.to_dict()
        report.high_nan_columns = sorted(
            nan_frac[nan_frac > nan_threshold].index.tolist()
        )

    # Determine pass/fail
    report.passed = (
        len(report.missing_columns) == 0
        and len(report.high_nan_columns) == 0
    )

    # Log results
    if report.passed:
        log.info("Ingestion completeness: PASS (%d columns)", report.total_columns)
    else:
        log.warning(
            "Ingestion completeness: FAIL — %d missing, %d high-NaN columns",
            len(report.missing_columns),
            len(report.high_nan_columns),
        )
        if report.missing_columns:
            log.warning("  Missing: %s", report.missing_columns[:10])
        if report.high_nan_columns:
            log.warning("  High NaN (>%.0f%%): %s", nan_threshold * 100, report.high_nan_columns[:10])

    return report
