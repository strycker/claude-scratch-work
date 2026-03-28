"""
monitoring — Pipeline monitoring helpers, validation summaries, diagnostic reports.

Phase C1: Steps 1-2 monitoring (ingestion + feature engineering).
Phase C2: Steps 3-4 monitoring (clustering + regime labeling).
Phase C3: Steps 5-7 monitoring (prediction + dashboard).
Phase C4: Pipeline-level validation and health.

All public names are re-exported here so existing imports continue to work:
    from trading_crab_lib.monitoring import validate_date_range
"""

from __future__ import annotations

# ── Step 1-2: Ingestion monitoring ───────────────────────────────────────
from trading_crab_lib.monitoring.ingestion import (
    DateRangeReport,
    SourceRowCounts,
    count_source_columns,
    format_completeness_table,
    validate_date_range,
)

# ── Step 2: Feature quality ──────────────────────────────────────────────
from trading_crab_lib.monitoring.features import (
    FeatureQualityReport,
    compute_feature_quality,
)

# ── Steps 3-4: Clustering & regime stability ─────────────────────────────
from trading_crab_lib.monitoring.clustering import (
    RegimeStabilityReport,
    compute_regime_stability,
    format_method_comparison,
)

# ── Steps 5-7: Prediction & dashboard QA ─────────────────────────────────
from trading_crab_lib.monitoring.prediction import (
    CVFoldReport,
    check_regime_probabilities,
    compute_cv_fold_scores,
)

# ── Pipeline-level validation ────────────────────────────────────────────
from trading_crab_lib.monitoring.pipeline import (
    PipelineHealthSummary,
    StepValidation,
    format_tactics_summary,
    validate_step_output,
)

__all__ = [
    # Ingestion
    "format_completeness_table",
    "DateRangeReport",
    "validate_date_range",
    "SourceRowCounts",
    "count_source_columns",
    # Features
    "FeatureQualityReport",
    "compute_feature_quality",
    # Clustering
    "format_method_comparison",
    "RegimeStabilityReport",
    "compute_regime_stability",
    # Prediction
    "CVFoldReport",
    "compute_cv_fold_scores",
    "check_regime_probabilities",
    # Pipeline
    "format_tactics_summary",
    "StepValidation",
    "validate_step_output",
    "PipelineHealthSummary",
]
