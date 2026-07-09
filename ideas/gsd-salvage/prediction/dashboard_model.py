"""Resolve which pickled current-regime model the dashboard uses (RF vs GB).

The training pipeline may save both ``current_regime.pkl`` (random forest) and
``current_regime_gb.pkl`` (gradient boosting). This module picks the path implied
by ``dashboard.regime_model`` in settings, with safe fallback to RF.
"""

from __future__ import annotations

import logging
from pathlib import Path


def resolve_current_regime_model_path(
    cfg: dict,
    model_dir: Path,
    log: logging.Logger | None = None,
) -> Path:
    """
    Choose ``current_regime.pkl`` (RF) or ``current_regime_gb.pkl`` for live scoring.

    Config: ``dashboard.regime_model`` — ``\"rf\"`` (default) or ``\"gb\"``.
    If ``gb`` is requested but the GB pickle is missing, fall back to RF with a warning.
    """
    raw = cfg.get("dashboard", {}).get("regime_model", "rf")
    regime_model = raw if isinstance(raw, str) else "rf"
    regime_model = regime_model.lower().strip()
    if regime_model not in ("rf", "gb"):
        if log:
            log.warning(
                "dashboard.regime_model=%r invalid (use rf or gb); using rf",
                raw,
            )
        regime_model = "rf"

    rf_path = model_dir / "current_regime.pkl"
    gb_path = model_dir / "current_regime_gb.pkl"
    # RF is default production model; GB is optional — both trained on same features, different bias–variance tradeoff.

    if regime_model == "gb" and gb_path.exists():
        if log:
            log.info("Dashboard using gradient-boosted current-regime model: %s", gb_path.name)
        return gb_path

    if regime_model == "gb" and not gb_path.exists():
        if log:
            log.warning(
                "dashboard.regime_model=gb but %s not found — falling back to %s",
                gb_path.name,
                rf_path.name,
            )

    if log:
        log.info("Dashboard using random-forest current-regime model: %s", rf_path.name)
    return rf_path
