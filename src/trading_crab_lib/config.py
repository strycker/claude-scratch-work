"""
Central config loader — call load() once at pipeline entry points.
Uses python-dotenv for secrets, PyYAML for settings.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from trading_crab_lib import CONFIG_DIR

log = logging.getLogger(__name__)


def load(settings_path: Path | None = None) -> dict:
    """Load settings.yaml and inject secrets from .env / environment."""
    load_dotenv()  # reads .env if present; env vars already set take priority

    path = settings_path or CONFIG_DIR / "settings.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Inject FRED API key from environment
    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        log.warning("FRED_API_KEY not set — FRED ingestion will fail")
    cfg.setdefault("fred", {})["api_key"] = fred_key

    return cfg


def load_portfolio(portfolio_path: Path | None = None) -> dict[str, float]:
    """
    Load current portfolio weights from YAML (ticker -> weight fraction).
    Weights are normalized to sum to 1. Missing or empty file returns {}.
    """
    path = portfolio_path or CONFIG_DIR / "portfolio.yaml"
    if not path.exists():
        log.debug("No portfolio file at %s", path)
        return {}
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not raw or not isinstance(raw, dict):
        return {}
    # Accept numeric values only; normalize to sum = 1
    weights: dict[str, float] = {}
    for k, v in raw.items():
        if str(k).startswith("#"):
            continue
        try:
            w = float(v)
            if w > 0:
                weights[str(k).strip()] = w
        except (TypeError, ValueError):
            continue
    if not weights:
        return {}
    total = sum(weights.values())
    if total <= 0:
        return {}
    return {t: w / total for t, w in weights.items()}


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )
