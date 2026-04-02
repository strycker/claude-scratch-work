"""
Central config loader — call load() once at pipeline entry points.
Uses python-dotenv for secrets, PyYAML for settings.

``load()`` accepts three forms:

* ``load()`` or ``load(None)``    — reads ``config/settings.yaml`` from the
  repo root (default, backward-compatible with git-clone installs).
* ``load("/path/to/settings.yaml")`` — reads a specific file.
* ``load({"data": {...}, ...})``  — accepts a pre-built config dict directly,
  skipping all file I/O.  Useful for ``pip install``-only users who manage
  config programmatically, or for test isolation.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from trading_crab_lib import CONFIG_DIR

log = logging.getLogger(__name__)

# ── Schema: required top-level sections ────────────────────────────────────────

_REQUIRED_SECTIONS: list[str] = [
    "data",
    "fred",
    "multpl",
    "features",
    "clustering",
    "prediction",
    "assets",
    "dashboard",
    "pipeline",
    "tactics",
]

# Dot-separated paths → (expected_type, human_label)
_REQUIRED_SCALARS: dict[str, tuple[type, str]] = {
    "data.frequency":                    (str,   "string"),
    "clustering.n_pca_components":       (int,   "integer"),
    "clustering.n_clusters_search":      (int,   "integer"),
    "clustering.k_cap":                  (int,   "integer"),
    "clustering.balanced_k":             (int,   "integer"),
    "clustering.random_state":           (int,   "integer"),
    "prediction.cv_splits":              (int,   "integer"),
    "prediction.dt_max_depth":           (int,   "integer"),
    "prediction.n_estimators":           (int,   "integer"),
    "pipeline.random_state":             (int,   "integer"),
    "dashboard.signal_thresholds.green": (float, "float"),
}


def _get_nested(cfg: dict, dotpath: str) -> Any:
    """Return the value at *dotpath* (e.g. ``'clustering.k_cap'``), or raise ``KeyError``."""
    keys = dotpath.split(".")
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            raise KeyError(dotpath)
        node = node[k]
    return node


def validate_config(cfg: dict[str, Any]) -> None:
    """Validate *cfg* for required sections and scalar types.

    Raises :exc:`ValueError` immediately with a clear, actionable message if:

    * A required top-level section is missing.
    * A required scalar key is absent or has the wrong type.

    Does **not** validate list contents or optional keys — the goal is
    fail-fast detection of broken / incomplete ``settings.yaml`` edits, not
    exhaustive schema enforcement.
    """
    errors: list[str] = []

    # 1. Required top-level sections
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            errors.append(
                f"Missing required section '{section}' in settings.yaml. "
                f"Add a '{section}:' block or restore it from settings.example.yaml."
            )

    # 2. Required scalar keys with type checks (only when parent section present)
    for dotpath, (expected_type, label) in _REQUIRED_SCALARS.items():
        try:
            value = _get_nested(cfg, dotpath)
        except KeyError:
            errors.append(
                f"Missing required key '{dotpath}' in settings.yaml. "
                f"Expected a {label} value."
            )
            continue
        if not isinstance(value, expected_type):
            errors.append(
                f"Wrong type for '{dotpath}' in settings.yaml: "
                f"got {type(value).__name__}, expected {label} ({expected_type.__name__}). "
                f"Current value: {value!r}"
            )

    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(
            f"settings.yaml has {len(errors)} validation error(s):\n{bullet_list}"
        )


def load(
    settings_path: dict[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Load config, validate its schema, and inject secrets from the environment.

    Args:
        settings_path: Controls how config is sourced:

            * ``None`` (default) — reads ``config/settings.yaml`` relative to the
              repo root detected at import time.  Equivalent to a git-clone install.
            * ``Path`` or ``str`` — reads from the given YAML file path.
            * ``dict`` — uses the provided mapping directly, bypassing all file I/O.
              The dict is validated and the FRED API key is still injected from the
              environment.  Useful for programmatic config, Docker / CI environments,
              and ``pip install`` users who manage settings in code.

    Returns:
        Validated config dict with ``fred.api_key`` injected.

    Raises:
        ValueError: If required keys are missing or have the wrong type.
        FileNotFoundError: If a file path is given but does not exist.
    """
    load_dotenv()  # reads .env if present; env vars already set take priority

    if isinstance(settings_path, dict):
        # Dict path — skip file I/O entirely; validate + inject only.
        cfg: dict[str, Any] = settings_path
    else:
        path = Path(settings_path) if settings_path is not None else CONFIG_DIR / "settings.yaml"
        with open(path) as f:
            cfg = yaml.safe_load(f)

    validate_config(cfg)

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
    """Configure root logger with a timestamped format at the given *level*."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
    )
