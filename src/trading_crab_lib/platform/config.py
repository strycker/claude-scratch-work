"""
Platform config loader — independent of the frozen incumbent's
``trading_crab_lib.config`` module and ``config/settings.yaml`` (D-02).

``load_platform_config()`` mirrors the incumbent ``load()`` conventions
(tri-mode dict | Path | str | None, collect-all-errors validation, FRED key
injection from the environment) but reads its own schema from
``config/platform_settings.yaml`` — the frozen incumbent's ``validate_config()``
schema is never touched.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from trading_crab_lib import CONFIG_DIR

log = logging.getLogger(__name__)

_REQUIRED_PLATFORM_SECTIONS: list[str] = [
    "data",
    "fred_monthly",
    "fred_vintage",
    "splice",
    "universe",
    "taxonomy",
]


def validate_platform_config(cfg: dict[str, Any]) -> None:
    """Validate *cfg* for required top-level platform sections.

    Raises :exc:`ValueError` listing every missing section at once
    (collect-all-errors-then-raise-once, mirroring incumbent ``validate_config()``).
    """
    errors: list[str] = []
    for section in _REQUIRED_PLATFORM_SECTIONS:
        if section not in cfg:
            errors.append(
                f"Missing required section '{section}' in platform_settings.yaml. "
                f"Add a '{section}:' block."
            )
    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(
            f"platform_settings.yaml has {len(errors)} validation error(s):\n{bullet_list}"
        )


def load_platform_config(
    settings_path: dict[str, Any] | Path | str | None = None,
) -> dict[str, Any]:
    """Load platform config, validate its schema, and inject the FRED API key.

    Args:
        settings_path: Controls how config is sourced:

            * ``None`` (default) — reads ``config/platform_settings.yaml``.
            * ``Path`` or ``str`` — reads from the given YAML file path.
            * ``dict`` — uses the provided mapping directly, bypassing all file I/O.

    Returns:
        Validated config dict with ``fred_monthly.api_key`` and
        ``fred_vintage.api_key`` injected from ``FRED_API_KEY`` (D-07: reuses
        the incumbent's existing credential, no new secret).

    Raises:
        ValueError: If required sections are missing.
        FileNotFoundError: If a file path is given but does not exist.
    """
    load_dotenv()

    if isinstance(settings_path, dict):
        cfg: dict[str, Any] = settings_path
    else:
        path = Path(settings_path) if settings_path is not None else CONFIG_DIR / "platform_settings.yaml"
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    validate_platform_config(cfg)

    fred_key = os.getenv("FRED_API_KEY")
    if not fred_key:
        log.warning("FRED_API_KEY not set — platform FRED/ALFRED ingestion will fail")
    cfg.setdefault("fred_monthly", {})["api_key"] = fred_key
    cfg.setdefault("fred_vintage", {})["api_key"] = fred_key

    return cfg
