"""
Feature taxonomy — fast / slow / agency classification (DATA-04).

Reads the ``taxonomy`` block of ``config/platform_settings.yaml``:

    taxonomy:
      fast:   [...]   # regime-detection layer, market-observed, unrevised
      slow:   [...]   # strategic-tilt layer (valuation anchors)
      agency: [...]   # vintage-corrected agency series, used sparingly

``validate_taxonomy()`` enforces the DATA-04 guarantee — every feature is
classified into *exactly* one tier — mirroring the incumbent
``trading_crab_lib.config.validate_config()`` collect-all-errors-then-raise-
once shape.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

log = logging.getLogger(__name__)

_TIERS = ("fast", "slow", "agency")


def _tiers_dict(cfg: dict[str, Any]) -> dict[str, list[str]]:
    taxonomy = cfg.get("taxonomy", {})
    return {tier: list(taxonomy.get(tier, [])) for tier in _TIERS}


def classify_feature(name: str, cfg: dict[str, Any]) -> str | None:
    """Return the tier ("fast"/"slow"/"agency") containing *name*, else ``None``."""
    for tier, features in _tiers_dict(cfg).items():
        if name in features:
            return tier
    return None


def lean_feature_set(cfg: dict[str, Any]) -> set[str]:
    """Return the union of the fast and slow tiers — the lean full-history labeling set.

    Agency-tier features are excluded (design §9: "used sparingly", not part
    of the lean 1962+ set).
    """
    tiers = _tiers_dict(cfg)
    return set(tiers["fast"]) | set(tiers["slow"])


def check_columns_tagged(columns: list[str], cfg: dict[str, Any]) -> list[str]:
    """Return the subset of *columns* not found in any taxonomy tier."""
    tagged = set(tiers for tier in _tiers_dict(cfg).values() for tiers in tier)
    return [c for c in columns if c not in tagged]


def validate_taxonomy(cfg: dict[str, Any]) -> None:
    """Validate that every feature in *cfg*'s taxonomy is tagged in exactly one tier.

    Raises :exc:`ValueError` listing every double-tagged feature (and the
    tiers it appears in) at once — collect-all-errors-then-raise-once,
    mirroring incumbent ``validate_config()``.
    """
    tiers = _tiers_dict(cfg)
    counts: Counter[str] = Counter()
    tier_membership: dict[str, list[str]] = {}
    for tier, features in tiers.items():
        for feature in features:
            counts[feature] += 1
            tier_membership.setdefault(feature, []).append(tier)

    errors: list[str] = []
    for feature, count in counts.items():
        if count > 1:
            tiers_named = ", ".join(tier_membership[feature])
            errors.append(
                f"Feature '{feature}' is classified in {count} taxonomy tiers "
                f"({tiers_named}) — must be classified in exactly one tier."
            )

    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(
            f"platform taxonomy has {len(errors)} double-tagged feature(s):\n{bullet_list}"
        )
