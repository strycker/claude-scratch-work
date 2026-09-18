"""
Leadership/relative-strength feature engineering, ported to monthly cadence (DATA-04/D-10).

The incumbent quarterly library's momentum/relative-strength implementations
(``src/trading_crab_lib/momentum.py``) all size their lookback windows in
QUARTERS — reusing them verbatim would silently compute 2/4/8-MONTH windows
where 6/12/24-MONTH windows are intended, defeating the whole purpose of a
monthly-cadence leadership classifier (mirrors the precedent set by
``platform/ingestion/macro_monthly.py``'s docstring for the same
"ported, never imported, every constant re-derived" pattern). This module
therefore reimplements the same four algorithms natively at monthly cadence,
on ``monthly_raw``'s real column names — never on the legacy library's
quarterly column names.

Every function body below is a line-for-line port (not an import) of its
named counterpart in ``src/trading_crab_lib/momentum.py``, verified to have
zero internal imports of its own, so this port does not add a single legacy
import site anywhere under ``platform/`` (``tests/unit/
test_platform_legacy_import_ratchet.py``'s AST scan is pinned at 31 and may
only decrease — porting bodies, not importing the module, is what keeps this
port ratchet-safe, per 07-regime-representation criterion 8).

These are classifier #2's RAW candidate columns (07-regime-representation
D-10): every column produced here is disjoint, by construction, from
classifier #1's 13 lean raw columns (``taxonomy.lean_feature_set``) — a
*ratio* of two of #1's raw columns (e.g. an oil/equity relative-strength
ratio) is a genuinely different, scale-invariant quantity and is admissible
under D-10 even though its inputs are not.

Usage::

    from trading_crab_lib.platform.features.relative import add_relative_features
    from trading_crab_lib.platform.config import load_platform_config

    cfg = load_platform_config()
    relative_features = add_relative_features(monthly_raw, cfg)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Window-constant re-derivation (quarterly -> monthly, D-10/D-11) ─────────
#
# Legacy defaults are sized in QUARTERS: ``momentum.py:44-50`` documents
# ``windows = [2, 4, 8]`` quarters for trailing momentum, and
# ``momentum.py:116-120`` documents ``window=8`` quarters for rolling
# cross-correlation. At 3 months to a quarter, the monthly equivalents are:
#   2 quarters * 3 months/quarter =  6 months
#   4 quarters * 3 months/quarter = 12 months
#   8 quarters * 3 months/quarter = 24 months
# The legacy numerals [2, 4, 8] and 8 must NOT be carried across the cadence
# boundary unconverted — that would silently compute 2/4/8-month windows
# instead of the intended 6/12/24-month windows.

#: Trailing-momentum lookback windows, in MONTHS (re-derived from the legacy
#: quarterly defaults [2, 4, 8] -- see comment above).
MONTHLY_MOMENTUM_WINDOWS: list[int] = [6, 12, 24]

#: Rolling cross-correlation window, in MONTHS (re-derived from the legacy
#: quarterly default of 8 -- see comment above).
MONTHLY_CORRELATION_WINDOW: int = 24


# ── Relative strength ratios ────────────────────────────────────────────────

# Default pairs: (numerator, denominator, output_name) over monthly_raw's
# real column names (verified present in the live checkpoint: equities_tr,
# long_duration_tr, oil).
#
# `gold` is deliberately excluded from every default here: its first valid
# month in monthly_raw is 1985-02, so a gold ratio cannot survive D-11's
# common-support freeze at the 1972+ decision window (07-CONTEXT.md D-11).
# `oil` runs from 1962-01 in monthly_raw and does survive the freeze. D-10
# (ratios of #1's raw columns are admissible) opens the door for both; D-11
# (the freeze rule) is what closes it for gold specifically.
DEFAULT_RELATIVE_PAIRS: list[tuple[str, str, str]] = [
    ("equities_tr", "long_duration_tr", "rs_equities_bonds"),
    ("oil", "equities_tr", "rs_oil_equities"),
]


def compute_relative_strength(
    df: pd.DataFrame,
    pairs: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """
    Compute cross-asset relative strength ratios.

    Each pair (num, denom, name) produces ``name = num / denom``. Only
    computes if both columns exist. Skips gracefully otherwise.

    Ported from ``src/trading_crab_lib/momentum.py:77-110``
    (``compute_relative_strength``) — ratio math is unit-agnostic, so no
    window constant needed re-deriving here; only the default pairs changed
    (monthly_raw's real column names, not the legacy quarterly library's).

    Args:
        df: DataFrame with monthly time-series columns.
        pairs: List of (numerator_col, denominator_col, output_name).
               Default: :data:`DEFAULT_RELATIVE_PAIRS`.

    Returns:
        DataFrame with relative strength columns appended.
    """
    if pairs is None:
        pairs = DEFAULT_RELATIVE_PAIRS

    result = df.copy()
    added = 0
    for num, denom, name in pairs:
        if num not in result.columns or denom not in result.columns:
            log.debug("Skipping relative strength %s: missing %s or %s", name, num, denom)
            continue
        denom_safe = result[denom].replace(0, np.nan)
        result[name] = result[num] / denom_safe
        added += 1

    if added > 0:
        log.info("Added %d relative strength ratios", added)
    return result


# ── Trailing momentum (percentage change over N months) ─────────────────────

def compute_trailing_momentum(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute trailing N-month percentage returns for specified columns.

    For each column and each window, creates ``{col}_mom_{window}m``.
    E.g., ``equities_tr_mom_12m`` = (equities_tr[t] - equities_tr[t-12]) / equities_tr[t-12].

    Ported from ``src/trading_crab_lib/momentum.py:30-64``
    (``compute_trailing_momentum``) — column-name suffix changed from ``q``
    (quarters) to ``m`` (months); default windows re-derived (see the
    module-level comment above).

    Args:
        df: DataFrame with monthly time-series columns.
        columns: Column names to compute momentum for.
        windows: Lookback windows in months. Default :data:`MONTHLY_MOMENTUM_WINDOWS`.

    Returns:
        DataFrame with new momentum columns appended.
    """
    if windows is None:
        windows = MONTHLY_MOMENTUM_WINDOWS

    result = df.copy()
    added = 0
    for col in columns:
        if col not in result.columns:
            continue
        for w in windows:
            name = f"{col}_mom_{w}m"
            result[name] = result[col].pct_change(periods=w)
            added += 1

    if added > 0:
        log.info("Added %d trailing momentum features", added)
    return result


# ── Rolling cross-asset correlation ─────────────────────────────────────────

def compute_rolling_cross_correlation(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """
    Compute rolling Pearson correlation between signal pairs.

    For each (col_a, col_b), produces ``corr_{a}_{b}_{MONTHLY_CORRELATION_WINDOW}m``.
    Uses pct_change of each series (returns) to avoid spurious trend correlation.

    Ported from ``src/trading_crab_lib/momentum.py:123-157``
    (``compute_rolling_cross_correlation``) — the legacy version carries a
    per-pair window; this monthly analog uses ONE re-derived window
    (:data:`MONTHLY_CORRELATION_WINDOW`, see the module-level comment above)
    for every pair, since D-10/D-11's candidate set has no reason to vary it
    per pair.

    Args:
        df: DataFrame with monthly time-series columns.
        pairs: List of (col_a, col_b). Default: the column pairs underlying
            :data:`DEFAULT_RELATIVE_PAIRS` (same asset pairs, one source of
            truth) — the equity/bond pair is, per the phase proposal, "the
            single most important missing variable."

    Returns:
        DataFrame with rolling correlation columns appended.
    """
    if pairs is None:
        pairs = [(num, denom) for num, denom, _name in DEFAULT_RELATIVE_PAIRS]

    result = df.copy()
    added = 0
    window = MONTHLY_CORRELATION_WINDOW
    for col_a, col_b in pairs:
        if col_a not in result.columns or col_b not in result.columns:
            log.debug("Skipping correlation %s/%s: column missing", col_a, col_b)
            continue
        ret_a = result[col_a].pct_change()
        ret_b = result[col_b].pct_change()
        name = f"corr_{col_a}_{col_b}_{window}m"
        result[name] = ret_a.rolling(window=window, min_periods=max(4, window // 2)).corr(ret_b)
        added += 1

    if added > 0:
        log.info("Added %d rolling cross-correlation features", added)
    return result


# ── Inflation acceleration (2nd derivative of CPI) ──────────────────────────

def compute_inflation_acceleration(df: pd.DataFrame, col: str = "fred_cpi") -> pd.DataFrame:
    """
    Compute inflation acceleration: the 2nd derivative of CPI.

    Captures whether inflation is accelerating (positive) or decelerating
    (negative), which is more predictive of regime changes than the CPI
    level. Creates ``cpi_acceleration`` = diff(diff(col)) / col.

    Ported from ``src/trading_crab_lib/momentum.py:162-180``
    (``compute_inflation_acceleration``) — period-agnostic (2nd derivative),
    no window constant to re-derive; the legacy version tries ``["cpi",
    "fred_cpi"]`` in sequence, whereas the platform's ``monthly_raw`` has
    exactly one CPI column name (``fred_cpi``), so the port takes an
    explicit ``col`` parameter instead of a hardcoded fallback list.

    Args:
        df: DataFrame with monthly time-series columns.
        col: Column to compute acceleration from. Default ``"fred_cpi"``.

    Returns:
        DataFrame with ``cpi_acceleration`` appended if ``col`` is present.
    """
    result = df.copy()
    if col in result.columns:
        d1 = result[col].diff()
        d2 = d1.diff()
        # Normalize by level to get percentage acceleration.
        result["cpi_acceleration"] = d2 / result[col].replace(0, np.nan)
        log.info("Added cpi_acceleration from %s", col)
    return result


# ── INV-01 invariant ratios (named features, never anonymous PCs — R4) ──────

def compute_invariant_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute INV-01's two named era-stable invariant candidates.

    ``m2_gdp`` = ``fred_m2sl`` / ``fred_gdp``; ``credit_gdp`` = ``fred_totalsl``
    / ``fred_gdp``. Market-cap/GDP ("Buffett indicator") is deliberately
    NOT computed here — no free 1962+ market-cap source exists (D-12,
    restated from ``config/platform_settings.yaml``'s ``buffett_indicator``
    comment, not worked around).

    Each ratio is guarded on the presence of BOTH source columns, logging at
    INFO and skipping when a source is absent, so this module is usable
    before the INV-01 ingestion (07-05 Task 3) has populated
    ``fred_m2sl``/``fred_totalsl`` in ``monthly_raw``.

    These are NAMED features (design decision R4) — never folded into an
    anonymous PCA/PC1 representation.

    Args:
        df: DataFrame with monthly time-series columns.

    Returns:
        DataFrame with ``m2_gdp`` / ``credit_gdp`` appended where computable.
    """
    result = df.copy()
    cols = set(result.columns)

    if {"fred_m2sl", "fred_gdp"} <= cols:
        result["m2_gdp"] = result["fred_m2sl"] / result["fred_gdp"].replace(0, np.nan)
        log.info("Added m2_gdp invariant ratio (INV-01)")
    else:
        log.info(
            "Skipping m2_gdp invariant ratio (INV-01): missing fred_m2sl and/or fred_gdp"
        )

    if {"fred_totalsl", "fred_gdp"} <= cols:
        result["credit_gdp"] = result["fred_totalsl"] / result["fred_gdp"].replace(0, np.nan)
        log.info("Added credit_gdp invariant ratio (INV-01)")
    else:
        log.info(
            "Skipping credit_gdp invariant ratio (INV-01): missing fred_totalsl and/or fred_gdp"
        )

    return result


# ── Master wrapper ───────────────────────────────────────────────────────────

#: Default columns to compute trailing momentum for — monthly_raw's core
#: research series, all with deep (1962+) history.
_DEFAULT_MOMENTUM_COLUMNS: list[str] = ["equities_tr", "long_duration_tr", "oil"]


def add_relative_features(monthly_raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Compute classifier #2's full candidate leadership/relative-strength feature set.

    Returns a NEW DataFrame indexed identically to ``monthly_raw``, carrying
    ONLY the derived columns (never a passthrough of ``monthly_raw``'s own
    columns, and never mutating ``monthly_raw`` itself).

    Config keys (optional, all read defensively via ``cfg.get(...)`` under
    ``features.relative`` — never added to a required-sections list, per the
    Phase 2/4 config pattern):
        momentum_columns: list of columns to compute trailing momentum for
            (default :data:`_DEFAULT_MOMENTUM_COLUMNS`)
        relative_pairs: list of [num, denom, name] triples
            (default :data:`DEFAULT_RELATIVE_PAIRS`)
        correlation_pairs: list of [col_a, col_b] pairs
            (default: the pairs underlying :data:`DEFAULT_RELATIVE_PAIRS`)

    Args:
        monthly_raw: the platform's monthly raw checkpoint.
        cfg: platform config (``load_platform_config()``'s return value).

    Returns:
        DataFrame with only the derived relative-strength/momentum/
        invariant-ratio columns, indexed identically to ``monthly_raw``.
    """
    relative_cfg = cfg.get("features", {}).get("relative", {})

    mom_cols = relative_cfg.get("momentum_columns", _DEFAULT_MOMENTUM_COLUMNS)
    rel_pairs_raw = relative_cfg.get("relative_pairs")
    rel_pairs = [tuple(p) for p in rel_pairs_raw] if rel_pairs_raw else None
    corr_pairs_raw = relative_cfg.get("correlation_pairs")
    corr_pairs = [tuple(p) for p in corr_pairs_raw] if corr_pairs_raw else None

    working = monthly_raw.copy()
    before_cols = set(working.columns)

    working = compute_trailing_momentum(working, mom_cols)
    working = compute_relative_strength(working, rel_pairs)
    working = compute_rolling_cross_correlation(working, corr_pairs)
    working = compute_inflation_acceleration(working)
    working = compute_invariant_ratios(working)

    derived_cols = [c for c in working.columns if c not in before_cols]
    return working[derived_cols]
