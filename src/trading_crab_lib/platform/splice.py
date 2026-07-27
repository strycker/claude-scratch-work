"""
Splicing engine — turns raw source segments into one long-history research
series per core asset class (D-03 "model the index, trade the ETF").

Implements the D-04 splice/synthesis techniques consumed by the ``splice``
block of ``config/platform_settings.yaml``:

  * ``ratio_splice()`` — scale an older segment to match a newer segment at
    a literal, documented ``join_date`` (never a computed default — RESEARCH
    Pitfall 2). None of the current 5 core classes need a 2-segment splice
    (see ``docs/splicing_rules.md``), but ``build_core_research_series()``
    dispatches to it for any future class whose ``method`` is
    ``"ratio_splice"``.
  * Treasury total-return synthesis from CMT yields ("par-bond repricing") —
    ``bond_price()`` + ``monthly_total_return()`` + ``build_treasury_tr_synthetic()``.
  * Equity total-return construction from price + dividend yield —
    ``build_equity_total_return()``.

``build_core_research_series()`` dispatches per-class on the ``method`` key
in ``cfg['splice']`` and returns the 5 ``research_name`` columns documented
in ``docs/splicing_rules.md`` (equities_tr, long_duration_tr, gold, oil, cash).

Usage:
    from trading_crab_lib.platform.splice import build_core_research_series
    research = build_core_research_series(raw, cfg)

Does not import or modify any incumbent transform module (D-01).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


# ── Ratio splice ─────────────────────────────────────────────────────────────

def ratio_splice(old: pd.Series, new: pd.Series, join_date: pd.Timestamp) -> pd.Series:
    """Scale `old` to match `new` at `join_date`, then concatenate.

    `old` (before `join_date`, scaled) is glued to `new` (at/after
    `join_date`, unscaled). The seam is continuous by construction — the
    value at `join_date` is `new`'s own value, not a scaled approximation.

    Args:
        old: Older segment; must have an observation at exactly `join_date`.
        new: Newer segment; must have an observation at exactly `join_date`.
        join_date: Literal, documented anchor date (RESEARCH Pitfall 2 —
            never compute this dynamically; pin it in config/docs).

    Returns:
        The spliced series, named after `new`.

    Raises:
        ValueError: If `join_date` is missing from either input's index —
            never silently interpolates the scale factor.
    """
    if join_date not in old.index:
        raise ValueError(f"join_date {join_date} not found in `old` series index")
    if join_date not in new.index:
        raise ValueError(f"join_date {join_date} not found in `new` series index")

    scale = new.loc[join_date] / old.loc[join_date]
    log.debug("ratio_splice: join_date=%s scale=%.6f", join_date, scale)
    scaled_old = old.loc[:join_date] * scale
    spliced = pd.concat([scaled_old.iloc[:-1], new.loc[join_date:]])
    spliced.name = new.name
    return spliced


# ── Treasury total-return synthetic (par-bond repricing) ────────────────────

def bond_price(yield_annual: float, coupon_annual: float, years_to_maturity: float, freq: int = 2) -> float:
    """PV of a bond given a flat yield curve assumption at `yield_annual`.

    A par bond priced at its own coupon yield (`yield_annual == coupon_annual`)
    returns ~1.0 by construction.
    """
    periods = int(round(years_to_maturity * freq))
    coupon = coupon_annual / freq
    r = yield_annual / freq
    pv_coupons = sum(coupon / (1 + r) ** t for t in range(1, periods + 1))
    pv_face = 1.0 / (1 + r) ** periods
    return pv_coupons + pv_face


def monthly_total_return(
    yield_t0: float, yield_t1: float, maturity_years: float = 10, *, freq: int = 2
) -> float:
    """One month of Treasury total return: issue a par bond at `yield_t0`,
    accrue one month of coupon, then reprice at `yield_t1` (RESEARCH Pattern 3,
    "par-bond repricing" — semiannual compounding by default per `freq`).

    Rising yields hurt total return (the price decline outweighs the coupon
    accrual); falling yields help.
    """
    price_t0 = bond_price(yield_t0, yield_t0, maturity_years, freq=freq)  # priced at par by construction
    price_t1 = bond_price(yield_t1, yield_t0, maturity_years - 1 / 12, freq=freq)  # coupon fixed, yield moves
    coupon_accrued = yield_t0 / 12
    return (price_t1 - price_t0 + coupon_accrued) / price_t0


def build_treasury_tr_synthetic(gs10_yield_series: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Chain `monthly_total_return()` across a monthly CMT yield series into a
    cumulative total-return index (base = 1.0 at the first observation).

    Reads `maturity_years`/`coupon_freq` from `cfg['splice']['long_duration']`.
    """
    params = cfg["splice"]["long_duration"]
    maturity_years = params["maturity_years"]
    freq = params["coupon_freq"]

    yields = gs10_yield_series.dropna()
    monthly_returns = [
        monthly_total_return(y0, y1, maturity_years, freq=freq) for y0, y1 in zip(yields.iloc[:-1], yields.iloc[1:])
    ]
    returns = pd.Series(monthly_returns, index=yields.index[1:])
    index_level = pd.concat([pd.Series([1.0], index=[yields.index[0]]), (1 + returns).cumprod()])
    index_level.name = params["research_name"]
    return index_level


# ── Equity total return ──────────────────────────────────────────────────────

def build_equity_total_return(price: pd.Series, div_yield: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Monthly total return = price return + (annual `div_yield` / 12),
    chained into a cumulative index (base = 1.0 at the first observation).

    Source: already-scraped multpl price + dividend yield, not a new Shiller
    fetcher (Claude's-Discretion choice — see docs/splicing_rules.md for the
    Shiller cross-check option).
    """
    params = cfg["splice"]["equities"]
    price = price.dropna()
    div = div_yield.reindex(price.index).ffill()
    monthly_returns = (price.pct_change() + div / 12).dropna()
    index_level = pd.concat([pd.Series([1.0], index=[price.index[0]]), (1 + monthly_returns).cumprod()])
    index_level.name = params["research_name"]
    return index_level


# ── Per-class assembly ────────────────────────────────────────────────────────

# Maps each splice ``method`` to the param keys whose values name required
# columns in the raw monthly ingest frame. Used to preflight-validate that every
# upstream source actually arrived before assembling research series.
_SPLICE_SOURCE_COL_KEYS: dict[str, tuple[str, ...]] = {
    "total_return_from_price_div": ("price_col", "div_yield_col"),
    "cmt_par_bond_repricing": ("yield_col",),
    "single_source": ("source_col",),
    "yield_as_return": ("yield_col",),
    "ratio_splice": ("old_col", "new_col"),
}


def build_core_research_series(raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Dispatch on each core class's `method` and assemble the 5
    `research_name` columns from `cfg['splice']` (D-03/D-04).

    Preflight-validates that every required source column is present before
    assembling, so a failed upstream fetch (e.g. macrotrends returning no
    gold_spot/wti_crude, or a yfinance rate-limit) yields one clear, actionable
    error naming every missing column instead of a bare ``KeyError`` deep in
    the loop.
    """
    splice_cfg = cfg["splice"]

    # ── Preflight: report all missing source columns at once ──
    # A single_source class whose primary source_col is absent is still
    # satisfiable if its optional ``fallback_col`` (e.g. a FRED cross-check
    # series) IS present — so it is only flagged missing when both are absent.
    available = set(raw.columns)
    missing: list[str] = []
    for class_name, params in splice_cfg.items():
        method = params.get("method", "")
        for key in _SPLICE_SOURCE_COL_KEYS.get(method, ()):
            col = params.get(key)
            if col is None or col in available:
                continue
            if method == "single_source" and params.get("fallback_col") in available:
                continue
            missing.append(
                f"  - {params.get('research_name', class_name)} "
                f"({method}): missing column '{col}'"
            )
    if missing:
        raise ValueError(
            "build_core_research_series: required source columns are missing from the "
            "monthly ingest frame:\n" + "\n".join(missing) + "\n\n"
            "These columns come from upstream ingestion (FRED / multpl.com / "
            "macrotrends.net / yfinance). One or more sources failed to fetch — check the "
            "WARNING logs above (e.g. 'No tables found' for macrotrends, or a yfinance "
            "rate-limit). Wait for the source(s) to become reachable, then re-run the data build."
        )

    columns: dict[str, pd.Series] = {}

    for class_name, params in splice_cfg.items():
        method = params["method"]
        research_name = params["research_name"]

        if method == "total_return_from_price_div":
            series = build_equity_total_return(raw[params["price_col"]], raw[params["div_yield_col"]], cfg)
        elif method == "cmt_par_bond_repricing":
            series = build_treasury_tr_synthetic(raw[params["yield_col"]], cfg)
        elif method == "single_source":
            col = params["source_col"]
            fallback = params.get("fallback_col")
            if col not in raw.columns and fallback in raw.columns:
                log.warning(
                    "splice '%s': primary source '%s' missing — falling back to '%s'",
                    research_name, col, fallback,
                )
                col = fallback
            series = raw[col].dropna().rename(research_name)
        elif method == "yield_as_return":
            series = raw[params["yield_col"]].dropna().rename(research_name)
        elif method == "ratio_splice":
            series = ratio_splice(
                raw[params["old_col"]], raw[params["new_col"]], pd.Timestamp(params["join_date"])
            ).rename(research_name)
        else:
            raise ValueError(f"Unknown splice method '{method}' for class '{class_name}'")

        columns[research_name] = series

    log.info("Built %d core research series: %s", len(columns), sorted(columns))
    return pd.concat(columns, axis=1)
