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

Source-column chains: ANY source-column key on ANY class (``source_col``,
``yield_col``, ``price_col``, ``div_yield_col``, ``old_col``, ``new_col``) may
be declared in config as a scalar or as an ordered list of candidate columns
— ``source_candidates()`` normalizes both spellings, and
``resolve_class_sources()`` is the single place that decides which column a
class actually uses (first candidate present in the raw frame wins). Every
resolution is logged at INFO on every run, and the full resolved-vs-tried
picture is recorded in the returned frame's ``.attrs["splice_provenance"]``
(persist it to JSON with ``write_splice_provenance()``).

IAU-is-not-spot-gold caveat: the ``gold`` class's chain falls back to the
``IAU`` ETF column when ``gold_spot`` (macrotrends) is unavailable. IAU is a
total-return gold ETF carrying an expense ratio and tracking drift — it is
**not** spot gold, and its price history begins around 2005 versus
``gold_spot``'s 1915+. Falling back to it changes what downstream models see
and truncates the gold research series by roughly nine decades. This is an
explicitly chosen compromise (see ``config/platform_settings.yaml``'s
``splice.gold`` comment and ``docs/splicing_rules.md`` §3), not an
equivalence claim.

Usage:
    from trading_crab_lib.platform.splice import build_core_research_series
    research = build_core_research_series(raw, cfg)

Does not import or modify any incumbent transform module (D-01).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
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


# ── Yield units ──────────────────────────────────────────────────────────────
#
# Raw columns keep whatever units their SOURCE publishes — ingestion does not
# rescale, and neither should it. That means `monthly_raw` legitimately mixes
# conventions: FRED's CMT and T-bill series are annualized PERCENT
# (`fred_gs10` = 4.68) while multpl's `div_yield` is already a DECIMAL fraction
# (0.011). `bond_price()` / `monthly_total_return()` are decimal-domain
# functions, so percent series MUST be converted here, at the splice boundary,
# and nowhere else.
#
# Getting this wrong is silent and catastrophic rather than loud: feeding 4.68
# to `monthly_total_return` accrues a 39%-per-month coupon and compounds
# `long_duration_tr` to ~1e128, which then propagates into every KPI without
# raising anything. Hence `assert_yield_units_plausible()` below.

_YIELD_UNIT_SCALES = {"percent": 0.01, "decimal": 1.0}
DEFAULT_YIELD_UNITS = "percent"

# A US Treasury yield has never exceeded 100% annualized. Anything above that
# after conversion is a units error, not a market.
_IMPLAUSIBLE_DECIMAL_YIELD = 1.0
# Below this the series is *suspicious* (a whole history median under 10bp),
# but ZIRP-era slices can legitimately sit here — warn, never fail.
_SUSPICIOUSLY_LOW_MEDIAN_YIELD = 0.001


def assert_yield_units_plausible(yields_decimal: pd.Series, *, source: str) -> None:
    """Fail loudly on an unambiguous yield-units error; warn on a suspicious one.

    Deliberately asymmetric, because the two directions are not equally
    knowable. A converted yield above 100% annualized cannot be a rate under
    any market conditions, so that is an error. A very low median can mean
    either double-converted units or a genuine ZIRP window, so that is only a
    warning — a hard failure there would reject real 2009-2021 T-bill data.

    Raises:
        ValueError: if any converted yield exceeds 100% annualized.
    """
    clean = yields_decimal.dropna()
    if clean.empty:
        return
    worst = float(clean.abs().max())
    if worst > _IMPLAUSIBLE_DECIMAL_YIELD:
        raise ValueError(
            f"{source}: yield of {worst:.4g} ({worst:.1%} annualized) after unit conversion is "
            f"not a plausible rate. The series is almost certainly in PERCENT while configured "
            f"as decimal — set `yield_units: percent` for this splice class. Left uncaught this "
            f"compounds into a nonsense total-return index rather than failing."
        )
    median = float(clean.median())
    if median < _SUSPICIOUSLY_LOW_MEDIAN_YIELD:
        log.warning(
            "%s: median yield after conversion is %.6g (%.4f%% annualized), which is very low. "
            "If this series is already a decimal fraction, it is being converted twice — check "
            "`yield_units` for this splice class.",
            source, median, median * 100,
        )


def yield_to_decimal(yields: pd.Series, *, units: str = DEFAULT_YIELD_UNITS, source: str = "yield") -> pd.Series:
    """Normalize an annualized yield series to a decimal fraction.

    ``units`` comes from the splice class's ``yield_units`` config key and
    defaults to ``percent`` — the convention of every yield column the
    platform currently ingests (all FRED).

    Raises:
        ValueError: on an unknown ``units`` value, or via
            ``assert_yield_units_plausible`` on an impossible converted rate.
    """
    if units not in _YIELD_UNIT_SCALES:
        raise ValueError(
            f"{source}: unknown yield_units {units!r}; expected one of {sorted(_YIELD_UNIT_SCALES)}"
        )
    converted = yields * _YIELD_UNIT_SCALES[units]
    assert_yield_units_plausible(converted, source=source)
    return converted


# ── Treasury total-return synthetic (par-bond repricing) ────────────────────

def bond_price(yield_annual: float, coupon_annual: float, years_to_maturity: float, freq: int = 2) -> float:
    """PV of a bond given a flat yield curve assumption at `yield_annual`.

    A par bond priced at its own coupon yield (`yield_annual == coupon_annual`)
    returns ~1.0 by construction.

    Units: `yield_annual` and `coupon_annual` are DECIMAL fractions (0.0468),
    never percent (4.68) — `pv_face = 1 / (1 + r)**n` only prices to par in the
    decimal domain. Callers reading raw FRED columns must run them through
    `yield_to_decimal()` first.
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

    Reads `maturity_years`/`coupon_freq`/`yield_units` from
    `cfg['splice']['long_duration']`. The incoming series is in its source's
    native units (FRED CMT is PERCENT) and is converted to decimal here —
    `monthly_total_return()` is decimal-domain.
    """
    params = cfg["splice"]["long_duration"]
    maturity_years = params["maturity_years"]
    freq = params["coupon_freq"]

    yields = yield_to_decimal(
        gs10_yield_series.dropna(),
        units=params.get("yield_units", DEFAULT_YIELD_UNITS),
        source="splice.long_duration",
    )
    monthly_returns = [
        monthly_total_return(y0, y1, maturity_years, freq=freq) for y0, y1 in zip(yields.iloc[:-1], yields.iloc[1:])
    ]
    returns = pd.Series(monthly_returns, index=yields.index[1:])
    index_level = pd.concat([pd.Series([1.0], index=[yields.index[0]]), (1 + returns).cumprod()])
    index_level.name = params["research_name"]
    return index_level


# ── Cash (T-bill) total return ───────────────────────────────────────────────

def build_cash_index(tbill_yield_series: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Chain a short-rate yield series into a cumulative total-return index
    (base = 1.0 at the first observation), like every other research class.

    Two things this must get right, both of which were previously wrong:

    * **Units.** The yield arrives in its source's native units (FRED TB3MS is
      PERCENT) and is converted to decimal here.
    * **Level, not rate.** Every consumer runs research series through
      ``compute_monthly_returns()``, i.e. ``pct_change()``. Returning the yield
      series itself therefore made "cash return" the month-over-month CHANGE IN
      THE YIELD — a T-bill going 0.5% -> 3.0% booked a +500% month. Cash must be
      an index level so that ``pct_change()`` recovers the accrual.

    Accrual convention matches ``monthly_total_return()``: the month ending at
    ``t`` earns the yield observed at ``t-1``, which is the rate actually known
    when that month began. Using the yield at ``t`` would be look-ahead.
    """
    params = cfg["splice"]["cash"]
    yields = yield_to_decimal(
        tbill_yield_series.dropna(),
        units=params.get("yield_units", DEFAULT_YIELD_UNITS),
        source="splice.cash",
    )
    if yields.empty:
        return pd.Series(dtype=float, name=params["research_name"])

    monthly_accrual = (yields.shift(1) / 12).dropna()
    index_level = pd.concat(
        [pd.Series([1.0], index=[yields.index[0]]), (1 + monthly_accrual).cumprod()]
    )
    index_level.name = params["research_name"]
    return index_level


# ── Equity total return ──────────────────────────────────────────────────────

def build_equity_total_return(price: pd.Series, div_yield: pd.Series, cfg: dict[str, Any]) -> pd.Series:
    """Monthly total return = price return + (annual `div_yield` / 12),
    chained into a cumulative index (base = 1.0 at the first observation).

    Source: already-scraped multpl price + dividend yield, not a new Shiller
    fetcher (Claude's-Discretion choice — see docs/splicing_rules.md for the
    Shiller cross-check option).

    Units: multpl's ``div_yield`` is already a DECIMAL fraction (0.011), unlike
    the FRED yield columns — hence no ``yield_to_decimal()`` call here. That
    asymmetry is exactly why the conversion is per-class config, not a blanket
    rule.
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


def source_candidates(params: dict[str, Any], key: str) -> list[str]:
    """Normalize `params[key]` into an ordered list of candidate column names.

    A list value is taken as-is; a scalar becomes a one-element list; a
    missing/None value becomes an empty list. When `key` is `"source_col"`,
    the legacy `fallback_col` value (if present and not already listed) is
    appended — the entire back-compat surface for the old one-off
    `single_source` fallback spelling, so a config still written the legacy
    way (`source_col: x` + `fallback_col: y`) keeps working unedited.
    De-duplicates while preserving order.
    """
    raw_value = params.get(key)
    if raw_value is None:
        candidates: list[Any] = []
    elif isinstance(raw_value, list):
        candidates = list(raw_value)
    else:
        candidates = [raw_value]

    if key == "source_col":
        fallback = params.get("fallback_col")
        if fallback is not None and fallback not in candidates:
            candidates.append(fallback)

    seen: set[Any] = set()
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            deduped.append(candidate)
    return deduped


def resolve_source_column(params: dict[str, Any], key: str, available: set[str]) -> str | None:
    """Return the first candidate for `key` that is present in `available`, else None."""
    for candidate in source_candidates(params, key):
        if candidate in available:
            return candidate
    return None


def resolve_class_sources(params: dict[str, Any], available: set[str]) -> dict[str, str] | None:
    """Resolve every source-column key a splice class's `method` needs.

    Returns a `key -> resolved column` mapping, or None if ANY required key
    has no resolvable candidate in `available`. This is the single place that
    decides which column a class uses — both the preflight validation and
    the assembly step below call it, so they can never disagree.
    """
    method = params.get("method", "")
    resolved: dict[str, str] = {}
    for key in _SPLICE_SOURCE_COL_KEYS.get(method, ()):
        col = resolve_source_column(params, key, available)
        if col is None:
            return None
        resolved[key] = col
    return resolved


def write_splice_provenance(provenance: dict[str, Any], path: Path) -> Path:
    """Write splice `provenance` (as produced on `build_core_research_series()`'s
    return value `.attrs["splice_provenance"]`) to `path` as indented JSON,
    stamped with a `captured_at` UTC ISO timestamp. Pure I/O helper — the
    rest of this module stays free of file I/O.
    """
    payload = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def build_core_research_series(raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Dispatch on each core class's `method` and assemble the 5
    `research_name` columns from `cfg['splice']` (D-03/D-04).

    Preflight-validates that every required class's source columns are
    resolvable before assembling, so a failed upstream fetch (e.g.
    macrotrends returning no gold_spot/wti_crude, or a yfinance rate-limit)
    yields one clear, actionable error naming every candidate tried instead
    of a bare ``KeyError`` deep in the loop.

    Every class's resolution is logged at INFO on every run (which column,
    and its 1-based position in its candidate chain) and recorded in the
    returned frame's ``.attrs["splice_provenance"]`` — a per-class dict with
    `class_name`, `method`, `status` (`primary`/`fallback`/`skipped`), and a
    `sources` map of each source-column key's candidate list, resolved
    column, and position.
    """
    splice_cfg = cfg["splice"]
    available = set(raw.columns)

    # ── Preflight: report every missing source column for required classes,
    # in one error, before assembling anything. Optional classes (e.g. gold
    # when its only free source is IP-blocked) never block the build — they
    # are skipped at assembly time with a WARNING instead.
    missing: list[str] = []
    for class_name, params in splice_cfg.items():
        if params.get("optional", False):
            continue
        method = params.get("method", "")
        if resolve_class_sources(params, available) is not None:
            continue
        research_name = params.get("research_name", class_name)
        for key in _SPLICE_SOURCE_COL_KEYS.get(method, ()):
            if resolve_source_column(params, key, available) is not None:
                continue
            candidates = source_candidates(params, key)
            missing.append(
                f"  - {research_name} ({method}): missing column for '{key}' "
                f"(tried: {candidates})"
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
    provenance: dict[str, Any] = {}

    for class_name, params in splice_cfg.items():
        method = params["method"]
        research_name = params["research_name"]
        keys = _SPLICE_SOURCE_COL_KEYS.get(method, ())
        resolved_map = resolve_class_sources(params, available)

        if resolved_map is None:
            # Optional class whose source data never arrived → skip entirely.
            tried = {key: source_candidates(params, key) for key in keys}
            log.warning(
                "splice '%s': OPTIONAL class skipped — no candidate resolvable (tried %s). "
                "The backtest will run without this asset.",
                research_name, tried,
            )
            provenance[research_name] = {
                "class_name": class_name,
                "method": method,
                "status": "skipped",
                "sources": {
                    key: {"candidates": tried[key], "resolved": None, "position": None}
                    for key in keys
                },
            }
            continue

        source_detail: dict[str, dict[str, Any]] = {}
        any_fallback = False
        for key in keys:
            candidates = source_candidates(params, key)
            resolved_col = resolved_map[key]
            position = candidates.index(resolved_col) + 1
            source_detail[key] = {
                "candidates": candidates,
                "resolved": resolved_col,
                "position": position,
            }
            any_fallback = any_fallback or position != 1
            log.info(
                "splice '%s': resolved %s -> '%s' (candidate %d of %d)",
                research_name, key, resolved_col, position, len(candidates),
            )
        status = "fallback" if any_fallback else "primary"

        if method == "total_return_from_price_div":
            series = build_equity_total_return(
                raw[resolved_map["price_col"]], raw[resolved_map["div_yield_col"]], cfg
            )
        elif method == "cmt_par_bond_repricing":
            series = build_treasury_tr_synthetic(raw[resolved_map["yield_col"]], cfg)
        elif method == "single_source":
            series = raw[resolved_map["source_col"]].dropna().rename(research_name)
        elif method == "yield_as_return":
            series = build_cash_index(raw[resolved_map["yield_col"]], cfg)
        elif method == "ratio_splice":
            series = ratio_splice(
                raw[resolved_map["old_col"]], raw[resolved_map["new_col"]], pd.Timestamp(params["join_date"])
            ).rename(research_name)
        else:
            raise ValueError(f"Unknown splice method '{method}' for class '{class_name}'")

        columns[research_name] = series
        provenance[research_name] = {
            "class_name": class_name,
            "method": method,
            "status": status,
            "sources": source_detail,
        }

    log.info("Built %d core research series: %s", len(columns), sorted(columns))
    result = pd.concat(columns, axis=1)
    result.attrs["splice_provenance"] = provenance
    return result
