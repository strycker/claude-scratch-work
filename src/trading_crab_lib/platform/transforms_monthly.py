"""
Monthly feature-table assembly — the phase's vertical join point (DATA-01, DATA-03, DATA-04).

Orchestrates the ingestion + splice + vintage layers built by Plans 01-01..01-06 into
ONE monthly feature table indexed by month-end back to ~1962:

  1. Fetch monthly macro/long-history raw data (``macro_monthly.fetch_macro_monthly``).
  2. Fetch daily universe prices + derive a monthly spine (``prices_daily.fetch_universe_prices``).
  3. Apply the measured publication lags once (``publication_lags.apply_publication_lags``,
     D-01), then build the 5 core research series from the lagged frame via
     ratio-splice/synthesis (``splice.build_core_research_series``).
  4. Point-in-time-align the D-06 agency series (``align_agency_monthly``) — value_as_of
     where ALFRED vintages exist, publication-lag shift fallback before the earliest
     recorded vintage (RESEARCH Pitfall 4: vintage-correction subsumes the shift once
     vintages exist).
  5. Merge everything NULL-tolerantly via ``pd.concat([...], axis=1)`` (RESEARCH
     Pitfall 5 — never ``pd.merge``/``.join`` defaults) onto a canonical month-end index
     spanning ``cfg['data']['start_date']`` forward.
  6. Compute the lean fast/slow taxonomy-tagged feature set (``compute_lean_features`` +
     ``tag_feature_columns``) and persist ``daily_raw``, ``monthly_raw``, and
     ``monthly_features`` checkpoints in the platform namespace.

Mirrors the incumbent ``transforms.py::engineer_all``'s "fixed step order, each step a
named helper" structure (frozen, not edited — D-01). Imports the Plan 02/03/05/06
modules by absolute path; never edits any frozen incumbent module.

Usage:
    from trading_crab_lib.platform.transforms_monthly import build_monthly_spine
    from trading_crab_lib.platform.config import load_platform_config

    cfg = load_platform_config()
    monthly_features = build_monthly_spine(cfg)
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform import splice, taxonomy
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.honesty.holdout import write_monthly_features_split
from trading_crab_lib.platform.ingestion import alfred, macro_monthly, prices_daily
from trading_crab_lib.platform.ingestion.publication_lags import (
    LAG_MARKER_FILENAME,
    apply_publication_lags,
    lag_marker_matches,
    lag_table,
    write_lag_marker,
)

log = logging.getLogger(__name__)


# ── Point-in-time agency alignment (DATA-03 runtime) ────────────────────────


def _shift_fallback_series(
    all_releases: pd.DataFrame, monthly_index: pd.DatetimeIndex, monthly_freq: str, lag: int = 1
) -> pd.Series:
    """Build the D-06 pre-vintage-era fallback: each reference period's value
    from the **latest** vintage, resampled onto the monthly spine and shifted
    by *lag* periods on the reference-month grid (the incumbent's
    publication-lag ``shift()`` convention — ``ingestion/fred.py`` ADR #7).

    *lag* is the series' ``publication_lags.<name>.fallback_months``. For a
    quarterly series dated at the quarter start it must cover the whole
    quarter plus the release delay: GDP's Q1 (reference Jan) is released about
    the end of April, so ``lag=3`` makes it visible at Apr 30. The old
    hard-coded 1 showed it at Feb 28 (08.1 ruling 1).

    Latest vintage, not first-published, and the distinction is the whole
    point. ALFRED's real-time database begins part-way through a series'
    history — for CPIAUCSL the earliest ``realtime_start`` is 1972-07-21, and
    that first vintage only covers reference periods from 1970-12 onward.
    Taking each period's *first-published* value therefore mixes bases at that
    boundary:

      * reference <= 1970-11 -> earliest available row is a 1994 vintage -> 39.6
        (1982-84=100)
      * reference >= 1970-12 -> earliest available row is the 1972 vintage -> 119.03
        (1967=100)

    which put a 3.0x cliff into ``fred_cpi`` at 1971-01 after the ``shift(1)``.
    Reading one consistent (latest) vintage cannot do that.

    Nothing point-in-time is lost. ``align_with_fallback`` consults this series
    only for as-of dates *before* the earliest recorded vintage — where no
    vintage existed, so "first published" was already a later revision anyway.
    Using today's revision there is D-06's documented accepted compromise, and
    it is what the incumbent shift convention this docstring cites actually
    does.
    """
    cols = alfred._detect_vintage_columns(all_releases)
    date_col, rs_col, value_col = cols["date"], cols["realtime_start"], cols["value"]

    releases = all_releases.copy()
    # fredapi emits NaT/NaN in `value` for a release that restated nothing.
    # Those rows must not win the per-period pick.
    releases[value_col] = pd.to_numeric(releases[value_col], errors="coerce")
    releases = releases.dropna(subset=[value_col])

    latest = (
        releases.sort_values(rs_col)
        .groupby(date_col)
        .tail(1)
        .set_index(date_col)[value_col]
        .sort_index()
    )
    latest.index = pd.to_datetime(latest.index)

    monthly = latest.resample(monthly_freq).last().reindex(monthly_index).ffill()
    return monthly.shift(lag)


_DISCONTINUITY_RATIO = 1.5


def _warn_on_level_discontinuity(series: pd.Series, name: str, *, kind: str = "index") -> list[pd.Timestamp]:
    """Flag month-over-month jumps too large to be economics (audit item A3).

    An index-level series does not move 50% in a month. When one does, it is a
    units or index-base error, and the specific failure this guards is the one
    that shipped: point-in-time vintages of a *rebased* index spliced as if the
    bases were comparable, which put two ~2.99x cliffs into ``fred_cpi``
    (1970-12 39.60 -> 1971-01 119.03; 1988-01 345.9 -> 1988-02 115.9) and drove
    ``real_rate_level`` to a range of -209..+74 — silently, for weeks.

    **Only applied to index-level series** (``kind="index"``, the default).
    Rates genuinely make moves of this size: UNRATE went 4.4% -> 14.7% in April
    2020 (3.3x) and the 10-year yield fell 42% in March 2020. Running a ratio
    threshold over them produces false positives on the most economically
    important months in the sample, which is the fastest way to teach everyone
    to ignore the warning. Declare a rate series with ``kind: rate`` in
    ``cfg['fred_vintage']['series']``.

    That distinction is a fact about the series, not a threshold fitted to the
    data — which is the answer to Phase 6 D-11's objection that plausibility
    bounds "get tuned to whatever the current data happens to look like".

    WARNING rather than raise: a genuine level break that the source itself
    publishes should be seen, not used to abort an otherwise good build. The A3
    finding is that nothing was *looking*, not that everything should be fatal.

    Returns:
        list: the dates whose step exceeded the threshold (empty when clean, or
        when ``kind`` is not an index-level series).
    """
    if kind != "index":
        return []
    clean = series.dropna()
    if len(clean) < 2:
        return []
    prev = clean.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (clean / prev).replace([np.inf, -np.inf], np.nan).dropna()
    breaks = ratio[(ratio > _DISCONTINUITY_RATIO) | (ratio < 1 / _DISCONTINUITY_RATIO)]
    if len(breaks):
        log.warning(
            "align_agency_monthly: %s has %d implausible month-over-month step(s) "
            "(>%.0f%% or <%.0f%%) at %s — a units or index-base error, not economics. "
            "First: %s -> %s.",
            name, len(breaks), (_DISCONTINUITY_RATIO - 1) * 100,
            (1 / _DISCONTINUITY_RATIO - 1) * 100,
            [str(d.date()) for d in breaks.index[:5]],
            round(float(prev.loc[breaks.index[0]]), 4),
            round(float(clean.loc[breaks.index[0]]), 4),
        )
    return list(breaks.index)


def _series_kind(cfg: dict[str, Any], friendly_name: str) -> str:
    """The declared ``kind`` for a vintage series — ``index`` unless config says
    otherwise. Index-level is the safe default: a rate mistakenly guarded costs
    a false warning, an index mistakenly unguarded costs a silent base splice."""
    for meta in cfg.get("fred_vintage", {}).get("series", {}).values():
        if meta.get("name") == friendly_name:
            return str(meta.get("kind", "index"))
    return "index"


def align_agency_monthly(
    monthly_index: pd.DatetimeIndex, cfg: dict[str, Any], fred_client: Any | None = None
) -> pd.DataFrame:
    """Point-in-time-align every ``cfg['fred_vintage']['series']`` column onto ``monthly_index``.

    Uses ``alfred.fetch_all_vintages`` (bulk, one call per series) then
    ``alfred.align_with_fallback`` per series: point-in-time reconstruction via
    ``value_as_of`` where ALFRED vintages exist, publication-lag-shift fallback
    before the earliest recorded vintage — no revision or timing look-ahead
    (DATA-03 runtime application; RESEARCH Pitfall 4).

    Args:
        monthly_index: canonical month-end spine to align every series onto.
        cfg: platform config (``cfg['fred_vintage']['series']``, ``cfg['data']``).
        fred_client: optional pre-built ``fredapi.Fred`` client. When given,
            each series is fetched individually via ``alfred.fetch_vintage_series``
            (bypassing ``fetch_all_vintages``' own internal client construction) —
            useful for callers that already hold a client. When omitted (the
            default), ``alfred.fetch_all_vintages(cfg)`` builds its own client.
    """
    monthly_freq = cfg["data"].get("monthly_freq", "ME")

    if fred_client is not None:
        series_cfg: dict = cfg["fred_vintage"]["series"]
        all_vintages: dict[str, pd.DataFrame] = {}
        for series_id, meta in series_cfg.items():
            friendly_name = meta["name"]
            try:
                all_vintages[friendly_name] = alfred.fetch_vintage_series(fred_client, series_id)
            except Exception as exc:  # noqa: BLE001 — fredapi raises various types
                log.warning("Failed to fetch vintage history for %s (%s): %s", friendly_name, series_id, exc)
    else:
        all_vintages = alfred.fetch_all_vintages(cfg)

    if not all_vintages:
        log.warning("align_agency_monthly: no vintage series fetched")
        return pd.DataFrame(index=monthly_index)

    table = lag_table(cfg)
    unlisted = [n for n in all_vintages if not isinstance(table.get(n), dict)]
    if unlisted:
        raise ValueError(
            f"align_agency_monthly: {unlisted} need a publication_lags entry "
            "{vintage: true, fallback_months: N} — the pre-vintage fallback lag is measured, not defaulted."
        )

    columns: dict[str, pd.Series] = {}
    for name, releases in all_vintages.items():
        shift_series = _shift_fallback_series(
            releases, monthly_index, monthly_freq, lag=table[name]["fallback_months"]
        )
        aligned = alfred.align_with_fallback(releases, monthly_index, shift_series)
        aligned.name = name
        _warn_on_level_discontinuity(aligned, name, kind=_series_kind(cfg, name))
        columns[name] = aligned

    df = pd.concat(columns, axis=1)
    df.index.name = "date"
    log.info("align_agency_monthly: aligned %d agency series onto %d months", len(columns), len(monthly_index))
    return df


# ── Lean feature computation + taxonomy tagging (DATA-04) ──────────────────


def compute_lean_features(monthly_raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute the taxonomy-tagged full-history (1962+) lean feature set (DATA-04).

    Simple, single-purpose derivations only — column names match
    ``cfg['taxonomy']`` exactly, no speculative multi-scale expansion beyond
    the concrete taxonomy list (design §9). A source column missing from
    ``monthly_raw`` (e.g. a failed ingestion source) simply skips the
    dependent feature rather than raising — mirrors the incumbent's
    graceful-degradation convention.
    """
    cols = set(monthly_raw.columns)
    features: dict[str, pd.Series] = {}

    if {"fred_gs10", "fred_tb3ms"} <= cols:
        features["curve_10y3m"] = monthly_raw["fred_gs10"] - monthly_raw["fred_tb3ms"]
    if "fred_t10y2y" in cols:
        # Already the 10Y-2Y spread as published by FRED — passthrough, no
        # re-derivation (no GS2 series is ingested separately).
        features["curve_10y2y"] = monthly_raw["fred_t10y2y"]
    if {"fred_baa", "fred_aaa"} <= cols:
        features["credit_spread_baa_aaa"] = monthly_raw["fred_baa"] - monthly_raw["fred_aaa"]
    if "fred_vix" in cols:
        features["fred_vix"] = monthly_raw["fred_vix"]
    if "gold" in cols:
        features["gold"] = monthly_raw["gold"]
    if "oil" in cols:
        features["oil"] = monthly_raw["oil"]

    if "equities_tr" in cols:
        equity_returns = monthly_raw["equities_tr"].pct_change(fill_method=None)
        features["trailing_return_1m"] = equity_returns
        features["trailing_return_3m"] = monthly_raw["equities_tr"].pct_change(3, fill_method=None)
        # ponytail: naive single-period vol proxy (|1m return|) for
        # realized_vol_1m — true intra-period vol needs daily equity prices,
        # not available for the multpl-derived monthly equities_tr research
        # series. Upgrade if/when a daily total-return series exists.
        features["realized_vol_1m"] = equity_returns.abs()
        features["realized_vol_3m"] = equity_returns.rolling(3).std(ddof=0)

    if "cape_shiller" in cols:
        features["cape_shiller"] = monthly_raw["cape_shiller"]
    if "div_yield" in cols:
        features["div_yield"] = monthly_raw["div_yield"]
    if {"fred_gs10", "fred_cpi"} <= cols:
        # Real rate = nominal 10Y yield minus trailing-12-month CPI inflation
        # (YoY % change, percentage points) — both series are 1962+ and
        # already present in monthly_raw (fred_gs10 fast-layer; fred_cpi via
        # align_agency_monthly). No free 1962+ market-cap source exists yet
        # for buffett_indicator (see taxonomy.slow comment in
        # config/platform_settings.yaml) — real_rate_level has no such gap.
        cpi_yoy_pct = monthly_raw["fred_cpi"].pct_change(periods=12, fill_method=None) * 100.0
        features["real_rate_level"] = monthly_raw["fred_gs10"] - cpi_yoy_pct

    if not features:
        return pd.DataFrame(index=monthly_raw.index)
    return pd.concat(features, axis=1)


def tag_feature_columns(features_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, str]:
    """Map every produced feature column to its taxonomy tier (DATA-04).

    Logs a WARNING listing any untagged column via
    ``taxonomy.check_columns_tagged`` — ``compute_lean_features`` is designed
    to only ever emit taxonomy-listed column names, so this is a defensive
    gate, not expected to fire in normal operation.
    """
    columns = list(features_df.columns)
    untagged = taxonomy.check_columns_tagged(columns, cfg)
    if untagged:
        log.warning("Untagged lean feature column(s) (DATA-04 gap): %s", untagged)
    return {col: taxonomy.classify_feature(col, cfg) for col in columns}


def features_from_raw(monthly_raw: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """The ONE ``monthly_features`` assembly, shared by ``build_monthly_spine`` and
    ``scripts/recompute_monthly_features.rebuild_monthly_features``.

    Drops the P&L-only raw columns (``splice.pnl_only_columns``) FIRST — the 08.3
    L2 leak guard: L2 fits on every ``monthly_features`` column, so a month-end
    close here would be a new model input (D-01 keeps features unchanged).
    ``monthly_raw`` itself keeps them; the P&L builder reads them there. Then the
    lean features are computed, concatenated onto the raw columns, and duplicate
    labels deduped keeping the lean copy.
    """
    pnl_only = splice.pnl_only_columns(cfg)
    raw = monthly_raw.drop(columns=[c for c in monthly_raw.columns if c in pnl_only])

    lean = compute_lean_features(raw, cfg)
    tag_feature_columns(lean, cfg)  # WARNING-only defensive taxonomy-coverage check

    monthly_features = pd.concat([raw, lean], axis=1)
    # Passthrough lean columns (gold/oil/fred_vix/cape_shiller/div_yield) are
    # identical to their monthly_raw source — dedupe, keeping the lean copy.
    monthly_features = monthly_features.loc[:, ~monthly_features.columns.duplicated(keep="last")]
    monthly_features.index.name = "date"
    return monthly_features


# ── Orchestrator ─────────────────────────────────────────────────────────────


def last_complete_month_end(today: date | pd.Timestamp) -> pd.Timestamp:
    """The last month-end strictly before *today*'s month — the newest month that
    is complete. On the last day of a month that month is still running (its
    multpl row is the current price, its daily prints are pre-close), so it is
    not a row: 2026-09-30 -> 2026-08-31, 2026-10-01 -> 2026-09-30."""
    return pd.Timestamp(today).normalize() - pd.offsets.MonthEnd(1)


def _assert_lag_marker_allows_merge(cm: Any, cfg: dict[str, Any]) -> None:
    """Refuse to merge onto a monthly_raw built under a different lag table.

    monthly_raw is merge-on-save: saving a lagged frame over an unlagged disk
    copy would refill the first L rows of every lagged column from disk with
    their UNLAGGED values (08.1 Pitfall 1)."""
    if not (cm.dir / "monthly_raw.parquet").exists():
        return
    marker = cm.dir / LAG_MARKER_FILENAME
    if lag_marker_matches(marker, cfg):
        return
    state = "missing" if not marker.exists() else "records a different lag table"
    # BuildFailed (a RuntimeError), so the build script exits 1 with this message (2026-10-09).
    raise BuildFailed(
        f"build_monthly_spine: nothing was written, because {marker} is {state}, so the on-disk "
        "monthly_raw was not built under the current publication_lags and merging onto it would "
        "reintroduce unlagged values. If it predates publication lags, run "
        "`python scripts/migrate_publication_lags.py` once. If a lag or source was changed on purpose, "
        "build into an empty data folder (TC_DATA_DIR) or delete monthly_raw (and the marker) and rebuild."
    )


# ── Fail-loud build gate (08.4, DECISIONS D-08) ──────────────────────────────


class BuildFailed(RuntimeError):
    """A source the build needs is missing; raised before anything is written."""


def expected_source_columns(cfg: dict[str, Any]) -> dict[str, str]:
    """Every raw column the build expects, mapped to the config section that names it."""
    expected: dict[str, str] = {}
    for meta in cfg.get("fred_monthly", {}).get("series", {}).values():
        expected[meta["name"]] = "fred_monthly"
    for row in cfg.get("multpl_monthly", {}).get("datasets", []):
        expected[row[0]] = "multpl_monthly"
    for entry in cfg.get("macrotrends_monthly", {}).get("series", []):
        expected[entry["name"]] = "macrotrends_monthly"
    for entry in (cfg.get("worldbank_monthly") or {}).get("series", []):
        expected[entry["name"]] = "worldbank_monthly"
    for meta in cfg.get("index_monthly", {}).values():
        expected[meta["name"]] = "index_monthly"
    for ticker in prices_daily.universe_fetch_tickers(cfg):
        expected[ticker] = "universe"
    for meta in cfg.get("fred_vintage", {}).get("series", {}).values():
        expected[meta["name"]] = "fred_vintage"
    return expected


def missing_sources(cfg: dict[str, Any], delivered: set[str]) -> list[str]:
    """Every expected raw column that was neither delivered nor allowed (allowed ones are logged)."""
    allowed = set(cfg.get("build", {}).get("allow_missing_sources", []))
    problems: list[str] = []
    for column, section in expected_source_columns(cfg).items():
        if column in delivered:
            continue
        if column in allowed:
            log.warning("build: source column '%s' (%s) is missing; build.allow_missing_sources allows it.", column, section)
        else:
            problems.append(f"source column '{column}' ({section}) was not delivered")
    return problems


def fallback_splices(cfg: dict[str, Any], provenance: dict[str, Any] | None) -> list[str]:
    """Every splice class that fell back from a primary column that is not allowed to be missing."""
    allowed = set(cfg.get("build", {}).get("allow_missing_sources", []))
    problems: list[str] = []
    for research_name, record in (provenance or {}).items():
        if record.get("status") != "fallback":
            continue
        for key, detail in record["sources"].items():
            primary = detail["candidates"][0]
            if detail["position"] != 1 and primary not in allowed:
                problems.append(f"splice '{research_name}' fell back from '{primary}' to '{detail['resolved']}' ({key})")
    return problems


_BLOCKED_HINT = (
    "macrotrends (HTTP 403) and Yahoo (rate limits) block many corporate VPNs and firewalls; if you are "
    "on one, run the build off it. "
)


def _raise_if_fred_key_rejected(cfg: dict[str, Any], delivered: set[str]) -> None:
    """Every FRED series missing means the key, not the series: say so, and do not suggest
    --allow-missing for all of them (08.4 UAT failure drill, 2026-10-09)."""
    fred = [column for column, section in expected_source_columns(cfg).items() if section.startswith("fred_")]
    if fred and not set(fred) & delivered:
        raise BuildFailed(
            f"build_monthly_spine: nothing was written, because every FRED series failed ({len(fred)} columns: "
            f"{', '.join(fred)}). That is FRED_API_KEY, not the series: check the key in your environment or .env "
            "(free key: https://fred.stlouisfed.org/docs/api/api_key.html), then re-run "
            "python scripts/build_platform_data.py"
        )


_COLUMN_IN_PROBLEM = re.compile(r"(?:source column|fell back from) '([^']+)'")


def _raise_build_failed(problems: list[str]) -> None:
    if problems:
        blocked = any("macrotrends_monthly" in p or "index_monthly" in p for p in problems)
        columns = ",".join(dict.fromkeys(_COLUMN_IN_PROBLEM.findall(" ".join(problems))))
        raise BuildFailed(
            "build_monthly_spine: nothing was written, because " + "; ".join(problems) + ". "
            + (_BLOCKED_HINT if blocked else "")
            + "Retry: python scripts/build_platform_data.py (a flaky source usually comes back). To build "
            f"without them on purpose (each one's class falls back, and the run logs it): python "
            f"scripts/build_platform_data.py --allow-missing {columns}"
        )


def build_monthly_spine(cfg: dict[str, Any]) -> pd.DataFrame:
    """Assemble the monthly feature table (DATA-01, DATA-03 runtime, DATA-04).

    Orchestrates, in order: monthly macro ingestion, daily universe prices +
    monthly spine, core research series (splice), point-in-time agency
    alignment, and lean feature computation — merged NULL-tolerantly via
    ``pd.concat([...], axis=1)`` onto a canonical month-end index spanning
    ``cfg['data']['start_date']`` forward. Persists ``daily_raw``,
    ``monthly_raw``, and ``monthly_features`` checkpoints in the platform
    namespace and returns the monthly_features frame.
    """
    monthly_freq = cfg["data"].get("monthly_freq", "ME")
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or last_complete_month_end(date.today())
    monthly_index = pd.date_range(start=start, end=end, freq=monthly_freq)

    macro = macro_monthly.fetch_macro_monthly(cfg)
    daily, monthly_prices = prices_daily.fetch_universe_prices(cfg)
    agency = align_agency_monthly(monthly_index, cfg)

    # Fail-loud gate, part 1 (08.4, D-08): every source checked BEFORE any splice runs, so a lost
    # or silently empty column stops the build with its name instead of a crash further down.
    fail_loud = bool(cfg.get("build", {}).get("fail_loud"))
    if fail_loud:
        # A column that arrived but is entirely NaN (a silent parse failure) was not delivered.
        delivered = {c for frame in (macro, monthly_prices, agency) for c in frame.columns if frame[c].notna().any()}
        _raise_if_fred_key_rejected(cfg, delivered)
        _raise_build_failed(missing_sources(cfg, delivered))

    # The splice input MUST include the monthly ticker columns (e.g. IAU), not
    # just macro-only columns — otherwise a fallback chain naming a tradable
    # ETF (gold: [gold_wb, IAU]) could never resolve, even though the ETF
    # data is right there in monthly_prices. Guarded only for the both-empty
    # case: if either macro or monthly_prices has data, splice still runs on
    # whatever is available (a class missing its required macro columns then
    # raises its own actionable preflight error, rather than silently
    # skipping the whole splice step).
    #
    # Publication lags (D-01) are applied HERE, once, to the whole ingest frame:
    # the research series and monthly_raw are both built from the lagged frame,
    # so no consumer can reach an unlagged copy. Agency columns are added
    # afterwards and never pass through it (D-02: vintage-aligned already).
    splice_input_frames = [f for f in (macro, monthly_prices) if not f.empty]
    if splice_input_frames:
        splice_input = pd.concat(splice_input_frames, axis=1)
        lagged = apply_publication_lags(splice_input, cfg)
        research = splice.build_core_research_series(lagged, cfg)
    else:
        lagged = pd.DataFrame()
        research = pd.DataFrame()

    frames = [f for f in (lagged, research, agency) if not f.empty]
    monthly_raw = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=monthly_index)
    monthly_raw = monthly_raw.reindex(monthly_index)
    monthly_raw.index.name = "date"

    # Fail-loud gate, part 2: a splice class that fell back from its primary column.
    if fail_loud:
        _raise_build_failed(fallback_splices(cfg, research.attrs.get("splice_provenance") if not research.empty else None))

    cm = get_platform_checkpoint_manager()
    _assert_lag_marker_allows_merge(cm, cfg)  # before ANY write
    cm.save(daily, "daily_raw", source="prices_daily.fetch_universe_prices (universe price chain)")
    raw_path = cm.save(
        monthly_raw, "monthly_raw",
        source="build_monthly_spine (combined monthly ingest: macro+prices+research+agency)",
        # The spliced research columns are derived: replaced by every build, never refilled from
        # an old disk copy (08.4, the 221defc -98% gold month).
        replace_columns=[params["research_name"] for params in cfg.get("splice", {}).values()],
    )
    write_lag_marker(cm.dir / LAG_MARKER_FILENAME, cfg)

    splice_provenance = research.attrs.get("splice_provenance") if not research.empty else None
    if splice_provenance:
        splice.write_splice_provenance(splice_provenance, cm.dir / "splice_provenance.json")

    # ── Derive features from the AS-SAVED raw, never the pre-merge frame ─────
    #
    # ``monthly_raw`` is a merge-on-save checkpoint: CheckpointManager.save()
    # merges the frame above with whatever is already on disk
    # (merge_preserving) so a degraded fetch cannot silently truncate history.
    # It returns a Path, not the merged frame — so deriving features from the
    # local ``monthly_raw`` variable builds them from data that is NOT what
    # landed on disk, and the two checkpoints are then free to disagree.
    #
    # They did. A build on 2026-09-18 resolved the oil splice to macrotrends
    # ``wti_crude`` (1985-02+), merge-on-save preserved an older 1962-01+ oil
    # column in monthly_raw, and monthly_features got the 1985+ version. The
    # two checkpoints, written milliseconds apart, disagreed by 23 years on a
    # frozen labeling feature, and the staleness was diagnosed backwards for a
    # week because each artifact looked self-consistent.
    #
    # Re-reading closes the loop: features are now a pure function of the raw
    # a reader can actually load. test_platform_monthly_spine_consistency.py
    # asserts it.
    monthly_raw = pd.read_parquet(raw_path).reindex(monthly_index)
    monthly_raw.index.name = "date"

    monthly_features = features_from_raw(monthly_raw, cfg)

    # HON-01: carve at the holdout boundary rather than writing one unfenced
    # checkpoint. Dev rows (<= 2020-12) land in the default platform namespace;
    # post-cutoff rows go to data/holdout/, which the default manager cannot
    # read. Anything that legitimately needs the full span — live weekly
    # scoring, the verification notebooks — opts in by name via
    # honesty.holdout.load_full_span(). The fence is on fitting, not looking.
    write_monthly_features_split(monthly_features, "monthly_features")

    log.info(
        "build_monthly_spine: assembled %d months, %d columns (monthly_features)",
        len(monthly_features), len(monthly_features.columns),
    )
    return monthly_features
