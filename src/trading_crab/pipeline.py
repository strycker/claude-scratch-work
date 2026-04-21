"""
trading_crab.pipeline — Master entry point for the Trading-Crab market regime pipeline.

Runs all pipeline steps in order, or any selected subset, with a consistent
RunConfig passed through every module.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PIPELINE STEPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1  ingest         Scrape multpl.com + FRED + macrotrends + ETF prices → macro_raw.parquet
  2  features       Log transforms, derivatives, gap-fill → features.parquet
  3  cluster        PCA + KMeans → cluster_labels.parquet
  4  regime_label   Statistical profiling + human-readable names → profiles.parquet
  5  predict        Supervised classifiers (current + forward horizons)
  6  asset_returns  ETF returns by regime (prices cached from step 1)
  7  dashboard      Print dashboard + save outputs/reports/dashboard.csv
  8  diagnostics    Ratio + RRG diagnostics → outputs/reports/diagnostics/
  9  tactics        Per-asset buy_hold / swing / stand_aside classification

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ALL CLI FLAGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --refresh            Re-scrape multpl.com + re-hit FRED API (~10 min).
                       Without this flag, steps 1-2 load from cached checkpoints
                       if they are less than 7 days old.

  --recompute          Recompute derived features (step 2) from cached raw data.
                       Use after editing config/settings.yaml feature lists or
                       transforms.py without wanting to re-scrape.

  --refresh-assets     Re-fetch ETF prices from yfinance (step 6 only).
                       Without this flag, step 6 reuses data/raw/asset_prices.parquet
                       if it already exists. Useful when behind a firewall or when
                       ETF data hasn't changed since the last run.

  --plots              Generate and save matplotlib figures to outputs/plots/.
                       Each step produces its own set of charts.

  --show-plots         Also call plt.show() after each figure.
                       Off by default; do NOT use in CI or headless environments.

  --verbose            Set logging level to DEBUG (very chatty).

  --steps 1,3,5        Run only the listed step numbers (comma-separated integers).
                       Example: --steps 3,4,5,6,7 skips ingestion and features.
                       Valid values: 1 2 3 4 5 6 7

  --no-constrained     Skip the k-means-constrained package even if installed.
                       Falls back to plain KMeans for balanced clustering.
                       Use if you haven't run: pip install k-means-constrained

  --no-drop-tail       Include the most-recent (potentially incomplete) quarter.
                       By default the trailing row is dropped when it contains NaN
                       in any feature column — a side effect of the centered
                       np.gradient edge window in step 2.

  --market-code NAME   Inject a market_code label column into macro_raw so that
                       downstream models and notebooks can overlay regime labels.
                       NAME must be one of:
                         grok        Load the original Grok AI-generated labels
                                     from data/grok_*.pickle (cached automatically
                                     to market_code_grok checkpoint on first use)
                         clustered   Load labels saved by a prior --save-market-code
                                     run (checkpoint: market_code_clustered)
                         predicted   Load labels auto-saved by step 5 on its last
                                     run (checkpoint: market_code_predicted)
                         <any name>  Load checkpoint "market_code_<NAME>"
                       Omit entirely for a fully data-driven run with no label seed.

  --save-market-code   After step 3 completes, save the balanced_cluster column as
                       the "market_code_clustered" checkpoint.  Use this so future
                       runs can reference these cluster assignments with
                       --market-code clustered.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AUTO-SAVED CHECKPOINTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Step 5 automatically saves the predicted current-regime labels as the
  "market_code_predicted" checkpoint every time it runs.  No flag needed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 COMMON WORKFLOWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 ① FRESH START — scrape everything, seed with Grok labels (recommended first run):
     python run_pipeline.py --refresh --recompute --plots \\
         --market-code grok --save-market-code

 ② FULLY DATA-DRIVEN — no label seed, cluster from data only:
     python run_pipeline.py --refresh --recompute --plots --save-market-code

 ③ FAST RE-RUN — skip scraping, use cached checkpoints, regenerate plots:
     python run_pipeline.py --steps 3,4,5,6,7 --plots

 ④ RE-CLUSTER ONLY — update cluster assignments, save for downstream:
     python run_pipeline.py --steps 3 --save-market-code --plots

 ⑤ DOWNSTREAM WITH NEW CLUSTER LABELS — use labels saved in ④:
     python run_pipeline.py --steps 4,5,6,7 --market-code clustered --plots

 ⑥ DOWNSTREAM WITH GROK SEED — overlay original AI labels:
     python run_pipeline.py --steps 4,5,6,7 --market-code grok --plots

 ⑦ DOWNSTREAM WITH PREDICTED LABELS — use last step-5 predictions:
     python run_pipeline.py --steps 4,5,6,7 --market-code predicted --plots

 ⑧ RECOMPUTE FEATURES WITHOUT RE-SCRAPING (e.g., after editing settings.yaml):
     python run_pipeline.py --recompute --steps 2,3,4,5,6,7 --plots

 ⑨ ETF DATA REFRESH ONLY (no macro re-scrape):
     python run_pipeline.py --steps 6,7 --refresh-assets --plots

 ⑩ DEBUG A SINGLE STEP:
     python run_pipeline.py --steps 3 --verbose --plots --show-plots

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MARKET CODE EXPLAINED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The "market_code" is a per-quarter integer label (0-4) that serves as the
  reference regime assignment.  It is attached to macro_raw in step 1 and
  propagated through all downstream steps as an overlay/reference column.

  Four sources are available:
    grok        Original AI-assisted labels (circa 2026-02-16).  Useful as a
                stable reference baseline — these never change.
    clustered   Labels from the most recent --save-market-code run.  Updated
                every time you run step 3 with --save-market-code.
    predicted   Labels from the most recent step 5 run.  Reflects the current
                trained classifier's best guess for historical quarters.
    (omitted)   Run without a market_code column.  Clustering is fully
                data-driven; no external label is injected.

  To list all available market_code checkpoints:
    python -c "
    from trading_crab_lib.io.checkpoints import CheckpointManager
    cm = CheckpointManager()
    mc = [e for e in cm.list() if e['name'].startswith('market_code_')]
    for e in mc: print(e['name'], '—', e.get('rows', '?'), 'rows')
    "
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd  # noqa: F401 — used in type annotations

from trading_crab_lib import CONFIG_DIR, DATA_DIR, OUTPUT_DIR
from trading_crab_lib.config import load, setup_logging
from trading_crab_lib.email import (
    build_weekly_email_body,
    load_email_config,
    send_weekly_email,
)
from trading_crab_lib.runtime import RunConfig

log = logging.getLogger(__name__)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_parquet(canonical_path: Path, checkpoint_name: str) -> pd.DataFrame:
    """
    Load a DataFrame from its canonical inter-step path, falling back to the
    CheckpointManager when the file doesn't exist.

    This lets steps 3-7 work even when the upstream step was run on a different
    machine and only its checkpoint was committed to the repo.
    """
    import pandas as pd

    from trading_crab_lib.checkpoints import CheckpointManager

    if canonical_path.exists():
        return pd.read_parquet(canonical_path)

    log.info(
        "%s not found — loading from checkpoint '%s'",
        canonical_path.name, checkpoint_name,
    )
    df = CheckpointManager().load(checkpoint_name)
    # Backfill the canonical file so subsequent reads are fast
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(canonical_path)
    return df


# ── market_code helpers ───────────────────────────────────────────────────────

def _load_market_code(
    source: str,
    cfg: dict,
) -> pd.Series | None:
    """
    Load a market_code Series from the specified source.

    Args:
        source: "grok" to load from the grok pickle, or any other string to
                load checkpoint "market_code_{source}".

    Returns:
        pd.Series of integer codes indexed by quarter-end dates, or None on failure.
    """
    from trading_crab_lib.checkpoints import CheckpointManager

    cm = CheckpointManager()

    if source == "grok":
        from trading_crab_lib.ingestion.grok import load_grok_labels
        mc = load_grok_labels(DATA_DIR)
        if mc is not None:
            # Cache so subsequent runs don't need to reload the pickle
            cm.save(mc.to_frame(), "market_code_grok")
        return mc

    # Load from checkpoint
    ckpt_name = f"market_code_{source}"
    try:
        df = cm.load(ckpt_name)
        mc = df.iloc[:, 0]  # single-column DataFrame → Series
        mc.name = "market_code"
        log.info("Loaded market_code from checkpoint: %s (%d rows)", ckpt_name, len(mc))
        return mc
    except FileNotFoundError:
        log.error(
            "market_code checkpoint '%s' not found. "
            "Available checkpoints: %s",
            ckpt_name,
            [e["name"] for e in cm.list() if e["name"].startswith("market_code_")],
        )
        return None


def _save_market_code(labels: pd.Series, name: str) -> None:
    """Persist a market_code variant (any integer-coded label Series) to a checkpoint."""

    from trading_crab_lib.checkpoints import CheckpointManager

    cm = CheckpointManager()
    ckpt_name = f"market_code_{name}"
    df = labels.rename("market_code").to_frame()
    cm.save(df, ckpt_name)
    log.info("Saved market_code checkpoint: %s (%d rows)", ckpt_name, len(labels))


# ── Step registry ──────────────────────────────────────────────────────────────

def _make_asset_prices_placeholder(cfg: dict) -> pd.DataFrame:
    """
    Build a minimal placeholder asset_prices DataFrame when all network sources fail.

    Uses the configured ETF tickers as columns and a quarterly DatetimeIndex
    spanning the configured date range.  All values are NaN — this signals to
    step 6 that real prices are unavailable and proxy returns should be used.

    The placeholder satisfies the two structural constraint tests:
      - columns are a subset of the configured ETF universe
      - the index is quarterly with midnight timestamps (not intraday)
    """
    from datetime import date

    import pandas as pd

    tickers: list[str] = cfg.get("assets", {}).get("etfs", [])
    start = cfg["data"]["start_date"]
    end = cfg["data"]["end_date"] or str(date.today())
    index = pd.date_range(start=start, end=end, freq="QE")
    if index.empty:
        index = pd.date_range(periods=1, end=pd.Timestamp.today(), freq="QE")

    import numpy as np
    data = {t: np.full(len(index), float("nan")) for t in tickers}
    df = pd.DataFrame(data, index=index)
    df.index.name = "date"
    return df


def _fetch_and_cache_asset_prices(
    cfg: dict, run_cfg: RunConfig
) -> pd.DataFrame:
    """
    Fetch ETF prices via yfinance/stooq/OpenBB and cache to asset_prices checkpoint.

    Called from step 1 (early ingestion) so that asset prices are available for
    feature engineering in step 2.  Step 6 reuses the cached checkpoint — it no
    longer re-fetches unless --refresh-assets is passed.

    Always writes an asset_prices checkpoint, even when all network sources fail.
    In that case a placeholder (correct tickers, quarterly index, NaN values) is
    written so that structural constraint tests never skip and step 6 can detect
    the absence of real data and fall back to proxy returns.

    Returns the prices DataFrame (placeholder if all sources fail).
    """
    import pandas as pd

    from trading_crab_lib.checkpoints import CheckpointManager
    from trading_crab_lib.ingestion.assets import fetch_all as fetch_prices

    cm = CheckpointManager()
    cache_path = DATA_DIR / "raw" / "asset_prices.parquet"

    prices: pd.DataFrame = pd.DataFrame()

    if run_cfg.refresh_asset_prices or not cache_path.exists():
        try:
            prices = fetch_prices(cfg)
            if not prices.empty:
                raw_dir = DATA_DIR / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                prices.to_parquet(cache_path)
                log.info(
                    "Step 1: fetched ETF prices — %d tickers, %d quarters",
                    len(prices.columns), len(prices),
                )
        except Exception as exc:
            log.warning("Step 1: ETF price fetch failed (non-fatal): %s", exc)

    if prices.empty and cache_path.exists():
        prices = pd.read_parquet(cache_path)
        log.info("Step 1: loaded cached ETF prices (%d tickers)", len(prices.columns))

    if prices.empty:
        # All sources failed and no raw cache — write a placeholder so that
        # (a) the asset_prices checkpoint always exists after step 1 runs, and
        # (b) structural constraint tests (columns ⊆ universe, no intraday) can run.
        # Step 6 detects the all-NaN placeholder and falls back to proxy returns.
        prices = _make_asset_prices_placeholder(cfg)
        log.warning(
            "Step 1: all ETF price sources failed — writing NaN placeholder "
            "(%d tickers).  Install yfinance/pandas-datareader for real prices.",
            len(prices.columns),
        )

    # Always sync to checkpoint so downstream steps can use cm.load("asset_prices")
    cm.save(prices, "asset_prices")

    return prices


def _merge_asset_prices_into_raw(
    combined: pd.DataFrame,
    prices: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Merge a curated subset of ETF quarterly prices into macro_raw.

    Only ETFs listed in cfg["features"]["asset_price_columns"] are included.
    The column names are lowercased (e.g., SPY → etf_spy) to avoid collisions
    with existing macro columns and to follow the snake_case convention.

    These columns then participate in step 2 feature engineering (log transform,
    gap fill, derivatives) like any other macro series.
    """

    asset_cols = cfg.get("features", {}).get("asset_price_columns", [])
    if not asset_cols or prices.empty:
        return combined

    merged_count = 0
    for ticker in asset_cols:
        if ticker in prices.columns:
            col_name = f"etf_{ticker.lower()}"
            combined[col_name] = prices[ticker].reindex(combined.index)
            merged_count += 1
            valid = combined[col_name].notna().sum()
            log.info(
                "Step 1: merged %s → %s (%d/%d quarters with data)",
                ticker, col_name, valid, len(combined),
            )

    if merged_count:
        log.info("Step 1: merged %d ETF price columns into macro_raw", merged_count)

    return combined


def step1_ingest(cfg: dict, run_cfg: RunConfig) -> None:
    """Scrape multpl.com + FRED + macrotrends + ETF prices → data/raw/macro_raw.parquet.
    Optionally attaches a market_code column from the configured source."""
    import pandas as pd

    from trading_crab_lib import plotting
    from trading_crab_lib.checkpoints import CheckpointManager
    from trading_crab_lib.ingestion import fred as fred_module
    from trading_crab_lib.ingestion import multpl as multpl_module

    cm = CheckpointManager()

    if not run_cfg.refresh_source_datasets and cm.is_fresh("macro_raw", max_age_days=7):
        log.info("Step 1: using cached macro_raw checkpoint")
        # Still need to re-attach market_code if source changed
        if run_cfg.market_code_source:
            raw_path = DATA_DIR / "raw" / "macro_raw.parquet"
            if raw_path.exists():
                combined = pd.read_parquet(raw_path)
                mc = _load_market_code(run_cfg.market_code_source, cfg)
                if mc is not None:
                    combined["market_code"] = mc.reindex(combined.index)
                    combined.to_parquet(raw_path)
                    cm.save(combined, "macro_raw")
                    log.info("Step 1: refreshed market_code=%s in cached macro_raw",
                             run_cfg.market_code_source)
        # Sync asset_prices checkpoint in case it was cleared without clearing the
        # raw cache.  _fetch_and_cache_asset_prices will load from the raw parquet
        # (no network hit) and write the checkpoint if it is missing.
        _fetch_and_cache_asset_prices(cfg, run_cfg)
        return

    log.info("Step 1: fetching FRED data …")
    fred_df = fred_module.fetch_all(cfg)

    log.info("Step 1: scraping multpl.com (%d series) …",
             len(cfg["multpl"]["datasets"]))
    multpl_df = multpl_module.fetch_all(cfg)

    combined = fred_df.join(multpl_df, how="outer") if not multpl_df.empty else fred_df

    # macrotrends.net — long-history commodity prices (gold, oil)
    if cfg.get("macrotrends", {}).get("series"):
        try:
            from trading_crab_lib.ingestion import macrotrends as mt_module
            log.info("Step 1: scraping macrotrends.net (%d series) …",
                     len(cfg["macrotrends"]["series"]))
            mt_df = mt_module.fetch_all(cfg)
            if not mt_df.empty:
                combined = combined.join(mt_df, how="outer")
                log.info("Step 1: macrotrends added %d columns", len(mt_df.columns))
        except Exception as exc:
            log.warning("Step 1: macrotrends fetch failed (non-fatal): %s", exc)

    # ETF prices — fetch and cache early so they're available for step 2
    prices = _fetch_and_cache_asset_prices(cfg, run_cfg)
    combined = _merge_asset_prices_into_raw(combined, prices, cfg)

    start = cfg["data"]["start_date"]
    combined = combined[combined.index >= start]

    # Optionally attach market_code
    if run_cfg.market_code_source:
        mc = _load_market_code(run_cfg.market_code_source, cfg)
        if mc is not None:
            combined["market_code"] = mc.reindex(combined.index)
            log.info(
                "Step 1: attached market_code (%s), %d/%d rows have labels",
                run_cfg.market_code_source,
                combined["market_code"].notna().sum(),
                len(combined),
            )

    raw_dir = DATA_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(raw_dir / "macro_raw.parquet")
    cm.save(combined, "macro_raw")

    # ── Preservation checkpoint (C7.3) ────────────────────────────────
    from trading_crab_lib.checkpoints import preservation_checkpoint_should_write
    if preservation_checkpoint_should_write(
        "macro_raw_secondary", cm,
        force=run_cfg.refresh_preservation_checkpoints,
    ):
        cm.save(combined, "macro_raw_secondary")
        log.info("Step 1: wrote preservation checkpoint macro_raw_secondary")

    if run_cfg.generate_plots:
        plotting.plot_raw_series_coverage(combined, run_cfg)
        # Sample a handful of economically meaningful raw series for a quick QC chart
        sample_series = [c for c in [
            "sp500", "fred_gdp", "us_infl", "10yr_ustreas", "div_yield", "fred_baa",
        ] if c in combined.columns]
        if sample_series:
            plotting.plot_raw_series_sample(combined, sample_series, run_cfg)

    # ── Completeness report (P23 + C1.1) ────────────────────────────────
    from trading_crab_lib.ingestion import ingestion_completeness_report
    from trading_crab_lib.monitoring import (
        count_source_columns,
        format_completeness_table,
        validate_date_range,
    )
    expected_cols: list[str] = []
    for series_id, meta in cfg.get("fred", {}).get("series", {}).items():
        expected_cols.append(meta.get("name", series_id.lower()))
    for ds in cfg.get("multpl", {}).get("datasets", []):
        expected_cols.append(ds[0] if isinstance(ds, list) else ds["name"])
    for ds in cfg.get("macrotrends", {}).get("series", []):
        expected_cols.append(ds["name"] if isinstance(ds, dict) else ds[0])
    for ticker in cfg.get("features", {}).get("asset_price_columns", []):
        expected_cols.append(f"etf_{ticker.lower()}")
    report = ingestion_completeness_report(combined, expected_columns=expected_cols)
    log.info("Ingestion completeness:\n%s", format_completeness_table(report))

    # ── Date-range validation (C1.2) ──────────────────────────────────
    date_report = validate_date_range(combined)
    log.info("Date-range check:\n%s", date_report.summary())

    # ── Per-source column count summary (C1.3) ────────────────────────
    source_counts = count_source_columns(combined, cfg)
    log.info("Source breakdown:\n%s", source_counts.summary())

    log.info("Step 1 done: %d rows × %d cols", len(combined), len(combined.columns))


def _generate_gap_fill_plots(raw, features, cfg, run_cfg, plotting) -> None:
    """Generate gap-fill before/after plots for 3 sample columns (C1.5).

    Compares raw data (after log transform but before gap fill) against
    the fully-engineered features to visualize where Bernstein interpolation
    and Taylor extrapolation filled gaps.
    """
    from trading_crab_lib.transforms import (
        add_cross_ratios,
        add_yield_curve_features,
        apply_log_transforms,
        select_features,
    )

    feat_cfg = cfg["features"]

    # Build pre-gap-fill snapshot: cross-ratios → log → select initial features
    pre_fill = add_cross_ratios(raw.copy())
    pre_fill = add_yield_curve_features(pre_fill)
    pre_fill = apply_log_transforms(pre_fill, feat_cfg["log_columns"])
    pre_fill = select_features(pre_fill, feat_cfg["initial_features"])

    # Columns to visualize — economically meaningful, likely to have gaps
    sample_cols = ["log_sp500", "log_us_cpi", "log_10yr_ustreas"]
    for col in sample_cols:
        if col not in pre_fill.columns or col not in features.columns:
            continue
        plotting.plot_gap_fill_before_after(
            pre_fill[col],
            features[col],
            run_cfg,
            series_name=col,
        )
    log.info("Step 2: generated gap-fill before/after plots for %s", sample_cols)


def step2_features(cfg: dict, run_cfg: RunConfig) -> None:
    """Engineer features from macro_raw → data/processed/features.parquet"""

    from trading_crab_lib import plotting
    from trading_crab_lib.checkpoints import CheckpointManager
    from trading_crab_lib.transforms import engineer_all

    cm = CheckpointManager()

    if not run_cfg.recompute_derived_datasets and cm.is_fresh("features", max_age_days=7):
        log.info("Step 2: using cached features checkpoint")
        return

    raw = _load_parquet(DATA_DIR / "raw" / "macro_raw.parquet", "macro_raw")

    log.info("Step 2: engineering features from %d × %d raw data …",
             len(raw), len(raw.columns))

    # Centered features (forward + backward window) — used for clustering (steps 3-4)
    features = engineer_all(raw, cfg, causal=False)
    out_dir = DATA_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "features.parquet")
    cm.save(features, "features")

    # Causal features (backward-only window) — used for supervised learning (steps 5-7).
    # Identical column names to features.parquet but no look-ahead in any derivative.
    features_sup = engineer_all(raw, cfg, causal=True)
    features_sup.to_parquet(out_dir / "features_supervised.parquet")
    cm.save(features_sup, "features_supervised")

    # Backwards-compatible aliases for plan-level artifact names.
    # These mirror the non-causal and causal feature sets produced above so
    # downstream plans can reference features_noncausal / features_causal
    # explicitly without changing the core pipeline semantics.
    cm.save(features, "features_noncausal")
    cm.save(features_sup, "features_causal")
    log.info(
        "Step 2: wrote features.parquet (centered) and features_supervised.parquet (causal)"
    )

    # ── Preservation checkpoints (C7.4) ───────────────────────────────
    from trading_crab_lib.checkpoints import preservation_checkpoint_should_write
    force_pres = run_cfg.refresh_preservation_checkpoints
    if preservation_checkpoint_should_write("features_secondary", cm, force=force_pres):
        cm.save(features, "features_secondary")
        log.info("Step 2: wrote preservation checkpoint features_secondary")
    if preservation_checkpoint_should_write(
        "features_supervised_secondary", cm, force=force_pres,
    ):
        cm.save(features_sup, "features_supervised_secondary")
        log.info("Step 2: wrote preservation checkpoint features_supervised_secondary")

    # ── Feature quality report (C1.4) ────────────────────────────────────
    from trading_crab_lib.monitoring import compute_feature_quality
    quality = compute_feature_quality(features)
    log.info("Step 2 quality:\n%s", quality.summary())

    if run_cfg.generate_plots:
        feat_only = features.drop(columns=["market_code"], errors="ignore")
        plotting.plot_feature_distributions(feat_only, run_cfg)
        plotting.plot_feature_correlations(feat_only, run_cfg)

        # ── Gap-fill before/after plots (C1.5) ────────────────────────
        # Compare raw (pre-gap-fill) vs filled for 3 sample columns
        _generate_gap_fill_plots(raw, features, cfg, run_cfg, plotting)

    log.info("Step 2 done: %d rows × %d feature cols", len(features), len(features.columns))


def step3_cluster(cfg: dict, run_cfg: RunConfig, save_market_code: bool = False) -> None:
    """PCA + KMeans clustering → data/regimes/cluster_labels.parquet.
    When save_market_code=True, also checkpoints balanced_cluster as market_code_clustered."""
    from sklearn.preprocessing import StandardScaler

    from trading_crab_lib import plotting
    from trading_crab_lib.checkpoints import CheckpointManager
    from trading_crab_lib.clustering import (
        evaluate_kmeans,
        fit_clusters,
        pick_best_k,
        reduce_pca,
    )

    cm = CheckpointManager()
    clust_cfg = cfg["clustering"]

    if (not run_cfg.recompute_derived_datasets
            and cm.is_fresh("cluster_labels", max_age_days=7)):
        log.info("Step 3: using cached cluster_labels checkpoint")
        return

    features = _load_parquet(DATA_DIR / "processed" / "features.parquet", "features")
    X = features.drop(columns=["market_code"], errors="ignore").dropna()
    n_dropped = len(features) - len(X)
    if n_dropped:
        log.info(
            "Step 3: dropped %d quarter(s) with NaN features before PCA "
            "(expected when market_code source doesn't cover all dates)",
            n_dropped,
        )

    pca_df, pca_model, scaler = reduce_pca(
        X,
        n_components=clust_cfg["n_pca_components"],
        random_state=clust_cfg["random_state"],
    )

    X_scaled = StandardScaler().fit_transform(pca_df.values)
    scores = evaluate_kmeans(
        X_scaled,
        k_range=range(2, clust_cfg["n_clusters_search"] + 1),
        random_state=clust_cfg["random_state"],
    )
    best_k = pick_best_k(scores, k_cap=clust_cfg["k_cap"])

    log.info("K-sweep: chose k=%d  (cap=%d)", best_k, clust_cfg["k_cap"])

    clustered = fit_clusters(
        pca_df,
        best_k=best_k,
        balanced_k=clust_cfg["balanced_k"],
        random_state=clust_cfg["random_state"],
        use_constrained=run_cfg.use_constrained_kmeans,
    )

    if "market_code" in features.columns:
        clustered["market_code"] = features["market_code"]

    out_dir = DATA_DIR / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)

    label_cols = ["cluster", "balanced_cluster"] + (
        ["market_code"] if "market_code" in clustered.columns else []
    )
    clustered[label_cols].to_parquet(out_dir / "cluster_labels.parquet")
    clustered.drop(columns=label_cols, errors="ignore").to_parquet(
        out_dir / "pca_components.parquet"
    )
    scores.to_parquet(out_dir / "kmeans_scores.parquet", index=False)

    cm.save(clustered[label_cols], "cluster_labels")
    cm.save(pca_df, "pca_components")

    # Optionally save balanced_cluster as a market_code checkpoint
    if save_market_code:
        _save_market_code(clustered["balanced_cluster"], "clustered")
        log.info(
            "Step 3: saved balanced_cluster as market_code_clustered checkpoint "
            "(use --market-code clustered on future runs)"
        )

    if run_cfg.generate_plots:
        regime_names: dict[int, str] = {}  # populated in step 4; use IDs for now
        plotting.plot_pca_scatter(pca_df, clustered["balanced_cluster"], regime_names, run_cfg)
        plotting.plot_elbow_curve(scores, best_k, run_cfg)
        plotting.plot_cluster_sizes(clustered["balanced_cluster"], regime_names, run_cfg)

        # ── C2.1: Scree + PCA loadings plots ─────────────────────────
        plotting.plot_scree(pca_model, run_cfg)
        feature_names = list(X.columns)
        plotting.plot_pca_loadings(pca_model, feature_names, run_cfg)

        # ── C2.2: Silhouette samples plot ─────────────────────────────
        plotting.plot_silhouette_samples(
            X_scaled, clustered["balanced_cluster"].loc[X.index], run_cfg,
        )

    # ── C2.3: Method comparison table ─────────────────────────────────
    # Compare KMeans balanced clustering against standard KMeans
    from trading_crab_lib.cluster_comparison import compare_all_methods
    from trading_crab_lib.monitoring import format_method_comparison
    labels_dict = {
        "KMeans (best-k)": clustered["cluster"].loc[X.index],
        "KMeans (balanced)": clustered["balanced_cluster"].loc[X.index],
    }
    comparison = compare_all_methods(pca_df, labels_dict)
    log.info("Step 3 method comparison:\n%s", format_method_comparison(comparison))
    if run_cfg.generate_plots:
        plotting.plot_method_comparison_table(comparison, run_cfg)

    log.info("Step 3 done: balanced_k=%d", clust_cfg["balanced_k"])


def step4_regime_label(cfg: dict, run_cfg: RunConfig) -> None:
    """Profile clusters → data/regimes/profiles.parquet + transition_matrix.parquet"""
    import yaml

    from trading_crab_lib import plotting
    from trading_crab_lib.regime import (
        build_profiles,
        build_transition_matrix,
        load_name_overrides,
        suggest_names,
    )

    features = _load_parquet(DATA_DIR / "processed" / "features.parquet", "features")
    labels = _load_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet", "cluster_labels")["balanced_cluster"]

    common = features.index.intersection(labels.index)
    features = features.loc[common]
    labels = labels.loc[common]

    profile = build_profiles(features, labels)
    profile.to_parquet(DATA_DIR / "regimes" / "profiles.parquet")

    auto_names = suggest_names(features, labels)
    overrides = load_name_overrides(CONFIG_DIR)
    regime_names = {**auto_names, **overrides}

    suggestions_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
    with open(suggestions_path, "w") as f:
        yaml.dump(regime_names, f, default_flow_style=False)

    tm = build_transition_matrix(labels)
    tm.to_parquet(DATA_DIR / "regimes" / "transition_matrix.parquet")

    # ── C2.4: Regime stability summary ───────────────────────────────
    from trading_crab_lib.monitoring import compute_regime_stability
    stability = compute_regime_stability(tm, labels)
    log.info("Step 4 regime stability:\n%s", stability.summary())

    if run_cfg.generate_plots:
        plotting.plot_transition_matrix(tm, regime_names, run_cfg)
        plotting.plot_regime_timeline(labels, regime_names, run_cfg)
        key_cols = [
            c for c in [
                "us_infl", "gdp_growth", "credit_spread", "sp500_pe",
                "log_cpi_d1", "10yr_ustreas_d1", "log_earn_d1",
            ] if c in features.columns
        ]
        if key_cols:
            plotting.plot_regime_profiles(features, labels, regime_names, key_cols, run_cfg)

        # ── C2.5: Feature-regime overlay for key indicators ──────────
        overlay_cols = [
            c for c in ["log_sp500_d1", "log_us_cpi_d1", "credit_spread", "10yr_ustreas_d1"]
            if c in features.columns
        ]
        for col in overlay_cols:
            plotting.plot_feature_regime_overlay(features[col], labels, regime_names, run_cfg)

    for rid, name in sorted(regime_names.items()):
        n = (labels == rid).sum()
        log.info("Cluster %d: %r  (%d quarters)", rid, name, n)

    log.info("Step 4 done")


def step5_predict(cfg: dict, run_cfg: RunConfig) -> None:
    """Train supervised classifiers → outputs/models/"""

    import pandas as pd

    from trading_crab_lib import plotting
    from trading_crab_lib.prediction import (
        HAS_LIGHTGBM,
        predict_current,
        train_current_regime,
        train_decision_tree,
        train_forward_classifiers,
        train_lightgbm,
    )

    # Step 5 uses causal (backward-window) features so no future data leaks into
    # training.  Falls back to centered features.parquet if supervised file absent
    # (e.g. after a partial run that pre-dates this change).
    sup_path = DATA_DIR / "processed" / "features_supervised.parquet"
    feat_path = sup_path if sup_path.exists() else DATA_DIR / "processed" / "features.parquet"
    if not sup_path.exists():
        log.warning(
            "Step 5: features_supervised.parquet not found — falling back to features.parquet. "
            "Re-run step 2 to generate causal features."
        )
    features = _load_parquet(feat_path, "features_supervised")
    labels = _load_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet", "cluster_labels")["balanced_cluster"]

    common = features.index.intersection(labels.index)
    X_raw = features.loc[common].drop(columns=["market_code"], errors="ignore")
    nan_cols = X_raw.columns[X_raw.isna().any()].tolist()
    if nan_cols:
        log.warning(
            "Step 5: dropping %d column(s) with NaN values (consider fixing gap-fill): %s",
            len(nan_cols), nan_cols,
        )
    X = X_raw.dropna(axis=1, how="any")
    y = labels.loc[common]

    current_model = train_current_regime(X, y, cfg)

    latest = predict_current(current_model, X)
    log.info("Latest quarter → regime %d", latest["regime"])
    for r, p in sorted(latest["probabilities"].items(), key=lambda x: -x[1]):
        log.info("  Regime %d: %.1f%%", r, p * 100)

    dt_model = train_decision_tree(X, y, cfg)

    lgbm_model = None
    if HAS_LIGHTGBM:
        lgbm_model = train_lightgbm(X, y, cfg)
    else:
        log.info("Step 5: LightGBM not installed — skipping (pip install lightgbm>=4.0)")

    forward_models = train_forward_classifiers(X, y, cfg)

    # ── C3.1: Per-fold CV accuracy summary ───────────────────────────────
    from trading_crab_lib.monitoring import CVFoldReport, compute_cv_fold_scores
    cv_report = CVFoldReport()
    n_splits = cfg.get("prediction", {}).get("cv_splits", 5)
    rf_scores = compute_cv_fold_scores(current_model, X, y, n_splits=n_splits)
    cv_report.add("RF", rf_scores)
    dt_scores = compute_cv_fold_scores(dt_model, X, y, n_splits=n_splits)
    cv_report.add("DT", dt_scores)
    if lgbm_model is not None:
        lgbm_scores = compute_cv_fold_scores(lgbm_model, X, y, n_splits=n_splits)
        cv_report.add("LGBM", lgbm_scores)
    log.info("Step 5 CV summary:\n%s", cv_report.summary())

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(current_model, model_dir / "current_regime.pkl")
    joblib.dump(dt_model, model_dir / "decision_tree.pkl")
    if lgbm_model is not None:
        joblib.dump(lgbm_model, model_dir / "lightgbm_regime.pkl")
    joblib.dump(forward_models, model_dir / "forward_classifiers.pkl")

    # Optionally save predicted labels as a market_code checkpoint
    predicted_labels = pd.Series(
        current_model.predict(X), index=X.index, name="market_code"
    ).astype(int)
    _save_market_code(predicted_labels, "predicted")
    log.info(
        "Step 5: saved predicted regime labels as market_code_predicted checkpoint "
        "(use --market-code predicted on future runs)"
    )

    if run_cfg.generate_plots:
        try:
            regime_names_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
            import yaml
            regime_names = {}
            if regime_names_path.exists():
                with open(regime_names_path) as f:
                    regime_names = yaml.safe_load(f) or {}
                regime_names = {int(k): v for k, v in regime_names.items()}
            plotting.plot_feature_importance(current_model, X.columns.tolist(), run_cfg)
            plotting.plot_forward_probabilities(latest, regime_names, run_cfg)
            plotting.plot_predicted_vs_actual(X, y, current_model, regime_names, run_cfg)

            # ── C3.2: CV fold accuracy + decision tree plots ─────────
            plotting.plot_cv_fold_accuracy(rf_scores, run_cfg, model_name="RF")
            plotting.plot_cv_fold_accuracy(
                dt_scores, run_cfg, model_name="DT",
                filename="05_cv_fold_accuracy_dt.png",
            )
            plotting.plot_decision_tree(
                dt_model, X.columns.tolist(), regime_names, run_cfg,
            )

            # ── C3.3: Calibration curve + model comparison bar ───────
            import numpy as _np
            y_proba_rf = current_model.predict_proba(X)
            plotting.plot_calibration_curve(y, y_proba_rf, regime_names, run_cfg)

            model_metrics: dict[str, dict[str, float]] = {
                "RF": {"accuracy": float(_np.mean(rf_scores))},
                "DT": {"accuracy": float(_np.mean(dt_scores))},
            }
            if lgbm_model is not None:
                model_metrics["LGBM"] = {"accuracy": float(_np.mean(lgbm_scores))}
            plotting.plot_model_comparison_bar(model_metrics, run_cfg)
        except Exception as exc:
            log.warning("Could not generate prediction plots: %s", exc)

    # ── Interpretability tree (Phase 9) ───────────────────────────────────────
    try:
        from sklearn.tree import export_text

        from trading_crab_lib.prediction.classifier import train_interpretability_tree
        tree_model, tree_features = train_interpretability_tree(current_model, X, y, cfg)
        tree_txt = export_text(tree_model, feature_names=tree_features)
        report_dir = OUTPUT_DIR / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        tree_path = report_dir / "current_regime_tree.txt"
        tree_path.write_text(tree_txt, encoding="utf-8")
        log.info("Wrote interpretability tree → %s", tree_path)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not generate interpretability tree: %s", exc)

    log.info("Step 5 done — models saved to %s", model_dir)


def step6_asset_returns(cfg: dict, run_cfg: RunConfig) -> None:
    """Load ETF prices (cached from step 1) → data/regimes/asset_return_profile.parquet.
    Falls back to macro-data proxy returns when prices are unavailable.
    Re-fetches only when --refresh-assets is passed."""
    import pandas as pd

    from trading_crab_lib import plotting
    from trading_crab_lib.asset_returns import (
        compute_proxy_returns,
        compute_quarterly_returns,
        returns_by_regime,
    )
    from trading_crab_lib.checkpoints import CheckpointManager

    cm = CheckpointManager()

    labels = _load_parquet(DATA_DIR / "regimes" / "cluster_labels.parquet", "cluster_labels")["balanced_cluster"]

    # Load cached prices (fetched by step 1); re-fetch only if --refresh-assets
    prices: pd.DataFrame | None = None
    cache_path = DATA_DIR / "raw" / "asset_prices.parquet"

    if run_cfg.refresh_asset_prices:
        # Re-fetch when explicitly requested
        prices = _fetch_and_cache_asset_prices(cfg, run_cfg)
    elif cache_path.exists():
        prices = pd.read_parquet(cache_path)
        # Ensure checkpoint is in sync — may be absent if step 1 took the
        # early-return path or if checkpoints were cleared without clearing raw.
        if not prices.empty:
            cm.save(prices, "asset_prices")
    else:
        # No cache — try fetching now
        prices = _fetch_and_cache_asset_prices(cfg, run_cfg)

    # Compute returns: use real ETF prices when available, macro proxies otherwise.
    # A prices DataFrame that is all-NaN is a placeholder written when all fetch
    # sources failed (no yfinance / network); treat it the same as missing data.
    returns: pd.DataFrame | None = None
    has_real_prices = (
        prices is not None
        and not prices.empty
        and prices.notna().any(axis=None)
    )
    if has_real_prices:
        returns = compute_quarterly_returns(prices)
        log.info("Step 6: using ETF price data (%d tickers)", len(returns.columns))
    else:
        log.warning(
            "Step 6: no ETF price data available — computing proxy returns from macro data"
        )
        macro_path = DATA_DIR / "raw" / "macro_raw.parquet"
        if macro_path.exists():
            macro_df = pd.read_parquet(macro_path)
            returns = compute_proxy_returns(macro_df)
            if returns.empty:
                log.warning("Step 6: proxy returns also empty — skipping")
                return
            log.info(
                "Step 6: proxy returns computed (%d quarters × %d assets)",
                len(returns), len(returns.columns),
            )
        else:
            log.warning("Step 6: macro_raw.parquet not found — skipping")
            return

    common = returns.index.intersection(labels.index)

    # profile: regime × ticker DataFrame of median returns
    profile = returns_by_regime(returns.loc[common], labels.loc[common])

    out_dir = DATA_DIR / "regimes"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(out_dir / "asset_return_profile.parquet")

    if run_cfg.generate_plots:
        try:
            regime_names_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
            import yaml
            regime_names = {}
            if regime_names_path.exists():
                with open(regime_names_path) as f:
                    regime_names = yaml.safe_load(f) or {}
                regime_names = {int(k): v for k, v in regime_names.items()}
            plotting.plot_asset_returns_by_regime(profile, regime_names, run_cfg)
            plotting.plot_asset_heatmap(profile, regime_names, run_cfg)
        except Exception as exc:
            log.warning("Could not generate asset plots: %s", exc)

    log.info("Step 6 done — asset return profile written")


def step7_dashboard(cfg: dict, run_cfg: RunConfig) -> None:
    """Print + save stoplight dashboard → outputs/reports/dashboard.csv
    Also computes portfolio weights and BUY/SELL/HOLD trade recommendations."""

    import pandas as pd
    import yaml

    from trading_crab_lib.asset_returns import rank_assets_by_regime
    from trading_crab_lib.prediction import predict_current
    from trading_crab_lib.reporting import (
        asset_signals,
        blended_regime_portfolio,
        generate_recommendation,
        print_dashboard,
        save_dashboard_csv,
        simple_regime_portfolio,
    )

    model_dir = OUTPUT_DIR / "models"
    current_model_path = model_dir / "current_regime.pkl"
    if not current_model_path.exists():
        log.warning("Step 7: current_regime.pkl not found — run step 5 first")
        return

    import joblib
    current_model = joblib.load(current_model_path)

    # Step 7 uses causal features for live scoring — same as step 5 training data.
    # Falls back to centered features.parquet when supervised file is absent.
    sup_path = DATA_DIR / "processed" / "features_supervised.parquet"
    feat_path = sup_path if sup_path.exists() else DATA_DIR / "processed" / "features.parquet"
    if not sup_path.exists():
        log.warning(
            "Step 7: features_supervised.parquet not found — falling back to features.parquet. "
            "Re-run step 2 to generate causal features."
        )
    features = _load_parquet(feat_path, "features_supervised")
    X = features.drop(columns=["market_code"], errors="ignore")
    # Align to the exact feature set the model was trained on
    if hasattr(current_model, "feature_names_in_"):
        X = X[current_model.feature_names_in_]
    else:
        X = X.dropna(axis=1, how="any")
    prediction = predict_current(current_model, X)

    tm = _load_parquet(DATA_DIR / "regimes" / "transition_matrix.parquet", "transition_matrix")

    # Load regime names (pinned overrides take precedence over auto-suggested)
    override_path = CONFIG_DIR / "regime_labels.yaml"
    suggested_path = DATA_DIR / "regimes" / "regime_names_suggested.yaml"
    regime_names: dict[int, str] = {}
    for path in [override_path, suggested_path]:
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            names = {int(k): v for k, v in raw.items() if not str(k).startswith("#")}
            if names:
                regime_names = names
                break

    # Load signal thresholds from config
    thresholds = cfg.get("dashboard", {}).get("signal_thresholds", None)

    asset_signals_df = pd.DataFrame()
    profile_path = DATA_DIR / "regimes" / "asset_return_profile.parquet"
    if profile_path.exists():
        # profile is regime × ticker; rank_assets_by_regime produces the flat form
        profile = pd.read_parquet(profile_path)
        ranked = rank_assets_by_regime(profile)
        asset_signals_df = asset_signals(ranked, prediction["regime"], thresholds=thresholds)

    # ── C3.5: QA gate — warn if any regime has suspiciously low probability ──
    from trading_crab_lib.monitoring import check_regime_probabilities
    qa_warnings = check_regime_probabilities(prediction["probabilities"])
    for w in qa_warnings:
        log.warning("Step 7 QA: %s", w)
    if not qa_warnings:
        log.info("Step 7 QA: all regime probabilities >= 5%% — OK")

    # ── C3.4: Forward probability evolution plot ─────────────────────────────
    from trading_crab_lib.regime import compute_forward_probabilities
    labels = _load_parquet(
        DATA_DIR / "regimes" / "cluster_labels.parquet", "cluster_labels"
    )["balanced_cluster"]
    forward_probs = compute_forward_probabilities(labels)
    if run_cfg.generate_plots and forward_probs:
        from trading_crab_lib import plotting
        plotting.plot_forward_prob_evolution(forward_probs, regime_names, run_cfg)

    print_dashboard(prediction, regime_names, asset_signals_df, tm)

    if not asset_signals_df.empty:
        save_dashboard_csv(asset_signals_df, OUTPUT_DIR / "reports")

    # ── Portfolio construction and trade recommendations ─────────────────────
    if profile_path.exists():
        profile = pd.read_parquet(profile_path)
        current_regime = prediction["regime"]
        probs = prediction["probabilities"]

        log.info("── Simple portfolio (top-3, regime %d) ──", current_regime)
        simple_weights = simple_regime_portfolio(profile, current_regime, top_n=3)

        log.info("── Blended portfolio (probability-weighted) ──")
        blended_weights = blended_regime_portfolio(profile, probs, top_n=3)

        log.info("── Trade recommendations (blended target vs all-cash) ──")
        recommendations = generate_recommendation(blended_weights)

        report_dir = OUTPUT_DIR / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        if not simple_weights.empty:
            simple_weights.to_frame("weight").to_csv(report_dir / "portfolio_simple.csv")
        if not blended_weights.empty:
            blended_weights.to_frame("weight").to_csv(report_dir / "portfolio_blended.csv")
        if not recommendations.empty:
            recommendations.to_csv(report_dir / "trade_recommendations.csv")
            log.info(
                "Trade recommendations saved to %s",
                report_dir / "trade_recommendations.csv",
            )

    log.info("Step 7 done")


from trading_crab_lib.diagnostics import (  # noqa: E402
    percentile_rank,
    rolling_zscore,
    rrg_for_benchmark,
)


def step8_diagnostics(cfg: dict, run_cfg: RunConfig) -> None:
    """Compute ratio and RRG diagnostics from ETF prices → outputs/reports/diagnostics/."""
    import pandas as pd

    prices_path = DATA_DIR / "raw" / "asset_prices.parquet"
    if not prices_path.exists():
        log.warning("Step 8: ETF prices %s not found; skipping diagnostics.", prices_path)
        return

    prices = pd.read_parquet(prices_path)
    tickers = cfg.get("assets", {}).get("etfs") or list(prices.columns)
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        log.warning("Step 8: no configured ETF columns in prices; skipping diagnostics.")
        return
    prices = prices[cols]

    diag_dir = OUTPUT_DIR / "reports" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    ratios_cfg = cfg.get("diagnostics", {}).get("ratios") or []
    if ratios_cfg:
        records = []
        for item in ratios_cfg:
            name = item.get("name")
            num = item.get("numerator")
            den = item.get("denominator")
            if not name or not num or not den or num not in prices.columns or den not in prices.columns:
                continue
            ratio_series = prices[num] / prices[den]
            z = rolling_zscore(ratio_series)
            pct = percentile_rank(ratio_series)
            latest = ratio_series.dropna().iloc[-1] if not ratio_series.dropna().empty else float("nan")
            latest_z = z.dropna().iloc[-1] if not z.dropna().empty else float("nan")
            latest_pct = pct.dropna().iloc[-1] if not pct.dropna().empty else float("nan")
            records.append({
                "name": name, "numerator": num, "denominator": den,
                "latest_value": latest, "latest_zscore": latest_z, "percentile": latest_pct,
            })
        if records:
            pd.DataFrame.from_records(records).to_parquet(diag_dir / "ratios_current.parquet", index=False)
            log.info("Step 8: wrote ratio diagnostics to %s", diag_dir / "ratios_current.parquet")

    benchmarks = cfg.get("diagnostics", {}).get("rrg_benchmarks") or ["SPY"]
    all_rrg = []
    for bench in benchmarks:
        df_b = rrg_for_benchmark(prices, bench)
        if not df_b.empty:
            all_rrg.append(df_b)
    if all_rrg:
        rrg_combined = pd.concat(all_rrg, ignore_index=True)
        rrg_path = diag_dir / "rrg_current.parquet"
        rrg_combined.to_parquet(rrg_path, index=False)
        log.info("Step 8: wrote RRG diagnostics to %s", rrg_path)

        # ── C4.1: RRG scatter plot ───────────────────────────────────
        if run_cfg.generate_plots:
            from trading_crab_lib import plotting
            plotting.plot_rrg_scatter(rrg_combined, run_cfg)

    log.info("Step 8 done")


from trading_crab_lib.tactics import classify_tactics, compute_tactics_metrics  # noqa: E402


def step9_tactics(cfg: dict, run_cfg: RunConfig) -> None:
    """Compute per-asset tactics signals and write tactics_signals.parquet."""
    import pandas as pd

    prices_path = DATA_DIR / "raw" / "asset_prices.parquet"
    labels_path = DATA_DIR / "regimes" / "cluster_labels.parquet"

    if not prices_path.exists():
        log.warning("Step 9: ETF prices checkpoint %s not found; skipping tactics.", prices_path)
        return
    if not labels_path.exists():
        log.warning("Step 9: cluster labels %s not found; skipping tactics.", labels_path)
        return

    prices = pd.read_parquet(prices_path)
    labels = pd.read_parquet(labels_path)["balanced_cluster"]

    metrics = compute_tactics_metrics(prices, labels, cfg)
    tactics_df = classify_tactics(metrics, cfg).reset_index()

    out_dir = OUTPUT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tactics_signals.parquet"
    tactics_df.to_parquet(out_path, index=False)

    # ── C4.2: Tactics summary ────────────────────────────────────────
    from trading_crab_lib.monitoring import format_tactics_summary
    log.info("Step 9 tactics summary:\n%s", format_tactics_summary(tactics_df))

    log.info("Step 9: tactics signals written to %s", out_path)


# ── Step dispatch table ────────────────────────────────────────────────────────

STEPS: dict[int, tuple[str, callable]] = {
    1: ("Ingest macro + ETF data",      step1_ingest),
    2: ("Engineer features",            step2_features),
    3: ("PCA + clustering",             step3_cluster),
    4: ("Regime profiling + labeling",  step4_regime_label),
    5: ("Supervised prediction",        step5_predict),
    6: ("Asset returns (cached prices)", step6_asset_returns),
    7: ("Dashboard",                    step7_dashboard),
    8: ("Diagnostics (ratios + RRG)",   step8_diagnostics),
    9: ("Tactics signals",              step9_tactics),
}


# ── Weekly report helpers (archive + email) ────────────────────────────────────

def archive_weekly_report(reports_dir: Path | None = None) -> None:
    """
    Copy weekly_report.md to weekly_YYYY-MM-DD.md and write email_body.txt.

    No-op if weekly_report.md does not exist. This mirrors the behaviour of
    scripts/run_weekly_report.py so that the full weekly flow can be driven
    directly via run_pipeline.
    """
    reports = reports_dir or (OUTPUT_DIR / "reports")
    report_path = reports / "weekly_report.md"
    if not report_path.exists():
        print(f"No weekly_report.md at {report_path} — skip archive/email body.")
        return

    today = date.today().isoformat()
    stamped = reports / f"weekly_{today}.md"
    stamped.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, stamped)
    print(f"Archived report → {stamped}")

    email_body_path = reports / "email_body.txt"
    body = report_path.read_text(encoding="utf-8")
    email_body_path.write_text(body, encoding="utf-8")
    print(f"Email body → {email_body_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trading-Crab market regime pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--refresh", action="store_true",
                   help="Re-scrape multpl.com + re-hit FRED API")
    p.add_argument("--recompute", action="store_true",
                   help="Recompute features from cached raw data")
    p.add_argument("--refresh-assets", action="store_true",
                   help=(
                       "Re-fetch ETF prices from yfinance (step 6). "
                       "Without this flag, step 6 loads from the cached "
                       "data/raw/asset_prices.parquet if it exists. "
                       "Useful behind firewalls: omit this flag to reuse "
                       "previously fetched prices without hitting the network."
                   ))
    p.add_argument("--refresh-preservation", action="store_true",
                   help=(
                       "Rewrite preservation checkpoints (*_secondary) even if they "
                       "already exist. Normally these are write-once and survive clear_all()."
                   ))
    p.add_argument("--plots", action="store_true",
                   help="Generate and save matplotlib figures")
    p.add_argument("--show-plots", action="store_true",
                   help="Call plt.show() after each figure")
    p.add_argument("--verbose", action="store_true",
                   help="Set logging to DEBUG")
    p.add_argument("--steps", type=str, default=None,
                   help="Comma-separated step numbers to run, e.g. 1,3,5")
    p.add_argument("--no-constrained", action="store_true",
                   help="Skip k-means-constrained (if package not installed)")
    p.add_argument("--no-drop-tail", action="store_true",
                   help=(
                       "Include the most-recent (potentially incomplete) quarter "
                       "in training and prediction rather than trimming it. "
                       "By default the trailing row is dropped when it contains "
                       "NaN in any feature column (centered np.gradient edge effect)."
                   ))
    p.add_argument("--market-code", type=str, default=None, metavar="NAME",
                   help=(
                       "Load market_code from this source. "
                       "'grok' loads the grok pickle; any other value loads "
                       "checkpoint 'market_code_{NAME}'. Omit to run without market_code."
                   ))
    p.add_argument("--save-market-code", action="store_true",
                   help=(
                       "After step 3, save balanced_cluster labels as the "
                       "'market_code_clustered' checkpoint for future use with "
                       "--market-code clustered."
                   ))
    p.add_argument(
        "--weekly-report",
        action="store_true",
        help=(
            "After running the selected steps, archive outputs/reports/weekly_report.md "
            "to a dated copy and write outputs/reports/email_body.txt."
        ),
    )
    p.add_argument(
        "--send-email",
        action="store_true",
        help=(
            "After weekly-report post-processing, send the weekly report email via "
            "config/email.local.yaml (see config/email.example.yaml)."
        ),
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging()
    run_cfg = RunConfig.from_args(args)
    run_cfg.apply_logging()

    cfg = load()

    # Set global random seeds for reproducibility.  Individual sklearn models
    # also use random_state=42, but np.random.seed() is needed for any code
    # path that calls np.random without an explicit seed (e.g. KMeans n_init).
    _seed = cfg.get("pipeline", {}).get("random_state", 42)
    import numpy as np
    np.random.seed(_seed)
    random.seed(_seed)
    log.info("Global random seed set to %d (pipeline.random_state)", _seed)

    # Determine which steps to run
    if args.steps:
        try:
            requested = {int(s.strip()) for s in args.steps.split(",")}
        except ValueError:
            parser.error("--steps must be comma-separated integers, e.g. 1,3,5")
    else:
        requested = set(STEPS.keys())

    invalid = requested - set(STEPS.keys())
    if invalid:
        parser.error(f"Unknown step numbers: {invalid}. Valid: {sorted(STEPS.keys())}")

    save_market_code = getattr(args, "save_market_code", False)

    print(f"\nTrading-Crab pipeline  [{run_cfg}]")
    print(f"Steps to run: {sorted(requested)}")
    if run_cfg.market_code_source:
        print(f"market_code source: {run_cfg.market_code_source}")
    print()

    # Ensure output dirs exist
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    # ── C4.4 + C4.5: Pipeline timing and health tracking ────────────────────
    import time as _time

    from trading_crab_lib.monitoring import PipelineHealthSummary
    health = PipelineHealthSummary()

    for step_num in sorted(requested):
        label, fn = STEPS[step_num]
        print(f"── Step {step_num}: {label} ──")
        t0 = _time.monotonic()
        try:
            # step3 needs the save_market_code flag
            if step_num == 3:
                fn(cfg, run_cfg, save_market_code=save_market_code)
            else:
                fn(cfg, run_cfg)
            elapsed = _time.monotonic() - t0
            health.record_step(step_num, elapsed)
            print(f"   ✓ done ({elapsed:.1f}s)\n")
        except Exception as exc:
            elapsed = _time.monotonic() - t0
            health.record_step(step_num, elapsed, failed=True)
            log.exception("Step %d failed: %s", step_num, exc)
            print(f"   ✗ FAILED: {exc}\n")
            sys.exit(1)

    # Optional weekly-report archive + email sending
    if getattr(args, "weekly_report", False) or getattr(args, "send_email", False):
        archive_weekly_report()

    if getattr(args, "send_email", False):
        email_cfg = load_email_config()
        if not email_cfg:
            print("Email config not found or invalid; skipping send.")
        else:
            subject, body = build_weekly_email_body(OUTPUT_DIR / "reports")
            ok = send_weekly_email(email_cfg, subject, body)
            if ok:
                print("Weekly report email sent.")
            else:
                print("Weekly report email failed to send (see logs).")

    # ── C4.5: Pipeline health summary ────────────────────────────────────
    print(health.summary())
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
