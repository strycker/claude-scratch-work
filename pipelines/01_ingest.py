"""
Pipeline step 1 — Data Ingestion (checkpointed)

Fetches macro data from FRED and multpl.com, merges into one wide DataFrame,
and writes data/raw/macro_raw.parquet.

Macro and ETF ingestion are fully config-driven.
Checkpoints are managed by CheckpointManager.
Behaviour and logging match `run_pipeline.py --steps 1`.

Run:
    python pipelines/01_ingest.py
    python pipelines/01_ingest.py --refresh
    python pipelines/01_ingest.py --refresh --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_crab import DATA_DIR
from trading_crab.config import load, setup_logging
from trading_crab.ingestion import fred as fred_module
from trading_crab.ingestion import multpl as multpl_module
from trading_crab.ingestion import ingestion_completeness_report
from trading_crab.runtime import RunConfig

import pandas as pd

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trading-Crab step 1 — data ingestion (FRED + multpl, with checkpoints)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scrape multpl.com + re-hit FRED API instead of using fresh checkpoints.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate and save ingestion QC plots.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Call plt.show() after each figure (avoid in headless/CI).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Set logging level to DEBUG for this step.",
    )
    parser.add_argument(
        "--market-code",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Attach a market_code column: 'grok' loads the Grok pickle; "
            "any other value loads checkpoint 'market_code_{NAME}'."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = load()

    # ── FRED ─────────────────────────────────────────────────────────────
    fred_df = fred_module.fetch_all(cfg)

    # ── multpl.com ────────────────────────────────────────────────────────
    multpl_df = multpl_module.fetch_all(cfg)

    # ── macrotrends.net ────────────────────────────────────────────────────
    mt_df = pd.DataFrame()
    if cfg.get("macrotrends", {}).get("series"):
        try:
            from trading_crab.ingestion import macrotrends as mt_module
            mt_df = mt_module.fetch_all(cfg)
        except Exception as exc:
            log.warning("macrotrends fetch failed (non-fatal): %s", exc)

    # ── Merge ─────────────────────────────────────────────────────────────
    if not multpl_df.empty:
        combined = fred_df.join(multpl_df, how="outer")
    else:
        combined = fred_df
    if not mt_df.empty:
        combined = combined.join(mt_df, how="outer")

    # Filter to configured date range
    start = cfg["data"]["start_date"]
    combined = combined[combined.index >= start]

    # Persist
    out_path = DATA_DIR / "raw" / "macro_raw.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path)
    print(f"Wrote {len(combined)} rows × {len(combined.columns)} cols → {out_path}")

    # ── Completeness report ─────────────────────────────────────────────
    # Build expected column list from config
    expected_cols: list[str] = []
    for series_id, meta in cfg.get("fred", {}).get("series", {}).items():
        expected_cols.append(meta.get("name", series_id.lower()))
    for ds in cfg.get("multpl", {}).get("datasets", []):
        expected_cols.append(ds[0] if isinstance(ds, list) else ds["name"])
    for ds in cfg.get("macrotrends", {}).get("series", []):
        expected_cols.append(ds["name"] if isinstance(ds, dict) else ds[0])

    report = ingestion_completeness_report(combined, expected_columns=expected_cols)
    print(report.summary())


if __name__ == "__main__":
    main()
