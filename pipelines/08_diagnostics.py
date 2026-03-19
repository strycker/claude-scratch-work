"""
Pipeline step 8 — Diagnostics (ratios + RRG-style rotation view).

This step is intentionally conservative: it only reads existing checkpoints
and ETF prices to compute diagnostic artifacts. It does not alter regimes,
features, or recommendations.

Run:
    python pipelines/08_diagnostics.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading_crab import DATA_DIR, OUTPUT_DIR  # noqa: E402
from trading_crab.config import load, setup_logging  # noqa: E402
from trading_crab.diagnostics import (  # noqa: E402
    normalize_100,
    percentile_rank,
    rolling_zscore,
    rrg_for_benchmark,
)

log = logging.getLogger(__name__)


def _load_etf_prices(cfg: dict) -> pd.DataFrame:
    """Load ETF price history from the existing checkpoint (no network)."""
    prices_path = DATA_DIR / "raw" / "asset_prices.parquet"
    if not prices_path.exists():
        log.warning("ETF prices checkpoint not found at %s", prices_path)
        return pd.DataFrame()
    prices = pd.read_parquet(prices_path)
    tickers = cfg.get("assets", {}).get("etfs") or list(prices.columns)
    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return pd.DataFrame()
    return prices[cols]


def _compute_ratios(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute configured ratios from ETF prices and summarize current readings.
    """
    ratios_cfg = cfg.get("diagnostics", {}).get("ratios") or []
    if prices.empty or not ratios_cfg:
        return pd.DataFrame()

    records: list[dict] = []
    for item in ratios_cfg:
        name = item.get("name")
        num = item.get("numerator")
        den = item.get("denominator")
        if not name or not num or not den:
            continue
        if num not in prices.columns or den not in prices.columns:
            continue
        ratio_series = prices[num] / prices[den]
        z = rolling_zscore(ratio_series)
        pct = percentile_rank(ratio_series)
        latest = ratio_series.dropna().iloc[-1] if not ratio_series.dropna().empty else float("nan")
        latest_z = z.dropna().iloc[-1] if not z.dropna().empty else float("nan")
        records.append(
            {
                "name": name,
                "numerator": num,
                "denominator": den,
                "latest_value": latest,
                "latest_zscore": latest_z,
                "percentile": pct,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    setup_logging()
    cfg = load()

    prices = _load_etf_prices(cfg)
    if prices.empty:
        print("No ETF prices available for diagnostics; skipping step 8.")
        return

    diag_dir = OUTPUT_DIR / "reports" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # Ratios / triggers
    ratios_df = _compute_ratios(prices, cfg)
    if not ratios_df.empty:
        out_ratios = diag_dir / "ratios_current.parquet"
        ratios_df.to_parquet(out_ratios, index=False)
        print(f"Wrote ratio diagnostics → {out_ratios}")

    # RRG diagnostics for configured benchmarks
    benchmarks = cfg.get("diagnostics", {}).get("rrg_benchmarks") or ["SPY"]
    all_rrg_frames: list[pd.DataFrame] = []
    for bench in benchmarks:
        df_b = rrg_for_benchmark(prices, bench)
        if not df_b.empty:
            all_rrg_frames.append(df_b)
    if all_rrg_frames:
        rrg_df = pd.concat(all_rrg_frames, ignore_index=True)
        out_rrg = diag_dir / "rrg_current.parquet"
        rrg_df.to_parquet(out_rrg, index=False)
        print(f"Wrote RRG diagnostics → {out_rrg}")


if __name__ == "__main__":
    main()
