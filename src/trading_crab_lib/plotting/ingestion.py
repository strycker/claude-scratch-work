from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trading_crab_lib.plotting.core import (
    CUSTOM_COLORS,
    PLOT_DIR,
    REGIME_CMAP,
    RunConfig,
    _regime_color,
    _save_or_show,
)

log = logging.getLogger(__name__)

# ── Step 01: Ingestion ─────────────────────────────────────────────────────────

def plot_raw_series_coverage(
    raw: pd.DataFrame,
    run_cfg: RunConfig,
    max_cols: int = 50,
) -> None:
    """
    Heatmap of non-NaN coverage across all raw series.
    Columns = series, rows = quarters — dark = data available.
    """
    # Binarize: 1 = has data, 0 = NaN
    coverage = raw.notna().astype(int)
    # Limit columns for legibility
    coverage = coverage.iloc[:, :max_cols]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(
        coverage.T.values,
        aspect="auto",
        cmap="Blues",
        vmin=0, vmax=1,
        interpolation="nearest",
    )
    n_quarters = len(raw)
    tick_step = max(1, n_quarters // 10)
    ax.set_xticks(range(0, n_quarters, tick_step))
    ax.set_xticklabels(
        [str(raw.index[i].year) for i in range(0, n_quarters, tick_step)],
        rotation=45, ha="right", fontsize=7,
    )
    ax.set_yticks(range(len(coverage.columns)))
    ax.set_yticklabels(coverage.columns, fontsize=6)
    ax.set_title("Raw Series Coverage (dark = data available)", fontsize=12)
    ax.set_xlabel("Quarter")
    plt.colorbar(im, ax=ax, shrink=0.5, label="Has data")
    fig.tight_layout()
    _save_or_show(fig, "01_raw_coverage.png", run_cfg)


def plot_raw_series_sample(
    raw: pd.DataFrame,
    series: list[str],
    run_cfg: RunConfig,
    filename: str = "01_raw_series_sample.png",
    title: str = "Raw Series Sample",
) -> None:
    """Line chart for a subset of raw series (for quick visual QC)."""
    series = [s for s in series if s in raw.columns]
    if not series:
        return

    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, series):
        ax.plot(raw.index, raw[col], linewidth=1.2)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Quarter")
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)

