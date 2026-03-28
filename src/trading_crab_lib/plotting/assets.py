from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# ── Step 06: Asset Returns ─────────────────────────────────────────────────────

def plot_asset_returns_by_regime(
    profile: pd.DataFrame,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
) -> None:
    """
    Grouped bar chart: median quarterly return per asset × regime.
    Assets on the x-axis, regime as the grouping variable.
    """
    if profile.empty:
        return

    # profile index = regime, columns = assets
    unique_regimes = sorted(profile.index.astype(int).unique())
    assets = profile.columns.tolist()
    n_regimes = len(unique_regimes)
    x = np.arange(len(assets))
    width = 0.7 / n_regimes

    fig, ax = plt.subplots(figsize=(max(10, len(assets) * 1.5), 5))

    for offset, rid in enumerate(unique_regimes):
        if rid not in profile.index:
            continue
        returns = profile.loc[rid, assets].values
        ax.bar(
            x + offset * width - width * n_regimes / 2,
            returns,
            width,
            label=regime_names.get(rid, f"Regime {rid}"),
            color=_regime_color(rid),
            alpha=0.85,
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(assets, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Median quarterly return")
    ax.set_title("Asset Returns by Regime", fontsize=12)
    ax.legend(fontsize=8, loc="best")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "06_asset_returns_by_regime.png", run_cfg)


def plot_asset_heatmap(
    profile: pd.DataFrame,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
) -> None:
    """
    Heatmap: regimes (rows) × assets (cols) — cell = median quarterly return.
    Green = positive, red = negative.
    """
    if profile.empty:
        return
    try:
        import seaborn as sns
    except ImportError:
        log.warning("seaborn not installed — skipping asset heatmap")
        return

    row_labels = [
        regime_names.get(int(i), f"Regime {int(i)}") for i in profile.index
    ]

    fig, ax = plt.subplots(figsize=(max(8, len(profile.columns)), max(4, len(profile) * 0.8)))
    sns.heatmap(
        profile.values,
        ax=ax,
        annot=True,
        fmt=".1%",
        cmap="RdYlGn",
        center=0,
        xticklabels=profile.columns.tolist(),
        yticklabels=row_labels,
        linewidths=0.5,
        cbar_kws={"label": "Median quarterly return"},
    )
    ax.set_title("Asset Returns Heatmap by Regime", fontsize=12)
    ax.tick_params(axis="x", labelsize=9, rotation=20)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    _save_or_show(fig, "06_asset_heatmap.png", run_cfg)


def plot_asset_return_distributions(
    returns: pd.DataFrame,
    labels: pd.Series,
    regime_names: dict[int, str],
    ticker: str,
    run_cfg: RunConfig,
) -> None:
    """
    Overlapping distribution (KDE or hist) of quarterly returns for one asset,
    one distribution per regime.
    """
    if ticker not in returns.columns:
        return

    unique_regimes = sorted(labels.dropna().astype(int).unique())
    fig, ax = plt.subplots(figsize=(9, 5))

    for cid in unique_regimes:
        mask = labels.astype(int) == cid
        data = returns.loc[mask & returns[ticker].notna(), ticker]
        if len(data) < 3:
            continue
        label = regime_names.get(cid, f"Regime {cid}")
        ax.hist(data, bins=15, density=True, alpha=0.45,
                color=_regime_color(cid), label=label, edgecolor="none")

    ax.set_xlabel(f"{ticker} quarterly return")
    ax.set_ylabel("Density")
    ax.set_title(f"{ticker} Return Distributions by Regime", fontsize=12)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, f"06_returns_dist_{ticker}.png", run_cfg)


