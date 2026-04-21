from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import pandas as pd

from trading_crab_lib.plotting.core import (
    CUSTOM_COLORS,
    RunConfig,
    _regime_color,
    _save_or_show,
)

log = logging.getLogger(__name__)

# ── Specialty Diagnostic Plots (Phase A4) ─────────────────────────────────────


def plot_rrg_scatter(
    rrg_df: pd.DataFrame,
    run_cfg: RunConfig,
    *,
    filename: str = "08_rrg_scatter.png",
) -> None:
    """RRG 4-quadrant scatter plot with asset labels.

    Parameters
    ----------
    rrg_df : DataFrame
        Must have columns ``rs`` (relative strength) and ``rm`` (relative momentum),
        indexed by asset ticker. Optionally ``quadrant`` for coloring.
    """
    required = {"rs", "rm"}
    if not required.issubset(rrg_df.columns) or rrg_df.empty:
        return

    quad_colors = {
        "LEADING": "#50a000", "WEAKENING": "#f48c06",
        "LAGGING": "#d00000", "IMPROVING": "#0000d0",
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    # draw quadrant background
    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(100, color="gray", linewidth=0.8, linestyle="--")
    ax.text(101, 101, "LEADING", fontsize=8, color="#50a000", alpha=0.6)
    ax.text(99, 101, "IMPROVING", fontsize=8, color="#0000d0", alpha=0.6, ha="right")
    ax.text(99, 99, "LAGGING", fontsize=8, color="#d00000", alpha=0.6, ha="right", va="top")
    ax.text(101, 99, "WEAKENING", fontsize=8, color="#f48c06", alpha=0.6, va="top")

    for ticker, row in rrg_df.iterrows():
        quad = row.get("quadrant", "")
        color = quad_colors.get(str(quad).upper(), "gray")
        ax.scatter(row["rs"], row["rm"], c=color, s=60, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(str(ticker), (row["rs"], row["rm"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Relative Strength (RS)")
    ax.set_ylabel("Relative Momentum (RM)")
    ax.set_title("Relative Rotation Graph", fontsize=12)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_divergence_timeseries(
    div_features: pd.DataFrame,
    labels: pd.Series,
    run_cfg: RunConfig,
    *,
    cols: list[str] | None = None,
    filename: str = "02_divergence_timeseries.png",
) -> None:
    """Z-score divergence features over time with regime-transition markers.

    Parameters
    ----------
    div_features : DataFrame
        Must contain z-score divergence columns (e.g. ``div_spy_tlt_z_4q``).
    labels : Series
        Regime labels for detecting transitions.
    cols : list[str], optional
        Columns to plot. If None, auto-detect columns containing ``_z_``.
    """
    if cols is None:
        cols = [c for c in div_features.columns if "_z_" in c]
    cols = [c for c in cols if c in div_features.columns]
    if not cols:
        return

    common = div_features.index.intersection(labels.index)
    div = div_features.loc[common, cols]
    lab = labels.loc[common].astype(int)

    # find transition points
    transitions = lab.index[lab.diff().fillna(0) != 0]

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True, squeeze=False)

    for i, col in enumerate(cols):
        ax = axes[i][0]
        ax.plot(div.index, div[col].values, color=CUSTOM_COLORS[0], linewidth=1)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axhline(2, color=CUSTOM_COLORS[1], linestyle="--", alpha=0.5, linewidth=0.8)
        ax.axhline(-2, color=CUSTOM_COLORS[1], linestyle="--", alpha=0.5, linewidth=0.8)
        for t in transitions:
            ax.axvline(t, color=CUSTOM_COLORS[3], alpha=0.3, linewidth=0.8)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(alpha=0.2)

    axes[0][0].set_title("Divergence Z-Scores with Regime Transitions", fontsize=12)
    axes[-1][0].set_xlabel("Date")
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_momentum_dashboard(
    momentum_features: pd.DataFrame,
    labels: pd.Series,
    run_cfg: RunConfig,
    *,
    cols: list[str] | None = None,
    filename: str = "02_momentum_dashboard.png",
) -> None:
    """Grid of trailing momentum + relative strength for key series.

    Parameters
    ----------
    momentum_features : DataFrame
        Must contain momentum columns (e.g. ``sp500_mom_4q``).
    cols : list[str], optional
        Columns to plot. If None, auto-detect columns containing ``_mom_`` or ``_rs_``.
    """
    if cols is None:
        cols = [c for c in momentum_features.columns
                if "_mom_" in c or "_rs_" in c or "acceleration" in c]
    cols = [c for c in cols if c in momentum_features.columns]
    if not cols:
        return

    common = momentum_features.index.intersection(labels.index)
    mom = momentum_features.loc[common, cols]
    lab = labels.loc[common].astype(int)

    n = len(cols)
    ncols_grid = min(3, n)
    nrows = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows, ncols_grid, figsize=(5 * ncols_grid, 3.5 * nrows),
                             squeeze=False)

    for i, col in enumerate(cols):
        ax = axes[i // ncols_grid][i % ncols_grid]
        for cid in sorted(lab.unique()):
            mask = lab == cid
            ax.scatter(mom.index[mask], mom.loc[mask, col],
                       c=_regime_color(cid), s=10, alpha=0.6)
        ax.plot(mom.index, mom[col].values, color="black", linewidth=0.7, alpha=0.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_title(col, fontsize=9)
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=7)

    for i in range(n, nrows * ncols_grid):
        axes[i // ncols_grid][i % ncols_grid].set_visible(False)

    fig.suptitle("Momentum & Relative Strength Dashboard", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)
