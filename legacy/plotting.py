"""
Plotting: all visualization helpers.

Designed to be called from notebooks or from the pipeline with --plots.
Every repeated plot pattern is extracted into a reusable function.
"""

import logging
from datetime import datetime
from itertools import combinations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from trading_crab.config import CUSTOM_COLORS, N_PCA_COMPONENTS

log = logging.getLogger(__name__)

MY_CMAP = ListedColormap(CUSTOM_COLORS)


# ===================================================================
# Core helpers
# ===================================================================

def _parse_dates(date_strings) -> list:
    """Convert string dates to datetime.date objects for matplotlib."""
    return [datetime.strptime(str(d), "%Y-%m-%d").date() for d in date_strings]


def plot_time_series(
    dates, values, colors=None, title: str = "",
    figsize: tuple = (8, 4), cmap=None, point_size: int = 10,
    ax=None,
) -> None:
    """
    Scatter-plot a time series colored by regime.

    Args:
        dates: sequence of date strings or date objects
        values: numeric values to plot
        colors: array for point coloring (regime labels)
        title: plot title
        figsize: figure size (ignored if ax is provided)
        cmap: colormap (defaults to MY_CMAP)
        point_size: marker size
        ax: optional axes to draw on (creates new figure if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Parse string dates if needed
    if len(dates) > 0 and isinstance(dates[0], str):
        dates = _parse_dates(dates)

    scatter_kw = dict(s=point_size, alpha=0.7, marker="o")
    if colors is not None:
        scatter_kw["c"] = colors
        scatter_kw["cmap"] = cmap or MY_CMAP

    ax.scatter(dates, values, **scatter_kw)

    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    if title:
        ax.set_title(title)
    ax.grid(True)


def plot_pc_pairs(
    reduced_df: pd.DataFrame,
    colors: np.ndarray,
    title_prefix: str = "",
    figsize: tuple = (4, 3),
) -> None:
    """
    Plot all pairwise combinations of principal components.

    Replaces ~200 lines of hand-written scatter plots with a loop
    over itertools.combinations.
    """
    pc_cols = [f"PC{i+1}" for i in range(N_PCA_COMPONENTS)]
    pc_cols = [c for c in pc_cols if c in reduced_df.columns]
    pairs = list(combinations(pc_cols, 2))

    log.info("Plotting %d PC pairs...", len(pairs))
    for pc_x, pc_y in pairs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            reduced_df[pc_x], reduced_df[pc_y],
            c=colors, cmap=MY_CMAP, s=10, alpha=0.7,
        )
        ax.set_xlabel(pc_x)
        ax.set_ylabel(pc_y)
        ax.set_title(f"{title_prefix}{pc_x} vs {pc_y}")
        ax.grid(True)
        plt.tight_layout()
        plt.show()


# ===================================================================
# Stage-specific plots
# ===================================================================

def plot_raw_series(df: pd.DataFrame, columns: list[str] = None) -> None:
    """Plot each column of a DataFrame as a time series."""
    columns = columns or list(df.columns)
    log.info("Plotting %d raw series...", len(columns))
    for col in columns:
        dates = _parse_dates(df.index.values)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(dates, df[col].values, color="blue", marker="o", markersize=2)
        formatter = mdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()
        ax.set_title(col)
        ax.grid(True)
        plt.tight_layout()
        plt.show()


def plot_interpolation_comparison(
    dates, original: np.ndarray, filled: np.ndarray,
    regimes: np.ndarray, title: str = "",
) -> None:
    """Show before/after interpolation for a single column."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(title)

    plot_time_series(dates, original, colors=regimes, title="Before", ax=ax1)
    plot_time_series(dates, filled, colors=regimes, title="After", ax=ax2)

    plt.tight_layout()
    plt.show()


def plot_derivatives(
    dates, values, d1, d2, d3, regimes, title: str = "",
) -> None:
    """Plot a feature and its smoothed 1st/2nd/3rd derivatives."""
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    fig.suptitle(title)

    labels = ["Value", "1st Derivative", "2nd Derivative", "3rd Derivative"]
    data = [values, d1, d2, d3]

    for ax, y, label in zip(axes, data, labels):
        plot_time_series(dates, y, colors=regimes, title=label, ax=ax)

    plt.tight_layout()
    plt.show()


def plot_kmeans_scores(scores: pd.DataFrame) -> None:
    """Plot silhouette, Calinski-Harabasz, and Davies-Bouldin scores vs k."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(scores["k"], scores["silhouette"], "o-", label="Silhouette")
    ax.plot(scores["k"], scores["calinski"] / scores["calinski"].max(),
            "s-", label="Calinski-Harabasz (scaled)")
    ax.plot(scores["k"], 1 / scores["davies_bouldin"], "^-", label="1 / Davies-Bouldin")
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title("Cluster Quality Metrics")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_regime_profile_heatmap(
    profile: pd.DataFrame, regime_names: dict[int, str] | None = None,
) -> None:
    """Heatmap of mean feature values per regime."""
    import seaborn as sns

    # Extract means only
    mean_cols = [col for col in profile.columns if col[1] == "mean"]
    means = profile[mean_cols].copy()
    means.columns = [col[0] for col in mean_cols]

    if regime_names:
        means.index = [regime_names.get(r, f"Regime {r}") for r in means.index]

    fig, ax = plt.subplots(figsize=(12, max(4, len(means))))
    sns.heatmap(means, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("Mean Feature Values by Regime")
    plt.tight_layout()
    plt.show()


def plot_transition_matrix(
    matrix: pd.DataFrame, regime_names: dict[int, str] | None = None,
) -> None:
    """Heatmap of regime transition probabilities."""
    import seaborn as sns

    display = matrix.copy()
    if regime_names:
        display.index = [regime_names.get(r, f"R{r}") for r in display.index]
        display.columns = [regime_names.get(r, f"R{r}") for r in display.columns]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(display, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Regime Transition Probabilities P(j | i)")
    ax.set_xlabel("To Regime")
    ax.set_ylabel("From Regime")
    plt.tight_layout()
    plt.show()


def plot_feature_importances(importances: pd.Series, top_n: int = 15) -> None:
    """Horizontal bar chart of top feature importances."""
    top = importances.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    ax.barh(top.index, top.values, color="#3a86ff")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()


def plot_portfolio_recommendation(rec: pd.DataFrame) -> None:
    """Bar chart showing current vs target allocation with trade signals."""
    colors = {"BUY": "#50a000", "SELL": "#d00000", "HOLD": "#888888"}
    bar_colors = [colors.get(s, "#888888") for s in rec["signal"]]

    fig, ax = plt.subplots(figsize=(10, max(4, len(rec) * 0.4)))
    y_pos = range(len(rec))

    ax.barh(y_pos, rec["target_pct"], color=bar_colors, alpha=0.8, label="Target")
    ax.barh(y_pos, rec["current_pct"], color="#cccccc", alpha=0.5, label="Current")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rec.index)
    ax.set_xlabel("Allocation %")
    ax.set_title("Portfolio Recommendation")
    ax.legend()
    plt.tight_layout()
    plt.show()
