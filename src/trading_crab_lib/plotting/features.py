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

# ── Step 02: Features ──────────────────────────────────────────────────────────

def plot_feature_correlations(
    features: pd.DataFrame,
    run_cfg: RunConfig,
    top_n: int = 40,
) -> None:
    """
    Correlation heatmap for the top_n most-variance clustering features.
    """
    try:
        import seaborn as sns
    except ImportError:
        log.warning("seaborn not installed — skipping correlation heatmap")
        return

    # Pick top-n by variance to keep the plot readable
    variances = features.var().sort_values(ascending=False)
    cols = variances.head(top_n).index.tolist()
    corr = features[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        annot=False,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(f"Feature Correlation Matrix (top {top_n} by variance)", fontsize=12)
    ax.tick_params(axis="both", labelsize=6)
    fig.tight_layout()
    _save_or_show(fig, "02_feature_correlations.png", run_cfg)


def plot_feature_distributions(
    features: pd.DataFrame,
    run_cfg: RunConfig,
    cols: list[str] | None = None,
) -> None:
    """Histogram grid for a subset of features."""
    if cols is None:
        # Use a readable default sample
        cols = [c for c in features.columns if not c.endswith("_d2") and not c.endswith("_d3")][:20]
    cols = [c for c in cols if c in features.columns]
    if not cols:
        return

    n = len(cols)
    ncols_grid = 4
    nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(16, 3 * nrows_grid))
    axes_flat = axes.flat

    for ax, col in zip(axes_flat, cols):
        data = features[col].dropna()
        ax.hist(data, bins=30, edgecolor="none", alpha=0.75, color="#4477aa")
        ax.set_title(col, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)

    # Hide unused panels
    for ax in list(axes_flat)[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Feature Distributions", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, "02_feature_distributions.png", run_cfg)


def plot_pairplot(
    features: pd.DataFrame,
    labels: pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    pca_cols: int = 5,
) -> None:
    """
    Seaborn pairplot of the first few PCA components (slow — opt-in via RunConfig).
    Only runs when run_cfg.generate_pairplot is True.
    """
    if not run_cfg.generate_pairplot:
        return
    try:
        import seaborn as sns
    except ImportError:
        log.warning("seaborn not installed — skipping pairplot")
        return

    cols = [c for c in features.columns if c.startswith("PC")][:pca_cols]
    if not cols:
        cols = list(features.columns)[:pca_cols]

    df = features[cols].copy()
    df["Regime"] = labels.reindex(df.index).map(
        lambda x: regime_names.get(int(x), f"R{int(x)}") if pd.notna(x) else "?"
    )

    palette = {
        regime_names.get(i, f"R{i}"): CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        for i in sorted(labels.dropna().astype(int).unique())
    }

    g = sns.pairplot(df, hue="Regime", palette=palette, plot_kws={"alpha": 0.5, "s": 15})
    g.figure.suptitle("PCA Pairplot by Regime", y=1.02, fontsize=13)
    _save_or_show(g.figure, "02_pca_pairplot.png", run_cfg)


def plot_gap_fill_before_after(
    raw_series: pd.Series,
    filled_series: pd.Series,
    run_cfg: RunConfig,
    *,
    series_name: str | None = None,
    filename: str | None = None,
) -> None:
    """Overlay of raw (with gaps) vs filled series, gaps highlighted."""
    name = series_name or getattr(raw_series, "name", "series") or "series"

    fig, ax = plt.subplots(figsize=(14, 4))

    # identify gap locations (NaN in raw, filled in result)
    gap_mask = raw_series.isna() & filled_series.notna()

    ax.plot(raw_series.index, raw_series.values, "o-", color=CUSTOM_COLORS[0],
            markersize=3, linewidth=1, label="Raw (with gaps)", alpha=0.7)
    ax.plot(filled_series.index, filled_series.values, "-", color=CUSTOM_COLORS[1],
            linewidth=1.5, label="After gap fill", alpha=0.8)

    # highlight filled gaps
    if gap_mask.any():
        gap_vals = filled_series[gap_mask]
        ax.scatter(gap_vals.index, gap_vals.values, color=CUSTOM_COLORS[2],
                   s=25, zorder=5, label="Filled gaps", edgecolors="black", linewidths=0.5)

    ax.set_title(f"Gap Fill: {name}", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = filename or f"02_gap_fill_{name}.png"
    _save_or_show(fig, fname, run_cfg)


def plot_regime_colored_pca_3d(
    pca_df: pd.DataFrame,
    labels: pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    *,
    filename: str = "03_pca_3d.png",
) -> None:
    """3D scatter of PC1 x PC2 x PC3 with regime colors."""
    cols = [c for c in pca_df.columns if c.startswith("PC")]
    if len(cols) < 3:
        log.warning("plot_regime_colored_pca_3d: need >= 3 PC columns, got %d", len(cols))
        return

    common = pca_df.index.intersection(labels.index)
    pca = pca_df.loc[common]
    lab = labels.loc[common].astype(int)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    unique_regimes = sorted(lab.unique())
    for cid in unique_regimes:
        mask = lab == cid
        subset = pca.loc[mask]
        ax.scatter(
            subset[cols[0]], subset[cols[1]], subset[cols[2]],
            c=_regime_color(cid), label=regime_names.get(cid, f"R{cid}"),
            s=20, alpha=0.7,
        )

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])
    ax.set_title("PCA 3D — Regime Clustering", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_feature_variance_ranking(
    features: pd.DataFrame,
    run_cfg: RunConfig,
    *,
    top_n: int = 30,
    filename: str = "02_feature_variance_ranking.png",
) -> None:
    """Horizontal bar chart of features ranked by variance."""
    numeric = features.select_dtypes(include="number")
    if numeric.empty:
        return

    variances = numeric.var().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.3)))
    colors = [CUSTOM_COLORS[0] if v > variances.median() else CUSTOM_COLORS[2]
              for v in variances.values[::-1]]
    ax.barh(range(len(variances)), variances.values[::-1], color=colors, alpha=0.8)
    ax.set_yticks(range(len(variances)))
    ax.set_yticklabels(variances.index[::-1], fontsize=8)
    ax.set_xlabel("Variance")
    ax.set_title(f"Feature Variance Ranking — Top {top_n}", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_nan_heatmap(
    df: pd.DataFrame,
    run_cfg: RunConfig,
    *,
    filename: str = "01_nan_heatmap.png",
) -> None:
    """Binary heatmap: which cells are NaN (data coverage map)."""
    if df.empty:
        return

    nan_matrix = df.isna().astype(int)

    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.25), max(4, len(df) * 0.04)))
    ax.imshow(nan_matrix.values, aspect="auto", cmap="Reds", interpolation="nearest",
              vmin=0, vmax=1)
    ax.set_xlabel("Features")
    ax.set_ylabel("Quarter")

    # x-axis labels
    if len(df.columns) <= 40:
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])

    # y-axis: show decade markers
    if hasattr(df.index, "year"):
        years = df.index.year
        decade_idx = [i for i in range(len(years)) if years[i] % 10 == 0 and
                      (i == 0 or years[i] != years[i - 1])]
        ax.set_yticks(decade_idx)
        ax.set_yticklabels([str(years[i]) for i in decade_idx], fontsize=8)

    ax.set_title("NaN Heatmap (red = missing)", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_centered_vs_causal_comparison(
    features_centered: pd.DataFrame,
    features_causal: pd.DataFrame,
    cols: list[str],
    run_cfg: RunConfig,
    *,
    filename: str = "02_centered_vs_causal.png",
) -> None:
    """Side-by-side: centered vs causal for same features (shows look-ahead effect)."""
    cols = [c for c in cols if c in features_centered.columns and c in features_causal.columns]
    if not cols:
        return

    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True, squeeze=False)

    for i, col in enumerate(cols):
        ax = axes[i][0]
        ax.plot(features_centered.index, features_centered[col].values,
                color=CUSTOM_COLORS[0], linewidth=1, label="Centered", alpha=0.8)
        ax.plot(features_causal.index, features_causal[col].values,
                color=CUSTOM_COLORS[1], linewidth=1, label="Causal", alpha=0.8)
        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)

    axes[0][0].set_title("Centered vs Causal Feature Comparison", fontsize=12)
    axes[-1][0].set_xlabel("Date")
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)
