"""
plotting.py — Shared visualization helpers for all pipeline stages.

All plot functions:
  - Accept run_cfg: RunConfig and honour save_plots / show_plots
  - Save to outputs/plots/{step}_{description}.png when save_plots=True
  - Are importable by notebooks without side-effects

Custom 5-regime color palette (from legacy/unified_script.py):
    CUSTOM_COLORS = ["#0000d0","#d00000","#f48c06","#8338ec","#50a000"]

Usage:
    from market_regime import plotting
    from market_regime.runtime import RunConfig
    run_cfg = RunConfig(generate_plots=True, save_plots=True)
    plotting.plot_pca_scatter(pca_df, labels, regime_names, run_cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

# Only force the Agg (headless) backend when NOT running inside Jupyter/IPython.
# In Jupyter, %matplotlib inline has already configured the inline backend and
# calling matplotlib.use("Agg") after that would break inline display and cause
# "FigureCanvasAgg is non-interactive" warnings when plt.show() is called.
def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import]
        return get_ipython() is not None
    except ImportError:
        return False

if not _in_jupyter():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from market_regime import OUTPUT_DIR
from market_regime.runtime import RunConfig
from market_regime.transforms import trim_incomplete_tail

log = logging.getLogger(__name__)

# ── Color palette ──────────────────────────────────────────────────────────────
CUSTOM_COLORS: list[str] = ["#0000d0", "#d00000", "#f48c06", "#8338ec", "#50a000"]
REGIME_CMAP = mcolors.ListedColormap(CUSTOM_COLORS)

PLOT_DIR = OUTPUT_DIR / "plots"


def _save_or_show(fig: plt.Figure, filename: str, run_cfg: RunConfig) -> None:
    """Finalize a figure: save to disk and/or display according to run_cfg.

    In Jupyter notebooks, plt.show() is always called so the figure appears
    inline — regardless of show_plots — because the inline backend handles
    display cleanly and plt.close() would otherwise prevent any inline output.
    """
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if run_cfg.save_plots:
        out = PLOT_DIR / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved plot: %s", out)
    if run_cfg.show_plots or _in_jupyter():
        plt.show()
    plt.close(fig)


def _regime_color(cluster_id: int) -> str:
    return CUSTOM_COLORS[cluster_id % len(CUSTOM_COLORS)]


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
    fig.suptitle("Raw Series Sample", fontsize=13, y=1.01)
    fig.tight_layout()
    _save_or_show(fig, "01_raw_series_sample.png", run_cfg)


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


# ── Step 03: Clustering ────────────────────────────────────────────────────────

def plot_elbow_curve(
    scores: pd.DataFrame,
    chosen_k: int,
    run_cfg: RunConfig,
) -> None:
    """
    Three-panel k-sweep plot: silhouette, Calinski-Harabasz, Davies-Bouldin.
    Vertical dashed line marks the chosen k.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    metrics = [
        ("silhouette", "Silhouette Score", "higher = better"),
        ("calinski", "Calinski-Harabasz", "higher = better"),
        ("davies_bouldin", "Davies-Bouldin", "lower = better"),
    ]

    for ax, (col, title, subtitle) in zip(axes, metrics):
        if col not in scores.columns:
            ax.set_visible(False)
            continue
        ax.plot(scores["k"], scores[col], "o-", linewidth=2, markersize=6, color="#3366cc")
        ax.axvline(chosen_k, color="#cc3300", linestyle="--", linewidth=1.5,
                   label=f"k={chosen_k}")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_title(f"{title}\n({subtitle})", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("K-Sweep Evaluation Metrics", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, "03_elbow_curves.png", run_cfg)


def plot_pca_scatter(
    pca_df: pd.DataFrame,
    labels: pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
) -> None:
    """
    2D scatter: PC1 vs PC2, coloured by cluster.
    Includes a secondary plot of PC3 vs PC4.
    """
    pca_cols = pca_df.columns.tolist()
    if len(pca_cols) < 2:
        log.warning("Need at least 2 PCA components for scatter — skipping")
        return

    aligned_labels = labels.reindex(pca_df.index)
    unique_clusters = sorted(aligned_labels.dropna().astype(int).unique())

    n_panels = 2 if len(pca_cols) >= 4 else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    pairs = [(pca_cols[0], pca_cols[1])]
    if n_panels == 2:
        pairs.append((pca_cols[2], pca_cols[3]))

    for ax, (xcol, ycol) in zip(axes, pairs):
        for cid in unique_clusters:
            mask = aligned_labels == cid
            label = regime_names.get(cid, f"Regime {cid}")
            ax.scatter(
                pca_df.loc[mask, xcol],
                pca_df.loc[mask, ycol],
                c=_regime_color(cid),
                label=label,
                s=25,
                alpha=0.75,
                edgecolors="none",
            )
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{xcol} vs {ycol}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("PCA Scatter — Cluster Assignments", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, "03_pca_scatter.png", run_cfg)


def plot_cluster_sizes(
    labels: pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    title: str = "Cluster Sizes",
) -> None:
    """Bar chart of how many quarters fall in each cluster."""
    counts = labels.dropna().astype(int).value_counts().sort_index()
    regime_labels = [regime_names.get(i, f"Regime {i}") for i in counts.index]
    colors = [_regime_color(i) for i in counts.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(counts)), counts.values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(regime_labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Number of quarters")
    ax.set_title(title, fontsize=12)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "03_cluster_sizes.png", run_cfg)


# ── Step 04: Regime Profiling ──────────────────────────────────────────────────

def plot_regime_timeline(
    labels: pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
) -> None:
    """
    Horizontal strip chart showing which regime was active each quarter,
    one row per unique cluster, shaded bands.
    """
    unique_clusters = sorted(labels.dropna().astype(int).unique())
    n = len(unique_clusters)

    fig, ax = plt.subplots(figsize=(16, max(3, n * 1.2)))

    # Draw a filled band for each active quarter
    for cid in unique_clusters:
        mask = labels.astype(int) == cid
        y = cid
        for idx in labels.index[mask]:
            ax.barh(y, width=92, left=idx, height=0.8,
                    color=_regime_color(cid), alpha=0.8)

    ax.set_yticks(unique_clusters)
    ax.set_yticklabels(
        [regime_names.get(i, f"Regime {i}") for i in unique_clusters],
        fontsize=9,
    )
    ax.set_xlabel("Date")
    ax.set_title("Regime Timeline", fontsize=13)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "04_regime_timeline.png", run_cfg)


def plot_transition_matrix(
    tm: pd.DataFrame,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
) -> None:
    """
    Heatmap of the regime transition probability matrix.
    Cell values are probabilities; diagonal = persistence.
    """
    try:
        import seaborn as sns
    except ImportError:
        log.warning("seaborn not installed — skipping transition matrix heatmap")
        return

    labels_map = [regime_names.get(int(i), f"R{i}") for i in tm.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        tm.values,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0, vmax=1,
        xticklabels=labels_map,
        yticklabels=labels_map,
        linewidths=0.5,
        cbar_kws={"label": "Transition probability"},
    )
    ax.set_xlabel("Next regime")
    ax.set_ylabel("Current regime")
    ax.set_title("Regime Transition Matrix", fontsize=12)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    _save_or_show(fig, "04_transition_matrix.png", run_cfg)


def plot_regime_profiles(
    features: pd.DataFrame,
    labels: pd.Series,
    regime_names: dict[int, str],
    key_cols: list[str],
    run_cfg: RunConfig,
) -> None:
    """
    Box-plot grid: one panel per key indicator, coloured by regime.
    Useful for visually verifying the naming heuristics fired correctly.
    """
    key_cols = [c for c in key_cols if c in features.columns]
    if not key_cols:
        return

    valid = labels.dropna()
    unique_clusters = sorted(valid.astype(int).unique())
    n = len(key_cols)
    ncols_grid = 3
    nrows_grid = (n + ncols_grid - 1) // ncols_grid
    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(14, 4 * nrows_grid))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, col in zip(axes_flat, key_cols):
        data_by_regime = [
            features.loc[valid.astype(int) == cid, col].dropna().values
            for cid in unique_clusters
        ]
        bp = ax.boxplot(
            data_by_regime,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
        )
        for patch, cid in zip(bp["boxes"], unique_clusters):
            patch.set_facecolor(_regime_color(cid))
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(unique_clusters) + 1))
        ax.set_xticklabels(
            [regime_names.get(i, f"R{i}") for i in unique_clusters],
            rotation=20, ha="right", fontsize=7,
        )
        ax.set_title(col, fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes_flat[len(key_cols):]:
        ax.set_visible(False)

    fig.suptitle("Regime Profiles — Key Indicators", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, "04_regime_profiles.png", run_cfg)


# ── Step 05: Prediction ────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: list[str],
    run_cfg: RunConfig,
    top_n: int = 25,
) -> None:
    """
    Horizontal bar chart of the top_n most important features from the
    current-regime RandomForest classifier.
    """
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.28)))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(top_features)))
    ax.barh(range(len(top_features)), top_values, color=colors, edgecolor="none")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=8)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_n} Feature Importances — Current Regime Classifier", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "05_feature_importance.png", run_cfg)


def plot_forward_probabilities(
    prediction: dict,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
) -> None:
    """
    Bar chart of predicted regime probabilities for the current quarter.
    """
    probs = prediction.get("probabilities", {})
    if not probs:
        return

    regimes = sorted(probs.keys())
    values = [probs[r] for r in regimes]
    labels = [regime_names.get(r, f"Regime {r}") for r in regimes]
    colors = [_regime_color(r) for r in regimes]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(regimes)), values, color=colors, edgecolor="white")
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Current Quarter Regime Probabilities\n"
        f"(predicted regime: {regime_names.get(prediction['regime'], prediction['regime'])})",
        fontsize=11,
    )
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, "05_current_regime_probs.png", run_cfg)


def plot_predicted_vs_actual(
    features: pd.DataFrame,
    labels: pd.Series,
    model,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
) -> None:
    """
    Side-by-side timeline of actual vs model-predicted regimes.
    """
    # Select the exact columns the model was trained on, then drop the trailing
    # incomplete quarter(s) where centered np.gradient leaves NaN (edge effect).
    if hasattr(model, "feature_names_in_"):
        train_cols = [c for c in model.feature_names_in_ if c in features.columns]
        X = trim_incomplete_tail(features[train_cols]).dropna(how="any")
    else:
        X = trim_incomplete_tail(features).dropna(how="any")
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    y_true = labels.loc[common]
    y_pred = pd.Series(model.predict(X), index=common, name="predicted")

    unique_clusters = sorted(labels.dropna().astype(int).unique())
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

    for ax, (series, title) in zip(axes, [(y_true, "Actual"), (y_pred, "Predicted")]):
        for cid in unique_clusters:
            mask = series.astype(int) == cid
            for idx in series.index[mask]:
                ax.barh(0, width=92, left=idx, height=0.8,
                        color=_regime_color(cid), alpha=0.85)
        ax.set_yticks([])
        ax.set_ylabel(title, fontsize=10)
        ax.set_xlim(common[0], common[-1])

    axes[1].set_xlabel("Quarter")
    fig.suptitle("Actual vs Predicted Regime Assignments", fontsize=13)

    # Legend
    patches = [
        matplotlib.patches.Patch(
            color=_regime_color(i),
            label=regime_names.get(i, f"Regime {i}"),
        )
        for i in unique_clusters
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(unique_clusters),
               fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    _save_or_show(fig, "05_predicted_vs_actual.png", run_cfg)


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
