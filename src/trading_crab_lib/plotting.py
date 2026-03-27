"""
plotting.py — Shared visualization helpers for all pipeline stages.

All plot functions:
  - Accept run_cfg: RunConfig and honour save_plots / show_plots
  - Save to outputs/plots/{step}_{description}.png when save_plots=True
  - Are importable by notebooks without side-effects

Custom 5-regime color palette (from legacy/unified_script.py):
    CUSTOM_COLORS = ["#0000d0","#d00000","#f48c06","#8338ec","#50a000"]

Usage:
    from trading_crab_lib import plotting
    from trading_crab_lib.runtime import RunConfig
    run_cfg = RunConfig(generate_plots=True, save_plots=True)
    plotting.plot_pca_scatter(pca_df, labels, regime_names, run_cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path

def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore[import]
        return get_ipython() is not None
    except ImportError:
        return False

try:
    import matplotlib
    # Only force the Agg (headless) backend when NOT running inside Jupyter/IPython.
    # In Jupyter, %matplotlib inline has already configured the inline backend and
    # calling matplotlib.use("Agg") after that would break inline display and cause
    # "FigureCanvasAgg is non-interactive" warnings when plt.show() is called.
    if not _in_jupyter():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
except ImportError as _matplotlib_err:
    raise ImportError(
        "matplotlib is required for plotting functions. "
        "Install with: pip install 'trading-crab-lib[plotting]'"
    ) from _matplotlib_err
import numpy as np
import pandas as pd

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib.runtime import RunConfig
from trading_crab_lib.transforms import trim_incomplete_tail

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


# ── Plot Reuse Infrastructure ─────────────────────────────────────────────────


def _plot_is_fresh(filename: str, checkpoint_name: str | None = None) -> bool:
    """Return True if the PNG exists and is newer than the relevant checkpoint.

    If *checkpoint_name* is None, only checks that the PNG exists (always fresh).
    If a checkpoint name is given, compares the PNG's mtime against the checkpoint's
    meta.json ``created`` timestamp — stale if the checkpoint was regenerated after
    the plot was last saved.
    """
    png_path = PLOT_DIR / filename
    if not png_path.exists():
        return False

    if checkpoint_name is None:
        return True

    from trading_crab_lib.checkpoints import CHECKPOINT_DIR

    meta_path = CHECKPOINT_DIR / f"{checkpoint_name}.meta.json"
    if not meta_path.exists():
        # No checkpoint to compare against — treat PNG as fresh
        return True

    import json
    from datetime import datetime

    try:
        meta = json.loads(meta_path.read_text())
        ckpt_created = datetime.fromisoformat(meta["created"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return True  # corrupt metadata — don't force regeneration

    png_mtime = datetime.fromtimestamp(png_path.stat().st_mtime)
    return png_mtime >= ckpt_created


def load_or_generate(
    plot_func: callable,
    *args: object,
    filename: str,
    run_cfg: RunConfig,
    checkpoint_name: str | None = None,
    **kwargs: object,
) -> None:
    """Show a cached PNG if fresh, otherwise call *plot_func* to regenerate.

    Designed for use in Jupyter notebooks to avoid re-running expensive plot
    functions when the underlying data hasn't changed.

    Args:
        plot_func: any ``plotting.plot_*`` function.
        *args: positional arguments forwarded to *plot_func*.
        filename: the PNG filename (e.g. ``"03_pca_scatter.png"``).
        run_cfg: runtime configuration.
        checkpoint_name: optional checkpoint name (e.g. ``"features"``) used
            to determine freshness.  If the checkpoint was regenerated after
            the PNG was last saved, the plot is considered stale.
        **kwargs: keyword arguments forwarded to *plot_func*.

    Example (in a Jupyter notebook)::

        from trading_crab_lib import plotting
        plotting.load_or_generate(
            plotting.plot_pca_scatter,
            pca_df, labels, regime_names,
            filename="03_pca_scatter.png",
            run_cfg=run_cfg,
            checkpoint_name="cluster_labels",
        )
    """
    if _plot_is_fresh(filename, checkpoint_name):
        png_path = PLOT_DIR / filename
        log.info("Plot fresh — loading cached: %s", png_path)
        if _in_jupyter():
            try:
                from IPython.display import Image, display  # type: ignore[import]
                display(Image(filename=str(png_path)))
            except ImportError:
                log.warning("IPython not available — regenerating plot")
                plot_func(*args, run_cfg=run_cfg, filename=filename, **kwargs)
        else:
            log.info("Cached plot available at: %s", png_path)
        return

    log.info("Plot stale or missing — regenerating: %s", filename)
    plot_func(*args, run_cfg=run_cfg, filename=filename, **kwargs)


def list_available_plots(plot_dir: Path | None = None) -> str:
    """Return a formatted table of all PNGs in the plot directory.

    Useful in notebooks and CLI to see what plots are available without
    opening a file browser.

    Returns:
        Human-readable table string with filename, modification time, and
        file size for each PNG found.  Returns a "no plots" message if the
        directory is empty or doesn't exist.
    """
    d = plot_dir or PLOT_DIR
    if not d.exists():
        return "No plots directory found."

    pngs = sorted(d.glob("*.png"))
    if not pngs:
        return "No plots found in %s" % d

    lines = [
        f"{'Filename':<50} {'Modified':<22} {'Size':>10}",
        "-" * 84,
    ]

    for p in pngs:
        stat = p.stat()
        from datetime import datetime
        mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_kb = stat.st_size / 1024
        if size_kb >= 1024:
            size_str = f"{size_kb / 1024:.1f} MB"
        else:
            size_str = f"{size_kb:.1f} KB"
        lines.append(f"{p.name:<50} {mtime:<22} {size_str:>10}")

    lines.append(f"\nTotal: {len(pngs)} plots")
    return "\n".join(lines)


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


def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray | pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    title: str = "Regime Classification — Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Confusion matrix heatmap for the current-regime classifier.

    Args:
        y_true       — ground-truth cluster labels
        y_pred       — model-predicted labels (same length / index as y_true)
        regime_names — {cluster_id: human name}
        run_cfg      — controls save/show behaviour
        title        — plot title
        normalize    — if True, show row-normalized (recall) percentages
    """
    from sklearn.metrics import confusion_matrix as _cm

    labels = sorted(y_true.dropna().astype(int).unique())
    cm = _cm(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm.astype(float) / row_sums
        fmt = ".1%"
    else:
        cm_display = cm
        fmt = "d"

    tick_labels = [regime_names.get(i, f"Regime {i}") for i in labels]

    try:
        import seaborn as sns
    except ImportError:
        log.warning("seaborn not installed — skipping confusion matrix plot")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        ax=ax,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        linewidths=0.5,
        cbar_kws={"label": "Recall" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted regime")
    ax.set_ylabel("Actual regime")
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    _save_or_show(fig, "05_confusion_matrix.png", run_cfg)


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


# ── Model Evaluation Plots (Phase A1) ────────────────────────────────────────


def plot_decision_tree(
    tree,
    feature_names: list[str],
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    *,
    max_depth: int | None = 4,
    filename: str = "05_decision_tree.png",
) -> None:
    """Render a sklearn DecisionTreeClassifier as a readable tree diagram."""
    from sklearn.tree import plot_tree

    class_names = [regime_names.get(i, f"R{i}") for i in sorted(regime_names)]
    depth = min(max_depth, tree.get_depth()) if max_depth else tree.get_depth()
    fig_h = max(6, 2 * depth)
    fig_w = max(14, 3 * (2 ** min(depth, 4)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree — Regime Classification", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_cv_fold_accuracy(
    fold_accuracies: list[float],
    run_cfg: RunConfig,
    *,
    model_name: str = "RF",
    filename: str = "05_cv_fold_accuracy.png",
) -> None:
    """Bar chart of per-fold accuracy from TimeSeriesSplit."""
    n = len(fold_accuracies)
    if n == 0:
        return
    mean_acc = np.mean(fold_accuracies)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(1, n + 1), fold_accuracies, color=CUSTOM_COLORS[0], alpha=0.8)
    ax.axhline(mean_acc, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_acc:.1%}")
    for bar, acc in zip(bars, fold_accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} — TimeSeriesSplit CV Accuracy per Fold", fontsize=12)
    ax.set_ylim(0, min(1.05, max(fold_accuracies) + 0.15))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_model_comparison_bar(
    metrics: dict[str, dict[str, float]],
    run_cfg: RunConfig,
    *,
    filename: str = "05_model_comparison.png",
) -> None:
    """Grouped bar chart comparing models (RF, DT, GB) on accuracy/F1.

    Parameters
    ----------
    metrics : dict
        ``{"RF": {"accuracy": 0.7, "f1": 0.65}, "DT": {...}, ...}``
    """
    if not metrics:
        return
    model_names = list(metrics.keys())
    metric_names = list(next(iter(metrics.values())).keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(model_names):
        vals = [metrics[model].get(m, 0) for m in metric_names]
        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_calibration_curve(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    *,
    n_bins: int = 8,
    filename: str = "05_calibration_curve.png",
) -> None:
    """Reliability diagram: predicted probability vs actual frequency per regime.

    Parameters
    ----------
    y_true : array-like of int
        True regime labels.
    y_proba : ndarray, shape (n_samples, n_classes)
        Predicted probabilities (e.g. from ``model.predict_proba(X)``).
    """
    y_true = np.asarray(y_true)
    classes = sorted(set(y_true))
    n_classes = len(classes)
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, cls in enumerate(classes):
        ax = axes[idx // cols][idx % cols]
        probs = y_proba[:, idx] if y_proba.shape[1] > idx else np.zeros(len(y_true))
        binary = (y_true == cls).astype(int)

        # bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_freqs = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            bin_centers.append((lo + hi) / 2)
            bin_freqs.append(binary[mask].mean())

        name = regime_names.get(cls, f"Regime {cls}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
        ax.plot(bin_centers, bin_freqs, "o-", color=_regime_color(cls), label=name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(name, fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # hide unused axes
    for idx in range(n_classes, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle("Calibration Curves (Reliability Diagram)", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_learning_curve(
    model,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    run_cfg: RunConfig,
    *,
    cv: int = 5,
    n_points: int = 8,
    filename: str = "05_learning_curve.png",
) -> None:
    """Train/test score vs training set size — detects overfitting.

    Uses TimeSeriesSplit to respect temporal ordering.
    """
    from sklearn.model_selection import TimeSeriesSplit

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n_total = len(X_arr)

    # generate increasing training sizes
    min_train = max(cv * 2, 10)
    sizes = np.linspace(min_train, n_total, n_points, dtype=int)
    sizes = np.unique(sizes)

    train_scores = []
    test_scores = []
    actual_sizes = []

    for size in sizes:
        X_sub, y_sub = X_arr[:size], y_arr[:size]
        if len(np.unique(y_sub)) < 2:
            continue
        tscv = TimeSeriesSplit(n_splits=min(cv, size // 3))
        fold_train, fold_test = [], []
        for tr_idx, te_idx in tscv.split(X_sub):
            if len(np.unique(y_sub[tr_idx])) < 2:
                continue
            m = model.__class__(**model.get_params())
            m.fit(X_sub[tr_idx], y_sub[tr_idx])
            fold_train.append(m.score(X_sub[tr_idx], y_sub[tr_idx]))
            fold_test.append(m.score(X_sub[te_idx], y_sub[te_idx]))
        if fold_train and fold_test:
            train_scores.append(np.mean(fold_train))
            test_scores.append(np.mean(fold_test))
            actual_sizes.append(size)

    if not actual_sizes:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(actual_sizes, train_scores, "o-", color=CUSTOM_COLORS[0], label="Train")
    ax.plot(actual_sizes, test_scores, "o-", color=CUSTOM_COLORS[1], label="Test (CV)")
    ax.fill_between(actual_sizes, train_scores, test_scores, alpha=0.1, color="gray")
    ax.set_xlabel("Training set size (quarters)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Learning Curve — Overfitting Detection", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


# ── PCA & Clustering Plots (Phase A2) ────────────────────────────────────────


def plot_scree(
    pca_obj,
    run_cfg: RunConfig,
    *,
    filename: str = "03_scree.png",
) -> None:
    """Scree plot: individual + cumulative explained variance per PCA component."""
    ratios = pca_obj.explained_variance_ratio_
    cum = np.cumsum(ratios)
    n = len(ratios)
    x = np.arange(1, n + 1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, ratios, color=CUSTOM_COLORS[0], alpha=0.7, label="Individual")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Explained Variance Ratio")
    ax1.set_xticks(x)

    ax2 = ax1.twinx()
    ax2.plot(x, cum, "o-", color=CUSTOM_COLORS[1], linewidth=2, label="Cumulative")
    ax2.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax2.set_ylabel("Cumulative Explained Variance")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("PCA Scree Plot", fontsize=12)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_pca_loadings(
    pca_obj,
    feature_names: list[str],
    run_cfg: RunConfig,
    *,
    top_n: int = 15,
    filename: str = "03_pca_loadings.png",
) -> None:
    """Heatmap of top features x PCA components (absolute loadings)."""
    import seaborn as sns

    components = pca_obj.components_  # (n_components, n_features)
    abs_loadings = np.abs(components)
    max_per_feature = abs_loadings.max(axis=0)
    top_idx = np.argsort(max_per_feature)[::-1][:top_n]

    selected_names = [feature_names[i] for i in top_idx]
    selected_loadings = components[:, top_idx].T  # (top_n, n_components)

    n_comp = components.shape[0]
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]

    df = pd.DataFrame(selected_loadings, index=selected_names, columns=comp_labels)

    fig, ax = plt.subplots(figsize=(max(6, n_comp + 2), max(6, top_n * 0.4)))
    sns.heatmap(
        df, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.5, cbar_kws={"label": "Loading"},
    )
    ax.set_title(f"PCA Loadings — Top {top_n} Features", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_silhouette_samples(
    X: np.ndarray | pd.DataFrame,
    labels: pd.Series | np.ndarray,
    run_cfg: RunConfig,
    *,
    filename: str = "03_silhouette_samples.png",
) -> None:
    """Per-sample silhouette width plot grouped by cluster."""
    from sklearn.metrics import silhouette_samples, silhouette_score

    X_arr = np.asarray(X)
    labels_arr = np.asarray(labels).astype(int)
    unique_labels = np.sort(np.unique(labels_arr))
    n_clusters = len(unique_labels)

    sil_vals = silhouette_samples(X_arr, labels_arr)
    avg_score = silhouette_score(X_arr, labels_arr)

    fig, ax = plt.subplots(figsize=(8, max(5, n_clusters * 1.5)))
    y_lower = 0

    for cid in unique_labels:
        cluster_sil = np.sort(sil_vals[labels_arr == cid])
        size = len(cluster_sil)
        y_upper = y_lower + size
        color = _regime_color(cid)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size, f"C{cid}", fontsize=9, va="center")
        y_lower = y_upper + 2

    ax.axvline(avg_score, color="red", linestyle="--",
               label=f"Mean = {avg_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Samples (grouped by cluster)")
    ax.set_title("Silhouette Analysis", fontsize=12)
    ax.set_yticks([])
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_gmm_bic_surface(
    bic_df: pd.DataFrame,
    run_cfg: RunConfig,
    *,
    filename: str = "03_gmm_bic_surface.png",
) -> None:
    """Heatmap of (k, covariance_type) -> BIC score from fit_gmm().

    Parameters
    ----------
    bic_df : DataFrame
        Must have columns 'k', 'covariance_type', 'bic'.
    """
    import seaborn as sns

    required = {"k", "covariance_type", "bic"}
    if not required.issubset(bic_df.columns):
        log.warning("plot_gmm_bic_surface: missing columns %s", required - set(bic_df.columns))
        return

    pivot = bic_df.pivot_table(index="covariance_type", columns="k", values="bic")

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) + 2), 4))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".0f", cmap="YlOrRd_r",
        linewidths=0.5, cbar_kws={"label": "BIC (lower = better)"},
    )
    ax.set_title("GMM — BIC by (k, Covariance Type)", fontsize=12)
    ax.set_xlabel("Number of Components (k)")
    ax.set_ylabel("Covariance Type")
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_method_comparison_table(
    comparison_df: pd.DataFrame,
    run_cfg: RunConfig,
    *,
    filename: str = "03_method_comparison.png",
) -> None:
    """Render a clustering method comparison DataFrame as a table figure.

    Parameters
    ----------
    comparison_df : DataFrame
        Output of ``compare_all_methods()``. Expected columns include
        'method', 'n_clusters', 'silhouette', 'davies_bouldin', 'calinski'.
    """
    if comparison_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(2, 0.5 * len(comparison_df) + 1)))
    ax.axis("off")

    display_df = comparison_df.copy()
    for col in ["silhouette", "davies_bouldin", "calinski"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    # header styling
    for j in range(len(display_df.columns)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # highlight best silhouette row
    if "silhouette" in comparison_df.columns:
        best_idx = comparison_df["silhouette"].idxmax()
        row_pos = list(comparison_df.index).index(best_idx) + 1
        for j in range(len(display_df.columns)):
            table[row_pos, j].set_facecolor("#D6EAF8")

    ax.set_title("Clustering Method Comparison", fontsize=13, pad=20)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


# ── Time-Series & Regime Plots (Phase A3) ─────────────────────────────────────


def plot_soft_probabilities(
    probs_df: pd.DataFrame,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    *,
    title: str = "Soft Regime Probabilities Over Time",
    filename: str = "03_soft_probabilities.png",
) -> None:
    """Stacked area chart of GMM/HMM posterior probabilities over time.

    Parameters
    ----------
    probs_df : DataFrame
        Rows = quarters (DatetimeIndex), columns = regime probability columns
        (e.g. ``gmm_prob_0``, ``hmm_prob_1``, or simply ``0, 1, 2, ...``).
    """
    if probs_df.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    cols = probs_df.columns.tolist()
    labels = []
    colors = []
    for i, col in enumerate(cols):
        # try to extract regime id from column name
        try:
            rid = int(str(col).rsplit("_", 1)[-1])
        except ValueError:
            rid = i
        labels.append(regime_names.get(rid, f"Regime {rid}"))
        colors.append(_regime_color(rid))

    ax.stackplot(
        probs_df.index, *[probs_df[c].values for c in cols],
        labels=labels, colors=colors, alpha=0.8,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="upper left", fontsize=8, ncol=min(len(cols), 5))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_feature_regime_overlay(
    feature_series: pd.Series,
    labels: pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    *,
    feature_name: str | None = None,
    filename: str | None = None,
) -> None:
    """Time-series line plot with regime-colored background bands."""
    common = feature_series.index.intersection(labels.index)
    if len(common) == 0:
        return

    feat = feature_series.loc[common]
    lab = labels.loc[common].astype(int)
    name = feature_name or getattr(feature_series, "name", "feature") or "feature"

    fig, ax = plt.subplots(figsize=(14, 4))

    # draw regime background bands
    unique_regimes = sorted(lab.unique())
    for i in range(len(common)):
        dt = common[i]
        cid = lab.iloc[i]
        # quarter width: approximate as 90 days
        left = dt - pd.Timedelta(days=45)
        right = dt + pd.Timedelta(days=45)
        ax.axvspan(left, right, color=_regime_color(cid), alpha=0.15)

    ax.plot(feat.index, feat.values, color="black", linewidth=1.2)
    ax.set_title(f"{name} with Regime Overlay", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel(name)
    ax.grid(alpha=0.3)

    # legend for regimes
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_regime_color(r), alpha=0.4,
                     label=regime_names.get(r, f"R{r}")) for r in unique_regimes]
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=min(len(unique_regimes), 5))

    fig.tight_layout()
    fname = filename or f"04_feature_overlay_{name}.png"
    _save_or_show(fig, fname, run_cfg)


def plot_forward_prob_evolution(
    forward_probs: dict[int, pd.DataFrame],
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    *,
    filename: str = "04_forward_prob_evolution.png",
) -> None:
    """Heatmap: regime x horizon showing P(transition) over 1Q/2Q/4Q/8Q.

    Parameters
    ----------
    forward_probs : dict
        ``{horizon: DataFrame}`` where each DataFrame has shape (n_regimes, n_regimes),
        rows = source regime, columns = target regime.  From ``compute_forward_probabilities()``.
    """
    import seaborn as sns

    if not forward_probs:
        return

    horizons = sorted(forward_probs.keys())
    n_h = len(horizons)
    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 4), squeeze=False)

    for i, h in enumerate(horizons):
        ax = axes[0][i]
        df = forward_probs[h]
        row_labels = [regime_names.get(r, f"R{r}") for r in df.index]
        col_labels = [regime_names.get(c, f"R{c}") for c in df.columns]
        sns.heatmap(
            df.values, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
            vmin=0, vmax=1, xticklabels=col_labels, yticklabels=row_labels,
            linewidths=0.5, cbar=i == n_h - 1,
        )
        ax.set_title(f"{h}Q Forward", fontsize=10)
        ax.set_xlabel("Target Regime")
        if i == 0:
            ax.set_ylabel("Current Regime")

    fig.suptitle("Forward Transition Probabilities by Horizon", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


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


def plot_feature_importance_comparison(
    models_dict: dict[str, object],
    feature_names: list[str],
    run_cfg: RunConfig,
    *,
    top_n: int = 20,
    filename: str = "05_feature_importance_comparison.png",
) -> None:
    """Side-by-side feature importance: RF vs DT vs GB in one figure.

    Parameters
    ----------
    models_dict : dict
        ``{"RF": fitted_rf, "DT": fitted_dt, ...}``. Each model must have
        ``feature_importances_`` attribute.
    """
    valid = {k: m for k, m in models_dict.items() if hasattr(m, "feature_importances_")}
    if not valid:
        return

    n_models = len(valid)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, max(5, top_n * 0.3)),
                             squeeze=False)

    for i, (name, model) in enumerate(valid.items()):
        ax = axes[0][i]
        imp = model.feature_importances_
        indices = np.argsort(imp)[::-1][:top_n]
        top_names = [feature_names[j] for j in indices]
        top_vals = imp[indices]

        color = CUSTOM_COLORS[i % len(CUSTOM_COLORS)]
        ax.barh(range(top_n), top_vals[::-1], color=color, alpha=0.8)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.set_xlabel("Importance")
        ax.set_title(f"{name} — Top {top_n}", fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Importance Comparison", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_regime_duration_histogram(
    labels: pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    *,
    filename: str = "04_regime_duration.png",
) -> None:
    """Histogram of consecutive quarters each regime persists."""
    lab = labels.dropna().astype(int)
    if lab.empty:
        return

    # compute run lengths
    durations: dict[int, list[int]] = {}
    current_regime = lab.iloc[0]
    run_len = 1
    for val in lab.iloc[1:]:
        if val == current_regime:
            run_len += 1
        else:
            durations.setdefault(current_regime, []).append(run_len)
            current_regime = val
            run_len = 1
    durations.setdefault(current_regime, []).append(run_len)

    unique_regimes = sorted(durations.keys())
    n = len(unique_regimes)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

    for i, cid in enumerate(unique_regimes):
        ax = axes[0][i]
        runs = durations[cid]
        name = regime_names.get(cid, f"Regime {cid}")
        ax.hist(runs, bins=range(1, max(runs) + 2), color=_regime_color(cid),
                alpha=0.8, edgecolor="white", align="left")
        ax.set_xlabel("Duration (quarters)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}\nMean={np.mean(runs):.1f}Q", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Regime Duration Distribution", fontsize=13)
    fig.tight_layout()
    _save_or_show(fig, filename, run_cfg)


def plot_correlation_change_heatmap(
    features: pd.DataFrame,
    labels: pd.Series,
    regime_names: dict[int, str],
    run_cfg: RunConfig,
    *,
    top_n: int = 15,
    filename: str = "04_correlation_change.png",
) -> None:
    """Per-regime feature correlation heatmap (shows structure changes).

    Selects top_n features by variance, then plots one correlation heatmap
    per regime side-by-side.
    """
    import seaborn as sns

    common = features.index.intersection(labels.index)
    if len(common) < 10:
        return

    feat = features.loc[common].select_dtypes(include="number")
    lab = labels.loc[common].astype(int)

    # pick top-N by variance
    variances = feat.var().sort_values(ascending=False)
    top_cols = variances.head(top_n).index.tolist()
    feat = feat[top_cols]

    unique_regimes = sorted(lab.unique())
    n = len(unique_regimes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, max(4, top_n * 0.35)), squeeze=False)

    for i, cid in enumerate(unique_regimes):
        ax = axes[0][i]
        subset = feat.loc[lab == cid]
        if len(subset) < 3:
            ax.set_title(f"{regime_names.get(cid, f'R{cid}')}\n(too few samples)")
            continue
        corr = subset.corr()
        sns.heatmap(corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    xticklabels=False, yticklabels=(i == 0),
                    linewidths=0.3, cbar=(i == n - 1))
        ax.set_title(regime_names.get(cid, f"R{cid}"), fontsize=10)
        ax.tick_params(axis="y", labelsize=7)

    fig.suptitle("Feature Correlation by Regime", fontsize=13)
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


# ── Feature Engineering & Selection Plots (Phase A5) ──────────────────────────


def plot_feature_selection_curve(
    importances: list[tuple[str, float]] | pd.Series,
    run_cfg: RunConfig,
    *,
    filename: str = "05_feature_selection_curve.png",
) -> None:
    """Cumulative importance vs # features — find diminishing returns.

    Parameters
    ----------
    importances : list of (name, importance) or pd.Series
        Feature importances sorted descending. If a Series, index=feature names.
    """
    if isinstance(importances, pd.Series):
        vals = importances.sort_values(ascending=False).values
    else:
        if not importances:
            return
        vals = np.array([v for _, v in importances])

    cum = np.cumsum(vals) / vals.sum()
    x = np.arange(1, len(cum) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, cum, "o-", color=CUSTOM_COLORS[0], markersize=3, linewidth=1.5)
    ax.axhline(0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
    ax.axhline(0.95, color="gray", linestyle=":", alpha=0.4, label="95% threshold")

    # mark where 90% is reached
    idx_90 = np.searchsorted(cum, 0.9)
    if idx_90 < len(cum):
        ax.axvline(idx_90 + 1, color=CUSTOM_COLORS[1], linestyle="--", alpha=0.6)
        ax.text(idx_90 + 1.5, 0.85, f"{idx_90 + 1} features\nfor 90%",
                fontsize=9, color=CUSTOM_COLORS[1])

    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Cumulative Importance (fraction)")
    ax.set_title("Feature Selection — Cumulative Importance", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
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
