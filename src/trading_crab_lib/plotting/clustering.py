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

