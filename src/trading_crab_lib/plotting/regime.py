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


