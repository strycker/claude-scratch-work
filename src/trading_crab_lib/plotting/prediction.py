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
from trading_crab_lib.transforms import trim_incomplete_tail

log = logging.getLogger(__name__)

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

