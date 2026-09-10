"""
platform/plotting/nowcaster.py — P4 nowcaster (L2) diagnostics and forecast-quality plots.

This module has two halves, deliberately kept distinct because they answer
different questions and must never be read as the same numbers:

1. **A diagnostic, in-process, NON-walk-forward fit**
   (:func:`fit_nowcaster_diagnostics`) whose only purpose is to make design
   §5.1's *persistence trap* visible: a classifier that simply repeats last
   month's state scores ~90% overall while being wrong at exactly the
   transitions guidance depends on. Overall, transition-window, and
   steady-state accuracy are therefore always reported together.
2. **Rendering of the REAL walk-forward's already-persisted forecast-quality
   artifacts** (:func:`plot_calibration_curve`, :func:`plot_confusion_matrix`),
   read from ``outputs/reports/platform/`` via
   :func:`trading_crab_lib.platform.plotting.loaders.load_report_artifact`.
   Nothing here recomputes them, and nothing here calls ``run_backtest`` or
   ``run_full_backtest_evaluation``.

Deliberate omission — why this is not a wrapper around ``evaluate_nowcaster``
-----------------------------------------------------------------------------
:func:`fit_nowcaster_diagnostics` reimplements only the *orchestration* of
:func:`trading_crab_lib.platform.prediction.nowcaster.evaluate_nowcaster`. None
of the math is re-derived: ``build_nowcaster_training_set``, ``fit_nowcaster``
and ``transition_window_accuracy`` are imported and called verbatim. What is
skipped is that function's two side effects:

* ``honesty.registry.append_trial`` — opening a notebook is not an *evaluated
  configuration*. Appending a trial per notebook open would inflate the
  git-tracked multiple-testing ledger that the deflated-Sharpe denominator is
  computed from at design freeze, corrupting the one artifact whose whole
  purpose is to be an honest count.
* ``get_platform_checkpoint_manager().save_model(model, "nowcaster")`` — a
  cheap in-process glance must never overwrite the production ``nowcaster``
  model checkpoint that a real evaluation run produced.

``tests/unit/test_platform_plotting_nowcaster.py`` monkeypatches both call
sites to raise and proves neither fires.

Relationship to audit item A13
------------------------------
The walk-forward per-step probability path that
``model_metrics_{brier,calibration,confusion}.parquet`` summarize is the SAME
"filtered" series that P3's side-by-side and P6's sojourn/lag headline compare
against the fixed-feature "smoothed reference". A13 found that comparison is
not interpretable, because the filtered path's active feature set changes
across the backtest. P4 characterizes *this module's own* forecast quality
only — it neither attempts nor resolves the A13 comparison, and
:data:`trading_crab_lib.platform.plotting.core.A13_CAVEAT` is rendered verbatim
wherever the filtered path is referenced.

Fresh-package boundary (D-01): this module imports nothing from the legacy
``trading_crab_lib.plotting`` package, and reaches matplotlib only through
:mod:`trading_crab_lib.platform.plotting.core`, which owns the Agg-backend guard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform import taxonomy
from trading_crab_lib.platform.plotting import core
from trading_crab_lib.platform.prediction.nowcaster import (
    build_nowcaster_training_set,
    fit_nowcaster,
    transition_window_accuracy,
)

log = logging.getLogger(__name__)

_NO_DATA_TEXT = "no data"

# The three accuracy figures design §5.1 insists are shown together — never
# `overall_accuracy` alone, which a trivial persistence classifier maximizes.
_ACCURACY_KEYS: tuple[str, ...] = (
    "overall_accuracy",
    "transition_accuracy",
    "steady_state_accuracy",
)


def _no_data_figure(title: str, *, save_path: Path | None, show: bool) -> core.plt.Figure:
    fig, ax = core.plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, _NO_DATA_TEXT, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title(title)
    return core._save_or_show(fig, save_path=save_path, show=show)


# ── The diagnostic (registry-free, checkpoint-free) nowcaster fit ─────────────


def fit_nowcaster_diagnostics(
    features_df: pd.DataFrame,
    labels: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Fit the shipped nowcaster once, for display only — no trial, no checkpoint.

    Composes the production primitives verbatim
    (``build_nowcaster_training_set`` -> ``fit_nowcaster`` ->
    ``transition_window_accuracy``) with the shipped config defaults. **No
    hyperparameter is swept** (D-14's no-sweep discipline extended to L2):
    ``embargo_months``, ``label_horizon``, ``embargo`` and ``n_splits`` are all
    read from *cfg*.

    Unlike
    :func:`trading_crab_lib.platform.prediction.nowcaster.evaluate_nowcaster`,
    this function calls **neither** ``honesty.registry.append_trial`` **nor**
    ``CheckpointManager.save_model`` — see the module docstring for why that
    omission is the entire reason this is a separate function rather than a thin
    wrapper.

    Feature columns are narrowed to ``taxonomy.lean_feature_set(cfg)`` (the same
    fast-plus-slow set ``backtest/driver.py`` refits L1/L2 on) when any lean
    column is present. Passing the whole 53-column ``monthly_features`` frame
    would make ``fit_nowcaster``'s internal non-finite row drop discard every
    month before the latest-starting ETF column.

    Returns:
        dict: ``model``, ``X``, ``y``, ``y_pred``, ``proba``, ``classes``,
        ``metrics``. ``X``/``y``/``y_pred``/``proba`` are all aligned on the
        finite-row subset actually scored (``fit_nowcaster`` drops non-finite
        rows internally; scoring must apply the same mask or
        ``predict`` raises on the NaN rows). ``metrics`` carries
        ``overall_accuracy``, ``transition_accuracy`` and
        ``steady_state_accuracy`` together.
    """
    embargo_months = int(cfg.get("labeling", {}).get("embargo_months", 12))
    label_horizon = int(cfg.get("cv", {}).get("default_label_horizon_months", 12))
    embargo = int(cfg.get("cv", {}).get("default_embargo_months", 1))
    n_splits = int(cfg.get("backtest", {}).get("nowcaster_cv_splits", 5))

    lean_cols = sorted(taxonomy.lean_feature_set(cfg) & set(features_df.columns))
    if lean_cols:
        feature_frame = features_df[lean_cols]
    else:
        log.warning(
            "fit_nowcaster_diagnostics: no lean-taxonomy column found in the supplied frame; "
            "falling back to all %d columns.",
            features_df.shape[1],
        )
        feature_frame = features_df

    X, y = build_nowcaster_training_set(feature_frame, labels, embargo_months=embargo_months)
    model = fit_nowcaster(X, y, label_horizon=label_horizon, embargo=embargo, n_splits=n_splits)

    # Same mask fit_nowcaster applies internally. Scoring the unfiltered X would
    # raise on the NaN months (VIX starts 1990-01, rolling features are NaN at
    # the series start) rather than silently mis-scoring, but either way the
    # returned frames must line up with proba's row count.
    finite = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    X_scored, y_scored = X.loc[finite], y.loc[finite]

    y_pred = pd.Series(model.predict(X_scored), index=y_scored.index)
    proba = np.asarray(model.predict_proba(X_scored), dtype=float)
    metrics = transition_window_accuracy(y_scored, y_pred)

    log.info(
        "fit_nowcaster_diagnostics: %d scored rows over %d lean features; overall=%.4f "
        "transition=%.4f steady=%.4f (diagnostic only — no trial logged, no checkpoint written)",
        len(X_scored),
        X_scored.shape[1],
        metrics["overall_accuracy"],
        metrics["transition_accuracy"],
        metrics["steady_state_accuracy"],
    )

    return {
        "model": model,
        "X": X_scored,
        "y": y_scored,
        "y_pred": y_pred,
        "proba": proba,
        "classes": list(model.classes_),
        "metrics": metrics,
    }


# ── The persistence-trap panel (design §5.1) ─────────────────────────────────


def plot_transition_window_accuracy(
    metrics: dict[str, float],
    *,
    title: str = "Overall vs. transition-window vs. steady-state accuracy",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Render the three §5.1 accuracy figures as one three-bar chart.

    A NaN entry (e.g. ``transition_accuracy`` on a constant-label fixture with
    no transitions) is drawn as a zero-height bar annotated ``n/a`` rather than
    omitted — an absent bar reads as "zero accuracy", which is a different and
    much worse claim than "this fixture contains no transitions to score".

    Returns:
        matplotlib Figure — "no data" annotated when *metrics* is empty.
    """
    if not metrics:
        return _no_data_figure(title, save_path=save_path, show=show)

    labels = [key.replace("_", " ") for key in _ACCURACY_KEYS]
    raw = [float(metrics.get(key, float("nan"))) for key in _ACCURACY_KEYS]
    heights = [0.0 if np.isnan(v) else v for v in raw]
    positions = np.arange(len(_ACCURACY_KEYS))
    colors = [core._regime_color(i) for i in range(len(_ACCURACY_KEYS))]

    fig, ax = core.plt.subplots(figsize=(8, 4.5))
    ax.bar(positions, heights, color=colors, alpha=0.85, width=0.6)
    for pos, value, height in zip(positions, raw, heights):
        text = "n/a" if np.isnan(value) else f"{value:.3f}"
        ax.text(pos, height + 0.02, text, ha="center", va="bottom", fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)


def plot_proba_over_time(
    dates,
    proba: np.ndarray,
    classes: list,
    *,
    title: str = "Nowcaster probability path (in-sample diagnostic fit)",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Stacked-area rendering of the per-state probability path over *dates*.

    One band per entry in *classes*, colored via
    :func:`trading_crab_lib.platform.plotting.core._regime_color`. Probabilities
    are shown as a distribution, never collapsed to an argmax (L2-01).

    Returns:
        matplotlib Figure — "no data" annotated for a zero-row ``proba``.
    """
    proba = np.asarray(proba, dtype=float)
    if proba.size == 0 or proba.ndim != 2 or proba.shape[0] == 0:
        return _no_data_figure(title, save_path=save_path, show=show)

    index = pd.Index(dates)
    colors = [core._regime_color(i) for i in range(proba.shape[1])]
    labels = [f"state {c}" for c in classes][: proba.shape[1]]

    fig, ax = core.plt.subplots(figsize=(12, 4.5))
    ax.stackplot(index, proba.T, colors=colors, labels=labels, alpha=0.85)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(index.min(), index.max())
    ax.set_ylabel("P(state)")
    ax.set_title(title)
    # Legend below the axes, never inside: a stacked area chart fills its whole
    # frame, so an in-axes legend occludes the very bands it labels.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=min(5, max(1, proba.shape[1])),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)


# ── The REAL walk-forward's persisted forecast-quality artifacts ─────────────
# Everything below renders parquet written by
# platform/evaluation/model_metrics.py during the honest walk-forward. Nothing
# here recomputes a metric, and nothing here calls run_backtest() or
# run_full_backtest_evaluation().


def _regime_color_for_label(class_label: Any, fallback_index: int) -> str:
    """Palette color for a ``class_label`` that parquet stores as ``str``."""
    try:
        return core._regime_color(int(class_label))
    except (TypeError, ValueError):
        return core._regime_color(fallback_index)


def plot_calibration_curve(
    calibration_df: pd.DataFrame,
    *,
    title: str = "Calibration (walk-forward, per-class reliability)",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Per-class reliability diagram from ``model_metrics_calibration.parquet``.

    One line per distinct ``class_label`` of ``observed_freq`` against
    ``predicted_prob_mean`` (ordered by ``bin``), over a dashed 45-degree
    perfect-calibration reference. Marker area scales with ``n_in_bin`` so a
    bin holding one observation cannot be mistaken for one holding 380.

    Returns:
        matplotlib Figure — "no data" annotated on an empty frame.
    """
    required = {"class_label", "bin", "predicted_prob_mean", "observed_freq"}
    if calibration_df.empty or not required.issubset(calibration_df.columns):
        return _no_data_figure(title, save_path=save_path, show=show)

    fig, ax = core.plt.subplots(figsize=(7.5, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, alpha=0.6,
            label="perfectly calibrated")

    for position, (class_label, group) in enumerate(calibration_df.groupby("class_label", sort=True)):
        ordered = group.sort_values("bin")
        sizes = (
            25.0 + 3.0 * np.sqrt(ordered["n_in_bin"].astype(float))
            if "n_in_bin" in ordered.columns
            else 40.0
        )
        color = _regime_color_for_label(class_label, position)
        ax.plot(
            ordered["predicted_prob_mean"],
            ordered["observed_freq"],
            marker="o",
            markersize=4,
            color=color,
            linewidth=1.4,
            label=f"state {class_label}",
        )
        ax.scatter(
            ordered["predicted_prob_mean"],
            ordered["observed_freq"],
            s=sizes,
            color=color,
            alpha=0.35,
            edgecolors="none",
        )

    # A hair beyond [0, 1] so a bin sitting exactly at 0.0 or 1.0 (the real
    # artifact has both) draws a whole marker instead of a clipped half.
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("mean predicted probability in bin")
    ax.set_ylabel("observed frequency in bin")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)


def plot_confusion_matrix(
    confusion_df: pd.DataFrame,
    *,
    title: str = "Confusion (walk-forward argmax predictions)",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Annotated heatmap of ``model_metrics_confusion.parquet``'s tidy counts.

    Pivots ``true_label`` x ``pred_label`` -> ``count`` (absent cells are zero,
    since ``confusion_tidy`` writes only nonzero cells). Cell text is the raw
    count; the color scale is the count, so an operator reads both the shape
    and the magnitude.

    Returns:
        matplotlib Figure — "no data" annotated on an empty frame.
    """
    required = {"true_label", "pred_label", "count"}
    if confusion_df.empty or not required.issubset(confusion_df.columns):
        return _no_data_figure(title, save_path=save_path, show=show)

    matrix = (
        confusion_df.pivot(index="true_label", columns="pred_label", values="count")
        .fillna(0)
        .sort_index()
        .sort_index(axis=1)
    )
    values = matrix.to_numpy(dtype=float)

    fig, ax = core.plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(values, cmap="Blues", aspect="auto")
    fig.colorbar(image, ax=ax, label="count")

    threshold = values.max() / 2.0 if values.size else 0.0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{int(values[i, j])}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if values[i, j] > threshold else "black",
            )

    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_xticklabels([str(c) for c in matrix.columns])
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([str(r) for r in matrix.index])
    ax.set_xlabel("predicted state")
    ax.set_ylabel("true state")
    ax.set_title(title)
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)
