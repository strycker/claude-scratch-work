"""
platform/plotting/regime.py — P3 regime-labeling panels and the A13
side-by-side comparison surface (Phase 6, D-13/D-14, CONTEXT Amendment 2 item F).

Two groups of functions live here:

**Diagnostic panels for the shipped configuration.** D-14 is binding: this
module renders diagnostics for the shipped ``K=5``, ``lambda=52.0``,
``n_restarts=10`` labeler and sweeps nothing. Nothing here fits a model,
selects a hyperparameter, or writes a checkpoint.

**The A13 comparison surface.** Audit item A13 found that the evaluation's
"smoothed reference" labeling is fit on a FIXED 9-column feature set while the
walk-forward's own per-window labeling uses an active set that CHANGES SEVEN
TIMES across the backtest (4 -> 6 -> 8 -> 9 -> 10 -> 12 -> 13 columns). The
functions :func:`active_feature_count_timeline`, :func:`feature_set_change_dates`,
:func:`label_disagreement`, and :func:`plot_label_comparison` make that
inspectable. **They do not resolve A13** — see
:data:`~trading_crab_lib.platform.plotting.core.A13_CAVEAT`, the single string
every module and notebook renders wherever the §5.4 sojourn/lag headline appears.

Signature convention (D-02): every ``plot_x`` takes
``(data, *, ..., save_path: Path | None = None, show: bool = False)`` and
returns the ``Figure`` through :func:`~trading_crab_lib.platform.plotting.core._save_or_show`.
No ``RunConfig`` — the platform has none.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read-only import of a private name, deliberately. `_window_active_features`
# is the single source of truth for "which features enter the regime model in
# this window", and reimplementing its body here would let the reconstruction
# and the driver drift apart the moment `min_history` semantics change.
# Promoting it to a public name was rejected: this phase has exactly ONE
# sanctioned edit to Phase-5 code and it is the artifact write in plan 06-02.
from trading_crab_lib.platform.backtest.driver import _window_active_features
from trading_crab_lib.platform.honesty.walkforward import expanding_steps
from trading_crab_lib.platform.labeling.diagnostics import occupancy_and_sojourns
from trading_crab_lib.platform.plotting.core import (
    CUSTOM_COLORS,
    _regime_color,
    _save_or_show,
)

log = logging.getLogger(__name__)


def _no_data_figure(title: str, *, save_path: Path | None, show: bool) -> plt.Figure:
    """Empty-input fallback: a titled figure carrying a 'no data' annotation."""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title(title)
    return _save_or_show(fig, save_path=save_path, show=show)


def _state_run_lengths(states: pd.Series) -> dict[int, list[int]]:
    """Per-state consecutive run lengths, for distribution (box-plot) rendering.

    The scalar summaries (median/mean/n_runs) always come from
    ``labeling.diagnostics.occupancy_and_sojourns``; this expansion exists only
    because a box plot needs the individual run lengths that the summary
    collapses.
    """
    values = pd.to_numeric(states, errors="coerce").dropna().astype(int).to_numpy()
    runs: dict[int, list[int]] = {}
    if values.size == 0:
        return runs
    run_state = int(values[0])
    run_len = 1
    for value in values[1:]:
        value = int(value)
        if value == run_state:
            run_len += 1
        else:
            runs.setdefault(run_state, []).append(run_len)
            run_state, run_len = value, 1
    runs.setdefault(run_state, []).append(run_len)
    return runs


# ── Diagnostic panels for the shipped configuration (D-14: render, never tune) ─


def plot_regime_timeline(
    states: pd.Series,
    *,
    recessions: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    events: tuple[tuple[str, str, str], ...] | None = None,
    title: str = "Regime timeline",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Draw the state series as a coloured band per month, with history overlays.

    This is the panel ROADMAP criterion 4 rests on, so the event labels are
    drawn alongside the bands rather than in a detached legend — an operator
    reading "does 1973-09 look like an oil shock" should not have to
    cross-reference a key.

    Args:
        states: int-valued state Series indexed by month-end date.
        recessions: ``(start, end)`` pairs from
            :func:`~trading_crab_lib.platform.plotting.history.recession_periods`;
            ``None`` means the USREC fetch degraded and no shading is drawn.
        events: ``(start, end, label)`` triples, normally
            :data:`~trading_crab_lib.platform.plotting.history.ECONOMIC_EVENTS`.
    """
    if states is None or len(states) == 0:
        return _no_data_figure(title, save_path=save_path, show=show)

    clean = pd.to_numeric(states, errors="coerce").dropna().astype(int)
    dates = pd.DatetimeIndex(clean.index)
    colors = [_regime_color(int(s)) for s in clean.to_numpy()]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.vlines(dates, 0.0, 1.0, colors=colors, linewidth=1.4)

    if recessions:
        for start, end in recessions:
            ax.axvspan(start, end, ymin=0.0, ymax=1.0, color="grey", alpha=0.28, zorder=0)

    if events:
        for offset, (start, end, label) in enumerate(events):
            start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
            ax.axvspan(start_ts, end_ts, ymin=1.02, ymax=1.30, color="black", alpha=0.55, clip_on=False)
            ax.annotate(
                label,
                xy=(start_ts, 1.34 + 0.13 * (offset % 2)),
                xycoords=("data", "axes fraction"),
                fontsize=7,
                rotation=0,
                ha="left",
                va="bottom",
                annotation_clip=False,
            )

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("month")
    ax.set_title(title, pad=44)

    n_states = int(clean.max()) + 1 if len(clean) else 0
    handles = [
        plt.Line2D([0], [0], color=_regime_color(k), lw=6, label=f"state {k}")
        for k in range(min(n_states, len(CUSTOM_COLORS)))
    ]
    if recessions:
        handles.append(plt.Line2D([0], [0], color="grey", lw=6, alpha=0.4, label="NBER recession"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=7, fontsize=8)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)


def plot_occupancy_and_sojourn(
    states: pd.Series,
    *,
    n_states: int = 5,
    title: str = "Occupancy and sojourn (shipped K=5 configuration)",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Occupancy shares beside median sojourn length, both from the labeler's own diagnostics.

    Every number rendered here comes from
    ``labeling.diagnostics.occupancy_and_sojourns`` rather than being
    recomputed, so the panel cannot disagree with the labeler's §4.4 report.
    """
    if states is None or len(states) == 0:
        return _no_data_figure(title, save_path=save_path, show=show)

    clean = pd.to_numeric(states, errors="coerce").dropna().astype(int)
    if len(clean) == 0:
        return _no_data_figure(title, save_path=save_path, show=show)

    diagnostics = occupancy_and_sojourns(clean.to_numpy(), n_states=n_states)
    occupancy = diagnostics["occupancy_pct"]
    sojourns = diagnostics["sojourns"]
    ids = list(range(n_states))
    colors = [_regime_color(k) for k in ids]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    shares = [occupancy[k] for k in ids]
    axes[0].bar(ids, shares, color=colors)
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[0].annotate(
        "0.05 §4.4 soft floor", xy=(ids[0] - 0.4, 0.055), fontsize=7, va="bottom"
    )
    for k, share in zip(ids, shares):
        axes[0].annotate(f"{share:.3f}", xy=(k, share), ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(ids)
    axes[0].set_xlabel("state")
    axes[0].set_ylabel("occupancy share")
    axes[0].set_title("Occupancy (sums to 1.0)")

    medians = [sojourns[k]["median_months"] for k in ids]
    axes[1].bar(ids, medians, color=colors)
    for k in ids:
        median = sojourns[k]["median_months"]
        label = "n/a" if np.isnan(median) else f"{median:.0f}mo\nn={sojourns[k]['n_runs']}"
        axes[1].annotate(
            label,
            xy=(k, 0.0 if np.isnan(median) else median),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].set_xticks(ids)
    axes[1].set_xlabel("state")
    axes[1].set_ylabel("median sojourn (months)")
    overall = diagnostics["overall_median_sojourn_months"]
    axes[1].set_title(f"Median sojourn — pooled median {overall:.1f} months")

    fig.suptitle(title)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)


def plot_transition_matrix(
    matrix: pd.DataFrame,
    *,
    title: str = "Empirical transition matrix",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Annotated heatmap of an ``empirical_transition_matrix`` output."""
    if matrix is None or matrix.empty:
        return _no_data_figure(title, save_path=save_path, show=show)

    grid = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.1 * grid.shape[1] + 3, 0.9 * grid.shape[0] + 2.5))
    image = ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_xticklabels([f"to {c}" for c in matrix.columns])
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_yticklabels([f"from {r}" for r in matrix.index])

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid[i, j]
            ax.text(
                j,
                i,
                "—" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if (not np.isnan(value) and value > 0.5) else "black",
            )

    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)


def plot_soft_confidences(
    confidences: pd.DataFrame,
    *,
    title: str = "Soft state confidences",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Stacked area chart over the confidence frame's ``state_{k}`` columns.

    Integer-named columns (the shape
    :func:`~trading_crab_lib.platform.plotting.loaders.load_filtered_state_probs`
    returns) are accepted too.
    """
    if confidences is None or confidences.empty or confidences.shape[1] == 0:
        return _no_data_figure(title, save_path=save_path, show=show)

    columns = list(confidences.columns)
    labels = [str(col).replace("state_", "state ") if isinstance(col, str) else f"state {col}" for col in columns]
    colors = [_regime_color(k) for k in range(len(columns))]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.stackplot(
        pd.DatetimeIndex(confidences.index),
        [confidences[col].astype(float).to_numpy() for col in columns],
        labels=labels,
        colors=colors,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("confidence")
    ax.set_xlabel("month")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=6, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)


def plot_regime_profiles(
    profiles_df: pd.DataFrame,
    *,
    title: str = "Per-state profiles (auto_profile)",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Render ``auto_profile``'s per-state description strings as a table figure."""
    if profiles_df is None or profiles_df.empty:
        return _no_data_figure(title, save_path=save_path, show=show)

    frame = profiles_df.copy()
    if "state" not in frame.columns:
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "state"})

    cells = [[str(row.get("state", "")), str(row.get("profile", ""))] for _, row in frame.iterrows()]

    fig, ax = plt.subplots(figsize=(13, 0.6 * len(cells) + 1.6))
    ax.set_axis_off()
    table = ax.table(cellText=cells, colLabels=["state", "profile"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for row_idx, row in enumerate(cells, start=1):
        try:
            table[(row_idx, 0)].set_facecolor(_regime_color(int(row[0])))
            table[(row_idx, 0)].set_text_props(color="white")
        except (ValueError, KeyError):
            pass
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)


def plot_sojourn_distribution(
    states: pd.Series,
    *,
    n_states: int = 5,
    title: str = "Sojourn length distribution by state",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Box plot of per-state consecutive run lengths, annotated with run counts."""
    if states is None or len(states) == 0:
        return _no_data_figure(title, save_path=save_path, show=show)

    runs = _state_run_lengths(states)
    if not runs:
        return _no_data_figure(title, save_path=save_path, show=show)

    ids = list(range(n_states))
    data = [runs.get(k, []) for k in ids]

    fig, ax = plt.subplots(figsize=(10, 4))
    positions = [k for k, lengths in zip(ids, data) if lengths]
    populated = [lengths for lengths in data if lengths]
    if populated:
        boxes = ax.boxplot(populated, positions=positions, patch_artist=True, widths=0.6)
        for patch, state_id in zip(boxes["boxes"], positions):
            patch.set_facecolor(_regime_color(state_id))
            patch.set_alpha(0.75)
    for k in ids:
        ax.annotate(
            f"n={len(runs.get(k, []))}",
            xy=(k, 0),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_xticks(ids)
    ax.set_xticklabels([f"state {k}" for k in ids])
    ax.set_ylabel("sojourn length (months)")
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)


# ── A13 comparison surface (Amendment 2 item F, sourced per Amendment 3 item H) ─


def active_feature_count_timeline(
    dev_features: pd.DataFrame,
    cols: list[str],
    *,
    min_history: int,
    min_train: int,
) -> pd.DataFrame:
    """Reconstruct the walk-forward's per-window active feature set, step by step.

    Walks :func:`~trading_crab_lib.platform.honesty.walkforward.expanding_steps`
    over *dev_features*'s index and, at each decision date, asks
    ``backtest.driver._window_active_features`` which of *cols* have accumulated
    ``min_history`` non-NaN months in that window's training block.

    ``_window_active_features`` is imported rather than reimplemented on purpose
    (see this module's import block): it is the single source of truth for the
    rule, and a local copy would silently diverge if ``min_history`` semantics
    ever change. The import reaches a private name deliberately and read-only;
    promoting it was rejected because this phase has exactly one sanctioned edit
    to Phase-5 code, and that budget was spent on plan 06-02's artifact write.

    Returns:
        pd.DataFrame indexed by decision date with columns ``n_active`` (int)
        and ``active`` (the sorted list of active column names). Empty frame if
        the index is shorter than *min_train*.
    """
    records: list[dict[str, object]] = []
    for decision_date, train_index, _test_index in expanding_steps(dev_features.index, min_train=min_train):
        active = _window_active_features(dev_features.loc[train_index], cols, min_history=min_history)
        records.append({"date": decision_date, "n_active": len(active), "active": sorted(active)})

    if not records:
        return pd.DataFrame(columns=["n_active", "active"], index=pd.DatetimeIndex([], name="date"))

    return pd.DataFrame(records).set_index("date")


def feature_set_change_dates(timeline: pd.DataFrame) -> list[pd.Timestamp]:
    """Decision dates at which ``n_active`` changes, including the first step.

    A constant-count timeline therefore returns an EMPTY list, not a
    single-element one — "the set never changed" is the honest reading, and
    reporting the first step as a change would inflate every change count by
    one. The first step IS reported when the count later changes, because that
    is the starting point the subsequent changes are relative to.
    """
    if timeline is None or timeline.empty or "n_active" not in timeline.columns:
        return []

    counts = timeline["n_active"].astype(int)
    if counts.nunique() <= 1:
        return []

    changed = counts.ne(counts.shift())
    changed.iloc[0] = True
    return [pd.Timestamp(stamp) for stamp in counts.index[changed.to_numpy()]]


def plot_active_feature_count(
    timeline: pd.DataFrame,
    *,
    title: str = "Walk-forward active feature count (audit item A13)",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Step chart of ``n_active`` with each change date annotated with its new count."""
    if timeline is None or timeline.empty or "n_active" not in timeline.columns:
        return _no_data_figure(title, save_path=save_path, show=show)

    counts = timeline["n_active"].astype(int)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.step(pd.DatetimeIndex(counts.index), counts.to_numpy(), where="post", color="#0000d0", linewidth=1.8)

    for stamp in feature_set_change_dates(timeline):
        value = int(counts.loc[stamp])
        ax.axvline(stamp, color="black", linestyle=":", linewidth=1, alpha=0.6)
        ax.annotate(
            f"{stamp.date()}\n{value}",
            xy=(stamp, value),
            xytext=(3, 6),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_ylabel("active features")
    ax.set_xlabel("decision date")
    ax.set_title(title)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)


def label_disagreement(reference_states: pd.Series, comparison_states: pd.Series) -> dict:
    """Quantify disagreement between two labelings over their common dates.

    Returns zeros rather than raising when the two indexes do not intersect —
    a disjoint span is itself a finding (the walk-forward's filtered path does
    not begin until 1974-02 while the smoothed reference starts 1963-02), not an
    error condition.

    Returns:
        dict with ``n_compared``, ``n_disagree``, ``pct_disagree`` (in ``[0, 1]``),
        ``first_common_date``, ``last_common_date``, and ``per_state_confusion``
        (reference state x comparison state counts).
    """
    empty = {
        "n_compared": 0,
        "n_disagree": 0,
        "pct_disagree": 0.0,
        "first_common_date": None,
        "last_common_date": None,
        "per_state_confusion": pd.DataFrame(),
    }
    if reference_states is None or comparison_states is None:
        return empty
    if len(reference_states) == 0 or len(comparison_states) == 0:
        return empty

    common = reference_states.index.intersection(comparison_states.index)
    if len(common) == 0:
        return empty

    ref = pd.to_numeric(reference_states.loc[common], errors="coerce")
    cmp_ = pd.to_numeric(comparison_states.loc[common], errors="coerce")
    both = pd.DataFrame({"reference": ref, "comparison": cmp_}).dropna().astype(int)
    if both.empty:
        return empty

    n_compared = int(len(both))
    n_disagree = int((both["reference"] != both["comparison"]).sum())
    confusion = pd.crosstab(both["reference"], both["comparison"])

    return {
        "n_compared": n_compared,
        "n_disagree": n_disagree,
        "pct_disagree": n_disagree / n_compared,
        "first_common_date": pd.Timestamp(both.index.min()),
        "last_common_date": pd.Timestamp(both.index.max()),
        "per_state_confusion": confusion,
    }


def plot_label_comparison(
    labelings: dict[str, pd.Series],
    *,
    change_dates: list[pd.Timestamp] | None = None,
    title: str = "Labelings side by side (audit item A13)",
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """One coloured band track per labeling on a shared time axis, plus a disagreement strip.

    The labelings are NOT silently reindexed onto a common index: a track is
    simply blank where its labeling has no value. That one labeling starts in
    1990 and another in 1963 is exactly what this panel exists to show, and
    reindexing would hide it behind forward-filled or NaN-dropped rows.

    Args:
        labelings: ordered mapping of display name -> state Series. Two or
            three entries with different spans and lengths are all supported.
        change_dates: dates to mark with a vertical rule, normally
            :func:`feature_set_change_dates`'s output.
    """
    present = {name: series for name, series in (labelings or {}).items() if series is not None and len(series) > 0}
    if not present:
        return _no_data_figure(title, save_path=save_path, show=show)

    cleaned = {
        name: pd.to_numeric(series, errors="coerce").dropna().astype(int)
        for name, series in present.items()
    }
    cleaned = {name: series for name, series in cleaned.items() if len(series) > 0}
    if not cleaned:
        return _no_data_figure(title, save_path=save_path, show=show)

    names = list(cleaned.keys())
    n_tracks = len(names)

    fig, ax = plt.subplots(figsize=(14, 1.1 * n_tracks + 3.0))

    for row, name in enumerate(names):
        series = cleaned[name]
        top = n_tracks - row
        bottom = top - 0.8
        ax.vlines(
            pd.DatetimeIndex(series.index),
            bottom,
            top,
            colors=[_regime_color(int(s)) for s in series.to_numpy()],
            linewidth=1.4,
        )

    # Disagreement strip: months where at least two labelings have a value and
    # those values are not all equal. Months covered by only one labeling are
    # not disagreements — they are coverage gaps, visible as blank track space.
    aligned = pd.concat(cleaned.values(), axis=1, keys=names)
    counts = aligned.notna().sum(axis=1)
    comparable = aligned[counts >= 2]
    if not comparable.empty:
        differs = comparable.nunique(axis=1, dropna=True) > 1
        disagreement_dates = comparable.index[differs.to_numpy()]
        ax.vlines(pd.DatetimeIndex(disagreement_dates), 0.05, 0.75, colors="black", linewidth=1.0)
        n_comparable = int(len(comparable))
        n_differ = int(differs.sum())
        strip_label = f"disagree ({n_differ}/{n_comparable} comparable months)"
    else:
        strip_label = "disagree (no comparable months)"

    if change_dates:
        for stamp in change_dates:
            ax.axvline(pd.Timestamp(stamp), color="red", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_ylim(0.0, n_tracks + 0.25)
    ax.set_yticks([n_tracks - row - 0.4 for row in range(n_tracks)] + [0.4])
    ax.set_yticklabels(names + [strip_label], fontsize=8)
    ax.set_xlabel("month")
    ax.set_title(title)

    handles = [plt.Line2D([0], [0], color=_regime_color(k), lw=6, label=f"state {k}") for k in range(len(CUSTOM_COLORS))]
    if change_dates:
        handles.append(
            plt.Line2D([0], [0], color="red", lw=1.5, linestyle="--", label="feature-set change")
        )
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=7, fontsize=8)
    fig.tight_layout()
    return _save_or_show(fig, save_path=save_path, show=show)
