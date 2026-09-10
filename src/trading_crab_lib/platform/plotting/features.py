"""
platform/plotting/features.py — P2 feature-taxonomy plots and quality guards (Phase 6).

Groups ``monthly_features``' columns into the fast / slow / agency / untagged
tiers the platform declares in ``config/platform_settings.yaml``, tabulates each
feature's observed range, and carries the two quality guards P2 runs on every
open:

* :func:`assert_feature_ranges_plausible` — a named-domain-bound check (D-11
  reinstated by CONTEXT amendment item A): a value that *cannot exist* stops the
  notebook, which drift-against-baseline structurally cannot catch because a
  uniformly wrong series shows zero drift.
* :func:`check_agency_level_discontinuities` — the regression guard for audit
  item A4 (the ALFRED rebasing discontinuity in ``fred_cpi``).

Tier membership comes exclusively from :mod:`trading_crab_lib.platform.taxonomy`
— never from a re-derived list read out of ``cfg["taxonomy"]`` here — and the
plausibility / drift math comes exclusively from
:mod:`trading_crab_lib.platform.plotting.drift`.

Fresh-package boundary (D-01): this module imports nothing from the legacy
``trading_crab_lib.plotting`` package. Matplotlib itself is reached only through
:mod:`trading_crab_lib.platform.plotting.core`, which owns the Agg-backend guard.

Causal-only note (CONTEXT amendment 3 item I): there is deliberately no
centered-feature rendering here. ``transforms_monthly.py`` has zero
``center``/``centered``/``causal`` occurrences and
``honesty.gating.FORBIDDEN_CENTERED_SUFFIXES`` refuses any ``_centered`` /
``_c5`` / ``_zerophase`` column outright, so no centered variant exists to plot
and creating one would trip the platform's own gating rail.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform import taxonomy
from trading_crab_lib.platform.plotting import core, drift

log = logging.getLogger(__name__)

_TIER_KEYS: tuple[str, ...] = ("fast", "slow", "agency", "untagged")
_UNTAGGED = "untagged"

# ── D-11 named domain bounds ─────────────────────────────────────────────────
# Every bound below is a fact about the domain, not a number tuned to whatever
# the current data happens to look like (CONTEXT amendment item A). Each is
# stated with headroom against the observed 1962+ dev range, recorded beside it.

_VIX_MIN_PLAUSIBLE = 0.0                 # observed min 9.51 — an index cannot be <= 0
_VOL_MIN_PLAUSIBLE = 0.0                 # observed mins 0.0001 / 0.0007 — a realized vol cannot be negative
_CREDIT_SPREAD_MIN_PLAUSIBLE = -0.005    # observed min 0.32 — Baa-Aaa is not meaningfully negative
_CAPE_MIN_PLAUSIBLE = 0.0                # observed min 6.64 — a P/E ratio is positive
_COMMODITY_MIN_PLAUSIBLE = 0.0           # observed mins gold 254.60, oil 10.25 — a spot price is positive
_DIV_YIELD_RANGE: tuple[float, float] = (0.0, 0.15)  # observed 0.0111 .. 0.0624

_KIND_MIN = "min"                        # observed min must be >= bound
_KIND_MIN_EXCLUSIVE = "min_exclusive"    # observed min must be strictly > bound
_KIND_RANGE = "range"                    # observed [min, max] must sit inside bound

_RANGE_CHECKS: dict[str, dict[str, Any]] = {
    "fred_vix": {
        "kind": _KIND_MIN_EXCLUSIVE,
        "bound": _VIX_MIN_PLAUSIBLE,
        "reason": "the VIX is an annualized volatility index and cannot be zero or negative",
    },
    "realized_vol_1m": {
        "kind": _KIND_MIN,
        "bound": _VOL_MIN_PLAUSIBLE,
        "reason": "a realized volatility is a standard deviation and cannot be negative",
    },
    "realized_vol_3m": {
        "kind": _KIND_MIN,
        "bound": _VOL_MIN_PLAUSIBLE,
        "reason": "a realized volatility is a standard deviation and cannot be negative",
    },
    "credit_spread_baa_aaa": {
        "kind": _KIND_MIN,
        "bound": _CREDIT_SPREAD_MIN_PLAUSIBLE,
        "reason": "Baa yields above Aaa by construction; only numerical tolerance is allowed below zero",
    },
    "cape_shiller": {
        "kind": _KIND_MIN_EXCLUSIVE,
        "bound": _CAPE_MIN_PLAUSIBLE,
        "reason": "a cyclically-adjusted price/earnings ratio is positive",
    },
    "gold": {
        "kind": _KIND_MIN_EXCLUSIVE,
        "bound": _COMMODITY_MIN_PLAUSIBLE,
        "reason": "a spot commodity price is positive",
    },
    "oil": {
        "kind": _KIND_MIN_EXCLUSIVE,
        "bound": _COMMODITY_MIN_PLAUSIBLE,
        "reason": "a spot commodity price is positive",
    },
    "div_yield": {
        "kind": _KIND_RANGE,
        "bound": _DIV_YIELD_RANGE,
        "reason": (
            "the S&P 500 dividend yield has never been negative nor exceeded 15% over the "
            "platform's 1962+ history; a value outside this band is a units error, not a market event"
        ),
    },
}

_RANGE_TABLE_COLUMNS: tuple[str, ...] = (
    "feature",
    "tier",
    "n_obs",
    "first_valid",
    "last_valid",
    "min",
    "max",
    "mean",
    "std",
)

_SYMLOG_LINTHRESH = 1e-3
_NO_DATA_TEXT = "no data"


# ── Tier grouping and the feature range table ────────────────────────────────


def tier_frames(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Group *df*'s columns into the four taxonomy tiers.

    Tier membership is resolved exclusively through
    :func:`trading_crab_lib.platform.taxonomy.classify_feature`; columns it
    classifies as ``None`` are grouped under ``"untagged"`` (raw pass-through —
    the ETF universe and intermediate splice columns).

    Returns:
        dict[str, pd.DataFrame]: exactly the keys ``"fast"``, ``"slow"``,
        ``"agency"``, ``"untagged"``. A tier with no members is an
        empty-column DataFrame (sharing *df*'s index), never ``None``.
    """
    members: dict[str, list[str]] = {key: [] for key in _TIER_KEYS}
    for column in df.columns:
        tier = taxonomy.classify_feature(str(column), cfg) or _UNTAGGED
        members.setdefault(tier, []).append(column)

    return {key: df[members[key]] for key in _TIER_KEYS}


def feature_range_table(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Tabulate one row per column of *df*: tier, observation count, span, and range.

    An all-NaN column reports ``n_obs=0`` with NaN range/mean/std rather than
    raising — a dead column is a finding to display, not a crash.

    Returns:
        pd.DataFrame: columns ``feature``, ``tier``, ``n_obs``, ``first_valid``,
        ``last_valid``, ``min``, ``max``, ``mean``, ``std``, sorted by
        ``tier`` then ``feature`` for a stable, readable table.
    """
    rows: list[dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        clean = series.dropna()
        rows.append(
            {
                "feature": column,
                "tier": taxonomy.classify_feature(str(column), cfg) or _UNTAGGED,
                "n_obs": int(clean.shape[0]),
                "first_valid": series.first_valid_index(),
                "last_valid": series.last_valid_index(),
                "min": float(clean.min()) if len(clean) else float("nan"),
                "max": float(clean.max()) if len(clean) else float("nan"),
                "mean": float(clean.mean()) if len(clean) else float("nan"),
                "std": float(clean.std()) if len(clean) > 1 else float("nan"),
            }
        )

    table = pd.DataFrame(rows, columns=list(_RANGE_TABLE_COLUMNS))
    if table.empty:
        return table
    return table.sort_values(["tier", "feature"]).reset_index(drop=True)


def _no_data_figure(title: str, *, save_path: Path | None, show: bool) -> core.plt.Figure:
    fig, ax = core.plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, _NO_DATA_TEXT, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    ax.set_title(title)
    return core._save_or_show(fig, save_path=save_path, show=show)


def plot_feature_ranges(
    range_table: pd.DataFrame,
    *,
    tier: str | None = None,
    title: str = "Feature ranges",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Render one horizontal bar per feature spanning ``[min, max]``, marked at ``mean``.

    The x-axis is symmetric-log: a single tier mixes quantities spanning six
    orders of magnitude (``realized_vol_1m`` at 1e-4 alongside ``gold`` near
    2e3) and a linear axis would collapse every small-valued feature onto the
    zero line, producing a chart that looks complete and shows nothing. Each
    row is also annotated with its literal min/max so the numbers do not depend
    on reading the axis.

    Args:
        range_table: output of :func:`feature_range_table`.
        tier: when given, render only rows whose ``tier`` equals it.

    Returns:
        matplotlib Figure — carrying a "no data" annotation when the table is
        empty (or empty after the tier filter) rather than raising.
    """
    plot_title = f"{title} — {tier} tier" if tier else title
    if range_table.empty or "feature" not in range_table.columns:
        return _no_data_figure(plot_title, save_path=save_path, show=show)

    subset = range_table if tier is None else range_table[range_table["tier"] == tier]
    subset = subset.dropna(subset=["min", "max"])
    if subset.empty:
        return _no_data_figure(plot_title, save_path=save_path, show=show)

    subset = subset.sort_values("feature", ascending=False).reset_index(drop=True)
    positions = np.arange(len(subset))

    fig, ax = core.plt.subplots(figsize=(11, max(2.5, 0.42 * len(subset))))
    for pos, row in zip(positions, subset.itertuples(index=False)):
        ax.plot([row.min, row.max], [pos, pos], color=core.CUSTOM_COLORS[0], linewidth=3, alpha=0.6,
                solid_capstyle="butt")
        ax.plot([row.min], [pos], marker="|", color=core.CUSTOM_COLORS[0], markersize=10)
        ax.plot([row.max], [pos], marker="|", color=core.CUSTOM_COLORS[0], markersize=10)
        ax.plot([row.mean], [pos], marker="o", color=core.CUSTOM_COLORS[1], markersize=5, zorder=3)

    ax.set_xscale("symlog", linthresh=_SYMLOG_LINTHRESH)
    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{row.feature}  [{row.min:.4g} .. {row.max:.4g}]" for row in subset.itertuples(index=False)],
        fontsize=8,
    )
    ax.set_ylim(-0.8, len(subset) - 0.2)
    ax.set_xlabel("value (symlog scale; dot = mean)")
    ax.set_title(plot_title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)


# ── D-11 plausibility guard ──────────────────────────────────────────────────


def _check_one_feature(feature: str, spec: dict[str, Any], row: pd.Series) -> tuple[dict[str, Any], str | None]:
    """Return (verdict row, violation message or None) for one named feature."""
    kind = spec["kind"]
    bound = spec["bound"]
    reason = spec["reason"]
    observed_min = float(row["min"])
    observed_max = float(row["max"])

    if kind == _KIND_RANGE:
        low, high = bound
        bound_text = f"in [{low:g}, {high:g}]"
        observed_text = f"min={observed_min:.6g}, max={observed_max:.6g}"
        violated = (observed_min < low) or (observed_max > high)
    elif kind == _KIND_MIN_EXCLUSIVE:
        bound_text = f"> {bound:g}"
        observed_text = f"min={observed_min:.6g}"
        violated = not (observed_min > bound)
    else:
        bound_text = f">= {bound:g}"
        observed_text = f"min={observed_min:.6g}"
        violated = observed_min < bound

    if np.isnan(observed_min) or np.isnan(observed_max):
        # An all-NaN column has no observed range to contradict a bound; the
        # coverage panel is where a dead column is a finding, not here.
        violated = False
        observed_text = "no observations"

    verdict_row = {
        "feature": feature,
        "bound": bound_text,
        "observed": observed_text,
        "verdict": "VIOLATION" if violated else "pass",
    }
    message = None
    if violated:
        message = f"{feature}: observed {observed_text} violates the domain bound {bound_text} — {reason}."
    return verdict_row, message


def assert_feature_ranges_plausible(range_table: pd.DataFrame) -> pd.DataFrame:
    """Raise if any named feature's observed range falls outside its domain bound (D-11).

    Every violation is collected before raising once — the
    ``config.validate_config`` / ``drift.assert_kpi_table_plausible``
    collect-then-raise idiom — so an operator sees every impossible value in
    one message rather than fixing them one build at a time.

    A feature named in :data:`_RANGE_CHECKS` but absent from *range_table* is
    skipped silently (a future taxonomy edit is not a failure here).

    Args:
        range_table: output of :func:`feature_range_table`.

    Returns:
        pd.DataFrame: verdict frame with columns ``feature``, ``bound``,
        ``observed``, ``verdict`` — one row per checked feature present.

    Raises:
        ValueError: naming every feature, its observed value, and the bound it
        violated.
    """
    rows: list[dict[str, Any]] = []
    violations: list[str] = []

    if range_table.empty or "feature" not in range_table.columns:
        return pd.DataFrame(rows, columns=["feature", "bound", "observed", "verdict"])

    indexed = range_table.set_index("feature")
    for feature, spec in _RANGE_CHECKS.items():
        if feature not in indexed.index:
            log.debug("assert_feature_ranges_plausible: '%s' absent from range table — skipped", feature)
            continue
        verdict_row, message = _check_one_feature(feature, spec, indexed.loc[feature])
        rows.append(verdict_row)
        if message is not None:
            violations.append(message)

    if violations:
        bullet_list = "\n".join(f"  • {v}" for v in violations)
        raise ValueError(
            f"assert_feature_ranges_plausible found {len(violations)} domain-impossible "
            f"feature range(s):\n{bullet_list}\n"
            f"A value that cannot exist is not caught by drift-against-baseline — a uniformly "
            f"wrong series shows zero drift against itself."
        )

    return pd.DataFrame(rows, columns=["feature", "bound", "observed", "verdict"])


def check_agency_level_discontinuities(
    monthly_raw: pd.DataFrame,
    *,
    columns: tuple[str, ...] = ("fred_cpi",),
) -> dict[str, list[pd.Timestamp]]:
    """Re-run the A4 level-discontinuity guard against agency-tier index columns.

    Audit item **A4**: ALFRED point-in-time vintages of a *rebased* index are
    not level-comparable across rebasings, which put two artificial ~3x jumps
    into ``fred_cpi`` (1970-12 -> 1971-01, 1988-01 -> 1988-02) and propagated
    into ``real_rate_level`` — a defining feature of two labeler states holding
    64.3% of occupancy. The defect is fixed; this function is the regression
    guard that keeps it fixed, run on every P2 open rather than trusted to have
    been repaired once.

    The underlying :func:`drift.assert_no_level_discontinuity` raises per call;
    this wrapper catches per column so the first violation never short-circuits
    the rest, then raises once listing every offending column.

    Args:
        monthly_raw: the raw monthly frame. Columns absent from it are skipped.
        columns: agency-tier index columns to check.

    Returns:
        dict[str, list[pd.Timestamp]]: empty when every checked column is clean.

    Raises:
        ValueError: listing every offending column and its dates.
    """
    violations: list[str] = []
    for column in columns:
        if column not in monthly_raw.columns:
            log.warning(
                "check_agency_level_discontinuities: column '%s' absent from monthly_raw — "
                "not checked. If this column is expected, rebuild with "
                "`python scripts/build_platform_data.py`.",
                column,
            )
            continue
        try:
            drift.assert_no_level_discontinuity(monthly_raw[column], name=column)
        except ValueError as exc:
            violations.append(str(exc))

    if violations:
        bullet_list = "\n".join(f"  • {v}" for v in violations)
        raise ValueError(
            f"check_agency_level_discontinuities found {len(violations)} column(s) carrying a "
            f"level discontinuity — this is the shape of audit item A4 (ALFRED rebasing), which "
            f"was fixed and has regressed:\n{bullet_list}"
        )

    return {}


# ── Drift summary rendering ──────────────────────────────────────────────────


def plot_drift_summary(
    drift_report_df: pd.DataFrame,
    *,
    title: str = "Feature drift vs pre-2021 baseline",
    save_path: Path | None = None,
    show: bool = False,
) -> core.plt.Figure:
    """Render one horizontal bar per feature of its standardized mean shift.

    Input order is preserved — :func:`drift.drift_report` already sorts by
    descending absolute shift. Bars at or beyond
    ``drift._DRIFT_FLAG_THRESHOLD`` are drawn in a distinct color from in-band
    bars, and the threshold is drawn as a pair of reference lines so an
    operator can see how far a flagged feature sits past it.

    Returns:
        matplotlib Figure — "no data" annotated when the report is empty.
    """
    if drift_report_df.empty or "standardized_mean_shift" not in drift_report_df.columns:
        return _no_data_figure(title, save_path=save_path, show=show)

    subset = drift_report_df.iloc[::-1].reset_index(drop=True)  # largest shift at the top
    shifts = subset["standardized_mean_shift"].astype(float).fillna(0.0).to_numpy()
    labels = subset["column"].astype(str).tolist() if "column" in subset.columns else [
        str(i) for i in range(len(subset))
    ]
    threshold = drift._DRIFT_FLAG_THRESHOLD
    colors = [core.CUSTOM_COLORS[1] if abs(s) >= threshold else core.CUSTOM_COLORS[0] for s in shifts]

    positions = np.arange(len(subset))
    fig, ax = core.plt.subplots(figsize=(10, max(2.5, 0.40 * len(subset))))
    ax.barh(positions, shifts, color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=0.8)
    for sign in (-1.0, 1.0):
        ax.axvline(sign * threshold, color=core.CUSTOM_COLORS[1], linestyle="--", linewidth=0.9, alpha=0.7)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.8, len(subset) - 0.2)
    ax.set_xlabel(f"standardized mean shift (baseline sigma); dashed = flag threshold +/-{threshold:g}")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return core._save_or_show(fig, save_path=save_path, show=show)
