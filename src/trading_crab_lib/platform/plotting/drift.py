"""
platform/plotting/drift.py — D-09 drift-against-baseline and D-11
plausibility bands (Phase 6, 06-VALIDATION.md).

Pure functions only — no matplotlib import — so this module is importable
and testable independently of any figure code.

**Drift and plausibility catch disjoint failure modes; neither substitutes
for the other.** A uniformly wrong series shows zero drift against itself —
this is exactly why the percent-vs-decimal yield defect (which compounded
``long_duration_tr`` to 2.3e128 across the entire 1962-2026 span) was
invisible to a drift-only check: the corrupted series never moved relative
to its own corrupted baseline. Plausibility catches "this value cannot
exist"; drift catches "this feature is quietly decaying" (e.g. a
seasonality effect that was predictive and stopped being predictive). Every
notebook that displays one of these numbers must run both checks, not
either.

The plausibility raise/warn idiom below is copied verbatim in shape from
``platform/splice.py::assert_yield_units_plausible``: module-level named
constants (never inline magic numbers), a hard ``ValueError`` whose f-string
states the observed value, the band it violated, and what to check, and
``log.warning`` only where the failure direction is genuinely ambiguous.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from trading_crab_lib.platform.honesty.holdout import (
    DEFAULT_HOLDOUT_CUTOFF,
    split_by_holdout_boundary,
)

log = logging.getLogger(__name__)

# ── D-11 plausibility bands (reinstated by 06-CONTEXT.md Amendment 1 item A) ──
# None of these bounds are fitted to the data — they are facts about the
# domain (a US Treasury yield has never exceeded 100% annualized; a 60/40
# portfolio cannot cross 1972-2020 with a -2.27% max drawdown; a total-return
# index cannot reach 10^128). See 06-VALIDATION.md's plausibility table.

_UNIVERSAL_LOG_WEALTH_ABS_MAX = 10.0
_DOMAIN_LOG_WEALTH_RANGE: tuple[float, float] = (-3.0, 12.0)

_UNIVERSAL_MAX_DRAWDOWN_RANGE: tuple[float, float] = (-1.0, 0.0)
_DOMAIN_MAX_DRAWDOWN_BANDS: dict[str, tuple[float, float]] = {
    "spy_buy_hold": (-0.90, -0.30),
    "sixty_forty": (-0.80, -0.10),
    "faber_sma": (-0.50, -0.05),
    "strategy": (-0.80, -0.05),
    "no_regime_ablation": (-0.80, -0.05),
}

# This codebase's multiclass Brier score is mean((p - onehot)^2) over the
# full (n, K) array (evaluation/model_metrics.py::compute_brier_multiclass) —
# the bound is [0, 1], NOT the textbook [0, 2].
_BRIER_RANGE: tuple[float, float] = (0.0, 1.0)

_TURNOVER_RANGE: tuple[float, float] = (0.0, 2.0)
_TURNOVER_OPERATIONAL_MAX = 0.30

_CVAR_RANGE: tuple[float, float] = (-0.5, 0.0)
_CVAR_OPERATIONAL_RANGE: tuple[float, float] = (-0.15, -0.01)

# Mirrors labeling/diagnostics.py's existing _MIN_OCCUPANCY_THRESHOLD (0.05).
_MIN_OCCUPANCY_WARN = 0.05
_OCCUPANCY_SUM_TOLERANCE = 1e-9

# Mirrors transforms_monthly.py's existing _DISCONTINUITY_RATIO (1.5) — the
# ALFRED rebasing defect closed as audit item A4 (fred_cpi ~3x cliffs).
_LEVEL_DISCONTINUITY_RATIO = 1.5

# D-09 drift flag threshold: |standardized mean shift| at or above this many
# baseline standard deviations is flagged as drifted.
_DRIFT_FLAG_THRESHOLD = 1.0


def no_skill_brier(n_classes: int) -> float:
    """No-skill (always-predict-the-prior) multiclass Brier floor: (K-1)/K^2."""
    return (n_classes - 1) / n_classes**2


def assert_terminal_log_wealth_plausible(value: float, *, leg: str) -> None:
    """Raise if *value* is not a plausible terminal log-wealth for *leg*.

    Checks the universal absolute bound first (``abs(x) < 10``, i.e.
    ``exp(x)`` under ~e^10 ≈ 22,000x), then the per-span domain band.

    Raises:
        ValueError: on either bound.
    """
    if abs(value) >= _UNIVERSAL_LOG_WEALTH_ABS_MAX:
        raise ValueError(
            f"terminal_log_wealth={value:.4f} for leg '{leg}' violates the universal "
            f"bound abs(x) < {_UNIVERSAL_LOG_WEALTH_ABS_MAX}. exp({value:.2f}) is not a "
            f"plausible total-return multiple over any real backtest span — check for a "
            f"compounding units error (e.g. a yield fed in as percent rather than decimal)."
        )
    lo, hi = _DOMAIN_LOG_WEALTH_RANGE
    if not (lo <= value <= hi):
        raise ValueError(
            f"terminal_log_wealth={value:.4f} for leg '{leg}' violates the domain band "
            f"[{lo}, {hi}] for this backtest's span, even though it passes the universal "
            f"abs(x) < {_UNIVERSAL_LOG_WEALTH_ABS_MAX} bound. Check the equity curve for "
            f"this leg for a scaling or compounding error."
        )


def assert_max_drawdown_plausible(value: float, *, leg: str) -> None:
    """Raise if *value* is not a plausible max drawdown for *leg*.

    Checks the universal range ``[-1, 0]`` first, then the per-leg domain
    band. When *leg* has no configured domain band, logs a warning and
    applies the universal bound only.

    Raises:
        ValueError: on either bound.
    """
    u_lo, u_hi = _UNIVERSAL_MAX_DRAWDOWN_RANGE
    if not (u_lo <= value <= u_hi):
        raise ValueError(
            f"max_drawdown={value:.4f} for leg '{leg}' violates the universal band "
            f"[{u_lo}, {u_hi}] — a drawdown cannot exceed -100% or be positive. Check for "
            f"an equity-curve sign error."
        )
    band = _DOMAIN_MAX_DRAWDOWN_BANDS.get(leg)
    if band is None:
        log.warning(
            "assert_max_drawdown_plausible: no domain band configured for leg '%s' — "
            "only the universal [-1, 0] bound was checked.",
            leg,
        )
        return
    floor, ceiling = band
    if not (floor <= value <= ceiling):
        raise ValueError(
            f"max_drawdown={value:.4%} for leg '{leg}' violates its domain band "
            f"[{floor:.0%}, {ceiling:.0%}], even though it passes the universal "
            f"[{u_lo:.0%}, {u_hi:.0%}] bound. This is exactly the failure mode a "
            f"universal-only check misses (e.g. the historical 60/40 leg reporting a "
            f"-2.27% max drawdown). Check for a percent-vs-decimal units error at the "
            f"splice boundary."
        )


def assert_brier_plausible(value: float, *, n_classes: int) -> dict[str, float | bool]:
    """Raise if *value* is outside this codebase's multiclass Brier range ``[0, 1]``.

    Returns:
        dict: ``{"value": value, "no_skill": no_skill_brier(n_classes),
        "beats_no_skill": value < no_skill}``.

    Raises:
        ValueError: if *value* is outside ``[0, 1]``.
    """
    lo, hi = _BRIER_RANGE
    if not (lo <= value <= hi):
        raise ValueError(
            f"brier={value:.6f} violates the plausible range [{lo}, {hi}] for this "
            f"codebase's multiclass Brier score (mean((p - onehot)^2) over the full "
            f"(n, K) array — bound is [0, 1], NOT the textbook [0, 2]). Check "
            f"compute_brier_multiclass's inputs."
        )
    baseline = no_skill_brier(n_classes)
    beats_no_skill = value < baseline
    if not beats_no_skill:
        log.warning(
            "assert_brier_plausible: brier=%.6f does not beat the no-skill reference "
            "%.4f for K=%d classes — this metric cannot presently claim the model beats "
            "random guessing.",
            value, baseline, n_classes,
        )
    return {"value": value, "no_skill": baseline, "beats_no_skill": beats_no_skill}


def assert_turnover_plausible(value: float) -> None:
    """Raise if *value* is not a plausible monthly portfolio turnover.

    Warns (does not raise) above the operational ceiling for a
    hysteresis-gated book.

    Raises:
        ValueError: outside ``[0, 2]``.
    """
    lo, hi = _TURNOVER_RANGE
    if not (lo <= value <= hi):
        raise ValueError(
            f"turnover={value:.4f} violates the plausible range [{lo}, {hi}] for monthly "
            f"portfolio turnover. Check the weights history for a units or sign error."
        )
    if value >= _TURNOVER_OPERATIONAL_MAX:
        log.warning(
            "assert_turnover_plausible: turnover=%.4f exceeds the operational ceiling "
            "%.2f for a hysteresis-gated book — check for excessive rebalancing.",
            value, _TURNOVER_OPERATIONAL_MAX,
        )


def assert_cvar_plausible(value: float) -> None:
    """Raise if *value* is not a plausible monthly CVaR(5%).

    Warns (does not raise) outside the operational range for a
    vol-targeted book.

    Raises:
        ValueError: outside ``[-0.5, 0]``.
    """
    lo, hi = _CVAR_RANGE
    if not (lo <= value <= hi):
        raise ValueError(
            f"cvar={value:.4f} violates the plausible range [{lo}, {hi}] for a monthly "
            f"CVaR(5%). Check for a units or sign error."
        )
    op_lo, op_hi = _CVAR_OPERATIONAL_RANGE
    if not (op_lo <= value <= op_hi):
        log.warning(
            "assert_cvar_plausible: cvar=%.4f is outside the operational range "
            "[%.2f, %.2f] for a vol-targeted book.",
            value, op_lo, op_hi,
        )


def assert_regime_occupancy_plausible(occupancy: pd.Series | dict[Any, float]) -> list[int]:
    """Validate regime occupancy shares: each in ``[0, 1]``, summing to 1.0.

    Args:
        occupancy: Series or mapping of state id -> occupancy share.

    Returns:
        list[int]: state ids below the soft warning threshold
        (:data:`_MIN_OCCUPANCY_WARN`, mirrors
        ``labeling/diagnostics.py::_MIN_OCCUPANCY_THRESHOLD``), logged at
        WARNING.

    Raises:
        ValueError: if any share is outside ``[0, 1]``, or the shares do not
            sum to 1.0 within :data:`_OCCUPANCY_SUM_TOLERANCE`.
    """
    items = list(occupancy.items()) if isinstance(occupancy, (pd.Series, dict)) else list(dict(occupancy).items())

    total = 0.0
    below_warn: list[int] = []
    for state, share in items:
        share = float(share)
        if not (0.0 <= share <= 1.0):
            raise ValueError(
                f"regime occupancy for state {state} is {share:.4f}, outside [0, 1]. "
                f"Check the occupancy computation."
            )
        total += share
        if share < _MIN_OCCUPANCY_WARN:
            below_warn.append(int(state))

    if abs(total - 1.0) > _OCCUPANCY_SUM_TOLERANCE:
        raise ValueError(
            f"regime occupancy shares sum to {total:.6f}, not 1.0 (tolerance "
            f"{_OCCUPANCY_SUM_TOLERANCE}). Check for a missing or double-counted state."
        )

    if below_warn:
        log.warning(
            "assert_regime_occupancy_plausible: state(s) %s below the %.0f%% sanity "
            "threshold (mirrors labeling/diagnostics._MIN_OCCUPANCY_THRESHOLD).",
            below_warn, _MIN_OCCUPANCY_WARN * 100,
        )
    return sorted(below_warn)


def assert_portfolio_weights_plausible(weights: pd.Series, *, cash_weight: float = 0.0) -> None:
    """Raise on any negative weight (long-only by design) or a total off 1.0.

    Raises:
        ValueError: on a negative weight, or if
            ``weights.sum() + cash_weight`` departs 1.0 beyond
            :data:`_OCCUPANCY_SUM_TOLERANCE`.
    """
    negative = weights[weights < 0]
    if len(negative):
        raise ValueError(
            f"portfolio weights contain negative value(s): {negative.to_dict()}. This "
            f"platform is long-only by design (no shorts/options per PROJECT.md)."
        )
    asset_total = float(weights.sum())
    total = asset_total + cash_weight
    if abs(total - 1.0) > _OCCUPANCY_SUM_TOLERANCE:
        raise ValueError(
            f"portfolio weights + cash sum to {total:.6f}, not 1.0 (tolerance "
            f"{_OCCUPANCY_SUM_TOLERANCE}). asset weights sum to {asset_total:.6f}, "
            f"cash_weight={cash_weight}."
        )


def assert_no_level_discontinuity(
    series: pd.Series,
    *,
    name: str,
    ratio_threshold: float = _LEVEL_DISCONTINUITY_RATIO,
) -> list[pd.Timestamp]:
    """Raise on a month-over-month level jump beyond *ratio_threshold* in either direction.

    Public counterpart of the private
    ``transforms_monthly.py::_warn_on_level_discontinuity`` detector — the
    shape of the ALFRED rebasing defect closed as audit item A4
    (``fred_cpi`` 1970-12 -> 1971-01, 1988-01 -> 1988-02).

    Returns:
        list[pd.Timestamp]: empty when the series is clean.

    Raises:
        ValueError: naming the offending dates, when a jump is found.
    """
    clean = series.dropna()
    if len(clean) < 2:
        return []
    prev = clean.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (clean / prev).replace([np.inf, -np.inf], np.nan).dropna()
    breaks = ratio[(ratio > ratio_threshold) | (ratio < 1 / ratio_threshold)]
    if len(breaks):
        raise ValueError(
            f"{name}: {len(breaks)} implausible month-over-month level jump(s) exceeding "
            f"a {ratio_threshold}x ratio at {[str(d.date()) for d in breaks.index[:5]]}. "
            f"An index-level series does not move this much in a month — this is the "
            f"shape of the ALFRED rebasing defect closed as audit item A4. Check for a "
            f"units or index-base error."
        )
    return []


def assert_kpi_table_plausible(kpi_table: pd.DataFrame) -> pd.DataFrame:
    """Validate every leg's ``terminal_log_wealth`` and ``max_drawdown``.

    Collects EVERY violation before raising a single ``ValueError`` listing
    them all (the ``config.validate_config`` collect-then-raise idiom).

    Args:
        kpi_table: the persisted ``backtest_kpi_table.parquet`` shape —
            columns ``leg``, ``terminal_log_wealth``, ``max_drawdown``.

    Returns:
        pd.DataFrame: on success, a tidy verdict frame with columns ``leg``,
        ``metric``, ``value``, ``universal_band``, ``domain_band``,
        ``verdict`` — one row per (leg, metric) pair.

    Raises:
        ValueError: listing every violated band, if any.
    """
    violations: list[str] = []
    rows: list[dict[str, Any]] = []

    for _, row in kpi_table.iterrows():
        leg = row["leg"]
        wealth = float(row["terminal_log_wealth"])
        drawdown = float(row["max_drawdown"])

        wealth_verdict = "pass"
        try:
            assert_terminal_log_wealth_plausible(wealth, leg=leg)
        except ValueError as exc:
            wealth_verdict = "FAIL"
            violations.append(str(exc))

        dd_verdict = "pass"
        try:
            assert_max_drawdown_plausible(drawdown, leg=leg)
        except ValueError as exc:
            dd_verdict = "FAIL"
            violations.append(str(exc))

        rows.append(
            {
                "leg": leg,
                "metric": "terminal_log_wealth",
                "value": wealth,
                "universal_band": f"abs(x) < {_UNIVERSAL_LOG_WEALTH_ABS_MAX}",
                "domain_band": str(_DOMAIN_LOG_WEALTH_RANGE),
                "verdict": wealth_verdict,
            }
        )
        rows.append(
            {
                "leg": leg,
                "metric": "max_drawdown",
                "value": drawdown,
                "universal_band": str(_UNIVERSAL_MAX_DRAWDOWN_RANGE),
                "domain_band": str(_DOMAIN_MAX_DRAWDOWN_BANDS.get(leg, "n/a — no domain band configured")),
                "verdict": dd_verdict,
            }
        )

    if violations:
        bullet_list = "\n".join(f"  - {v}" for v in violations)
        raise ValueError(
            f"assert_kpi_table_plausible found {len(violations)} plausibility "
            f"violation(s):\n{bullet_list}"
        )

    return pd.DataFrame(rows, columns=["leg", "metric", "value", "universal_band", "domain_band", "verdict"])


# ── D-09 drift-against-baseline ────────────────────────────────────────────


def baseline_and_current(
    full_df: pd.DataFrame, cutoff: str = DEFAULT_HOLDOUT_CUTOFF
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *full_df* into (baseline, current) at *cutoff* (D-09).

    Delegates to ``honesty.holdout.split_by_holdout_boundary`` so the
    baseline is always the pre-2021 fitted window and never a rolling
    trailing window — a rolling baseline decays alongside the feature and
    never trips on gradual decay, which is the failure this check exists to
    catch.
    """
    return split_by_holdout_boundary(full_df, cutoff=cutoff)


def compute_drift(current: pd.Series, baseline: pd.Series) -> dict[str, float | int | bool]:
    """Standardized mean shift, KS statistic, and 2-sigma exceedance of *current* vs *baseline*.

    Returns NaN-valued keys rather than raising when either side is empty
    or the baseline standard deviation is zero.

    Returns:
        dict with keys ``n_baseline``, ``n_current``, ``baseline_mean``,
        ``baseline_std``, ``current_mean``, ``standardized_mean_shift``,
        ``ks_statistic``, ``ks_pvalue``, ``pct_beyond_2sigma``, ``flag``
        (True when ``abs(standardized_mean_shift) >= _DRIFT_FLAG_THRESHOLD``).
    """
    baseline_clean = baseline.dropna()
    current_clean = current.dropna()

    result: dict[str, float | int | bool] = {
        "n_baseline": len(baseline_clean),
        "n_current": len(current_clean),
        "baseline_mean": float("nan"),
        "baseline_std": float("nan"),
        "current_mean": float("nan"),
        "standardized_mean_shift": float("nan"),
        "ks_statistic": float("nan"),
        "ks_pvalue": float("nan"),
        "pct_beyond_2sigma": float("nan"),
        "flag": False,
    }
    if len(baseline_clean) == 0 or len(current_clean) == 0:
        return result

    baseline_mean = float(baseline_clean.mean())
    baseline_std = float(baseline_clean.std())
    current_mean = float(current_clean.mean())
    result["baseline_mean"] = baseline_mean
    result["baseline_std"] = baseline_std
    result["current_mean"] = current_mean

    if baseline_std == 0.0:
        return result

    standardized_shift = (current_mean - baseline_mean) / baseline_std
    ks_stat, ks_pvalue = ks_2samp(current_clean.to_numpy(), baseline_clean.to_numpy())
    pct_beyond = float((np.abs(current_clean - baseline_mean) > 2 * baseline_std).mean())

    result["standardized_mean_shift"] = standardized_shift
    result["ks_statistic"] = float(ks_stat)
    result["ks_pvalue"] = float(ks_pvalue)
    result["pct_beyond_2sigma"] = pct_beyond
    result["flag"] = bool(abs(standardized_shift) >= _DRIFT_FLAG_THRESHOLD)
    return result


def drift_report(
    full_df: pd.DataFrame, columns: list[str], *, cutoff: str = DEFAULT_HOLDOUT_CUTOFF
) -> pd.DataFrame:
    """One row per column in *columns*, sorted by descending absolute standardized mean shift."""
    baseline_df, current_df = baseline_and_current(full_df, cutoff=cutoff)
    empty = pd.Series(dtype=float)
    rows = []
    for col in columns:
        current_series = current_df[col] if col in current_df.columns else empty
        baseline_series = baseline_df[col] if col in baseline_df.columns else empty
        stats = compute_drift(current_series, baseline_series)
        rows.append({"column": col, **stats})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    order = result["standardized_mean_shift"].abs().sort_values(ascending=False, na_position="last").index
    return result.loc[order].reset_index(drop=True)
