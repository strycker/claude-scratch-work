"""
Walk-forward backtest driver (EVAL-01, design §14 Phase 1 exit).

``run_backtest()`` steps the frozen ``honesty/walkforward.py::expanding_steps``
generator across the monthly index, refitting L1 (jump-model labeler,
``labeling/jump_model.py``) and L2 (calibrated nowcaster,
``prediction/nowcaster.py``) on ``train_index``-only data at EVERY step —
never the Phase 3/4 one-shot nowcaster checkpoint (that model has seen 2020
data; scoring 1972 with it is look-ahead, RESEARCH Anti-Pattern) — then
computes L3 returns-by-regime stats (``assets/returns.py``), applies the L4
hysteresis-gated vol-targeted regime tilt (``allocation/hysteresis.py`` +
``allocation/tilt.py``), and compounds a cost-adjusted
(``backtest/costs.py``) equity curve.

Three honesty fixes from cross-AI review are load-bearing here (see the
plan's ``must_haves``):

- F2: the accumulated ``per_step_metrics`` never sources ``y_true`` from
  this in-progress loop (the label at ``t`` was never in ``train_index``,
  which is strictly before ``t`` — sourcing "ground truth" from here would
  be off-by-one). The report layer (Plan 06) joins ``y_true`` retroactively
  against the full-sample smoothed reference labeling by decision date.
- F4: the strategy's cash residual (``tilt["cash"]``) earns the SAME
  ``cash_returns`` series the baseline legs use for their own cash sleeve —
  not a hard 0% — so the strategy/baseline comparison is symmetric.
- F5: when ``use_regime_tilt=False`` and
  ``cfg["backtest"].get("skip_l1l2_for_ablation", True)``, the expensive
  L1/L2 refits are skipped entirely for that step (their output would be
  discarded anyway once the tilt is off) — this halves the ~588-refit
  walk-forward gauntlet without changing the tilt-off equity curve
  (``TestAblationSkipInvariant`` proves skip=True == skip=False
  byte-for-byte).

Exactly one registry trial is logged per full run (mirroring, not
duplicating, ``run_walkforward``'s single-trial convention) — this loop is
NOT a call to ``run_walkforward`` (A1: the per-step body is new code; only
the ``expanding_steps`` spine is reused).

``split_by_holdout_boundary`` is applied to BOTH ``monthly_features`` and
``asset_returns`` BEFORE ``expanding_steps`` is constructed (T-05-03) — the
visited index never exceeds ``DEFAULT_HOLDOUT_CUTOFF`` (2020-12-31) even if
the input frames physically extend past it; no code path in this module
calls the holdout-namespace checkpoint manager getter. A per-step L2 refit failure
(T-05-05, RESEARCH.md Pitfall 2 — an early small post-embargo window
starving a K-fold) is caught, logged at WARNING, and degrades that step to
holding the previous weights/active regime rather than crashing the
1972-2020 run.

Usage::

    from trading_crab_lib.platform.backtest.driver import run_backtest

    equity_curve, per_step_metrics = run_backtest(
        monthly_features, asset_returns, cfg, cash_returns=cash_returns,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform.allocation.hysteresis import update_active_regime
from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt
from trading_crab_lib.platform.assets.returns import returns_by_regime_stats
from trading_crab_lib.platform.backtest.costs import apply_transaction_cost, compute_turnover
from trading_crab_lib.platform.honesty import registry
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.honesty.walkforward import expanding_steps
from trading_crab_lib.platform.labeling.jump_model import (
    canonicalize_states,
    fit_jump_model,
    standardize_features,
)
from trading_crab_lib.platform.prediction.nowcaster import build_nowcaster_training_set, fit_nowcaster
from trading_crab_lib.platform.taxonomy import lean_feature_set

log = logging.getLogger(__name__)

# Report progress this often. 12 = once per simulated year, which keeps a
# multi-hundred-step run visibly alive without flooding the log.
_PROGRESS_EVERY_STEPS = 12

# The only exception CalibratedClassifierCV/LogisticRegression raise for a
# degenerate (too-few-samples-per-class) fold — see RESEARCH.md Pitfall 2.
# ValueError: degenerate fold fit (0 samples / n_samples < n_clusters).
# IndexError: an empty CV test fold on a tiny early window (belt-and-suspenders;
#   cv.PurgedEmbargoedKFold now skips empty folds, but a data-starved early
#   window — e.g. VIX-driven NaN collapse before 1990 — must still degrade
#   gracefully rather than crash the whole 1972-2020 run).
_L2_DEGRADE_EXCEPTIONS: tuple[type[Exception], ...] = (ValueError, IndexError)


def _window_active_features(
    features: pd.DataFrame, cols: list[str], *, min_history: int
) -> list[str]:
    """Features usable in this window: those with ≥ ``min_history`` non-NaN months.

    A late-starting feature (``fred_vix`` from 1990, extra bond-curve tenors like
    a 20Y/30Y, any post-1990 ticker) ENTERS the regime model once it has
    accumulated ``min_history`` months of data; earlier windows simply use the
    features that exist then. So recent features are leveraged wherever they have
    enough history, without discarding pre-1990 rows in the windows where they do
    not yet qualify.

    The caller then trains on the rectangular block of rows where every active
    feature is present (governed by the latest-starting active feature), so
    ``min_history`` also floors the training-row count — a feature never enters
    with too little data to fit on. Computed from in-window data only
    (``train_index`` is strictly before the decision date) — causal, no
    look-ahead.
    """
    return [c for c in cols if int(features[c].notna().sum()) >= min_history]


def _cv_safe_active_features(
    X: pd.DataFrame,
    y: pd.Series,
    cols: list[str],
    *,
    min_history: int,
    n_splits: int,
) -> list[str]:
    """``_window_active_features`` narrowed until the induced training block can
    actually support the calibration CV.

    A late feature does not just need ``min_history`` months of its OWN data —
    admitting it truncates the training block to the rows where every active
    feature is present (``fit_nowcaster`` drops non-finite rows). That recent-only
    block is what the fit sees, and ``CalibratedClassifierCV`` raises
    ``ValueError: Requesting {n}-fold cross-validation but provided less than {n}
    examples for at least one class`` whenever ANY class present in the block has
    fewer than ``n_splits`` rows. That is the real trigger for the degrade cluster
    around a late feature's activation date — and it is independent of block SIZE:
    a 240-row block with a 3-example regime degrades exactly like a 120-row one.
    Raising ``min_history`` therefore only smooths it by accident (a later
    activation tends to have accumulated more of each class), never reliably; a
    regime that simply did not occur since the feature's start date stays rare no
    matter how long you wait.

    So admission is gated on the induced block itself: candidates are dropped
    newest-first (the newest feature is the one truncating the block hardest)
    until every class present has ``>= n_splits`` examples. A late feature enters
    as early as it is SAFE to, not merely as early as it is old enough — and the
    step falls back to the long-history feature set instead of degrading to a
    held position. No rows are imputed and no class is silently dropped, so the
    fit stays honest.

    Returns:
        list[str]: the admissible active columns (possibly empty — the caller's
        fit then raises and the step degrades, as before).
    """
    active = _window_active_features(X, cols, min_history=min_history)
    while active:
        block = np.isfinite(X[active].to_numpy(dtype=float)).all(axis=1)
        counts = y[block].value_counts()
        # >= 2 classes is a floor for classification at all; the CV needs
        # n_splits examples of each class that IS present.
        if len(counts) >= 2 and int(counts.min()) >= n_splits:
            return active
        # Drop the newest-starting candidate — it is the one truncating the block.
        active.remove(max(active, key=lambda c: _first_valid_position(X, c)))
    return active


def _first_valid_position(X: pd.DataFrame, col: str) -> int:
    """Positional index of ``col``'s first non-NaN row (``len(X)`` if all-NaN)."""
    first = X[col].first_valid_index()
    return len(X) if first is None else int(X.index.get_loc(first))


def _refit_l1(train_features: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    """Refit the L1 jump-model labeler on ``train_features`` ONLY.

    Mirrors ``labeling/diagnostics.py::label_regimes``'s wiring order (pull
    lean cols -> standardize -> fit -> canonicalize) but on a train-window
    slice, never the full-sample checkpoint.

    Returns:
        pd.Series: canonical state labels (0..K-1), indexed by the
        (post-dropna) subset of ``train_features.index``.
    """
    labeling_cfg = cfg.get("labeling", {})
    K = labeling_cfg.get("K", 5)
    lam = labeling_cfg.get("lambda", 52.0)
    n_restarts = labeling_cfg.get("n_restarts", 10)

    lean_cols = sorted(lean_feature_set(cfg) & set(train_features.columns))
    # Use the features that have ≥ feature_min_history months of data in THIS
    # window (a late feature like VIX enters once it qualifies), then train on the
    # rectangular block where every active feature is present — the block start is
    # governed by the latest-starting active feature. Labels stay comparable
    # across feature-set transitions because canonicalize_states sorts by
    # trailing_return_1m, present in every window.
    min_history = int(cfg.get("backtest", {}).get("feature_min_history", 120))
    active = _window_active_features(train_features, lean_cols, min_history=min_history)
    X_df = train_features[active].dropna(axis=0, how="any")
    used_cols = list(X_df.columns)
    X = standardize_features(X_df)

    fit = fit_jump_model(X, K=K, lam=lam, n_restarts=n_restarts)
    states, _centroids = canonicalize_states(fit["states"], fit["centroids"], used_cols)
    return pd.Series(states, index=X_df.index, name="state")


def _refit_l2(
    train_features: pd.DataFrame,
    train_states: pd.Series,
    feature_row: pd.DataFrame,
    cfg: dict[str, Any],
) -> pd.Series:
    """Refit the L2 nowcaster on ``(train_features, train_states)``, score ``feature_row``.

    ``feature_row`` is a single-row DataFrame for the CURRENT decision date
    ``t`` (causal features already available at ``t`` — a PREDICTION input,
    never a training row: ``train_states`` never contains a label at or
    after ``t``, since ``train_features``/``train_states`` are sliced to
    ``train_index`` which ``expanding_steps`` guarantees is strictly before
    ``t``).

    Returns:
        pd.Series: regime_probs indexed by ``model.classes_`` (the class
        order this step's fit produced).

    Raises:
        ValueError: propagated from a degenerate fold fit (RESEARCH.md
            Pitfall 2) — the caller (``run_backtest``) decides whether to
            degrade gracefully (added in Task 3).
    """
    embargo_months = cfg.get("labeling", {}).get("embargo_months", 12)
    X, y = build_nowcaster_training_set(train_features, train_states, embargo_months=embargo_months)
    # Use the features with ≥ feature_min_history months in this window (late
    # features like VIX enter once they qualify), applied to BOTH the training
    # matrix and the prediction row so their columns align. fit_nowcaster's own
    # non-finite-row drop then restricts training to the rectangular block where
    # every active feature is present. Causal — in-window data only (before t).
    min_history = int(cfg.get("backtest", {}).get("feature_min_history", 120))
    n_splits = int(cfg.get("backtest", {}).get("nowcaster_cv_splits", 5))
    active = _cv_safe_active_features(X, y, list(X.columns), min_history=min_history, n_splits=n_splits)
    X = X[active]
    model = fit_nowcaster(X, y, n_splits=n_splits)
    proba = model.predict_proba(feature_row[active])
    return pd.Series(proba[0], index=model.classes_)


def _realized_return(
    weights: pd.Series,
    cash_weight: float,
    asset_return_row: pd.Series,
    *,
    cash_return: float = 0.0,
) -> float:
    """Weighted asset-leg return + the cash residual's OWN return (review F4).

    ``cash_weight`` (``tilt["cash"]``) earns ``cash_return`` — the SAME
    series the baseline legs use for their cash sleeve — never a hard 0%.
    Callers with no ``cash_returns`` series available (e.g. synthetic
    tests) pass the documented default ``cash_return=0.0``.
    """
    common = weights.index.intersection(asset_return_row.index)
    asset_leg = float((weights[common] * asset_return_row[common]).sum()) if len(common) else 0.0
    return asset_leg + cash_weight * cash_return


def run_backtest(
    monthly_features: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    min_train: int | None = None,
    cash_returns: pd.Series | None = None,
    use_regime_tilt: bool = True,
    registry_path: Any = None,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Run the L1->L4 expanding-window walk-forward loop, log exactly one trial.

    Applies ``split_by_holdout_boundary`` to both ``monthly_features`` and
    ``asset_returns`` BEFORE constructing ``expanding_steps`` (T-05-03) — the
    visited index never exceeds ``DEFAULT_HOLDOUT_CUTOFF`` (2020-12-31) even
    when the input frames physically extend past it; this module never
    calls the holdout-namespace checkpoint manager getter. Refits L1 (``_refit_l1``)
    and L2 (``_refit_l2``) on ``train_index``-only data at every step
    (T-05-04) — no code path in this module loads the single Phase 3/4
    nowcaster checkpoint.

    When ``use_regime_tilt`` is False, a degenerate constant single-state
    label/probability pair feeds the SAME tilt code (regime-agnostic /
    vol-target-only ablation — never a hand-rolled parallel
    implementation). When additionally
    ``cfg["backtest"].get("skip_l1l2_for_ablation", True)``, the L1/L2
    refits are skipped entirely for that step (review F5) since their
    output would be discarded anyway.

    An L2 refit failure (RESEARCH.md Pitfall 2 — an early small
    post-embargo window starving a K-fold) is caught per step, logged at
    WARNING, and degrades that step to holding the previous
    weights/active-regime/cash (T-05-05) rather than crashing the run; the
    record's ``degraded`` field is set True and the step is excluded from
    ``per_step_metrics``.

    Args:
        monthly_features: causal monthly features (Phase 1 checkpoint
            shape) — may physically extend past the holdout cutoff.
        asset_returns: monthly simple returns per tradable asset, aligned
            (by index) to ``monthly_features``.
        cfg: platform config (``load_platform_config()`` output).
        min_train: overrides ``cfg["backtest"]["min_train_months"]``.
        cash_returns: the cash-sleeve return series the strategy's cash
            residual earns (review F4). Defaults to a documented 0.0 when
            omitted (synthetic tests only — never omit in a real run).
        use_regime_tilt: False runs the no-regime ablation (design §8.7).
        registry_path: overrides the default trial registry ledger path.

    Returns:
        tuple[pd.DataFrame, dict[str, list]]: ``(equity_curve, per_step_metrics)``.
        ``equity_curve`` is indexed by decision date with columns
        ``return``, ``turnover``, ``cost``, ``active_regime``, ``scale``,
        ``degraded``. ``per_step_metrics`` has keys ``dates``, ``proba``,
        ``classes`` — NO loop-sourced ``y_true`` (review F2); the report
        layer joins ``y_true`` from the smoothed reference by date.
    """
    backtest_cfg = cfg.get("backtest", {})
    if min_train is None:
        min_train = backtest_cfg.get("min_train_months", 120)
    cost_bps = backtest_cfg.get("cost_bps", 10)
    skip_l1l2 = backtest_cfg.get("skip_l1l2_for_ablation", True)

    allocation_cfg = cfg.get("allocation", {})
    target_vol_annual = allocation_cfg.get("target_vol_annual", 0.10)
    halflife = allocation_cfg.get("ewma_halflife_months", 6)
    portfolio_vol_min_obs = allocation_cfg.get("portfolio_vol_min_obs", 12)
    hysteresis_cfg = allocation_cfg.get("hysteresis", {})
    act_threshold = hysteresis_cfg.get("act_threshold", 0.70)
    unwind_threshold = hysteresis_cfg.get("unwind_threshold", 0.40)

    # T-05-03: apply the holdout boundary BEFORE constructing expanding_steps
    # — no call to the holdout-namespace checkpoint manager getter anywhere in this module,
    # and no reliance on the config end_date to bound the loop.
    dev_features, _ = split_by_holdout_boundary(monthly_features, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_asset_returns, _ = split_by_holdout_boundary(asset_returns, cutoff=DEFAULT_HOLDOUT_CUTOFF)

    records: list[dict[str, Any]] = []
    per_step_metrics: dict[str, list] = {"dates": [], "proba": [], "classes": []}

    prev_weights: pd.Series = pd.Series(dtype=float)
    prev_active_regime: int | None = None
    prev_cash: float = 1.0

    # Materialize the step list so progress can be reported as N/total. Every
    # step refits L1+L2 on an expanding window, so a full run is minutes of
    # silent sklearn work — without this the process looks hung, especially
    # after the early degraded-refit warnings stop and output goes quiet.
    steps = list(expanding_steps(dev_features.index, min_train=min_train))
    total_steps = len(steps)
    started = time.monotonic()
    log.info(
        "Backtest: %d monthly steps from %s to %s (each step refits L1+L2 on an expanding window)",
        total_steps,
        steps[0][0].date() if steps else "n/a",
        steps[-1][0].date() if steps else "n/a",
    )

    for step_no, (t, train_index, test_index) in enumerate(steps, start=1):
        if step_no % _PROGRESS_EVERY_STEPS == 0 or step_no == total_steps:
            elapsed = time.monotonic() - started
            rate = step_no / elapsed if elapsed > 0 else 0.0
            remaining = (total_steps - step_no) / rate if rate > 0 else 0.0
            log.info(
                "Backtest progress: %d/%d steps (%.0f%%) — %s — %.1fs elapsed, ~%.0fs remaining",
                step_no, total_steps, 100.0 * step_no / total_steps,
                t.date(), elapsed, remaining,
            )
        train_features = dev_features.loc[train_index]
        degraded = False

        if not use_regime_tilt:
            if not skip_l1l2:
                # Still pay the refit cost (for debugging/comparison
                # parity) but the output below is discarded either way —
                # review F5.
                try:
                    train_states = _refit_l1(train_features, cfg)
                    feature_row = dev_features.loc[[t]]
                    _refit_l2(train_features, train_states, feature_row, cfg)
                except _L2_DEGRADE_EXCEPTIONS as exc:
                    log.warning(
                        "Step %s: L1/L2 refit failed on the tilt-off (non-skip) path (%s) — "
                        "output discarded anyway", t, exc,
                    )
            states = pd.Series(0, index=train_index)
            regime_probs = pd.Series({0: 1.0})
        else:
            try:
                states = _refit_l1(train_features, cfg)
                feature_row = dev_features.loc[[t]]
                regime_probs = _refit_l2(train_features, states, feature_row, cfg)
            except _L2_DEGRADE_EXCEPTIONS as exc:
                log.warning(
                    "Step %s: L2 refit degraded (early small post-embargo window, "
                    "RESEARCH Pitfall 2) — holding previous weights: %s", t, exc,
                )
                degraded = True
                states = pd.Series(dtype=float)
                regime_probs = pd.Series(dtype=float)

        if degraded:
            new_weights = prev_weights
            new_active_regime = prev_active_regime
            new_cash = prev_cash
        else:
            stats = returns_by_regime_stats(dev_asset_returns.loc[train_index], states)
            new_active_regime = update_active_regime(
                regime_probs,
                prev_active_regime,
                act_threshold=act_threshold,
                unwind_threshold=unwind_threshold,
            )
            tilt = vol_targeted_tilt(
                regime_probs,
                stats,
                dev_asset_returns.loc[train_index],
                target_vol_annual=target_vol_annual,
                halflife=halflife,
                min_obs=portfolio_vol_min_obs,
            )
            new_weights = tilt["weights"]
            new_cash = tilt["cash"]

        turnover = compute_turnover(prev_weights, new_weights)
        test_date = test_index[0]
        asset_return_row = dev_asset_returns.loc[test_date]
        cash_ret = float(cash_returns.loc[test_date]) if cash_returns is not None else 0.0
        gross = _realized_return(new_weights, new_cash, asset_return_row, cash_return=cash_ret)
        net = apply_transaction_cost(gross, turnover, cost_bps)

        records.append(
            {
                "date": test_date,
                "return": net,
                "turnover": turnover,
                "cost": gross - net,
                "active_regime": new_active_regime,
                "scale": float(new_weights.sum()) if len(new_weights) else 0.0,
                "degraded": degraded,
            }
        )

        if not degraded:
            # EVAL-04 accumulators (review F2) — NO loop-sourced y_true; the
            # report layer (Plan 06) joins ground truth retroactively.
            per_step_metrics["dates"].append(t)
            per_step_metrics["proba"].append(regime_probs.values)
            per_step_metrics["classes"].append(list(regime_probs.index))

        prev_weights = new_weights
        prev_active_regime = new_active_regime
        prev_cash = new_cash

    equity_curve = (
        pd.DataFrame(records).set_index("date")
        if records
        else pd.DataFrame(columns=["return", "turnover", "cost", "active_regime", "scale", "degraded"])
    )

    terminal_log_wealth = float(np.log1p(equity_curve["return"]).sum()) if not equity_curve.empty else 0.0
    trial_config = {
        "phase": "05-backtest",
        "use_regime_tilt": use_regime_tilt,
        "min_train": min_train,
        "cost_bps": cost_bps,
    }
    registry.append_trial(
        config=trial_config,
        features=list(monthly_features.columns),
        metrics={"n_steps": int(len(equity_curve)), "terminal_log_wealth": terminal_log_wealth},
        path=registry_path,
    )

    return equity_curve, per_step_metrics


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency.
    _rng = np.random.default_rng(42)
    _idx = pd.date_range("2010-01-31", periods=48, freq="ME")
    _lean_cols = [
        "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
        "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
        "cape_shiller", "div_yield", "real_rate_level",
    ]
    _monthly_features = pd.DataFrame({c: _rng.normal(0, 1, 48) for c in _lean_cols}, index=_idx)
    _asset_returns = pd.DataFrame(
        {"SPY": _rng.normal(0.006, 0.03, 48), "TLT": _rng.normal(0.001, 0.02, 48)}, index=_idx
    )
    _cash_returns = pd.Series(_rng.normal(0.001, 0.0005, 48), index=_idx)
    _cfg = {
        "taxonomy": {
            "fast": _lean_cols[:10],
            "slow": _lean_cols[10:],
            "agency": [],
        },
        "labeling": {"K": 2, "lambda": 5.0, "n_restarts": 2, "embargo_months": 3},
        "allocation": {
            "target_vol_annual": 0.10, "ewma_halflife_months": 6, "portfolio_vol_min_obs": 3,
            "hysteresis": {"act_threshold": 0.70, "unwind_threshold": 0.40},
        },
        "backtest": {"cost_bps": 10, "min_train_months": 24, "skip_l1l2_for_ablation": True},
    }
    _equity_curve, _per_step_metrics = run_backtest(_monthly_features, _asset_returns, _cfg, cash_returns=_cash_returns)
    print(_equity_curve)  # noqa: T201 — first-class self-check output
