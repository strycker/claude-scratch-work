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

Two honesty fixes from cross-AI review are load-bearing here (see the
plan's ``must_haves``):

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

The holdout boundary (F2/T-05-03) and per-step L2-failure resilience
(T-05-05, RESEARCH.md Pitfall 2) are added in Plan 05-02 Task 3 — see that
task's docstring updates for the ``must_haves`` they close.

Usage::

    from trading_crab_lib.platform.backtest.driver import run_backtest

    equity_curve, per_step_metrics = run_backtest(
        monthly_features, asset_returns, cfg, cash_returns=cash_returns,
    )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform.allocation.hysteresis import update_active_regime
from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt
from trading_crab_lib.platform.assets.returns import returns_by_regime_stats
from trading_crab_lib.platform.backtest.costs import apply_transaction_cost, compute_turnover
from trading_crab_lib.platform.honesty import registry
from trading_crab_lib.platform.honesty.walkforward import expanding_steps
from trading_crab_lib.platform.labeling.jump_model import (
    canonicalize_states,
    fit_jump_model,
    standardize_features,
)
from trading_crab_lib.platform.prediction.nowcaster import build_nowcaster_training_set, fit_nowcaster
from trading_crab_lib.platform.taxonomy import lean_feature_set

log = logging.getLogger(__name__)


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
    X_df = train_features[lean_cols].dropna()
    X = standardize_features(X_df)

    fit = fit_jump_model(X, K=K, lam=lam, n_restarts=n_restarts)
    states, _centroids = canonicalize_states(fit["states"], fit["centroids"], lean_cols)
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
    model = fit_nowcaster(X, y)
    proba = model.predict_proba(feature_row[X.columns])
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

    Refits L1 (``_refit_l1``) and L2 (``_refit_l2``) on ``train_index``-only
    data at every step (T-05-04) — no code path in this module loads the
    single Phase 3/4 nowcaster checkpoint.

    When ``use_regime_tilt`` is False, a degenerate constant single-state
    label/probability pair feeds the SAME tilt code (regime-agnostic /
    vol-target-only ablation — never a hand-rolled parallel
    implementation). When additionally
    ``cfg["backtest"].get("skip_l1l2_for_ablation", True)``, the L1/L2
    refits are skipped entirely for that step (review F5) since their
    output would be discarded anyway.

    Args:
        monthly_features: causal monthly features (Phase 1 checkpoint shape).
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

    records: list[dict[str, Any]] = []
    per_step_metrics: dict[str, list] = {"dates": [], "proba": [], "classes": []}

    prev_weights: pd.Series = pd.Series(dtype=float)
    prev_active_regime: int | None = None
    prev_cash: float = 1.0

    for t, train_index, test_index in expanding_steps(monthly_features.index, min_train=min_train):
        train_features = monthly_features.loc[train_index]
        degraded = False

        if not use_regime_tilt:
            if not skip_l1l2:
                # Still pay the refit cost (for debugging/comparison
                # parity) but the output below is discarded either way —
                # review F5.
                train_states = _refit_l1(train_features, cfg)
                feature_row = monthly_features.loc[[t]]
                _refit_l2(train_features, train_states, feature_row, cfg)
            states = pd.Series(0, index=train_index)
            regime_probs = pd.Series({0: 1.0})
        else:
            states = _refit_l1(train_features, cfg)
            feature_row = monthly_features.loc[[t]]
            regime_probs = _refit_l2(train_features, states, feature_row, cfg)

        stats = returns_by_regime_stats(asset_returns.loc[train_index], states)
        new_active_regime = update_active_regime(
            regime_probs,
            prev_active_regime,
            act_threshold=act_threshold,
            unwind_threshold=unwind_threshold,
        )
        tilt = vol_targeted_tilt(
            regime_probs,
            stats,
            asset_returns.loc[train_index],
            target_vol_annual=target_vol_annual,
            halflife=halflife,
            min_obs=portfolio_vol_min_obs,
        )
        new_weights = tilt["weights"]
        new_cash = tilt["cash"]

        turnover = compute_turnover(prev_weights, new_weights)
        test_date = test_index[0]
        asset_return_row = asset_returns.loc[test_date]
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
        "labeling": {"K": 2, "lambda": 5.0, "n_restarts": 2, "embargo_months": 3},
        "allocation": {
            "target_vol_annual": 0.10, "ewma_halflife_months": 6, "portfolio_vol_min_obs": 3,
            "hysteresis": {"act_threshold": 0.70, "unwind_threshold": 0.40},
        },
        "backtest": {"cost_bps": 10, "min_train_months": 24, "skip_l1l2_for_ablation": True},
    }
    _equity_curve, _per_step_metrics = run_backtest(_monthly_features, _asset_returns, _cfg, cash_returns=_cash_returns)
    print(_equity_curve)  # noqa: T201 — first-class self-check output
