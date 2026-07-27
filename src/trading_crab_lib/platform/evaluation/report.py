"""
Honest backtest report — the EVAL-01..04 capstone (design §5.4, §8.7-9,
§23.1). This is where the whole evaluation chain comes together into the
artifact Glenn (and the later L4-upgrade milestone) reads.

``assemble_backtest_report()`` is a pure markdown builder over precomputed
inputs (mirrors ``report/weekly.py``'s ``assemble_weekly_report`` pattern —
no I/O, fully testable against synthetic dicts). Section order is fixed and
load-bearing (D-01a): the sojourn/detection-lag ratio is the FIRST metrics
section — the go/no-go headline — before the Faber comparison (§23.1), the
no-regime-ablation delta ("does the regime layer pay rent", design §8.7),
the smoothed-vs-filtered gap (HON-05), the baseline gauntlet table, and the
full strategy KPI table (with a documented cash-return convention note,
review F4).

``write_backtest_report()`` ALWAYS writes the markdown under
``(output_dir or OUTPUT_DIR / "reports" / "platform") / "backtest_report.md"``
— a distinct filename from the incumbent weekly report and from
``report/weekly.py``'s own ``weekly_report.md`` (no collision) — and persists
any per-leg equity-curve / KPI-table parquet artifacts the caller supplies.

``run_full_backtest_evaluation()`` is the actual orchestration: drives the
regime-tilt strategy backtest (``backtest/driver.py::run_backtest``), the
no-regime ablation (``backtest/baselines.py::no_regime_ablation``), the
three price baselines (``spy_buy_hold``/``sixty_forty``/``faber_sma``), the
sojourn/lag headline (``evaluation/sojourn_lag.py``), the strategy KPIs
(``evaluation/kpis.py``), and the model-metrics artifacts
(``evaluation/model_metrics.py``) — then assembles + writes the report.

Two honesty fixes from cross-AI review are load-bearing here:

- Pitfall 1 / review F1: the SMOOTHED reference labeling is built as ONE
  full-sample (non-walk-forward) ``fit_jump_model`` + ``canonicalize_states``
  call over the holdout-bounded dev window — a genuinely DISTINCT series from
  the walk-forward driver's per-step ``regime_probs`` (collected in
  ``per_step_metrics``). The headline's filtered-probs input is the
  MULTICLASS matrix built by ``sojourn_lag.build_filtered_probs_matrix`` —
  never a class-agnostic max.
- Review F2: ``y_true`` for the model-metrics artifacts is joined by
  REINDEXING the smoothed reference states onto ``per_step_metrics["dates"]``
  (the smoothed label AT each decision month) — never sourced from the
  walk-forward loop itself.
- Review F4: ``cash_ret`` (from ``splice.build_core_research_series`` ->
  ``assets/returns.compute_monthly_returns``) is passed into
  ``run_backtest`` as ``cash_returns`` so the strategy's cash residual earns
  the SAME series every baseline's non-invested leg earns.

D-04: this module never computes or displays the design-Phase-6
registry-denominator risk-adjusted statistic — the trial registry is
written (by ``run_backtest``/``no_regime_ablation``) but never read for
that statistic's multiple-testing denominator in this phase.

Usage::

    python3 -m trading_crab_lib.platform.evaluation.report
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt
from trading_crab_lib.platform.assets.returns import compute_monthly_returns, returns_by_regime_stats
from trading_crab_lib.platform.backtest.baselines import faber_sma, no_regime_ablation, sixty_forty, spy_buy_hold
from trading_crab_lib.platform.backtest.driver import run_backtest
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.evaluation.kpis import (
    crisis_capture_ratio,
    cvar,
    max_drawdown_and_duration,
    terminal_log_wealth,
)
from trading_crab_lib.platform.evaluation.model_metrics import report_model_metrics
from trading_crab_lib.platform.evaluation.sojourn_lag import build_filtered_probs_matrix, compute_sojourn_lag_headline
from trading_crab_lib.platform.honesty.gap_lag import compute_gap
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.labeling.jump_model import canonicalize_states, fit_jump_model, standardize_features
from trading_crab_lib.platform.splice import build_core_research_series
from trading_crab_lib.platform.taxonomy import lean_feature_set

log = logging.getLogger(__name__)


# ── assemble_backtest_report ──────────────────────────────────────────────────


def assemble_backtest_report(
    *,
    sojourn_lag: dict,
    strategy_kpis: dict,
    ablation_kpis: dict,
    baseline_kpis: dict,
    gap: float,
    excluded_assets: list[str] | None = None,
) -> str:
    """Assemble the honest backtest report markdown from precomputed inputs.

    A pure function, no I/O — mirrors ``report/weekly.py``'s
    ``assemble_weekly_report`` isolation pattern so this is testable against
    synthetic dicts with no real backtest run.

    Section order (fixed, D-01a — the headline goes FIRST):

    1. Headline: sojourn/detection-lag ratio (§5.4).
    2. Baseline comparison: Faber 10-month SMA (§23.1 standing target, NOT a
       pass/fail gate).
    3. No-regime-ablation delta ("does the regime layer pay rent", §8.7).
    4. Smoothed-vs-filtered gap (HON-05).
    5. Baseline gauntlet table (SPY, 60/40, Faber).
    6. Strategy KPI table + a documented "Conventions" note (cash-return
       symmetry, review F4; turnover/cost symmetry, A5).

    Args:
        sojourn_lag: ``evaluation/sojourn_lag.py::compute_sojourn_lag_headline``
            output — keys ``median_sojourn``, ``median_lag``, ``ratio``.
        strategy_kpis: dict with ``terminal_log_wealth``, ``max_drawdown``,
            optionally ``duration_months``, ``cvar``, ``turnover``,
            ``crisis_capture`` (dict[str, float]).
        ablation_kpis: dict with (at least) ``terminal_log_wealth``,
            ``max_drawdown`` for the no-regime-tilt ablation leg.
        baseline_kpis: dict keyed by ``"spy_buy_hold"``/``"sixty_forty"``/
            ``"faber_sma"``, each a dict with ``terminal_log_wealth``,
            ``max_drawdown``.
        gap: the smoothed-vs-filtered performance gap
            (``honesty/gap_lag.py::compute_gap`` output).

    Returns:
        str: the assembled markdown.
    """
    lines: list[str] = ["# Honest Backtest Report (EVAL-01..04, design §5.4/§8.7-9/§23.1)", ""]

    # ── 1. Headline: sojourn/detection-lag ratio (D-01a, FIRST metrics section) ──
    lines.append("## Headline: Sojourn / Detection-Lag Ratio (§5.4)")
    lines.append("")
    lines.append(
        "This is the go/no-go number: median regime sojourn (how long a "
        "regime typically lasts) divided by the median real-time detection "
        "lag (how long the walk-forward nowcaster took to notice a "
        "transition, checked against P(its own target state) — review F1). "
        "A high ratio means most of a regime's life is capturable after "
        "detection; a ratio near 1 means the lag eats the trade."
    )
    lines.append("")
    lines.append(f"- median sojourn (months): {sojourn_lag['median_sojourn']}")
    lines.append(f"- median detection lag (months): {sojourn_lag['median_lag']}")
    lines.append(f"- **ratio: {sojourn_lag['ratio']}**")
    lines.append("")

    # ── 2. Faber comparison (§23.1 standing target, not pass/fail) ──
    lines.append("## Baseline Comparison: Faber 10-Month SMA (§23.1)")
    lines.append("")
    lines.append(
        "The Faber 10-month SMA is design §23.1's STANDING TARGET for the "
        "regime strategy to beat on both log wealth AND max drawdown — this "
        "is recorded, not a pass/fail gate (D-01a)."
    )
    lines.append("")
    faber = baseline_kpis.get("faber_sma", {})
    lines.append(
        f"- strategy: terminal log wealth={strategy_kpis['terminal_log_wealth']:.4f}, "
        f"max drawdown={strategy_kpis['max_drawdown']:.2%}"
    )
    lines.append(
        f"- faber_sma: terminal log wealth={faber.get('terminal_log_wealth', float('nan')):.4f}, "
        f"max drawdown={faber.get('max_drawdown', float('nan')):.2%}"
    )
    lines.append("")

    # ── 3. No-regime-ablation delta ("does the regime layer pay rent") ──
    lines.append("## No-Regime-Ablation Delta (Does the Regime Layer Pay Rent?)")
    lines.append("")
    wealth_delta = strategy_kpis["terminal_log_wealth"] - ablation_kpis["terminal_log_wealth"]
    dd_delta = strategy_kpis["max_drawdown"] - ablation_kpis["max_drawdown"]
    lines.append(
        "The no-regime ablation (design §8.7) is the SAME L1-L4 code path "
        "with the regime tilt disabled (backtest/baselines.py::no_regime_ablation) "
        "— never a hand-rolled parallel implementation (D-02)."
    )
    lines.append("")
    lines.append(
        f"- terminal log wealth delta (strategy - ablation): {wealth_delta:+.4f} "
        f"(strategy={strategy_kpis['terminal_log_wealth']:.4f}, "
        f"ablation={ablation_kpis['terminal_log_wealth']:.4f})"
    )
    lines.append(
        f"- max drawdown delta (strategy - ablation): {dd_delta:+.2%} "
        f"(strategy={strategy_kpis['max_drawdown']:.2%}, "
        f"ablation={ablation_kpis['max_drawdown']:.2%})"
    )
    lines.append("")

    # ── 4. Smoothed-vs-filtered gap (HON-05) ──
    lines.append("## Smoothed-vs-Filtered Gap")
    lines.append("")
    lines.append(
        f"- gap (smoothed hindsight performance - real-time filtered performance): "
        f"{gap:.4f} — the measured hindsight content of the strategy (§5.4). "
        "The smoothed reference is ONE full-sample labeler fit; the filtered "
        "series is the walk-forward driver's actual per-step decisions — "
        "genuinely distinct series (Pitfall 1), never the same object reused."
    )
    lines.append("")

    # ── 5. Baseline gauntlet table (SPY, 60/40, Faber) ──
    lines.append("## Baseline Gauntlet")
    lines.append("")
    lines.append("| Leg | Terminal Log Wealth | Max Drawdown |")
    lines.append("|-----|---------------------|--------------|")
    for name, label in (
        ("spy_buy_hold", "SPY Buy & Hold"),
        ("sixty_forty", "60/40"),
        ("faber_sma", "Faber 10-Month SMA"),
    ):
        kpi = baseline_kpis.get(name, {})
        lines.append(
            f"| {label} | {kpi.get('terminal_log_wealth', float('nan')):.4f} | "
            f"{kpi.get('max_drawdown', float('nan')):.2%} |"
        )
    lines.append(
        f"| Strategy (regime tilt) | {strategy_kpis['terminal_log_wealth']:.4f} | "
        f"{strategy_kpis['max_drawdown']:.2%} |"
    )
    lines.append(
        f"| No-Regime Ablation | {ablation_kpis['terminal_log_wealth']:.4f} | "
        f"{ablation_kpis['max_drawdown']:.2%} |"
    )
    lines.append("")
    lines.append(
        f"- no-regime-ablation delta vs. strategy: {wealth_delta:+.4f} terminal log "
        f"wealth ({dd_delta:+.2%} max drawdown) — does the regime layer pay rent?"
    )
    lines.append("")

    # ── 6. Strategy KPI table + Conventions note ──
    lines.append("## Strategy KPIs")
    lines.append("")
    lines.append(f"- terminal log wealth: {strategy_kpis['terminal_log_wealth']:.4f}")
    lines.append(
        f"- max drawdown: {strategy_kpis['max_drawdown']:.2%} "
        f"({strategy_kpis.get('duration_months', 'n/a')} months underwater)"
    )
    lines.append(f"- CVaR(5%): {strategy_kpis.get('cvar', float('nan')):.4f}")
    lines.append(f"- turnover (mean monthly): {strategy_kpis.get('turnover', float('nan')):.4f}")
    crisis_capture = strategy_kpis.get("crisis_capture") or {}
    if crisis_capture:
        lines.append("- in-sample crisis capture ratios (down-capture, A6):")
        for name, ratio in crisis_capture.items():
            lines.append(f"  - {name}: {ratio:.2f}")
    lines.append("")

    lines.append("### Conventions")
    lines.append("")
    lines.append(
        "Cash-return convention (review F4): the strategy's cash residual "
        "and every baseline's non-invested leg (60/40's implicit reconstitution "
        "carry, Faber's out-of-market months) all earn the SAME `cash_ret` "
        "series built by `splice.build_core_research_series` -> "
        "`assets.returns.compute_monthly_returns` — so the §23.1 Faber "
        "comparison is cost-symmetric, never quietly biased by a strategy "
        "that earns 0% cash against a baseline earning the real 1970s-80s "
        "double-digit T-bill rate."
    )
    lines.append(
        "Turnover/cost convention (A5): `cost_bps` is applied identically to "
        "every rebalancing leg (strategy, 60/40, Faber) via "
        "`backtest/costs.py::apply_transaction_cost` — only SPY buy-and-hold "
        "is cost-free by construction (no rebalancing ever occurs)."
    )
    if excluded_assets:
        lines.append(
            "**⚠ Excluded assets:** the following research classes were EXCLUDED "
            "from this run because their source data was unavailable "
            "(e.g. gold when macrotrends is IP-blocked): "
            f"`{', '.join(excluded_assets)}`. The backtest ran on the remaining "
            "assets only — treat cross-run comparisons that include these assets "
            "as not directly comparable."
        )
    lines.append("")

    return "\n".join(lines)


def write_backtest_report(
    markdown: str,
    artifacts: dict[str, pd.DataFrame],
    *,
    output_dir: Path | None = None,
) -> Path:
    """ALWAYS write the markdown, plus persist any per-leg equity-curve / KPI
    -table parquet artifacts the caller supplies.

    Writes under ``(output_dir or OUTPUT_DIR / "reports" / "platform")``:

    - ``backtest_report.md`` — the markdown (a distinct filename from
      ``report/weekly.py``'s ``weekly_report.md``, no collision).
    - ``backtest_{name}.parquet`` — one file per key in ``artifacts``.

    Args:
        markdown: the assembled report markdown.
        artifacts: dict mapping an artifact name (e.g.
            ``"equity_curve_strategy"``, ``"kpi_table"``) to a DataFrame to
            persist (schema-stable empty-safe idiom is the CALLER's
            responsibility — this function just writes what it's given).
        output_dir: overrides the default artifact directory (for tests).

    Returns:
        Path: the written markdown path.
    """
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "platform"
    target_dir.mkdir(parents=True, exist_ok=True)

    report_path = target_dir / "backtest_report.md"
    report_path.write_text(markdown, encoding="utf-8")

    artifact_paths: dict[str, Path] = {}
    for name, df in artifacts.items():
        artifact_path = target_dir / f"backtest_{name}.parquet"
        df.to_parquet(artifact_path, index=True)
        artifact_paths[name] = artifact_path

    log.info("Backtest report written: %s (%d artifacts)", report_path, len(artifacts))
    return report_path


# ── run_full_backtest_evaluation ──────────────────────────────────────────────


def _leg_kpis(returns: pd.Series) -> dict[str, float]:
    """terminal_log_wealth + max_drawdown + duration_months for one baseline/ablation leg."""
    clean = returns.dropna()
    dd = max_drawdown_and_duration(clean)
    return {
        "terminal_log_wealth": terminal_log_wealth(clean),
        "max_drawdown": dd["max_drawdown"],
        "duration_months": dd["duration_months"],
    }


def _build_kpi_table(strategy_kpis: dict, ablation_kpis: dict, baseline_kpis: dict) -> pd.DataFrame:
    """One row per leg: leg/terminal_log_wealth/max_drawdown (KPI-table artifact)."""
    rows = [
        {
            "leg": "strategy",
            "terminal_log_wealth": strategy_kpis["terminal_log_wealth"],
            "max_drawdown": strategy_kpis["max_drawdown"],
        },
        {
            "leg": "no_regime_ablation",
            "terminal_log_wealth": ablation_kpis["terminal_log_wealth"],
            "max_drawdown": ablation_kpis["max_drawdown"],
        },
    ]
    for name, kpi in baseline_kpis.items():
        rows.append(
            {
                "leg": name,
                "terminal_log_wealth": kpi["terminal_log_wealth"],
                "max_drawdown": kpi["max_drawdown"],
            }
        )
    return pd.DataFrame(rows, columns=["leg", "terminal_log_wealth", "max_drawdown"])


def _smoothed_hindsight_perf(
    full_sample_states: pd.Series,
    asset_returns: pd.DataFrame,
    cash_ret: pd.Series,
    decision_dates: list,
    allocation_cfg: dict[str, Any],
) -> float:
    """Terminal log wealth of a hindsight oracle that knows the FULL-SAMPLE
    smoothed regime label at every decision date (but not future returns) —
    the "smoothed" half of the smoothed-vs-filtered gap (§5.4, Pitfall 1).

    At each of the SAME decision dates the walk-forward driver actually
    visited, drives ``vol_targeted_tilt`` (the SAME allocation math, no new
    formula) with a one-hot probability on the smoothed state instead of the
    real-time nowcaster's probability — using regime-conditional stats
    computed over the ENTIRE smoothed labeling (non-causal by construction,
    exactly the "labeler is intentionally non-causal at the batch level"
    behavior the labeler's own docstring documents).
    """
    smoothed_stats = returns_by_regime_stats(asset_returns, full_sample_states)
    step_returns: list[float] = []
    for t in decision_dates:
        state = int(full_sample_states.loc[t])
        tilt = vol_targeted_tilt(
            {state: 1.0},
            smoothed_stats,
            asset_returns.loc[:t],
            target_vol_annual=allocation_cfg.get("target_vol_annual", 0.10),
            halflife=allocation_cfg.get("ewma_halflife_months", 6),
            min_obs=allocation_cfg.get("portfolio_vol_min_obs", 12),
        )
        common = tilt["weights"].index.intersection(asset_returns.columns)
        asset_leg = float((tilt["weights"][common] * asset_returns.loc[t, common]).sum()) if len(common) else 0.0
        cash_leg = tilt["cash"] * float(cash_ret.loc[t]) if t in cash_ret.index else 0.0
        step_returns.append(asset_leg + cash_leg)

    return terminal_log_wealth(pd.Series(step_returns))


def run_full_backtest_evaluation(
    monthly_features: pd.DataFrame,
    monthly_raw: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    registry_path: Any = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Drive the whole EVAL-01..04 chain end-to-end and write the report.

    (a) ``build_core_research_series`` + ``compute_monthly_returns`` ->
        equity/bond/cash return series (and the equity LEVEL for Faber);
        ``run_backtest`` for the regime-tilt strategy, passing
        ``cash_returns=cash_ret`` so the strategy's cash residual earns the
        SAME series the baselines earn (review F4).
    (b) ``no_regime_ablation`` for the ablation leg (same ``cash_ret``).
    (c) the three price baselines over the SAME holdout-bounded window
        (baselines do not enforce the cutoff internally — the report layer
        must slice them, per ``backtest/baselines.py``'s own docstring).
    (d) ONE full-sample ``fit_jump_model`` + ``canonicalize_states`` over the
        holdout-bounded dev window for the SMOOTHED reference labeling — a
        genuinely distinct series from the walk-forward ``per_step_metrics``
        (Pitfall 1). The multiclass filtered-probs matrix
        (``build_filtered_probs_matrix``) feeds ``compute_sojourn_lag_headline``
        so each transition is scored against P(its own target state), never
        a class-agnostic max (review F1).
    (e) ``y_true`` for the model-metrics artifacts is joined by REINDEXING
        the smoothed reference states onto ``per_step_metrics["dates"]``
        (review F2) — asserted to align length-for-length before proceeding.
    (f) per-leg KPIs (``evaluation/kpis.py``) + the smoothed-vs-filtered gap
        (``honesty/gap_lag.py::compute_gap``, using a hindsight-oracle
        performance number distinct from the actual walk-forward run).
    (g) ``report_model_metrics`` on the date-joined ``per_step_metrics``.
    (h) assemble + write the report + artifacts.

    Args:
        monthly_features: causal monthly features (Phase 1 checkpoint
            shape) — may physically extend past the holdout cutoff.
        monthly_raw: raw monthly ingest DataFrame (Phase 1 checkpoint
            shape) — the source for ``build_core_research_series``.
        cfg: platform config (``load_platform_config()`` output).
        registry_path: overrides the default trial registry ledger path.
        output_dir: overrides the default artifact directory (for tests).

    Returns:
        dict with keys ``report_path``, ``sojourn_lag``, ``strategy_kpis``,
        ``ablation_kpis``, ``baseline_kpis``, ``gap``, ``model_metrics_paths``,
        ``equity_curve``, ``ablation_curve``, ``per_step_metrics``,
        ``full_sample_states``.
    """
    backtest_cfg = cfg.get("backtest", {})
    allocation_cfg = cfg.get("allocation", {})
    labeling_cfg = cfg.get("labeling", {})
    cost_bps = backtest_cfg.get("cost_bps", 10)
    baseline_cost_bps = cost_bps if backtest_cfg.get("apply_cost_to_baselines", True) else 0.0
    rebalance = backtest_cfg.get("sixty_forty_rebalance", "monthly")
    crisis_windows = backtest_cfg.get("crisis_windows", [])
    act_threshold = allocation_cfg.get("hysteresis", {}).get("act_threshold", 0.70)

    splice_cfg = cfg["splice"]
    research = build_core_research_series(monthly_raw, cfg)
    returns = compute_monthly_returns(research)

    equity_name = splice_cfg["equities"]["research_name"]
    bond_name = splice_cfg["long_duration"]["research_name"]
    cash_name = splice_cfg["cash"]["research_name"]

    equity_ret = returns[equity_name]
    bond_ret = returns[bond_name]
    cash_ret = returns[cash_name]
    equity_level = research[equity_name]

    # (a) Investable asset universe for run_backtest — the risk classes only
    # (excludes "cash": cash is never tilted into as a position, it is the
    # vol-target residual that earns cash_ret via cash_returns, review F4).
    # An optional research class (e.g. gold when macrotrends is blocked) may be
    # absent from `returns` — skip it here so the backtest runs on the assets
    # that DO have data instead of KeyError-ing on the missing series.
    asset_returns = pd.DataFrame(
        {
            params["tradable"]: returns[params["research_name"]]
            for name, params in splice_cfg.items()
            if name != "cash" and params["research_name"] in returns.columns
        }
    )
    _excluded = [
        params["research_name"]
        for name, params in splice_cfg.items()
        if name != "cash" and params["research_name"] not in returns.columns
    ]
    if _excluded:
        log.warning("Backtest asset universe EXCLUDES unavailable research classes: %s", _excluded)

    equity_curve, per_step_metrics = run_backtest(
        monthly_features, asset_returns, cfg, cash_returns=cash_ret, use_regime_tilt=True, registry_path=registry_path,
    )

    # (b) No-regime ablation — same cash_ret series (review F4).
    ablation_curve, _ablation_metrics = no_regime_ablation(
        monthly_features, asset_returns, cfg, cash_returns=cash_ret, registry_path=registry_path,
    )

    # (c) Three price baselines — the report layer holdout-bounds them
    # (baselines.py does not enforce the cutoff internally).
    dev_equity_ret, _ = split_by_holdout_boundary(equity_ret, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_bond_ret, _ = split_by_holdout_boundary(bond_ret, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_cash_ret, _ = split_by_holdout_boundary(cash_ret, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_equity_level, _ = split_by_holdout_boundary(equity_level, cutoff=DEFAULT_HOLDOUT_CUTOFF)

    spy_ret = spy_buy_hold(dev_equity_ret)
    sixty_forty_ret = sixty_forty(dev_equity_ret, dev_bond_ret, rebalance=rebalance, cost_bps=baseline_cost_bps)
    faber_ret = faber_sma(dev_equity_level, dev_cash_ret, cost_bps=baseline_cost_bps)

    # (d) The SMOOTHED reference labeling — ONE full-sample fit, distinct
    # from the walk-forward per_step_metrics (Pitfall 1).
    dev_features, _ = split_by_holdout_boundary(monthly_features, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    lean_cols = sorted(lean_feature_set(cfg) & set(dev_features.columns))
    X_df = dev_features[lean_cols].dropna()
    X = standardize_features(X_df)
    fit = fit_jump_model(
        X,
        K=labeling_cfg.get("K", 5),
        lam=labeling_cfg.get("lambda", 52.0),
        n_restarts=labeling_cfg.get("n_restarts", 10),
    )
    smoothed_states_arr, _centroids = canonicalize_states(fit["states"], fit["centroids"], lean_cols)
    full_sample_states = pd.Series(smoothed_states_arr, index=X_df.index, name="state")

    filtered_probs_matrix = build_filtered_probs_matrix(per_step_metrics)
    headline = compute_sojourn_lag_headline(full_sample_states, filtered_probs_matrix, act_threshold=act_threshold)

    # (e) Join y_true by REINDEXING the smoothed reference onto the
    # walk-forward's own decision dates (review F2) — never loop-sourced.
    dates_index = pd.DatetimeIndex(per_step_metrics["dates"])
    y_true_series = full_sample_states.reindex(dates_index)
    if y_true_series.isna().any():
        missing = list(dates_index[y_true_series.isna()])
        raise ValueError(
            f"y_true date-join failed — no smoothed reference state for decision "
            f"dates: {missing} (review F2: y_true must be joined by date, never "
            "loop-sourced)."
        )
    per_step_metrics = dict(per_step_metrics)
    per_step_metrics["y_true"] = [int(v) for v in y_true_series.tolist()]
    if not (len(per_step_metrics["y_true"]) == len(per_step_metrics["dates"]) == len(per_step_metrics["proba"])):
        raise ValueError("y_true/dates/proba length mismatch after date-join (review F2).")

    # (f) Per-leg KPIs + the smoothed-vs-filtered gap.
    strategy_returns = equity_curve["return"] if not equity_curve.empty else pd.Series(dtype=float)
    ablation_returns = ablation_curve["return"] if not ablation_curve.empty else pd.Series(dtype=float)

    strategy_dd = max_drawdown_and_duration(strategy_returns) if len(strategy_returns) else {
        "max_drawdown": 0.0, "duration_months": 0,
    }
    strategy_kpis = {
        "terminal_log_wealth": terminal_log_wealth(strategy_returns),
        "max_drawdown": strategy_dd["max_drawdown"],
        "duration_months": strategy_dd["duration_months"],
        "cvar": cvar(strategy_returns) if len(strategy_returns) else float("nan"),
        "turnover": float(equity_curve["turnover"].mean()) if not equity_curve.empty else float("nan"),
        "crisis_capture": (
            crisis_capture_ratio(strategy_returns, spy_ret, crisis_windows)
            if crisis_windows and len(strategy_returns)
            else {}
        ),
    }
    ablation_kpis = _leg_kpis(ablation_returns) if len(ablation_returns) else {
        "terminal_log_wealth": 0.0, "max_drawdown": 0.0, "duration_months": 0,
    }
    baseline_kpis = {
        "spy_buy_hold": _leg_kpis(spy_ret),
        "sixty_forty": _leg_kpis(sixty_forty_ret),
        "faber_sma": _leg_kpis(faber_ret),
    }

    smoothed_perf = _smoothed_hindsight_perf(
        full_sample_states, asset_returns, cash_ret, list(per_step_metrics["dates"]), allocation_cfg,
    )
    filtered_perf = strategy_kpis["terminal_log_wealth"]
    gap = compute_gap(smoothed_perf, filtered_perf)

    # (g) Model-metrics artifacts (Brier/calibration/confusion).
    model_metrics_paths = report_model_metrics(per_step_metrics, output_dir=output_dir)

    # (h) Assemble + write the report + artifacts.
    markdown = assemble_backtest_report(
        sojourn_lag=headline,
        strategy_kpis=strategy_kpis,
        ablation_kpis=ablation_kpis,
        baseline_kpis=baseline_kpis,
        gap=gap,
        excluded_assets=_excluded,
    )
    kpi_table = _build_kpi_table(strategy_kpis, ablation_kpis, baseline_kpis)
    report_path = write_backtest_report(
        markdown,
        {
            "equity_curve_strategy": equity_curve,
            "equity_curve_ablation": ablation_curve,
            "kpi_table": kpi_table,
        },
        output_dir=output_dir,
    )

    return {
        "report_path": report_path,
        "sojourn_lag": headline,
        "strategy_kpis": strategy_kpis,
        "ablation_kpis": ablation_kpis,
        "baseline_kpis": baseline_kpis,
        "gap": gap,
        "model_metrics_paths": model_metrics_paths,
        "equity_curve": equity_curve,
        "ablation_curve": ablation_curve,
        "per_step_metrics": per_step_metrics,
        "full_sample_states": full_sample_states,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: load platform config + checkpoints, run the full
    EVAL-01..04 evaluation, write the report."""
    parser = argparse.ArgumentParser(
        description="Assemble the honest backtest evaluation report (EVAL-01..04)"
    )
    parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    cfg = load_platform_config()
    cm = get_platform_checkpoint_manager()
    monthly_features = cm.load("monthly_features")
    monthly_raw = cm.load("monthly_raw")

    result = run_full_backtest_evaluation(monthly_features, monthly_raw, cfg)
    print(f"Backtest report written: {result['report_path']}")  # noqa: T201 — first-class CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
