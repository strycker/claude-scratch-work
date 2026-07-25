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

import logging
from pathlib import Path

import pandas as pd

from trading_crab_lib import OUTPUT_DIR

log = logging.getLogger(__name__)


# ── assemble_backtest_report ──────────────────────────────────────────────────


def assemble_backtest_report(
    *,
    sojourn_lag: dict,
    strategy_kpis: dict,
    ablation_kpis: dict,
    baseline_kpis: dict,
    gap: float,
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency. Mirrors
    # the unit test's precomputed-inputs convention.
    import tempfile

    _demo_sojourn_lag = {"median_sojourn": 18.0, "median_lag": 2.0, "ratio": 9.0}
    _demo_strategy_kpis = {
        "terminal_log_wealth": 0.55,
        "max_drawdown": -0.12,
        "duration_months": 8,
        "cvar": -0.08,
        "turnover": 0.15,
        "crisis_capture": {"2008-09_gfc": 0.6},
    }
    _demo_ablation_kpis = {"terminal_log_wealth": 0.40, "max_drawdown": -0.18}
    _demo_baseline_kpis = {
        "spy_buy_hold": {"terminal_log_wealth": 0.45, "max_drawdown": -0.30},
        "sixty_forty": {"terminal_log_wealth": 0.35, "max_drawdown": -0.15},
        "faber_sma": {"terminal_log_wealth": 0.42, "max_drawdown": -0.10},
    }

    _markdown = assemble_backtest_report(
        sojourn_lag=_demo_sojourn_lag,
        strategy_kpis=_demo_strategy_kpis,
        ablation_kpis=_demo_ablation_kpis,
        baseline_kpis=_demo_baseline_kpis,
        gap=0.03,
    )

    with tempfile.TemporaryDirectory() as _tmp:
        _path = write_backtest_report(_markdown, {}, output_dir=Path(_tmp))
        print(f"Self-check report written to: {_path}")  # noqa: T201
        print(_markdown)  # noqa: T201 — first-class self-check output
