"""
Strategy KPIs — terminal log wealth, max drawdown + duration, CVaR(5%), and
in-sample crisis-window capture ratios (design §8.9, EVAL-04).

- ``terminal_log_wealth(returns)``: sum of log(1+r) — the honest,
  compounding-consistent measure of total strategy growth.
- ``max_drawdown_and_duration(returns)``: peak-to-trough drawdown magnitude
  plus the longest consecutive run of periods spent below the prior running
  peak (mirrors the inline ``(cum / cum.cummax() - 1).min()`` one-liner in
  ``assets/returns.py``, extended with a duration measure).
- ``cvar(returns, alpha=0.05)``: mean of the worst ``floor(alpha * n)``
  returns — the down-tail convention (at least one observation is always
  averaged, even when ``alpha * n < 1``).
- ``crisis_capture_ratio(strategy_returns, benchmark_returns, windows,
  cutoff=DEFAULT_HOLDOUT_CUTOFF)``: strategy cumulative return divided by
  benchmark cumulative return over each configured in-sample crisis window
  (down-capture convention, A6 — see 05-RESEARCH.md). Any window whose end
  date exceeds ``cutoff`` raises ``ValueError`` — crisis windows are
  config-driven (Claude's discretion) and a typo listing a 2021+ window is
  exactly the kind of silent holdout contamination the honesty framework
  exists to prevent (RESEARCH.md Pitfall 4).

Usage::

    from trading_crab_lib.platform.evaluation.kpis import (
        crisis_capture_ratio, cvar, max_drawdown_and_duration, terminal_log_wealth,
    )

    wealth = terminal_log_wealth(equity_curve["return"])
    dd = max_drawdown_and_duration(equity_curve["return"])
    tail_risk = cvar(equity_curve["return"], alpha=0.05)
    captures = crisis_capture_ratio(strategy_returns, benchmark_returns, cfg["backtest"]["crisis_windows"])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF


def terminal_log_wealth(returns: pd.Series) -> float:
    """Sum of log(1 + r) over the full return series — additive, compounding-
    consistent measure of total growth (design §8.9)."""
    return float(np.log1p(returns).sum())


def max_drawdown_and_duration(returns: pd.Series) -> dict:
    """Peak-to-trough drawdown magnitude plus the longest underwater run.

    Mirrors the ``(cum / cum.cummax() - 1).min()`` one-liner from
    ``assets/returns.py``. ``duration_months`` is the longest run of
    CONSECUTIVE periods strictly below the prior running peak (drawdown <
    0) — not necessarily the run containing the single worst trough, but
    for a series with one drawdown episode the two coincide. A period
    exactly at a new high (drawdown == 0) is never counted as underwater.

    Args:
        returns: periodic (monthly) simple return Series.

    Returns:
        dict with keys ``max_drawdown`` (negative float magnitude, or 0.0 if
        the series never dips below its running peak) and
        ``duration_months`` (int, longest underwater run length).
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_drawdown = float(drawdown.min())

    underwater = drawdown < 0
    duration = 0
    current = 0
    for flag in underwater:
        if flag:
            current += 1
            duration = max(duration, current)
        else:
            current = 0

    return {"max_drawdown": max_drawdown, "duration_months": duration}


def cvar(returns: pd.Series, *, alpha: float = 0.05) -> float:
    """Conditional Value at Risk (down-tail convention).

    CVaR is the mean of the worst ``floor(alpha * n)`` returns. The floor is
    clamped to at least 1 — a sample too small for ``alpha * n >= 1`` still
    reports the single worst observed return rather than an empty-mean NaN.

    Args:
        returns: periodic (monthly) simple return Series.
        alpha: tail fraction (design default: 0.05, i.e. worst 5%).

    Returns:
        float — mean of the worst ``k = max(1, floor(alpha * n))`` returns.
    """
    values = np.sort(np.asarray(returns, dtype=float))
    n = len(values)
    k = max(1, int(np.floor(alpha * n)))
    return float(np.mean(values[:k]))


def crisis_capture_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    windows: list[dict],
    *,
    cutoff: str = DEFAULT_HOLDOUT_CUTOFF,
) -> dict[str, float]:
    """Strategy-cumulative / benchmark-cumulative return over each in-sample
    crisis window (down-capture convention, A6 — 05-RESEARCH.md).

    Every window's ``end`` date MUST be on or before ``cutoff`` — any window
    end exceeding it raises ``ValueError`` rather than silently including
    holdout-era data (RESEARCH.md Pitfall 4). Windows are config-driven
    dicts with ``name``/``start``/``end`` keys (matches
    ``config/platform_settings.yaml``'s ``backtest.crisis_windows`` shape).

    Args:
        strategy_returns: periodic (monthly) strategy simple return Series.
        benchmark_returns: periodic (monthly) benchmark simple return Series
            (design §8.9/A6: typically SPY / ``equities_tr``).
        windows: list of dicts, each with ``name``, ``start``, ``end``.
        cutoff: the holdout boundary (default: ``DEFAULT_HOLDOUT_CUTOFF``,
            2020-12-31).

    Returns:
        dict[str, float] mapping each window's ``name`` to its capture
        ratio (strategy cumulative return / benchmark cumulative return;
        both are typically negative during a crisis window, so a ratio < 1
        means the strategy lost less than the benchmark).

    Raises:
        ValueError: if any window's ``end`` date is after ``cutoff``.
    """
    cutoff_ts = pd.Timestamp(cutoff)
    results: dict[str, float] = {}
    for window in windows:
        name = window["name"]
        start = pd.Timestamp(window["start"])
        end = pd.Timestamp(window["end"])
        if end > cutoff_ts:
            raise ValueError(
                f"crisis window '{name}' ends {end.date()}, after the holdout cutoff "
                f"{cutoff_ts.date()} — 2021+ dates must never be used for in-sample "
                "crisis capture ratios (HON-01 / Pitfall 4)."
            )
        strat_slice = strategy_returns.loc[start:end]
        bench_slice = benchmark_returns.loc[start:end]
        strat_cum = float((1 + strat_slice).prod() - 1)
        bench_cum = float((1 + bench_slice).prod() - 1)
        results[name] = strat_cum / bench_cum

    return results


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint dependency.
    rng = np.random.default_rng(42)
    idx = pd.date_range("1972-01-31", periods=120, freq="ME")
    demo_returns = pd.Series(rng.normal(0.006, 0.03, 120), index=idx)
    demo_benchmark = pd.Series(rng.normal(0.005, 0.04, 120), index=idx)

    wealth = terminal_log_wealth(demo_returns)
    dd = max_drawdown_and_duration(demo_returns)
    tail = cvar(demo_returns, alpha=0.05)
    windows = [{"name": "demo_window", "start": "1973-01-01", "end": "1974-12-31"}]
    captures = crisis_capture_ratio(demo_returns, demo_benchmark, windows)

    print(  # noqa: T201 — first-class self-check output
        "Strategy KPIs self-check (design §8.9)\n"
        f"  terminal log wealth:  {wealth:.4f}\n"
        f"  max drawdown:         {dd['max_drawdown']:.4f} ({dd['duration_months']} months underwater)\n"
        f"  CVaR(5%):             {tail:.4f}\n"
        f"  crisis capture:       {captures}"
    )
