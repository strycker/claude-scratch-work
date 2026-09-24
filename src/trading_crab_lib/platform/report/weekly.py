"""
Weekly report assembly + trades-implied + opt-in email delivery (L4-02, design
§7, 04-CONTEXT.md D-02).

``assemble_weekly_report()`` builds the markdown Glenn reads before trading in
Fidelity: current regime distribution + trajectory (nowcaster probabilities +
the empirical transition matrix), per-asset signals (the returns-by-regime
table, flagging D11 short-history cells as low-confidence), and
target-vs-current with explicit per-account trades implied.

``write_weekly_report()`` ALWAYS writes the markdown to
``outputs/reports/platform/weekly_report.md`` — a distinct path from the
incumbent's ``outputs/reports/weekly_report.md`` (D-02, avoids collision).
Email delivery is opt-in behind ``--send-email`` and reuses the incumbent
``email.py`` machinery READ-ONLY (``build_weekly_email_body`` /
``load_email_config`` / ``send_weekly_email``) — never forked, never
modified, and never invoked on the default path.

``main()`` orchestrates the full allocation cycle before assembly: load the
previous hysteresis state, update it with the current nowcaster
probabilities, compute target weights via ``vol_targeted_tilt``, pass them
through the no-trade band, and persist the new state (load-before-save order,
Pitfall 3) — all isolated in
``_build_report_inputs()`` so it can be swapped out in tests without real
Phase 1/3 checkpoint data on disk.

**The Bayes filter at serve (plan 08-08).** The nowcaster's posterior is filtered
into a belief (``prediction/regime_filter.py``) before the hysteresis and the tilt
see it — the same recursion ``backtest/driver.py`` and ``joint_driver.py`` run in
their loops, so the allocator consumes the same kind of object at train and serve.
The belief persists across runs in the ``regime_belief`` checkpoint, loaded BEFORE
it is saved (the Pitfall 3 ordering the hysteresis state uses), in the same block
as the hysteresis and immediately before it. The cold start — no checkpoint, or a
persisted null — is ``unconditional_belief`` on the ``regime_labels`` the nowcaster
was trained on: the SAME function object the drivers use, because a cold start that
differed between train and serve would itself be train/serve skew. ``A`` is
``transition_matrix_for`` on those labels. One filter step is one MONTH: the belief
carries the as-of date of the feature row it absorbed, a weekly re-run inside the
same month reuses it unchanged (re-filtering would count the same month's evidence
again), and a gap of several months advances by ``predict_only_step`` for each
unobserved month. Known approximation (plan 08-06): the class prior is the whole
label series' distribution, while the nowcaster trains on the D-01 embargoed subset.

**The no-trade band and the active regime at serve (plan 08-09, ``08-A7.md``).** The
tilt's target passes through ``allocation/hysteresis.py::execute_rebalance`` — the 5pp
no-trade band, NOT SWEPT, the same function both backtest drivers call — so the weights
the report shows are the EXECUTED book, not the raw target. ``held`` is the last
executed book, persisted in the ``executed_weights`` checkpoint (load before save). One
band step is one MONTH, like the belief: a same-month re-run re-bands this month's
target against the SAME held book the first run used (the previous month's execution),
so repeated weekly runs neither compound the band nor drift. No checkpoint yet trades in
full. ``assemble_weekly_report`` now receives the hysteresis output as
``active_regime`` instead of recomputing ``probs.idxmax()`` — it no longer narrates a
state machine whose output it does not show. ``active_regime`` gates no weight: A7
closed by rewording, and the report says so beside the value.

Usage::

    python3 -m trading_crab_lib.platform.report.weekly [--send-email]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib.email import build_weekly_email_body, load_email_config, send_weekly_email
from trading_crab_lib.platform.allocation.hysteresis import (
    execute_rebalance,
    hysteresis_thresholds,
    load_active_regime,
    no_trade_band_from_config,
    save_active_regime,
    update_active_regime,
)
from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.honesty.holdout import load_full_span
from trading_crab_lib.platform.prediction.regime_filter import (
    filter_step,
    predict_only_step,
    transition_matrix_for,
    unconditional_belief,
)
from trading_crab_lib.platform.prediction.transition_matrix import empirical_transition_matrix
from trading_crab_lib.platform.report.holdings import load_account_weights

log = logging.getLogger(__name__)

# report.trade_threshold_pct / report.min_obs_flag defaults (config/platform_settings.yaml
# `report:` section) — used only when cfg omits the key (defensive .get() pattern).
_DEFAULT_TRADE_THRESHOLD_PCT = 0.03
_DEFAULT_MIN_OBS_FLAG = 6

_BELIEF_CHECKPOINT = "regime_belief"
_EXECUTED_CHECKPOINT = "executed_weights"


def load_regime_belief(cm=None) -> pd.Series | None:
    """Load the previous run's filtered regime belief. Cold start (no checkpoint yet,
    or a persisted null) returns None — the caller then uses ``unconditional_belief``.

    The returned Series is indexed by integer state and its ``name`` is the as-of
    month (``pd.Timestamp``) of the feature row it absorbed, or None if unrecorded.
    """
    cm = cm or get_platform_checkpoint_manager()
    try:
        frame = cm.load(_BELIEF_CHECKPOINT)
    except FileNotFoundError:
        return None
    if frame.empty or frame["belief"].isna().all() or frame["state"].isna().all():
        return None
    belief = pd.Series(frame["belief"].to_numpy(dtype=float), index=[int(v) for v in frame["state"]])
    as_of = frame["as_of"].iloc[0] if "as_of" in frame.columns else None
    belief.name = None if as_of is None or pd.isna(as_of) else pd.Timestamp(as_of)
    return belief


def save_regime_belief(belief: pd.Series | None, cm=None, *, as_of: pd.Timestamp | None = None) -> None:
    """Persist the filtered belief (or None) with the as-of month it absorbed."""
    cm = cm or get_platform_checkpoint_manager()
    if belief is None:
        frame = pd.DataFrame([{"state": None, "belief": None, "as_of": as_of}])
    else:
        frame = pd.DataFrame({
            "state": [int(v) for v in belief.index],
            "belief": belief.to_numpy(dtype=float),
            "as_of": [as_of] * len(belief),
        })
    cm.save(frame, _BELIEF_CHECKPOINT)


def _months_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _weights_rows(weights: pd.Series | None, basis: str, as_of: pd.Timestamp) -> list[dict]:
    """Rows for one book. An empty book (all cash) is one null-asset marker row, so it
    stays distinguishable from "no book" (no rows at all)."""
    if weights is None:
        return []
    if len(weights) == 0:
        return [{"asset": None, "weight": None, "basis": basis, "as_of": as_of}]
    return [
        {"asset": str(a), "weight": float(w), "basis": basis, "as_of": as_of} for a, w in weights.items()
    ]


def load_held_weights(cm=None, *, as_of: pd.Timestamp) -> pd.Series | None:
    """The ``held`` book the no-trade band compares this month's target against.

    - no checkpoint: None — nothing executed yet, the first execution trades in full;
    - checkpoint from an EARLIER month: that month's executed book;
    - checkpoint from THIS month (a weekly re-run): the held book that run used, so the
      band steps once per month and a re-run reproduces rather than compounds;
    - checkpoint dated AFTER ``as_of``: WARNING and None, mirroring the belief's rule.
    """
    cm = cm or get_platform_checkpoint_manager()
    try:
        frame = cm.load(_EXECUTED_CHECKPOINT)
    except FileNotFoundError:
        return None
    if frame.empty:
        return None
    saved_as_of = pd.Timestamp(frame["as_of"].iloc[0])
    gap = _months_between(saved_as_of, as_of)
    if gap < 0:
        log.warning("executed_weights checkpoint is dated %s, after this run's %s — no held book", saved_as_of, as_of)
        return None
    rows = frame[frame["basis"] == ("executed" if gap > 0 else "held_in")]
    if rows.empty:
        return None
    rows = rows[rows["asset"].notna()]
    return pd.Series(rows["weight"].to_numpy(dtype=float), index=[str(a) for a in rows["asset"]], dtype=float)


def save_executed_weights(
    executed: pd.Series, held_in: pd.Series | None, cm=None, *, as_of: pd.Timestamp
) -> None:
    """Persist this month's executed book and the held book it was banded against."""
    cm = cm or get_platform_checkpoint_manager()
    rows = _weights_rows(executed, "executed", as_of) + _weights_rows(held_in, "held_in", as_of)
    cm.save(pd.DataFrame(rows, columns=["asset", "weight", "basis", "as_of"]), _EXECUTED_CHECKPOINT)


def advance_regime_belief(
    prev_belief: pd.Series | None,
    regime_labels: pd.Series,
    regime_probs: pd.Series,
    *,
    state_index: list[int],
    as_of: pd.Timestamp,
) -> pd.Series:
    """This run's belief from the loaded one: cold start, same-month reuse, or filter.

    - no previous belief (or its states differ from ``state_index``): cold start from
      ``unconditional_belief(regime_labels)`` — the drivers' rule — then one filter step;
    - previous belief already absorbed ``as_of``: returned unchanged (no double count);
    - otherwise ``predict_only_step`` once per unobserved month in between, then one
      ``filter_step`` with this month's posterior.
    """
    prior = unconditional_belief(regime_labels, state_index=state_index)
    transition = transition_matrix_for(regime_labels, state_index=state_index)
    start = prior
    if prev_belief is not None and sorted(int(v) for v in prev_belief.index) == sorted(state_index):
        prev_as_of = prev_belief.name
        if prev_as_of is not None and _months_between(pd.Timestamp(prev_as_of), as_of) == 0:
            return prev_belief.rename(None)
        if prev_as_of is not None and _months_between(pd.Timestamp(prev_as_of), as_of) < 0:
            log.warning(
                "regime_belief checkpoint is dated %s, after this run's %s — cold-starting the filter",
                prev_as_of, as_of,
            )
        else:
            start = prev_belief.rename(None)
            gap = 1 if prev_as_of is None else _months_between(pd.Timestamp(prev_as_of), as_of)
            for _ in range(gap - 1):
                start = predict_only_step(start, transition)
    elif prev_belief is not None:
        log.warning(
            "regime_belief checkpoint covers states %s, not %s — cold-starting the filter",
            list(prev_belief.index), state_index,
        )
    return filter_step(start, transition, regime_probs, prior)


def trades_implied(
    target_weights: pd.Series,
    current_weights: pd.Series,
    *,
    threshold: float = _DEFAULT_TRADE_THRESHOLD_PCT,
) -> pd.DataFrame:
    """One row per asset (union of target and current index): current_pct,
    target_pct, delta_pct, signal in {BUY, SELL, HOLD}.

    Flat no-trade band (design §21's full band-width math is deferred to v2,
    L4-V2-01): ``|delta| < threshold`` -> HOLD, ``delta >= +threshold`` ->
    BUY, ``delta <= -threshold`` -> SELL. New implementation — inspiration
    from incumbent ``reporting.generate_recommendation`` but NOT imported
    (D-01).
    """
    all_assets = sorted(set(target_weights.index) | set(current_weights.index))
    rows = []
    for asset in all_assets:
        current = float(current_weights.get(asset, 0.0))
        target = float(target_weights.get(asset, 0.0))
        delta = target - current
        if delta >= threshold:
            signal = "BUY"
        elif delta <= -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        rows.append(
            {"asset": asset, "current_pct": current, "target_pct": target, "delta_pct": delta, "signal": signal}
        )
    return pd.DataFrame(rows, columns=["asset", "current_pct", "target_pct", "delta_pct", "signal"])


def assemble_weekly_report(
    *,
    regime_probs: pd.Series | dict,
    transition_matrix: pd.DataFrame,
    returns_by_regime: pd.DataFrame,
    target_weights: pd.Series,
    accounts: list[str],
    active_regime: int | None,
    regime_belief: pd.Series | dict | None = None,
    cash: float | None = None,
    no_trade_band: float | None = None,
    accounts_dir: Path | None = None,
    min_obs_flag: int = _DEFAULT_MIN_OBS_FLAG,
    trade_threshold_pct: float = _DEFAULT_TRADE_THRESHOLD_PCT,
) -> str:
    """Assemble the weekly report markdown (design §7 output list) from
    pre-computed inputs — a pure function, no I/O beyond the per-account
    holdings YAML reads (``load_account_weights``).

    Sections: (1) current regime distribution, then the hysteresis state
    machine's OUTPUT (``active_regime``) with the cold-start rule that produced it,
    (2) trajectory from the empirical transition matrix out of that regime, (3)
    per-asset signals from ``returns_by_regime``, flagging cells with ``n_obs <
    min_obs_flag`` as low-confidence (D11), (4) target-vs-current + trades implied
    PER account, each asset annotated with a one-line regime rationale.

    ``active_regime`` is ``update_active_regime``'s output, passed in — plan 08-09
    removed the internal ``probs.idxmax()`` recomputation. ``None`` is the neutral
    posture. ``target_weights`` should be the EXECUTED book (after the no-trade band);
    ``no_trade_band`` names the band so the report says so.
    """
    probs = pd.Series(regime_probs, dtype=float)

    lines: list[str] = ["# Trading-Crab Platform Weekly Report", ""]

    # ── 1. Current regime distribution ────────────────────────────────────
    lines.append("## Current Regime Distribution")
    lines.append("")
    if probs.empty:
        lines.append("(no regime probabilities available)")
    else:
        for regime_id, p in probs.sort_values(ascending=False).items():
            lines.append(f"- regime {regime_id}: {p:.1%}")
    lines.append("")
    if regime_belief is not None:
        belief = pd.Series(regime_belief, dtype=float)
        lines.append("## Filtered Regime Belief (what the allocation consumed)")
        lines.append("")
        for regime_id, p in belief.sort_values(ascending=False).items():
            lines.append(f"- regime {regime_id}: {p:.1%}")
        lines.append("")

    # The hysteresis OUTPUT, and — adjacent to it — the rule that produced it.
    lines.append("## Active Regime (hysteresis state machine output)")
    lines.append("")
    if active_regime is None:
        lines.append("- active regime: none (neutral posture)")
    else:
        lines.append(f"- active regime: regime {active_regime}")
    lines.append("")
    lines.append(
        "Hysteresis cold-start rule (A1): with no prior active regime, the "
        "platform acts immediately on the highest-probability regime if it "
        "already clears the act threshold; otherwise it stays neutral "
        "until some regime first crosses the act threshold. Once active, a regime "
        "is held until its own probability falls below the unwind threshold."
    )
    lines.append("")
    lines.append(
        "The active regime is a reported label: it selects the trajectory and "
        "per-asset rows below and gates no weight (audit item A7, 08-A7.md). "
        "The weights come from the filtered belief through the no-trade band."
    )
    lines.append("")

    # ── 2. Trajectory (empirical transition matrix) ───────────────────────
    lines.append("## Trajectory (Empirical Transition Matrix)")
    lines.append("")
    if active_regime is not None and active_regime in transition_matrix.index:
        row = transition_matrix.loc[active_regime].sort_values(ascending=False)
        lines.append(f"From regime {active_regime}, next-regime probabilities:")
        for to_regime, p in row.items():
            lines.append(f"- -> regime {to_regime}: {p:.1%}")
    else:
        lines.append("(no trajectory available for the current regime)")
    lines.append("")

    # ── 3. Per-asset signals (returns-by-regime), flagging D11 cells ──────
    lines.append("## Per-Asset Signals (Returns by Regime)")
    lines.append("")
    if returns_by_regime.empty:
        lines.append("(no returns-by-regime data available)")
    else:
        sub = returns_by_regime
        if active_regime is not None:
            sub = returns_by_regime[returns_by_regime["regime"] == active_regime]
        for row in sub.itertuples():
            flag = " [LOW-CONFIDENCE — short history, D11]" if row.n_obs < min_obs_flag else ""
            lines.append(
                f"- {row.asset}: mean={row.mean_monthly_return:.2%} "
                f"sharpe={row.sharpe_annualized:.2f} n_obs={row.n_obs}{flag}"
            )
    lines.append("")

    # ── 4. Target-vs-current + trades implied, per account ────────────────
    lines.append("## Target vs. Current — Trades Implied")
    lines.append("")
    if no_trade_band is not None:
        lines.append(
            f"Targets below are the EXECUTED book after the {no_trade_band:.1%} no-trade band "
            "(design §5.3 bounded turnover, 08-A7.md): an asset whose target moved by no more "
            "than the band from its last executed weight keeps that weight."
        )
        lines.append("")
    rationale = f"regime {active_regime}" if active_regime is not None else "neutral posture"
    for account in accounts:
        holdings = load_account_weights(account, accounts_dir=accounts_dir)
        current_weights = pd.Series(holdings["weights"], dtype=float)
        implied = trades_implied(target_weights, current_weights, threshold=trade_threshold_pct)
        lines.append(f"### Account: {account} (cash on file: {holdings['cash']:.1%})")
        lines.append("")
        if implied.empty:
            lines.append("(no target or current holdings)")
        for row in implied.itertuples():
            lines.append(
                f"- {row.asset}: {row.signal} current={row.current_pct:.1%} "
                f"target={row.target_pct:.1%} delta={row.delta_pct:+.1%} ({rationale})"
            )
        lines.append("")

    if cash is not None:
        lines.append(f"_Target allocation cash residual: {cash:.1%}_")
        lines.append("")

    return "\n".join(lines)


def write_weekly_report(markdown: str, *, output_dir: Path | None = None) -> Path:
    """ALWAYS write the markdown to
    ``(output_dir or OUTPUT_DIR / "reports" / "platform") / "weekly_report.md"``
    (mkdir parents) — D-02: this happens unconditionally, regardless of
    ``--send-email``."""
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "platform"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "weekly_report.md"
    path.write_text(markdown, encoding="utf-8")
    log.info("Weekly report written: %s", path)
    return path


def _build_report_inputs(cfg: dict, cm=None) -> dict:
    """The full allocation-cycle orchestration (load -> update -> tilt ->
    save, load-before-save order per Pitfall 3): load the previous
    hysteresis state, update it with the current nowcaster probabilities,
    compute target weights via ``vol_targeted_tilt``, and persist the new
    state.

    Isolated as its own function — not inlined in ``main()`` — so tests can
    monkeypatch it directly without needing real Phase 1/3 checkpoint data
    (nowcaster model, monthly_features, regime_labels, returns_by_regime,
    asset_returns) on disk.

    Returns:
        dict with keys ``regime_probs`` (raw posterior), ``regime_belief``
        (the filtered belief the allocation consumed), ``active_regime`` (the
        hysteresis output), ``transition_matrix``, ``returns_by_regime``,
        ``target_weights`` / ``cash`` (the EXECUTED book, after the no-trade band),
        ``pre_band_target_weights`` (the tilt's target) and ``no_trade_band``.
    """
    cm = cm or get_platform_checkpoint_manager()

    nowcaster = cm.load_model("nowcaster")
    # Live scoring is "looking", not "fitting", so it takes the explicit
    # full-span opt-in. The dev checkpoint stops at the 2020-12 holdout
    # boundary; loading it here would silently score December 2020 as "today",
    # every week, forever.
    monthly_features = load_full_span("monthly_features")
    regime_labels = cm.load("regime_labels")["state"]
    returns_by_regime = cm.load("returns_by_regime")
    asset_returns = cm.load("asset_returns")

    proba = nowcaster.predict_proba(monthly_features.iloc[[-1]])[0]
    regime_probs = pd.Series(proba, index=nowcaster.classes_)

    allocation_cfg = cfg.get("allocation", {})
    act_threshold, unwind_threshold = hysteresis_thresholds(cfg)
    no_trade_band = no_trade_band_from_config(cfg)

    # Bayes filter (plan 08-08): load BEFORE save, immediately before the hysteresis,
    # so the belief the hysteresis sees is this run's.
    state_index = list(range(int(cfg.get("labeling", {}).get("K", 5))))
    as_of = pd.Timestamp(monthly_features.index[-1])
    prev_belief = load_regime_belief(cm)  # load BEFORE save (Pitfall 3)
    regime_belief = advance_regime_belief(
        prev_belief, regime_labels, regime_probs, state_index=state_index, as_of=as_of
    )
    save_regime_belief(regime_belief, cm, as_of=as_of)  # save AFTER load

    prev_active = load_active_regime(cm)  # load BEFORE save (Pitfall 3)
    active_regime = update_active_regime(
        regime_belief,
        prev_active,
        act_threshold=act_threshold,
        unwind_threshold=unwind_threshold,
    )
    save_active_regime(active_regime, cm)  # save AFTER load

    tilt = vol_targeted_tilt(
        regime_belief,
        returns_by_regime,
        asset_returns,
        target_vol_annual=allocation_cfg.get("target_vol_annual", 0.10),
        halflife=allocation_cfg.get("ewma_halflife_months", 6),
        min_obs=allocation_cfg.get("portfolio_vol_min_obs", 12),
    )

    # The no-trade band (plan 08-09): the SAME function the drivers call. Held-book I/O
    # happens only when a band is configured — with none, the executed book is the target.
    held = load_held_weights(cm, as_of=as_of) if no_trade_band is not None else None  # load BEFORE save
    executed = execute_rebalance(tilt["weights"], tilt["cash"], held, band=no_trade_band)
    if no_trade_band is not None:
        save_executed_weights(executed["weights"], held, cm, as_of=as_of)  # save AFTER load

    return {
        "regime_probs": regime_probs,
        "regime_belief": regime_belief,
        "active_regime": active_regime,
        "transition_matrix": empirical_transition_matrix(regime_labels),
        "returns_by_regime": returns_by_regime,
        "target_weights": executed["weights"],
        "cash": executed["cash"],
        "pre_band_target_weights": tilt["weights"],
        "no_trade_band": no_trade_band,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: assemble + always write the markdown; email opt-in.

    ``main([])`` writes the markdown and does NOT call
    ``send_weekly_email``. ``main(["--send-email"])`` calls
    ``build_weekly_email_body`` / ``load_email_config`` /
    ``send_weekly_email`` exactly once each (D-02) — never on the default
    path.
    """
    parser = argparse.ArgumentParser(
        description="Assemble (and optionally email) the platform weekly report"
    )
    parser.add_argument(
        "--send-email", action="store_true", help="Send the report via SMTP (opt-in, D-02)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    cfg = load_platform_config()
    report_cfg = cfg.get("report", {})
    accounts = report_cfg.get("accounts", [])

    inputs = _build_report_inputs(cfg)
    markdown = assemble_weekly_report(
        regime_probs=inputs["regime_probs"],
        regime_belief=inputs.get("regime_belief"),
        active_regime=inputs["active_regime"],
        transition_matrix=inputs["transition_matrix"],
        returns_by_regime=inputs["returns_by_regime"],
        target_weights=inputs["target_weights"],
        accounts=accounts,
        cash=inputs["cash"],
        no_trade_band=inputs.get("no_trade_band"),
        min_obs_flag=report_cfg.get("min_obs_flag", _DEFAULT_MIN_OBS_FLAG),
        trade_threshold_pct=report_cfg.get("trade_threshold_pct", _DEFAULT_TRADE_THRESHOLD_PCT),
    )
    report_path = write_weekly_report(markdown)

    if args.send_email:
        subject, body = build_weekly_email_body(
            report_path.parent, subject_prefix="Trading-Crab Platform Weekly Report"
        )
        email_cfg = load_email_config()
        if email_cfg:
            send_weekly_email(email_cfg, subject, body)  # plot_paths=None in v1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
