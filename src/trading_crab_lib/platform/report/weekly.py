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
probabilities, compute target weights via ``vol_targeted_tilt``, and persist
the new state (load-before-save order, Pitfall 3) — all isolated in
``_build_report_inputs()`` so it can be swapped out in tests without real
Phase 1/3 checkpoint data on disk.

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
    load_active_regime,
    save_active_regime,
    update_active_regime,
)
from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.prediction.transition_matrix import empirical_transition_matrix
from trading_crab_lib.platform.report.holdings import load_account_weights

log = logging.getLogger(__name__)

# report.trade_threshold_pct / report.min_obs_flag defaults (config/platform_settings.yaml
# `report:` section) — used only when cfg omits the key (defensive .get() pattern).
_DEFAULT_TRADE_THRESHOLD_PCT = 0.03
_DEFAULT_MIN_OBS_FLAG = 6


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
    cash: float | None = None,
    accounts_dir: Path | None = None,
    min_obs_flag: int = _DEFAULT_MIN_OBS_FLAG,
    trade_threshold_pct: float = _DEFAULT_TRADE_THRESHOLD_PCT,
) -> str:
    """Assemble the weekly report markdown (design §7 output list) from
    pre-computed inputs — a pure function, no I/O beyond the per-account
    holdings YAML reads (``load_account_weights``).

    Sections: (1) current regime distribution, (2) trajectory from the
    empirical transition matrix, (3) per-asset signals from
    ``returns_by_regime``, flagging cells with ``n_obs < min_obs_flag`` as
    low-confidence (D11), (4) target-vs-current + trades implied PER
    account, each asset annotated with a one-line regime rationale.
    """
    probs = pd.Series(regime_probs, dtype=float)
    active_regime = probs.idxmax() if not probs.empty else None

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

    # ── 2. Trajectory (empirical transition matrix) ───────────────────────
    lines.append("## Trajectory (Empirical Transition Matrix)")
    lines.append("")
    lines.append(
        "Hysteresis cold-start rule (A1): with no prior active regime, the "
        "platform acts immediately on the highest-probability regime if it "
        "already clears the act threshold; otherwise it stays neutral "
        "(cash-heavy) until some regime first crosses the act threshold."
    )
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
        dict with keys ``regime_probs``, ``transition_matrix``,
        ``returns_by_regime``, ``target_weights``, ``cash``.
    """
    cm = cm or get_platform_checkpoint_manager()

    nowcaster = cm.load_model("nowcaster")
    monthly_features = cm.load("monthly_features")
    regime_labels = cm.load("regime_labels")["state"]
    returns_by_regime = cm.load("returns_by_regime")
    asset_returns = cm.load("asset_returns")

    proba = nowcaster.predict_proba(monthly_features.iloc[[-1]])[0]
    regime_probs = pd.Series(proba, index=nowcaster.classes_)

    allocation_cfg = cfg.get("allocation", {})
    hysteresis_cfg = allocation_cfg.get("hysteresis", {})

    prev_active = load_active_regime(cm)  # load BEFORE save (Pitfall 3)
    active_regime = update_active_regime(
        regime_probs,
        prev_active,
        act_threshold=hysteresis_cfg.get("act_threshold", 0.70),
        unwind_threshold=hysteresis_cfg.get("unwind_threshold", 0.40),
    )
    save_active_regime(active_regime, cm)  # save AFTER load

    tilt = vol_targeted_tilt(
        regime_probs,
        returns_by_regime,
        asset_returns,
        target_vol_annual=allocation_cfg.get("target_vol_annual", 0.10),
        halflife=allocation_cfg.get("ewma_halflife_months", 6),
        min_obs=allocation_cfg.get("portfolio_vol_min_obs", 12),
    )

    return {
        "regime_probs": regime_probs,
        "transition_matrix": empirical_transition_matrix(regime_labels),
        "returns_by_regime": returns_by_regime,
        "target_weights": tilt["weights"],
        "cash": tilt["cash"],
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
        transition_matrix=inputs["transition_matrix"],
        returns_by_regime=inputs["returns_by_regime"],
        target_weights=inputs["target_weights"],
        accounts=accounts,
        cash=inputs["cash"],
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
