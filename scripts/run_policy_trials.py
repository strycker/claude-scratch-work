#!/usr/bin/env python
"""run_policy_trials.py — wave-1 policy trials (07-CONTEXT.md D-03/D-04,
07-VALIDATION.md criteria 2-4).

Runs ONE of the two wave-1 evaluations to completion on the real dev
checkpoints and prints a JSON measurement record to stdout:

``--variant frozen``
    The frozen ten-column policy (07-CONTEXT.md D-02-A) — the decision.
    Regenerates the PUBLISHED artifacts under ``outputs/reports/platform/``.

``--variant impute``
    The 13-feature + back-fill-imputation alternative (D-03) — the recorded
    REJECTION, run once so the ADR's rejection is evidence-backed rather
    than argued. Writes to a SEPARATE ``output_dir``
    (``outputs/reports/platform/trials/impute-13col/``) and NEVER overwrites
    the published artifacts.

Variant B's imputation is **non-causal by construction**: for each of the
four historically late-starting lean columns (``curve_10y2y``, ``gold``,
``oil``, ``fred_vix``), the pre-start gap is back-filled with that column's
own first observed value. There is no past to impute from before a series
begins, so any pre-start fill fabricates data the model could not have had
at the time — this is the substantive reason D-03 rejects the alternative
on the merits (fabricating pre-1990 VIX levels would put an invented stress
feature inside a crisis classifier). **Note:** after the D-02-A checkpoint
recompute (plan 07-02), ``oil`` already has full 1962+ coverage in the dev
checkpoint and needs no imputation in practice — it stays in the list below
so the historical four-column exclusion set is complete; the back-fill is a
documented no-op for ``oil`` specifically.

Cost is not a constraint: a full 588-step L1+L2 walk-forward measures ~2
minutes. Both variants run for real, on the real checkpoints, never as a
patched artifact (07-03-PLAN.md: "Do not patch persisted artifacts in
place — that is the exact staleness class that produced D-02-A.").

**Hard vs. advisory checks (D-07).** This script asserts (raises) only on
values that are ARITHMETICALLY IMPOSSIBLE or structurally wrong — the class
of bug ``UAT-AUDIT-2026-09-09`` found undetected (terminal log wealth
111.06, e^111 ~ 10^48). It never hard-fails on the four ``[ASSUMED]``
domain bands from ``07-VALIDATION.md`` (``wealth_delta``, ``dd_delta``,
``n_transitions``, ``pct_disagree``) — a breach of one of those is recorded
as a flagged verdict in the returned dict, never a crash, and is NEVER used
to select or re-run a policy (D-04).

Usage::

    python scripts/run_policy_trials.py --variant frozen
    python scripts/run_policy_trials.py --variant impute
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement
from trading_crab_lib.platform.evaluation.report import run_full_backtest_evaluation
from trading_crab_lib.platform.honesty.registry import read_trials
from trading_crab_lib.platform.plotting.backtest import compute_ablation_delta

log = logging.getLogger(__name__)

FROZEN_TRIAL_TAG = "P7-W1-frozen-10col"
IMPUTE_TRIAL_TAG = "P7-W1-impute-13col-REJECTED"
IMPUTE_OUTPUT_DIR = OUTPUT_DIR / "reports" / "platform" / "trials" / "impute-13col"

# The four lean-feature columns excluded from the pre-D-02-A common-support
# set (07-CONTEXT.md D-02). See module docstring re: `oil`'s post-D-02-A no-op.
LATE_START_COLUMNS = ["curve_10y2y", "gold", "oil", "fred_vix"]

# UAT-AUDIT-2026-09-09 regression pin: 111.06 (e^111 ~ 10^48) is the rejected value.
TERMINAL_LOG_WEALTH_BOUND = 10.0

EXPECTED_DEV_SHAPE = (708, 53)
EXPECTED_OIL_NON_NAN = 708


def _impute_late_start_columns(monthly_features: pd.DataFrame) -> pd.DataFrame:
    """Back-fill each late-starting column's pre-start gap with its own
    first observed value (D-03's rejected 13-feature alternative).

    NON-CAUSAL BY CONSTRUCTION — see module docstring. Exists only to make
    D-03's rejection evidence-backed; never a candidate policy.
    """
    imputed = monthly_features.copy()
    for col in LATE_START_COLUMNS:
        if col not in imputed.columns:
            continue
        before_na = int(imputed[col].isna().sum())
        imputed[col] = imputed[col].bfill()
        after_na = int(imputed[col].isna().sum())
        log.info(
            "Imputed %s: %d -> %d NaN (back-filled with the column's own first "
            "observed value — non-causal by construction).",
            col, before_na, after_na,
        )
    return imputed


def _assert_leg_kpis_sane(leg_name: str, kpis: dict[str, float]) -> None:
    """Hard, physically-impossible-value guards — never advisory, never a
    policy-selection input (D-04). The UAT-AUDIT-2026-09-09 regression class."""
    wealth = kpis["terminal_log_wealth"]
    dd = kpis["max_drawdown"]
    assert abs(wealth) < TERMINAL_LOG_WEALTH_BOUND, (
        f"{leg_name} terminal_log_wealth={wealth} exceeds the physical-possibility "
        f"bound abs(x) < {TERMINAL_LOG_WEALTH_BOUND} — this is the 111.06 regression "
        "class (UAT-AUDIT-2026-09-09): e^111 is about 10^48, an arithmetically "
        "impossible wealth level."
    )
    assert -1.0 <= dd <= 0.0, (
        f"{leg_name} max_drawdown={dd} is outside [-1, 0] — drawdowns are negative "
        "fractions bounded below by -1 (100% loss); this value is arithmetically "
        "impossible."
    )


def _occupancy(full_sample_states: pd.Series) -> dict[int, float]:
    counts = full_sample_states.value_counts(normalize=True)
    return {int(k): float(v) for k, v in counts.items()}


def run_variant(variant: str, *, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run ONE wave-1 policy variant to completion; return its measured
    numbers as a plain, JSON-serializable (modulo dates) dict."""
    if variant not in ("frozen", "impute"):
        raise ValueError(f"unknown variant {variant!r} — must be 'frozen' or 'impute'")

    cfg = cfg or load_platform_config()
    cm = get_platform_checkpoint_manager()
    monthly_features = cm.load("monthly_features")
    monthly_raw = cm.load("monthly_raw")

    # Precondition (07-03-PLAN.md): the D-02-A recompute (plan 07-02) must
    # have already landed — a run against the stale checkpoint produces
    # numbers that cannot be used.
    assert monthly_features.shape == EXPECTED_DEV_SHAPE, (
        f"monthly_features checkpoint is {monthly_features.shape}, expected "
        f"{EXPECTED_DEV_SHAPE} — the D-02-A recompute (07-02) has not landed."
    )
    oil_non_nan = int(monthly_features["oil"].notna().sum())
    assert oil_non_nan == EXPECTED_OIL_NON_NAN, (
        f"monthly_features['oil'] has {oil_non_nan} non-NaN months, expected "
        f"{EXPECTED_OIL_NON_NAN} — the D-02-A recompute (07-02) has not landed."
    )

    registry_count_before = len(read_trials())

    if variant == "frozen":
        trial_tag = FROZEN_TRIAL_TAG
        output_dir: Path | None = None  # published location (OUTPUT_DIR/reports/platform)
        run_features = monthly_features
    else:
        trial_tag = IMPUTE_TRIAL_TAG
        output_dir = IMPUTE_OUTPUT_DIR
        run_features = _impute_late_start_columns(monthly_features)

    result = run_full_backtest_evaluation(
        run_features, monthly_raw, cfg, output_dir=output_dir, trial_tag=trial_tag,
    )

    registry_count_after = len(read_trials())
    registry_delta = registry_count_after - registry_count_before
    assert registry_delta == 2, (
        f"one run_full_backtest_evaluation call must append exactly 2 registry rows "
        f"(strategy + ablation); observed delta={registry_delta}. The rejected values "
        "are 0/1 (a leg failed to log) and 8 (a double-append regression)."
    )

    strategy_kpis = result["strategy_kpis"]
    ablation_kpis = result["ablation_kpis"]
    _assert_leg_kpis_sane("strategy", strategy_kpis)
    _assert_leg_kpis_sane("ablation", ablation_kpis)

    # Read the PERSISTED artifacts back (rather than reusing the in-memory
    # objects) for every downstream measurement, so each one uses the exact
    # on-disk shape — state_{k} string columns and all — that any later,
    # independent reader would use.
    persisted_dir = Path(output_dir) if output_dir is not None else (OUTPUT_DIR / "reports" / "platform")

    # plotting/backtest.py's own compute_ablation_delta, not a re-derived
    # subtraction — one formula, one place.
    kpi_table = pd.read_parquet(persisted_dir / "backtest_kpi_table.parquet")
    deltas = compute_ablation_delta(kpi_table)
    wealth_delta = deltas["wealth_delta"]
    dd_delta = deltas["dd_delta"]

    sojourn_lag = result["sojourn_lag"]
    assert sojourn_lag["ratio"] > 0, f"§5.4 ratio must be > 0, got {sojourn_lag['ratio']}"
    assert 0 <= sojourn_lag["median_lag"] <= 588, f"median_lag out of [0, 588]: {sojourn_lag['median_lag']}"
    assert 0 <= sojourn_lag["median_sojourn"] <= 588, (
        f"median_sojourn out of [0, 588]: {sojourn_lag['median_sojourn']}"
    )
    assert sojourn_lag["n_resolved"] <= sojourn_lag["n_transitions"], (
        f"n_resolved ({sojourn_lag['n_resolved']}) must be <= n_transitions "
        f"({sojourn_lag['n_transitions']})"
    )

    full_sample_states = result["full_sample_states"]

    full_sample_states_df = pd.read_parquet(persisted_dir / "backtest_full_sample_states.parquet")
    filtered_state_probs_df = pd.read_parquet(persisted_dir / "backtest_filtered_state_probs.parquet")
    disagreement = measure_label_disagreement(full_sample_states_df, filtered_state_probs_df)

    assert 0.0 <= disagreement["pct_disagree"] <= 1.0, (
        f"pct_disagree out of [0, 1]: {disagreement['pct_disagree']}"
    )
    assert not (disagreement["pct_disagree"] == 0.0 and disagreement["n_compared"] == 0), (
        "pct_disagree == 0.0 with n_compared == 0 is the coercion-bug signature "
        "(state_N strings passed through uncoerced) — never a legitimate 'perfect "
        "agreement' result."
    )

    occupancy = _occupancy(full_sample_states)
    occ_sum = sum(occupancy.values())
    assert abs(occ_sum - 1.0) < 1e-9, f"regime occupancy does not sum to 1.0: {occ_sum}"
    for state, frac in occupancy.items():
        assert 0.0 <= frac <= 1.0, f"state {state} occupancy {frac} outside [0, 1]"
    occupancy_below_floor = [state for state, frac in occupancy.items() if frac < 0.05]

    # Advisory-only flags (D-07): recorded, never raised, never used to select
    # or re-run a policy (D-04).
    wealth_delta_out_of_band_assumed = abs(wealth_delta) >= 5
    dd_delta_out_of_band_assumed = not (-0.5 <= dd_delta <= 0.5)
    n_transitions_implausible_assumed = sojourn_lag["n_transitions"] > 30

    summary: dict[str, Any] = {
        "variant": variant,
        "trial_tag": trial_tag,
        "output_dir": str(output_dir) if output_dir is not None else "outputs/reports/platform (published)",
        "registry_count_before": registry_count_before,
        "registry_count_after": registry_count_after,
        "registry_delta": registry_delta,
        "strategy_kpis": strategy_kpis,
        "ablation_kpis": ablation_kpis,
        "wealth_delta": wealth_delta,
        "dd_delta": dd_delta,
        "wealth_delta_out_of_band_ASSUMED": wealth_delta_out_of_band_assumed,
        "dd_delta_out_of_band_ASSUMED": dd_delta_out_of_band_assumed,
        "sojourn_lag": sojourn_lag,
        "n_transitions_implausible_ASSUMED": n_transitions_implausible_assumed,
        "disagreement": {k: v for k, v in disagreement.items() if k != "per_state_confusion"},
        "occupancy": occupancy,
        "occupancy_below_5pct_floor": occupancy_below_floor,
        "report_path": str(result["report_path"]),
        "frozen_l1_features": result["frozen_l1_features"],
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run one wave-1 policy variant, print its measurement
    record as JSON."""
    parser = argparse.ArgumentParser(
        description="Run one wave-1 policy trial variant (07-CONTEXT.md D-03/D-04)"
    )
    parser.add_argument("--variant", choices=["frozen", "impute"], required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    summary = run_variant(args.variant)
    print(json.dumps(summary, indent=2, default=str))  # noqa: T201 — first-class CLI output (the measurement record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
