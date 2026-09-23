#!/usr/bin/env python
"""run_joint_lift.py — criterion 7's joint-allocation-lift measurement (plan 07-11).

Runs ``backtest/joint_driver.py::run_joint_backtest`` twice per routing — once at
``blend_weight_1 = 1.0`` (the classifier-#1-alone baseline leg) and once at
ADR-0002's pinned ``blend_weight_1 = 0.50`` (the joint leg) — on the live dev
checkpoints, and prints one JSON measurement record to stdout.

**Routing (ADR-0002 decision (e)).**

``--routing l1``
    The DECISION-BEARING leg. Both runs append exactly one registry row each
    under an explicit ``trial_tag``; they count toward D-16's deflated-Sharpe
    denominator and D-17's ceiling.

``--routing l2``
    The OBSERVATIONAL, FIREWALLED leg. Both runs are appended with
    ``registry.NO_REGISTRY`` and contribute zero rows. **Nothing downstream in
    phase 7 may change on the basis of a number produced here** — it is reported
    and not acted on.

``--dry-run`` forces ``NO_REGISTRY`` on every run whatever the routing, for
wiring verification. Wave 1 appended four untagged wiring rows that permanently
raised D-16's denominator; that is why a dry run exists as a first-class flag
rather than as a habit.

**No ``sharpe`` key is written into the registry metrics, deliberately.**
``evaluation/deflated_sharpe.py::registry_sharpe_variance`` estimates the
cross-trial Sharpe variance from every row carrying ``metrics["sharpe"]``, and
falls back to the ``1.0`` placeholder below two observations. Writing these two
near-identical legs' Sharpes would flip that estimator from the placeholder to a
sample variance of order 1e-6, collapsing ``expected_max_sharpe`` to roughly zero
and SILENTLY DISABLING the multiple-testing correction for every future DSR in
this project — the "systematically under-penalize search" direction
``07-RESEARCH.md`` names as the closest analog to a security defect here. The
Sharpes are reported in this script's output and in ``07-JOINT-LIFT.md`` instead.
Revisiting this is an ADR-0002 amendment, not a script edit.

Usage::

    python scripts/run_joint_lift.py --routing l1 --dry-run
    python scripts/run_joint_lift.py --routing l2            # firewalled, 0 rows
    python scripts/run_joint_lift.py --routing l1            # 2 registry rows
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform.assets.returns import compute_monthly_returns
from trading_crab_lib.platform.backtest.joint_driver import (
    ROUTING_L1_ONLY,
    ROUTING_L2_NOWCAST,
    joint_lift_table,
    run_joint_backtest,
)
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.evaluation.churn import write_probability_matrix
from trading_crab_lib.platform.evaluation.deflated_sharpe import (
    deflated_sharpe_ratio,
    format_dsr_verdict,
    registry_sharpe_variance,
)
from trading_crab_lib.platform.evaluation.kpis import max_drawdown_and_duration, terminal_log_wealth
from trading_crab_lib.platform.evaluation.report import _reference_label_columns
from trading_crab_lib.platform.features.relative import add_relative_features
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.honesty.registry import NO_REGISTRY, total_trial_count
from trading_crab_lib.platform.labeling.classifier2 import (
    classifier2_config,
    freeze_classifier2_columns,
)
from trading_crab_lib.platform.splice import build_core_research_series
from trading_crab_lib.platform.taxonomy import lean_feature_set

log = logging.getLogger(__name__)

BASELINE_TAG = "07-11-c1-alone-L1only"
JOINT_TAG = "07-11-joint-c1xc2-L1only"

#: ADR-0002 § Trial ceiling, stated before the runs it budgets for (D-17).
ADR_0002_CEILING = 44

#: UAT-AUDIT-2026-09-09's 111.06 regression class: e^111 is about 10^48.
TERMINAL_LOG_WEALTH_BOUND = 10.0


def _annualized_sharpe(returns: pd.Series) -> float:
    """``assets/returns.py``'s own convention: ``(mean / std) * sqrt(12)``.

    Reused rather than re-derived so the leg-level Sharpe is the same quantity
    the per-regime tables report.
    """
    clean = returns.dropna()
    sd = float(clean.std())
    return float((clean.mean() / sd) * np.sqrt(12)) if sd > 0 else float("nan")


def _leg_kpis(equity_curve: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    """Every leg number, each reported WITH the window it was measured on."""
    returns = equity_curve["return"].dropna()
    dd = max_drawdown_and_duration(returns)
    tlw = terminal_log_wealth(returns)
    # Hard, arithmetically-impossible guards only (D-07). Never advisory bands.
    assert abs(tlw) < TERMINAL_LOG_WEALTH_BOUND, (
        f"terminal_log_wealth={tlw} exceeds abs(x) < {TERMINAL_LOG_WEALTH_BOUND} — "
        "the 111.06 regression class (UAT-AUDIT-2026-09-09)."
    )
    assert -1.0 <= dd["max_drawdown"] <= 0.0, (
        f"max_drawdown={dd['max_drawdown']} outside [-1, 0] — arithmetically impossible."
    )
    return {
        "routing": meta["routing"],
        "blend_weight_1": meta["blend_weight_1"],
        "terminal_log_wealth": tlw,
        "max_drawdown": dd["max_drawdown"],
        "underwater_duration_months": dd["duration_months"],
        "sharpe_annualized": _annualized_sharpe(returns),
        "skew": float(returns.skew()),
        "kurtosis_raw": float(returns.kurtosis() + 3.0),  # pandas reports EXCESS kurtosis
        "n_obs": int(len(returns)),
        "n_steps": meta["n_steps"],
        "first_date": str(pd.Timestamp(meta["first_date"]).date()),
        "last_date": str(pd.Timestamp(meta["last_date"]).date()),
        "n_degraded": meta["n_degraded"],
        "n_degraded_classifier_1": meta["n_degraded_classifier_1"],
        "n_degraded_classifier_2": meta["n_degraded_classifier_2"],
        "n_state_1_transitions": _n_transitions(equity_curve["state_1"]),
        "n_state_2_transitions": _n_transitions(equity_curve["state_2"]),
        "state_1_transition_rate": _n_transitions(equity_curve["state_1"]) / max(1, meta["n_steps"]),
        "state_2_transition_rate": _n_transitions(equity_curve["state_2"]) / max(1, meta["n_steps"]),
        "mean_turnover": float(equity_curve["turnover"].mean()),
        "total_cost": float(equity_curve["cost"].sum()),
        "min_monthly_return": float(returns.min()),
        "min_monthly_return_date": str(returns.idxmin().date()),
        "max_monthly_return": float(returns.max()),
        "max_monthly_return_date": str(returns.idxmax().date()),
        "mean_scale": float(equity_curve["scale"].mean()),
        "max_scale": float(equity_curve["scale"].max()),
        "registry_row_written": meta["registry_row_written"],
    }


def _n_transitions(states: pd.Series) -> int:
    """Run-length changes across the visited decision months (07-BANDS.md band 3b)."""
    clean = states.dropna()
    return int((clean != clean.shift()).sum() - 1) if len(clean) else 0


def build_inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Assemble both classifiers' feature frames, the asset universe and the frozen lists.

    Mirrors ``evaluation/report.py::run_full_backtest_evaluation``'s own
    assembly exactly — the same ``split_by_holdout_boundary``, the same
    ``first_decision = dev_features.index[min_train]``, the same
    ``_reference_label_columns`` call — so classifier #1's feature space in this
    harness is identical to the one every wave-1 number was measured on.
    """
    cm = get_platform_checkpoint_manager()
    monthly_features = cm.load("monthly_features")
    monthly_raw = cm.load("monthly_raw")

    dev_features, _ = split_by_holdout_boundary(monthly_features, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    lean_cols = sorted(lean_feature_set(cfg) & set(dev_features.columns))
    min_train = int(cfg.get("backtest", {}).get("min_train_months", 120))
    first_decision = dev_features.index[min_train]
    frozen_1 = _reference_label_columns(dev_features, lean_cols, first_decision)

    # Classifier #2's frame: derived columns only, reindexed onto classifier #1's
    # index so the two labelings are fit on the SAME months (D-11).
    relative = add_relative_features(monthly_raw, cfg).reindex(monthly_features.index)
    frozen_2 = freeze_classifier2_columns(relative, cfg, first_decision)

    splice_cfg = cfg["splice"]
    research = build_core_research_series(monthly_raw, cfg)
    returns = compute_monthly_returns(research)
    cash_ret = returns[splice_cfg["cash"]["research_name"]]
    asset_returns = pd.DataFrame(
        {
            params["tradable"]: returns[params["research_name"]]
            for name, params in splice_cfg.items()
            if name != "cash" and params["research_name"] in returns.columns
        }
    )
    return {
        "features_1": monthly_features,
        "features_2": relative,
        "asset_returns": asset_returns,
        "cash_returns": cash_ret,
        "frozen_1": frozen_1,
        "frozen_2": frozen_2,
        "first_decision": first_decision,
        "min_train": min_train,
    }


def run(routing_flag: str, *, dry_run: bool, dump_dir: str | None = None) -> dict[str, Any]:
    """Run both legs under one routing; return the full measurement record."""
    cfg = load_platform_config()
    routing = ROUTING_L1_ONLY if routing_flag == "l1" else ROUTING_L2_NOWCAST
    decision_bearing = (routing == ROUTING_L1_ONLY) and not dry_run
    ledger = None if decision_bearing else NO_REGISTRY

    inputs = build_inputs(cfg)
    blend_weight = float(cfg.get("allocation", {}).get("blend_weight_1", 0.50))
    c2 = classifier2_config(cfg)

    count_before = total_trial_count()
    ts_before = datetime.now(timezone.utc).isoformat()

    common = dict(
        cfg=cfg,
        features_2=inputs["features_2"],
        frozen_features_1=inputs["frozen_1"],
        frozen_features_2=inputs["frozen_2"],
        routing=routing,
        cash_returns=inputs["cash_returns"],
        registry_path=ledger,
    )
    baseline_curve, baseline_meta = run_joint_backtest(
        inputs["features_1"], inputs["asset_returns"],
        blend_weight_1=1.0, trial_tag=BASELINE_TAG, **common,
    )
    joint_curve, joint_meta = run_joint_backtest(
        inputs["features_1"], inputs["asset_returns"],
        blend_weight_1=blend_weight, trial_tag=JOINT_TAG, **common,
    )

    count_after = total_trial_count()
    ts_after = datetime.now(timezone.utc).isoformat()
    rows_added = count_after - count_before
    expected_rows = 2 if decision_bearing else 0
    assert rows_added == expected_rows, (
        f"registry moved by {rows_added} rows; expected exactly {expected_rows} "
        f"({'two tagged runs, one row each' if decision_bearing else 'NO_REGISTRY sentinel'}). "
        "An accounting mismatch blocks reporting any number from this run."
    )
    assert count_after <= ADR_0002_CEILING, (
        f"total_trial_count() is {count_after}, above ADR-0002's stated ceiling of "
        f"{ADR_0002_CEILING}. Exceeding it requires an explicit ADR amendment — this "
        "is the silent search-creep D-17 exists to make visible."
    )

    if dump_dir:
        # Persist both legs so every number below can be independently recomputed
        # from the on-disk curves rather than trusted from this script's output.
        from pathlib import Path as _Path

        out = _Path(dump_dir)
        out.mkdir(parents=True, exist_ok=True)
        suffix = "l1only" if routing == ROUTING_L1_ONLY else "l2"
        baseline_curve.to_parquet(out / f"joint_lift_baseline_{suffix}.parquet")
        joint_curve.to_parquet(out / f"joint_lift_joint_{suffix}.parquet")

        # ROADMAP criterion 0: the per-step probability matrix the driver
        # accumulates (joint_driver.py:508-510) reaches disk, so every number
        # about the nowcaster's filtered path can be RECOMPUTED from an artifact
        # rather than trusted from a log.
        #
        # Only the JOINT leg's matrices are written. The two legs differ solely
        # in `blend_weight_1`, and the suite already pins that they share the
        # state path exactly (test_joint_and_baseline_legs_share_the_state_path_exactly),
        # so a second pair of files would be two names for one object.
        for clf, per_step in (
            (1, joint_meta["per_step_metrics_1"]),
            (2, joint_meta["per_step_metrics_2"]),
        ):
            info = write_probability_matrix(
                per_step, out / f"joint_lift_probs_{clf}_{suffix}.parquet"
            )
            # Pitfall 6: a churn rate quoted without its degraded count is not
            # quotable. The driver appends only on NON-degraded steps, so the
            # difference below IS the degraded-step count (100 of 588 under l2).
            log.info(
                "probability matrix classifier #%d (%s): %d rows of %d steps; "
                "difference %d = degraded steps -> %s",
                clf, suffix, info["n_rows"], joint_meta["n_steps"],
                int(joint_meta["n_steps"]) - info["n_rows"], info["path"],
            )

    lift = joint_lift_table(joint_curve, baseline_curve)
    baseline_kpis = _leg_kpis(baseline_curve, baseline_meta)
    joint_kpis = _leg_kpis(joint_curve, joint_meta)

    sharpe_variance = registry_sharpe_variance()
    dsr = {}
    for name, kpis in (("baseline", baseline_kpis), ("joint", joint_kpis)):
        value = deflated_sharpe_ratio(
            observed_sharpe=kpis["sharpe_annualized"],
            n_trials=count_after,
            sharpe_variance=sharpe_variance,
            skew=kpis["skew"],
            kurtosis=kpis["kurtosis_raw"],
            n_obs=kpis["n_obs"],
        )
        dsr[name] = {
            "dsr": value,
            "verdict": format_dsr_verdict(value),
            "n_trials": count_after,
            "n_trials_read_at": ts_after,
            "sharpe_variance": sharpe_variance,
            "observed_sharpe": kpis["sharpe_annualized"],
            "n_obs": kpis["n_obs"],
            "window": f"{kpis['n_steps']} steps, {kpis['first_date']} -> {kpis['last_date']}",
        }

    return {
        "routing": routing,
        "decision_bearing": decision_bearing,
        "dry_run": dry_run,
        "blend_weight_1": blend_weight,
        "K_1": cfg["labeling"]["K"],
        "lambda_1": cfg["labeling"]["lambda"],
        "K_2": c2["K"],
        "lambda_2": c2["lam"],
        "frozen_features_1": inputs["frozen_1"],
        "frozen_features_2": inputs["frozen_2"],
        "first_decision": str(pd.Timestamp(inputs["first_decision"]).date()),
        "registry": {
            "count_before": count_before,
            "read_before_at": ts_before,
            "count_after": count_after,
            "read_after_at": ts_after,
            "rows_added": rows_added,
            "adr_0002_ceiling": ADR_0002_CEILING,
            "ceiling_respected": count_after <= ADR_0002_CEILING,
            "baseline_tag": BASELINE_TAG if decision_bearing else "(NO_REGISTRY)",
            "joint_tag": JOINT_TAG if decision_bearing else "(NO_REGISTRY)",
        },
        "baseline_leg": baseline_kpis,
        "joint_leg": joint_kpis,
        "lift": lift,
        "deflated_sharpe": dsr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Criterion 7's joint-lift measurement (plan 07-11)")
    parser.add_argument("--routing", choices=["l1", "l2"], required=True)
    parser.add_argument("--dry-run", action="store_true", help="force NO_REGISTRY on every run")
    parser.add_argument("--out", default=None, help="also write the JSON record to this path")
    parser.add_argument("--dump-curves", default=None, help="write both legs' equity curves here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    record = run(args.routing, dry_run=args.dry_run, dump_dir=args.dump_curves)
    text = json.dumps(record, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)  # noqa: T201 — first-class CLI output (the measurement record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
