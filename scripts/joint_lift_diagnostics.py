#!/usr/bin/env python
"""joint_lift_diagnostics.py — labeling diagnostics for criterion 7's two legs.

Reads the equity curves ``run_joint_lift.py`` persisted (``--dump-curves``) and
the full-sample smoothed label checkpoints, and reports the two **labeling**
quantities ``07-BANDS.md`` §8 governs — which criterion 7's joint-lift
comparison does NOT itself produce (§0.2 of that document names this):

- **Band 3a** (``sojourn_lag`` within-window): the definitional
  ``n_resolved <= n_transitions <= n_label_transitions``. A breach is a counting
  bug, never a strategy outcome.
- **Band 3b** (full-sample labeling): the RATE band, implausible above
  ``0.10 x n_months``.
- **Band 4** (``pct_disagree``): ``< 0.02`` **OR** ``n_compared == 0`` **OR**
  ``n_compared`` materially below expectation without a recorded reason. The
  third clause is passed live via ``expected_n_compared``, so a short window is
  caught by the mechanism rather than by coincidence.

**This script touches the trial registry not at all.** It re-reads persisted
artifacts; running it can never add a row or move D-16's denominator. That is
why the diagnostics live here rather than inside ``run_joint_lift.py``, whose
``--routing l1`` path appends two rows every time it is invoked.

Usage::

    python scripts/joint_lift_diagnostics.py --curves outputs/reports/platform/joint_lift
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.evaluation.churn import (
    argmax_churn,
    churn_rate,
    read_probability_matrix,
)
from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement
from trading_crab_lib.platform.evaluation.sojourn_lag import (
    classify_negative_offsets,
    compute_signed_detection_offsets,
    compute_sojourn_lag_headline,
)
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.prediction.nowcaster import transition_window_accuracy

log = logging.getLogger(__name__)

#: 07-BANDS.md §8 band 3b: implausible above one state change per ten months.
LABEL_TRANSITION_RATE_BAND = 0.10

#: The two churn series (08-RESEARCH.md § F-1), named IN the record so a reader
#: who opens the JSON in six months cannot mistake one for the other. They are
#: elementwise identical under ``l1only`` by construction
#: (``joint_driver.py:431``) — which is a degeneracy the ``series_identity``
#: block below records, not a licence to report either one as the other.
TRACK_A = (
    "A — L1 jump model terminal-month label (state_N); NOT movable by an L2 change"
)
TRACK_B = (
    "B — argmax of the L2 nowcaster's calibrated posterior; the object design §5.1 changes"
)
#: Plan 08-08: Track B splits in two. B0 (``walk_forward_nowcast``) is the RAW
#: posterior's argmax churn — the CONTROL: the nowcaster is untouched, so it must
#: not move, and that invariance is what makes any movement in B1 attributable to
#: the filter. B1 is the argmax of the FILTERED BELIEF the allocator now consumes.
TRACK_B1 = (
    "B1 — argmax of the Bayes-filtered belief (regime_filter over the L2 posterior); "
    "what the allocator consumes under l2. Compare against B0 (walk_forward_nowcast), the control"
)


def _max_prob_distribution(matrix: pd.DataFrame, act_threshold: float) -> dict[str, Any]:
    """Row-max distribution — the direct measurement that replaces F-2's derived bound."""
    row_max = matrix.max(axis=1)
    quantiles = row_max.quantile([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    return {
        "n_rows": int(len(row_max)),
        "n_below_act_threshold": int((row_max < act_threshold).sum()),
        "act_threshold": act_threshold,
        "quantiles": {f"q{int(round(q * 100)):02d}": float(v) for q, v in quantiles.items()},
    }


def _persistence_readings(
    full_dev: pd.Series, matrix: pd.DataFrame, act_threshold: float
) -> dict[str, Any]:
    """S-3 readings (reported, never gated) and the S-1 signed offsets for one matrix."""
    headline = compute_sojourn_lag_headline(full_dev, matrix, act_threshold=act_threshold)
    offsets = compute_signed_detection_offsets(full_dev, matrix, act_threshold=act_threshold)
    classified = classify_negative_offsets(full_dev, matrix, offsets, act_threshold)
    y_true = full_dev.reindex(matrix.index)
    keep = y_true.notna()
    accuracy = transition_window_accuracy(
        y_true[keep].astype(int), matrix.idxmax(axis=1)[keep].astype(int)
    )
    return {
        "sojourn_lag": {k: headline[k] for k in ("median_sojourn", "median_lag", "ratio", "n_transitions", "n_resolved")},
        "transition_window_accuracy": accuracy,
        "max_prob": _max_prob_distribution(matrix, act_threshold),
        "signed_offsets": {
            "min_offset": offsets["min_offset"],
            "median_offset": offsets["median_offset"],
            "n_transitions": offsets["n_transitions"],
            "n_resolved": offsets["n_resolved"],
            "n_negative": offsets["n_negative"],
            "n_zero_or_negative": offsets["n_zero_or_negative"],
        },
        "negative_offset_classification": {
            "n_lead": classified["n_lead"],
            "n_held_through_miss": classified["n_held_through_miss"],
            "lead_positions": classified["lead_positions"],
            "held_through_miss_positions": classified["held_through_miss_positions"],
            "details": [
                {**d, "date": str(pd.Timestamp(d["date"]).date()), "preceding_run": list(d["preceding_run"])}
                for d in classified["details"]
            ],
        },
    }


def _as_states(frame: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(frame, pd.Series):
        return frame
    for col in ("state", "regime", "label"):
        if col in frame.columns:
            return frame[col]
    if frame.shape[1] == 1:
        return frame.iloc[:, 0]
    raise ValueError(f"cannot derive a state column from {list(frame.columns)}")


def _n_transitions(states: pd.Series) -> int:
    clean = states.dropna()
    return int((clean != clean.shift()).sum() - 1) if len(clean) else 0


def _one_hot_state_probs(states: pd.Series, *, string_columns: bool) -> pd.DataFrame:
    """The walk-forward filtered probability matrix, in ONE of two column shapes.

    Under the L1-only routing the per-step probability vector IS a degenerate
    one-hot on the last filtered state, so this reconstruction is exact, not an
    approximation.

    **Two consumers, two incompatible column conventions — and mixing them fails
    SILENTLY.** ``measure_label_disagreement`` expects ``state_{k}`` STRING
    columns (it strips the prefix itself; an integer column would survive
    ``idxmax`` and then break the int cast). ``compute_sojourn_lag_headline``
    expects INTEGER columns and treats "transitions into a state with no
    observed column" as fully unresolved — so handing it string columns returns
    ``n_resolved = 0`` and ``median_lag = NaN`` with no error, which reads as
    "detection never happened" rather than "the matrix was the wrong shape".
    That is this project's signature defect (a check that can only confirm) in a
    new costume; :func:`diagnose` asserts against it explicitly.
    """
    clean = states.dropna().astype(int)
    levels = sorted(clean.unique())
    key = (lambda k: f"state_{k}") if string_columns else (lambda k: int(k))
    return pd.DataFrame({key(k): (clean == k).astype(float) for k in levels}, index=clean.index)


def diagnose(curves_dir: Path, *, suffix: str) -> dict[str, Any]:
    cfg = load_platform_config()
    cm = get_platform_checkpoint_manager()
    act_threshold = cfg.get("allocation", {}).get("hysteresis", {}).get("act_threshold", 0.70)

    joint = pd.read_parquet(curves_dir / f"joint_lift_joint_{suffix}.parquet")
    out: dict[str, Any] = {"suffix": suffix, "n_steps": int(len(joint))}

    identity: dict[str, Any] = {}
    for tag, curve_col, checkpoint, clf in (
        ("classifier_1", "state_1", "regime_labels", 1),
        ("classifier_2", "state_2", "regime_labels_2", 2),
    ):
        full = _as_states(cm.load(checkpoint))
        full_dev, _ = split_by_holdout_boundary(full.to_frame("state"), cutoff=DEFAULT_HOLDOUT_CUTOFF)
        full_dev = full_dev["state"]

        n_label_transitions = _n_transitions(full)
        n_months = int(full.notna().sum())

        filtered = joint[curve_col]
        probs_str = _one_hot_state_probs(filtered, string_columns=True)
        probs_int = _one_hot_state_probs(filtered, string_columns=False)
        disagreement = measure_label_disagreement(
            full_dev.to_frame("state"), probs_str, expected_n_compared=int(len(joint)),
        )
        soj = compute_sojourn_lag_headline(full_dev, probs_int, act_threshold=act_threshold)
        # A zero here is ambiguous between "nothing was ever detected" and "the
        # probability matrix had the wrong column shape". Refuse the ambiguity.
        overlap = set(probs_int.columns) & set(int(v) for v in full_dev.dropna().unique())
        assert overlap, (
            f"{tag}: the filtered probability matrix shares NO state column with the "
            f"full-sample labeling ({sorted(probs_int.columns)} vs "
            f"{sorted(full_dev.dropna().unique())}) — a sojourn/lag readout computed "
            "from it would report n_resolved = 0 for a reason that is not a finding."
        )

        # ── Track B: the nowcaster's own filtered path, read from its artifact ──
        #
        # There is deliberately NO fallback that derives this from `curve_col`.
        # Under ROUTING_L1_ONLY the state column and the argmax of the posterior
        # are the same object (joint_driver.py:431), so a fallback would let
        # Track B silently become Track A wearing a different key — the exact
        # masquerade this split exists to prevent. Absent artifact => raise.
        probs_path = curves_dir / f"joint_lift_probs_{clf}_{suffix}.parquet"
        if not probs_path.is_file():
            raise FileNotFoundError(
                f"{probs_path} is missing. Track B (the L2 nowcaster's argmax churn) is "
                "computed ONLY from the persisted per-step probability matrix. This "
                "function will not fall back to the state_N column, because under "
                "ROUTING_L1_ONLY that column IS the argmax by construction "
                "(joint_driver.py:431) and the fallback would report the L1 label churn "
                "under the L2 series' name. Produce the artifact with:\n"
                f"    python scripts/run_joint_lift.py --routing "
                f"{'l1 --dry-run' if suffix == 'l1only' else 'l2'} "
                "--dump-curves outputs/reports/platform/joint_lift"
            )
        nowcast_matrix = read_probability_matrix(probs_path)
        nowcast = argmax_churn(nowcast_matrix)
        # The driver appends per-step rows only on NON-degraded steps
        # (joint_driver.py:508-510), so this difference IS the degraded count.
        # Pitfall 6: a churn rate quoted without it is not quotable.
        n_degraded = int(len(joint) - len(nowcast_matrix))

        # Are the two series the same object? Answered by measurement, on the
        # non-degraded rows the nowcast matrix actually covers.
        argmax_labels = nowcast_matrix.idxmax(axis=1).astype("float64")
        aligned_state = pd.to_numeric(
            joint[curve_col].reindex(argmax_labels.index), errors="coerce"
        )
        n_mismatched = int((argmax_labels.to_numpy() != aligned_state.to_numpy()).sum())
        identity[tag] = {
            "argmax_equals_state_elementwise": bool(n_mismatched == 0),
            "n_mismatched_months": n_mismatched,
            "n_compared": int(len(argmax_labels)),
            "state_column": curve_col,
            "probs_source": str(probs_path),
        }

        # ── Track B1 (plan 08-08): the filtered belief, read from ITS artifact ──
        #
        # Present only where the filter ran (the l2 routing). Under l1only the filter
        # is never applied, and the block says so rather than borrowing another
        # series. Under l2 a missing artifact raises, exactly like Track B's.
        belief_path = curves_dir / f"joint_lift_belief_{clf}_{suffix}.parquet"
        if suffix == "l1only":
            belief_block: dict[str, Any] = {
                "track": TRACK_B1,
                "applicable": False,
                "reason": "the Bayes filter is not applied under ROUTING_L1_ONLY (a one-hot is not a likelihood)",
            }
        else:
            if not belief_path.is_file():
                raise FileNotFoundError(
                    f"{belief_path} is missing. Track B1 (the filtered belief's argmax churn) is "
                    "computed ONLY from the persisted belief matrix; there is no fallback. "
                    "Produce it with:\n    python scripts/run_joint_lift.py --routing l2 "
                    "--dump-curves outputs/reports/platform/joint_lift"
                )
            belief_matrix = read_probability_matrix(belief_path)
            belief = argmax_churn(belief_matrix)
            common = belief_matrix.index.intersection(nowcast_matrix.index)
            belief_argmax = belief_matrix.loc[common].idxmax(axis=1).astype(int)
            posterior_argmax = nowcast_matrix.loc[common].idxmax(axis=1).astype(int)
            belief_block = {
                "track": TRACK_B1,
                "applicable": True,
                "n_changes": belief["n_changes"],
                "n_rows": belief["n_rows"],
                "n_pairs": belief["n_pairs"],
                "rate": belief["rate"],
                "n_degraded": int(len(joint) - len(belief_matrix)),
                "first_date": belief["first_date"],
                "last_date": belief["last_date"],
                "source": str(belief_path),
                "index_equals_nowcast_index": bool(belief_matrix.index.equals(nowcast_matrix.index)),
                "n_mismatched_months_vs_posterior": int((belief_argmax != posterior_argmax).sum()),
                "n_compared_vs_posterior": int(len(common)),
            }
            # S-3 readings and the S-1 real-data guard, for BOTH the raw posterior
            # (the "before" object on this routing) and the belief (the "after").
            # Reported, never gated here; the gate is the test suite's.
            out.setdefault("persistence_readings", {})[tag] = {
                "s1_status": (
                    "OBSERVATIONAL since the 2026-09-23 ruling (08-08-PLAN.md): causal invariance is the "
                    "governing leakage guard; every negative offset is adjudicated by a truncation cut "
                    "recorded in s1_truncation_invariance.json"
                ),
                "act_threshold": act_threshold,
                "reference": f"{checkpoint} split at {DEFAULT_HOLDOUT_CUTOFF}",
                "raw_posterior": _persistence_readings(full_dev, nowcast_matrix, act_threshold),
                "belief": _persistence_readings(full_dev, belief_matrix, act_threshold),
            }

        rate = n_label_transitions / n_months if n_months else float("nan")
        out[tag] = {
            "full_sample": {
                "n_label_transitions": n_label_transitions,
                "n_months": n_months,
                "first_month": str(full.dropna().index.min().date()),
                "last_month": str(full.dropna().index.max().date()),
                "transition_rate": rate,
                "band_3b_threshold_rate": LABEL_TRANSITION_RATE_BAND,
                "band_3b_ok": bool(rate <= LABEL_TRANSITION_RATE_BAND),
            },
            "walk_forward_filtered": {
                "track": TRACK_A,
                "n_transitions": _n_transitions(filtered),
                "n_steps": int(len(joint)),
                # F-4 (plan 08-01): a change is a property of an adjacent PAIR, so
                # the rate divides by n_steps - 1. Recorded as 246/588 = 41.84%
                # before this fix; 246/587 = 41.91% is the rate. Routed through
                # evaluation/churn.py so this phase has ONE churn-rate definition.
                "n_pairs": int(len(joint)) - 1,
                "first_date": str(joint.index.min().date()),
                "last_date": str(joint.index.max().date()),
                "transition_rate": churn_rate(_n_transitions(filtered), int(len(joint))),
            },
            "walk_forward_nowcast": {
                "track": TRACK_B,
                "n_changes": nowcast["n_changes"],
                "n_rows": nowcast["n_rows"],
                "n_pairs": nowcast["n_pairs"],
                "rate": nowcast["rate"],
                "n_degraded": n_degraded,
                "first_date": nowcast["first_date"],
                "last_date": nowcast["last_date"],
                "source": str(probs_path),
            },
            "walk_forward_belief": belief_block,
            "sojourn_lag": {
                k: v for k, v in soj.items() if k not in ("per_state_lags", "lags")
            },
            "band_3a_ok": bool(
                soj["n_resolved"] <= soj["n_transitions"] <= n_label_transitions
            ),
            "disagreement": {
                k: v for k, v in disagreement.items() if k != "per_state_confusion"
            },
        }
    out["series_identity"] = identity
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Criterion-7 labeling diagnostics (plan 07-11)")
    parser.add_argument("--curves", required=True, help="directory holding the dumped equity curves")
    parser.add_argument("--suffix", default="l1only", choices=["l1only", "l2"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    record = diagnose(Path(args.curves), suffix=args.suffix)
    text = json.dumps(record, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)  # noqa: T201 — first-class CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
