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
    read_probability_matrix,
)
from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement
from trading_crab_lib.platform.evaluation.sojourn_lag import compute_sojourn_lag_headline
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary

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
                "first_date": str(joint.index.min().date()),
                "last_date": str(joint.index.max().date()),
                "transition_rate": _n_transitions(filtered) / max(1, len(joint)),
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
