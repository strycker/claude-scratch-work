#!/usr/bin/env python
"""terminal_month_diagnostic.py — the zero-trial terminal-month churn diagnostic (plan 08-02).

**What this measures, and why the question is not rhetorical.**

The L1 jump model minimises ``Σ_t d[t, s_t] + λ · Σ_t 1[s_t != s_{t-1}]``
[``labeling/jump_model.py:43-45``]. The **terminal month** of a decode is the only
month with no right-hand neighbour, so deviating there costs **λ once**, where an
interior deviation costs **2λ** (jump in and jump out). The filtered labeling reads
exactly that month, every step (``joint_driver.py:502``, ``state_1 = states_1.iloc[-1]``).
The *existence* of that asymmetry follows from the objective. Its *magnitude* is
unmeasured, and this script is the measurement.

For each walk-forward step *t*, this harness asks the step's **own** L1 fit for the
label it assigns to the k-th-from-last month of its **own** training window, for
k = 1 … ``--max-lag``, and churns each of those six series across steps. k = 1 is
today's ``state_1``. If churn falls sharply with k, a material share of classifier #1's
41.91 % filtered churn is an **edge artefact of reading ``iloc[-1]``**. If churn is
flat in k, the edge story is **refuted** and λ/d is the whole story.

**The anchor is the point.** Before any k > 1 number may be reported, the k = 1 column
must reproduce the tracked curve's ``state_1`` / ``state_2`` columns ELEMENTWISE
(246 and 24 changes respectively). If it does not, the harness is fitting something
other than the classifier every recorded number came from, and no k > 1 column is
interpretable. :func:`assert_anchor` raises with that message rather than softening to
a tolerance — the DP decode is deterministic at a fixed seed
(``fit_jump_model(random_state=42)``), so elementwise equality is the correct assertion.

**Zero registry trials, by construction.** Both classifiers are fit at their **pinned**
(K, λ) read from live config — K=6/λ=10.0 for #1, K=5/λ=16.0 for #2. No configuration is
varied, therefore no configuration is evaluated, therefore nothing is registered. A λ
sweep is NOT authorized (``08-CONTEXT.md`` Q-7) and this script cannot perform one: it has
no λ flag. ``registry`` is never imported here for writing.

**This script changes no production code.** ``driver.py``, ``joint_driver.py`` and
everything under ``labeling/`` are read-only to it, deliberately: a diagnostic that
modified the thing it measures could not be compared against the tracked curves.

Usage::

    python scripts/terminal_month_diagnostic.py --limit-steps 40 --out-dir /tmp/smoke
    python scripts/terminal_month_diagnostic.py --out-dir outputs/reports/platform/track_a
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:  # pragma: no cover — import plumbing
    sys.path.insert(0, str(_SCRIPTS_DIR))

from run_joint_lift import build_inputs  # noqa: E402 — needs the sys.path line above

from trading_crab_lib.platform.backtest.driver import (  # noqa: E402
    _L2_DEGRADE_EXCEPTIONS,
    _refit_l1,
)
from trading_crab_lib.platform.backtest.joint_driver import _refit_classifier2  # noqa: E402
from trading_crab_lib.platform.config import load_platform_config  # noqa: E402
from trading_crab_lib.platform.honesty.holdout import (  # noqa: E402
    DEFAULT_HOLDOUT_CUTOFF,
    split_by_holdout_boundary,
)
from trading_crab_lib.platform.honesty.walkforward import expanding_steps  # noqa: E402
from trading_crab_lib.platform.labeling.classifier2 import classifier2_config  # noqa: E402

log = logging.getLogger(__name__)

#: Mirrors ``joint_driver._PROGRESS_EVERY_STEPS`` so the two runs' logs read alike.
_PROGRESS_EVERY_STEPS = 24

#: The tracked curve the k=1 column must reproduce elementwise. This is the file every
#: recorded criterion-7 number was measured on; anchoring against anything else would
#: make the anchor a self-comparison.
DEFAULT_TRACKED_CURVE = Path("outputs/reports/platform/joint_lift/joint_lift_joint_l1only.parquet")

#: The counts the anchor must reproduce, from the tracked curve at ``b1c3519``.
#: Re-derived from the parquet at assert time; these are the expectation, not the source.
ANCHOR_N_CHANGES = {"classifier_1": 246, "classifier_2": 24}

DEFAULT_MAX_LAG = 6
DEFAULT_OUT_DIR = Path("outputs/reports/platform/track_a")

_CLASSIFIER_KEYS = {1: "classifier_1", 2: "classifier_2"}


# ── churn ──────────────────────────────────────────────────────────────────────

def state_change_count(states: pd.Series) -> dict[str, Any]:
    """Adjacent-pair state changes under the dropna-then-compare rule.

    NaN rows are DROPPED before the adjacent-pair comparison, never compared with
    ``!=`` (``np.nan != np.nan`` is True and would count every missing lag as a
    change). ``n_pairs = n_rows - 1`` after the drop, so the rate is
    pair-denominated — the off-by-one ``08-RESEARCH.md`` F-4 records against the
    41.84 %/41.91 % pair.

    Duplicated here on purpose. Plan 08-01 owns
    ``platform/evaluation/churn.py`` and runs in the SAME wave as this plan;
    importing it would be a cross-wave dependency this plan does not declare.
    **08-01's ``churn.py`` is the consolidation point** — when both plans have
    landed, this function should be deleted in favour of it, and the identical
    rule is what makes that a safe swap.
    """
    clean = pd.Series(states).dropna()
    n_rows = int(len(clean))
    n_pairs = max(0, n_rows - 1)
    n_changes = int((clean != clean.shift()).sum() - 1) if n_rows else 0
    return {
        "n_changes": n_changes,
        "n_rows": n_rows,
        "n_pairs": n_pairs,
        "rate": (n_changes / n_pairs) if n_pairs else 0.0,
    }


def churn_by_lag(labels_df: pd.DataFrame, *, max_lag: int = DEFAULT_MAX_LAG) -> dict[str, list[dict]]:
    """Per classifier, the churn of the ``iloc[-k]`` series across steps, k = 1 … max_lag."""
    out: dict[str, list[dict]] = {}
    for n, key in _CLASSIFIER_KEYS.items():
        rows = []
        for k in range(1, max_lag + 1):
            col = f"c{n}_lag{k}_state"
            if col not in labels_df.columns:
                continue
            entry = {"k": k}
            entry.update(state_change_count(labels_df[col]))
            rows.append(entry)
        out[key] = rows
    return out


def fixed_month_revision(labels_df: pd.DataFrame, *, max_lag: int = DEFAULT_MAX_LAG) -> dict[str, list[dict]]:
    """The SAME matrix read down the other axis — NOT a second experiment.

    ``churn_by_lag`` reads across steps at a fixed lag ("how much does the k-th-from-last
    label move as the window slides?"). This reads across lags at a fixed CALENDAR MONTH
    ("once month *m* is no longer the terminal month, does its label get revised?").
    Same cells, different traversal. It is reported as a secondary panel and carries no
    independent evidential weight.

    For each k ≥ 2 and each calendar month *m* labeled at both lag 1 and lag k, compares
    the two labels and counts disagreements.
    """
    out: dict[str, list[dict]] = {}
    for n, key in _CLASSIFIER_KEYS.items():
        base = _lag_month_map(labels_df, n, 1)
        rows = []
        for k in range(2, max_lag + 1):
            later = _lag_month_map(labels_df, n, k)
            shared = sorted(set(base) & set(later))
            n_differs = sum(1 for m in shared if base[m] != later[m])
            rows.append(
                {
                    "k": k,
                    "n_compared": len(shared),
                    "n_differs": int(n_differs),
                    "rate": (n_differs / len(shared)) if shared else 0.0,
                }
            )
        out[key] = rows
    return out


def _lag_month_map(labels_df: pd.DataFrame, n: int, k: int) -> dict[pd.Timestamp, float]:
    """``{calendar month -> the state the lag-k cell assigned it}``, NaN cells dropped."""
    date_col, state_col = f"c{n}_lag{k}_date", f"c{n}_lag{k}_state"
    if date_col not in labels_df.columns or state_col not in labels_df.columns:
        return {}
    sub = labels_df[[date_col, state_col]].dropna()
    return {pd.Timestamp(d): float(s) for d, s in zip(sub[date_col], sub[state_col], strict=True)}


# ── per-step recording ─────────────────────────────────────────────────────────

def lag_record(
    states: pd.Series,
    train_index: pd.Index,
    *,
    max_lag: int,
    prefix: str,
) -> tuple[dict[str, Any], bool]:
    """One step's ``iloc[-k]`` state AND date for k = 1 … max_lag.

    ``_refit_l1`` returns states indexed by the **post-dropna** subset of
    ``train_features.index``, so ``iloc[-1]`` is the last month in the window with
    complete data on the frozen columns — *usually* ``train_index[-1]`` but not
    guaranteed to be. The date of every cell is therefore RECORDED, and the k=1
    date is checked against ``train_index[-1]``.

    Returns:
        ``(columns, k1_date_misaligned)``. Lags beyond ``len(states)`` record NaN for
        both state and date; ``k1_date_misaligned`` is True when the fit's terminal
        month is not the window's last month — a finding, not a rounding detail.
    """
    cols: dict[str, Any] = {}
    n = len(states)
    for k in range(1, max_lag + 1):
        if n >= k:
            cols[f"{prefix}_lag{k}_state"] = float(states.iloc[-k])
            cols[f"{prefix}_lag{k}_date"] = pd.Timestamp(states.index[-k])
        else:
            cols[f"{prefix}_lag{k}_state"] = np.nan
            cols[f"{prefix}_lag{k}_date"] = pd.NaT

    misaligned = False
    if n and len(train_index):
        misaligned = pd.Timestamp(states.index[-1]) != pd.Timestamp(train_index[-1])
    return cols, bool(misaligned)


def _empty_lag_record(*, max_lag: int, prefix: str) -> dict[str, Any]:
    """All-NaN columns for a step whose refit degraded."""
    cols: dict[str, Any] = {}
    for k in range(1, max_lag + 1):
        cols[f"{prefix}_lag{k}_state"] = np.nan
        cols[f"{prefix}_lag{k}_date"] = pd.NaT
    return cols


# ── the harness ────────────────────────────────────────────────────────────────

def collect_terminal_labels(
    cfg: dict[str, Any],
    *,
    max_lag: int = DEFAULT_MAX_LAG,
    limit_steps: int | None = None,
    inputs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Refit BOTH L1 classifiers at every walk-forward step and record six lags each.

    Reuses ``run_joint_lift.build_inputs`` so the frozen column lists and the feature
    frames are byte-for-byte the ones the tracked curves were measured on. Re-deriving
    them independently is how a harness ends up measuring a different classifier than
    the recorded numbers came from.

    The holdout carve and the ``expanding_steps`` construction mirror
    ``joint_driver.run_joint_backtest`` exactly, including its degrade semantics: a
    classifier #1 failure short-circuits classifier #2 for that step, because that is
    what the tracked run did.

    Returns:
        pd.DataFrame, one row per step, with a ``t`` column and
        ``c{N}_lag{k}_state`` / ``c{N}_lag{k}_date`` per classifier per lag. Carries
        ``.attrs``: ``date_misalignments`` (list of dicts), ``n_degraded_1``,
        ``n_degraded_2``, ``max_lag``.
    """
    inputs = build_inputs(cfg) if inputs is None else inputs
    c2 = classifier2_config(cfg)

    dev_1, _ = split_by_holdout_boundary(inputs["features_1"], cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_2, _ = split_by_holdout_boundary(inputs["features_2"], cutoff=DEFAULT_HOLDOUT_CUTOFF)

    steps = list(expanding_steps(dev_1.index, min_train=inputs["min_train"]))
    if limit_steps is not None:
        steps = steps[:limit_steps]
    total = len(steps)
    log.info(
        "Terminal-month diagnostic: %d steps from %s to %s, lags 1..%d, both classifiers "
        "at their PINNED (K, lambda) — zero registry trials",
        total,
        steps[0][0].date() if steps else "n/a",
        steps[-1][0].date() if steps else "n/a",
        max_lag,
    )

    rows: list[dict[str, Any]] = []
    misalignments: list[dict[str, Any]] = []
    n_degraded_1 = 0
    n_degraded_2 = 0
    started = time.monotonic()

    for step_no, (t, train_index, _test_index) in enumerate(steps, start=1):
        if step_no % _PROGRESS_EVERY_STEPS == 0 or step_no == total:
            elapsed = time.monotonic() - started
            rate = step_no / elapsed if elapsed > 0 else 0.0
            log.info(
                "Terminal-month progress: %d/%d (%.0f%%) — %s — %.1fs elapsed, ~%.0fs remaining",
                step_no, total, 100.0 * step_no / total, t.date(), elapsed,
                (total - step_no) / rate if rate > 0 else 0.0,
            )

        row: dict[str, Any] = {"t": pd.Timestamp(t)}
        degraded = False

        try:
            states_1 = _refit_l1(dev_1.loc[train_index], cfg, frozen_features=inputs["frozen_1"])
        except _L2_DEGRADE_EXCEPTIONS as exc:
            log.warning("Step %s: classifier #1 refit degraded: %s", t, exc)
            degraded = True
            n_degraded_1 += 1
            row.update(_empty_lag_record(max_lag=max_lag, prefix="c1"))
        else:
            cols, misaligned = lag_record(states_1, train_index, max_lag=max_lag, prefix="c1")
            row.update(cols)
            if misaligned:
                misalignments.append(
                    {
                        "t": str(pd.Timestamp(t).date()),
                        "classifier": 1,
                        "fit_terminal_month": str(pd.Timestamp(states_1.index[-1]).date()),
                        "train_index_last": str(pd.Timestamp(train_index[-1]).date()),
                    }
                )

        if degraded:
            # Mirrors joint_driver: a classifier #1 failure short-circuits #2 for the
            # step, so the two harnesses visit the same (step, classifier) cells.
            row.update(_empty_lag_record(max_lag=max_lag, prefix="c2"))
        else:
            try:
                states_2 = _refit_classifier2(
                    dev_2.loc[train_index],
                    frozen_features=inputs["frozen_2"],
                    K=c2["K"], lam=c2["lam"], n_restarts=c2["n_restarts"],
                    sort_column=c2["sort_column"],
                )
            except _L2_DEGRADE_EXCEPTIONS as exc:
                log.warning("Step %s: classifier #2 refit degraded: %s", t, exc)
                n_degraded_2 += 1
                row.update(_empty_lag_record(max_lag=max_lag, prefix="c2"))
            else:
                cols, misaligned = lag_record(states_2, train_index, max_lag=max_lag, prefix="c2")
                row.update(cols)
                if misaligned:
                    misalignments.append(
                        {
                            "t": str(pd.Timestamp(t).date()),
                            "classifier": 2,
                            "fit_terminal_month": str(pd.Timestamp(states_2.index[-1]).date()),
                            "train_index_last": str(pd.Timestamp(train_index[-1]).date()),
                        }
                    )

        rows.append(row)

    labels_df = pd.DataFrame(rows)
    labels_df.attrs["date_misalignments"] = misalignments
    labels_df.attrs["n_degraded_1"] = n_degraded_1
    labels_df.attrs["n_degraded_2"] = n_degraded_2
    labels_df.attrs["max_lag"] = max_lag
    return labels_df


class AnchorError(AssertionError):
    """The k=1 column does not reproduce the tracked curve. No k>1 number is reportable."""


def assert_anchor(
    labels_df: pd.DataFrame,
    *,
    tracked_path: Path = DEFAULT_TRACKED_CURVE,
    require_full_counts: bool = True,
) -> dict[str, Any]:
    """The k=1 anchor — elementwise, no tolerance, before any k>1 number is reported.

    Asserts, for both classifiers, that the harness's step dates and its
    ``c{N}_lag1_state`` column reproduce ``tracked_path``'s index and ``state_{N}``
    column ELEMENTWISE, and (when ``require_full_counts``) that the derived churn is
    exactly 246 / 24.

    A failure here means the harness is fitting something other than the classifier
    every recorded number came from, so the raise says exactly that.

    Args:
        require_full_counts: False only for a ``--limit-steps`` smoke run, where the
            elementwise comparison is still made on the collected prefix but the
            full-window counts cannot exist. It never relaxes the elementwise check.
    """
    tracked = pd.read_parquet(tracked_path)
    harness_dates = pd.DatetimeIndex(pd.to_datetime(labels_df["t"]))
    tracked_dates = pd.DatetimeIndex(pd.to_datetime(tracked.index))

    if require_full_counts and not harness_dates.equals(tracked_dates):
        raise AnchorError(
            f"harness step index does not equal {tracked_path}'s index elementwise "
            f"({len(harness_dates)} vs {len(tracked_dates)} rows). The harness is "
            "walking a different window than the classifier every recorded number "
            "came from — NO k>1 number may be reported."
        )
    prefix = tracked_dates[: len(harness_dates)]
    if not harness_dates.equals(prefix):
        raise AnchorError(
            f"harness step dates diverge from {tracked_path}'s leading "
            f"{len(harness_dates)} dates — NO k>1 number may be reported."
        )

    result: dict[str, Any] = {}
    for n, key in _CLASSIFIER_KEYS.items():
        col = f"c{n}_lag1_state"
        tracked_col = tracked[f"state_{n}"].iloc[: len(labels_df)]
        mine = pd.to_numeric(labels_df[col], errors="coerce").to_numpy(dtype=float)
        theirs = pd.to_numeric(pd.Series(tracked_col.to_numpy()), errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(mine) & np.isnan(theirs)
        mismatch_mask = ~(both_nan | (mine == theirs))
        n_mismatched = int(mismatch_mask.sum())
        entry: dict[str, Any] = {
            "k1_matches_tracked_curve": n_mismatched == 0,
            "n_mismatched": n_mismatched,
            "n_compared": int(len(mine)),
            "tracked_source": str(tracked_path),
        }
        if n_mismatched:
            where = harness_dates[mismatch_mask][:5]
            raise AnchorError(
                f"{key}: c{n}_lag1_state differs from {tracked_path}'s state_{n} at "
                f"{n_mismatched} of {len(mine)} steps (first: "
                f"{[str(d.date()) for d in where]}). The harness is fitting something "
                "other than the classifier every recorded number came from — NO k>1 "
                "number may be reported. This is NOT a tolerance question: the DP "
                "decode is deterministic at random_state=42."
            )

        measured = state_change_count(labels_df[col])["n_changes"]
        entry["k1_n_changes"] = int(measured)
        if require_full_counts:
            expected = ANCHOR_N_CHANGES[key]
            if measured != expected:
                raise AnchorError(
                    f"{key}: k=1 churn is {measured}, expected {expected} from the "
                    "tracked curve. NO k>1 number may be reported."
                )
            entry["k1_n_changes_expected"] = expected
        result[key] = entry
    return result


def build_record(
    cfg: dict[str, Any],
    labels_df: pd.DataFrame,
    inputs: dict[str, Any],
    anchor: dict[str, Any],
    *,
    max_lag: int = DEFAULT_MAX_LAG,
) -> dict[str, Any]:
    """The persisted JSON: per classifier, its pinned config, λ/d, churn-by-lag, degrades.

    ``lambda_over_d`` is computed from the LIVE config and the RESOLVED frozen lists,
    never from the literals 1.0 / 2.0 — so a re-pin of either λ or a feature list makes
    the artifact-level test fail loudly instead of leaving a stale number in a record.
    """
    labeling = cfg.get("labeling", {})
    c2 = classifier2_config(cfg)
    churn = churn_by_lag(labels_df, max_lag=max_lag)
    revision = fixed_month_revision(labels_df, max_lag=max_lag)

    n_steps = int(len(labels_df))
    window = {
        "n_steps": n_steps,
        "first_date": str(pd.Timestamp(labels_df["t"].iloc[0]).date()) if n_steps else None,
        "last_date": str(pd.Timestamp(labels_df["t"].iloc[-1]).date()) if n_steps else None,
    }

    specs = {
        "classifier_1": {
            "K": int(labeling.get("K", 5)),
            "lambda": float(labeling.get("lambda", 52.0)),
            "frozen_columns": list(inputs["frozen_1"]),
            "n_degraded": int(labels_df.attrs.get("n_degraded_1", 0)),
        },
        "classifier_2": {
            "K": int(c2["K"]),
            "lambda": float(c2["lam"]),
            "frozen_columns": list(inputs["frozen_2"]),
            "n_degraded": int(labels_df.attrs.get("n_degraded_2", 0)),
        },
    }

    record: dict[str, Any] = {
        "generated_by": "scripts/terminal_month_diagnostic.py (plan 08-02)",
        "max_lag": int(max_lag),
        "window": window,
        "registry_trials_consumed": 0,
        "lambda_swept": False,
        "date_misalignments": list(labels_df.attrs.get("date_misalignments", [])),
    }
    for key, spec in specs.items():
        d = len(spec["frozen_columns"])
        record[key] = {
            "K": spec["K"],
            "lambda": spec["lambda"],
            "d": int(d),
            "lambda_over_d": (spec["lambda"] / d) if d else None,
            "frozen_columns": spec["frozen_columns"],
            "n_steps": n_steps,
            "n_degraded": spec["n_degraded"],
            "by_lag": churn[key],
            "fixed_month_revision": revision[key],
            "anchor": anchor.get(key, {}),
        }
    return record


def _format_table(record: dict[str, Any]) -> str:
    """The churn-vs-k table as it goes into 08-TRACK-A.md — every rate with its denominator."""
    lines = []
    for key in ("classifier_1", "classifier_2"):
        block = record[key]
        lines.append(
            f"{key}: K={block['K']} lambda={block['lambda']} d={block['d']} "
            f"lambda/d={block['lambda_over_d']:.2f} n_degraded={block['n_degraded']}"
        )
        lines.append("  k | n_changes / n_pairs | rate")
        for entry in block["by_lag"]:
            lines.append(
                f"  {entry['k']} | {entry['n_changes']} / {entry['n_pairs']} | "
                f"{100.0 * entry['rate']:.2f}%"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Terminal-month churn-vs-k diagnostic (plan 08-02). Zero registry trials."
    )
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--limit-steps", type=int, default=None,
        help="smoke run over the first N steps; the elementwise anchor still runs, the "
             "full-window 246/24 counts cannot and are skipped loudly.",
    )
    parser.add_argument("--tracked-curve", default=str(DEFAULT_TRACKED_CURVE))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    cfg = load_platform_config()
    inputs = build_inputs(cfg)
    labels_df = collect_terminal_labels(
        cfg, max_lag=args.max_lag, limit_steps=args.limit_steps, inputs=inputs
    )

    full_run = args.limit_steps is None
    if not full_run:
        log.warning(
            "--limit-steps %d: the elementwise k=1 anchor runs on the collected prefix, "
            "but the full-window 246/24 counts are NOT asserted. This run's k>1 numbers "
            "are a wiring check, NOT a measurement.",
            args.limit_steps,
        )
    anchor = assert_anchor(
        labels_df, tracked_path=Path(args.tracked_curve), require_full_counts=full_run
    )
    log.info("k=1 anchor holds elementwise for both classifiers: %s", json.dumps(anchor, default=str))

    record = build_record(cfg, labels_df, inputs, anchor, max_lag=args.max_lag)
    record["full_window_run"] = full_run

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / "terminal_month_labels.parquet"
    churn_path = out_dir / "terminal_month_churn.json"
    labels_df.to_parquet(labels_path)
    churn_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")

    log.info("wrote %s (%d x %d) and %s", labels_path, *labels_df.shape, churn_path)
    print(_format_table(record))  # noqa: T201 — first-class CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
