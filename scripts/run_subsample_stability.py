#!/usr/bin/env python
"""run_subsample_stability.py — design §4.4 criterion 3, RUN for both classifiers (plan 08-07).

Criterion 3, verbatim: *"re-estimate on subsamples (drop first decade / last decade /
block bootstrap); states persist with matched emission parameters (match via
Hungarian algorithm on distribution distances to defeat label switching). A
'regime' that evaporates when 2008-09 is dropped is an episode, not a regime."*

This project had never run it. This script does, under four schemes: drop first
decade, drop last decade, circular block bootstrap across the
``BLOCK_LENGTH_LADDER`` (6, 12, 24, 48 months), and leave-one-episode-out (the
design's own "drop 2008-09" example generalised, per ``08-RESEARCH.md`` §5.1).

**Criterion 3 is cheap. There has never been a cost reason it was not run.**
``08-RESEARCH.md`` §5.7 priced it at ~1,620 fits of ~0.111 s. Measured here the
fits take ~0.3-0.5 s each, so the full run is a few minutes across the worker
pool — still cheap, and still no reason.

**The reference is reproduced, not assumed.** Before any subsample is fit, both
full-sample labelings are refit in process and asserted ELEMENTWISE, with no
tolerance, against the tracked checkpoints (``regime_labels`` — 695 months, K=6;
``regime_labels_2`` — 696 months, K=5). ``fit_jump_model`` is deterministic at a
fixed seed and the DP decode is deterministic, so equality is the correct
assertion; a tolerance would hide a real divergence. A mismatch raises and
nothing else runs: a stability table measured against a baseline other than the
one every recorded number uses is worthless.

**Classifier #1 is fit on the TEN frozen columns, not ``lean_feature_set``'s 13.**
``diagnostics.label_regimes`` selects ``lean_feature_set(cfg)`` (13 columns), but
the tracked ``regime_labels`` checkpoint reproduces elementwise from
``evaluation/report.py::_reference_label_columns``' ten frozen columns — the list
every walk-forward number is measured on. The ten are used because that is what
reproduces the checkpoint; the fact is recorded in the JSON, not left implicit.

**The holdout is carved FIRST.** ``scripts/run_joint_lift.py::build_inputs``
derives the same frozen lists, but computes classifier #2's relative features and
the asset returns on the un-carved ``monthly_raw``. This script calls the same
functions in the same order after carving both frames at
``DEFAULT_HOLDOUT_CUTOFF``, so no post-2020-12-31 row reaches any computation.
Classifier #2's features are causal, so the carve changes nothing — and the
elementwise checkpoint identity is what proves that, rather than an assumption.

**No threshold, no verdict, no selection.** Every row carries its subsample
occupancy, its ``evaporated`` flag (built from occupancy alone — it OUTRANKS the
distance: ``_recompute_centroids`` freezes a zero-occupancy state at its previous
centroid, which then scores as perfectly stable), and its within-state
split-half null at the subsample's own per-state n. The only pass/fail rendered
is §4.4 AMENDMENT condition (i) — at least three temporally separated episodes —
on classifier #1's sub-floor state 0 alone. (K, lambda) are pinned; nothing here
selects, nothing here may re-pin, and nothing here writes to the trial registry
(asserted: ``total_trial_count()`` before == after).

**Occupancy, the null and episodes are keyed on the MATCHED PARTNER.**
``stability.run_stability`` keys them on the subsample state carrying the same id
as the reference state, which agrees with the partner only when the Hungarian
assignment is the identity. Under a non-identity assignment that would report the
occupancy of a different state than the one whose distance is on the row — and
could mask an evaporated partner. This runner therefore composes the same public
building blocks (``fit_for_stability``, ``match_states``, ``split_half_null``,
``state_episodes``, ``stability_row``) keyed on the partner; a test pins that it
reproduces ``run_stability``'s rows exactly whenever the assignment is the identity.

Usage::

    python scripts/run_subsample_stability.py --classifier 1 --schemes drop_first_decade \\
        --out-dir /tmp/stability_smoke
    python scripts/run_subsample_stability.py --classifier both \\
        --out-dir outputs/reports/platform/stability
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.evaluation.report import _reference_label_columns
from trading_crab_lib.platform.features.relative import add_relative_features
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.honesty.registry import total_trial_count
from trading_crab_lib.platform.labeling.classifier2 import (
    CLASSIFIER2_LABELS_CHECKPOINT,
    classifier2_config,
    freeze_classifier2_columns,
)
from trading_crab_lib.platform.labeling.stability import (
    BLOCK_LENGTH_LADDER,
    DEFAULT_STABILITY_SEED,
    EVAPORATED_OCCUPANCY_MONTHS,
    StabilityFit,
    _check_frozen_columns,
    fit_for_stability,
    match_states,
    scheme_circular_block_bootstrap,
    scheme_drop_first_decade,
    scheme_drop_last_decade,
    scheme_leave_one_episode_out,
    split_half_null,
    stability_row,
    state_episodes,
)
from trading_crab_lib.platform.taxonomy import lean_feature_set

log = logging.getLogger(__name__)

# ── constants ──

SCHEME_FAMILIES: tuple[str, ...] = (
    "drop_first_decade",
    "drop_last_decade",
    "circular_block_bootstrap",
    "leave_one_episode_out",
)

#: The base seed passed to ``fit_jump_model`` by every production fit
#: (``fit_jump_model``'s own default). Subsample fits use the SAME seed as the
#: reference so the only thing that differs between the two is the data.
FIT_RANDOM_STATE = 42

DEFAULT_N_BOOTSTRAP = 200
DEFAULT_NULL_REPS = 200
DEFAULT_OUT_DIR = Path("outputs/reports/platform/stability")

#: Classifier #1's canonicalization key when ``labeling.sort_column`` is absent
#: from config — ``canonicalize_states``' own default, which is what
#: ``diagnostics.label_regimes`` relies on (it passes no sort_column).
CLASSIFIER1_DEFAULT_SORT_COLUMN = "trailing_return_1m"
CLASSIFIER1_LABELS_CHECKPOINT = "regime_labels"

#: §4.4 AMENDMENT condition (i), verbatim. The only pass/fail this plan renders.
AMENDMENT_CONDITION_I_TEXT = (
    "(i) it recurs in at least three temporally separated episodes, so that removing "
    "any one leaves the state intact — criterion 3 applied directly, not by proxy"
)
AMENDMENT_CONDITION_I_MIN_EPISODES = 3  # quoted from the AMENDMENT, not chosen here
#: The sub-floor state invoking the recurrence exemption — the only row set the
#: condition binds on. Applying it to any other state would invent a threshold
#: by generalisation.
AMENDMENT_CONDITION_I_SCOPE = {"classifier": 1, "reference_state": 0}
#: Schemes whose subsample preserves calendar order, so "temporally separated
#: episode" is defined. A block-bootstrapped series is not in calendar order.
TEMPORAL_ORDER_SCHEMES = ("drop_first_decade", "drop_last_decade", "leave_one_episode_out")


# ── inputs and references ──


def build_frames(cfg: dict[str, Any]) -> dict[str, Any]:
    """Both classifiers' holdout-carved feature frames and frozen lists.

    Calls the functions ``scripts/run_joint_lift.py::build_inputs`` calls, in the
    same order, for the two frozen lists (``_reference_label_columns`` over the
    sorted lean set at ``first_decision = dev_features.index[min_train]``;
    ``freeze_classifier2_columns`` over ``add_relative_features``) — but carves
    ``monthly_raw`` at the holdout boundary BEFORE deriving anything from it.
    """
    cm = get_platform_checkpoint_manager()
    dev_features, _ = split_by_holdout_boundary(cm.load("monthly_features"), cutoff=DEFAULT_HOLDOUT_CUTOFF)
    dev_raw, _ = split_by_holdout_boundary(cm.load("monthly_raw"), cutoff=DEFAULT_HOLDOUT_CUTOFF)

    lean_cols = sorted(lean_feature_set(cfg) & set(dev_features.columns))
    min_train = int(cfg.get("backtest", {}).get("min_train_months", 120))
    first_decision = dev_features.index[min_train]
    frozen_1 = _reference_label_columns(dev_features, lean_cols, first_decision)

    relative = add_relative_features(dev_raw, cfg).reindex(dev_features.index)
    frozen_2 = freeze_classifier2_columns(relative, cfg, first_decision)
    return {
        "features_1": dev_features,
        "features_2": relative,
        "frozen_1": frozen_1,
        "frozen_2": frozen_2,
        "lean_cols": lean_cols,
        "first_decision": first_decision,
    }


def pinned_constants(cfg: dict[str, Any], classifier: int) -> dict[str, Any]:
    """(K, lam, n_restarts, sort_column, checkpoint) exactly as production reads them."""
    if classifier == 1:
        section = cfg.get("labeling", {}) or {}
        return {
            "K": int(section.get("K", 5)),
            "lam": float(section.get("lambda", 52.0)),
            "n_restarts": int(section.get("n_restarts", 10)),
            "sort_column": section.get("sort_column", CLASSIFIER1_DEFAULT_SORT_COLUMN),
            "checkpoint": CLASSIFIER1_LABELS_CHECKPOINT,
        }
    if classifier == 2:
        c2 = classifier2_config(cfg)
        return {
            "K": c2["K"],
            "lam": c2["lam"],
            "n_restarts": c2["n_restarts"],
            "sort_column": c2["sort_column"],
            "checkpoint": CLASSIFIER2_LABELS_CHECKPOINT,
        }
    raise ValueError(f"classifier must be 1 or 2, got {classifier!r}")


def build_reference(cfg: dict[str, Any], classifier: int, frames: dict[str, Any] | None = None) -> dict[str, Any]:
    """Refit one classifier's full-sample reference at its pinned constants.

    Returns:
        dict with ``fit`` (:class:`StabilityFit`), ``X`` (the complete-row frozen
        frame the fit saw — subsample positions index into it), ``frozen_columns``,
        ``classifier`` and the pinned ``K``/``lam``/``n_restarts``/``sort_column``/
        ``checkpoint``.
    """
    frames = frames if frames is not None else build_frames(cfg)
    pinned = pinned_constants(cfg, classifier)
    frozen = list(frames[f"frozen_{classifier}"])
    if pinned["sort_column"] not in frozen:
        raise ValueError(f"classifier #{classifier}: sort_column {pinned['sort_column']!r} not in {frozen}")
    X = frames[f"features_{classifier}"][frozen].dropna(axis=0, how="any")
    fit = fit_for_stability(
        X, K=pinned["K"], lam=pinned["lam"], n_restarts=pinned["n_restarts"],
        sort_column=pinned["sort_column"], random_state=FIT_RANDOM_STATE,
    )
    if len(fit.states) != len(X):
        raise RuntimeError("reference fit dropped rows from an already-complete frame")
    return {"classifier": classifier, "fit": fit, "X": X, "frozen_columns": frozen, **pinned}


def assert_reference_matches_checkpoint(
    reference: dict[str, Any], checkpoint_name: str, *, checkpoint_states: pd.Series | None = None
) -> dict[str, Any]:
    """Raise unless the reproduced states equal the tracked checkpoint ELEMENTWISE.

    No tolerance: the fit and the DP decode are deterministic at a fixed seed.
    ``checkpoint_states`` overrides the load (for tests of the failure path).

    Returns:
        the identity record: checkpoint, n_months, n_states, start, end, mismatches.

    Raises:
        AssertionError: on an index difference or any state mismatch, with the
            mismatch count and the first five mismatched dates in the message.
    """
    ours = reference["fit"].states
    if checkpoint_states is None:
        checkpoint_states = get_platform_checkpoint_manager().load(checkpoint_name)["state"]
    theirs = pd.Series(np.asarray(checkpoint_states, dtype=int), index=checkpoint_states.index)
    if not ours.index.equals(theirs.index):
        raise AssertionError(
            f"reference identity FAILED for {checkpoint_name}: index differs — reproduced "
            f"{len(ours)} months {ours.index.min()}..{ours.index.max()} vs checkpoint "
            f"{len(theirs)} months {theirs.index.min()}..{theirs.index.max()}. Every "
            "subsample would be compared against a baseline no recorded number uses."
        )
    diff = ours.to_numpy(dtype=int) != theirs.to_numpy(dtype=int)
    n_mismatch = int(diff.sum())
    if n_mismatch:
        first = [str(d.date()) for d in ours.index[diff][:5]]
        raise AssertionError(
            f"reference identity FAILED for {checkpoint_name}: {n_mismatch} mismatches of "
            f"{len(ours)} months (first five: {first}). The frozen list, K, lambda, "
            "n_restarts or sort_column has drifted from what produced the checkpoint."
        )
    record = {
        "checkpoint": checkpoint_name,
        "n_months": int(len(ours)),
        "n_states": int(len(np.unique(ours.to_numpy()))),
        "start": str(ours.index.min().date()),
        "end": str(ours.index.max().date()),
        "mismatches": 0,
        "identical": True,
    }
    log.info(
        "reference identity: classifier #%s reproduces %s elementwise — %d months, %d states, "
        "%s..%s, 0 mismatches",
        reference["classifier"], checkpoint_name, record["n_months"], record["n_states"],
        record["start"], record["end"],
    )
    return record


# ── one subsample, keyed on the matched partner ──


def summarize_subsample(
    reference_fit: StabilityFit,
    sub_fit: StabilityFit,
    *,
    reference_states_in_subsample: np.ndarray,
    classifier: int,
    scheme: str,
    null_reps: int,
    null_seed: int,
    extra: dict[str, Any] | None = None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """One row per reference state, plus the full K x K cost matrices in both unit spaces.

    Occupancy, the split-half null and the episode count are read off the MATCHED
    PARTNER — the subsample state the Hungarian assignment pairs with this
    reference state — so the distance, the occupancy and the null on a row all
    describe the same state. ``evaporated`` is built by ``stability_row`` from the
    partner's occupancy alone.

    ``reference_months_in_subsample`` counts subsample months the reference
    labeled this state (a property of the scheme mask, not the fit);
    ``partner_overlap_months`` counts those among the partner's months. For a
    degenerate leave-one-episode-out both are 0: whatever the partner is, it is
    built from months the reference never gave this state.

    **Companion distance in reference-SD units (``refscaled_*``).** The primary
    distance is plan 08-03's, in winsorized feature units, and is reported as
    built. It is, however, dominated by whichever columns have the largest raw
    scale — measured on the references: ``oil`` and ``cape_shiller`` carry 98.7%
    of classifier #1's squared between-centroid distance and ``rs_equities_bonds``
    100% of classifier #2's — so it is nearly blind to every other frozen column.
    ``08-RESEARCH.md`` §5.2 chose centroid distance on a benchmark with
    unit-scale columns. The companion divides BOTH de-standardized centroids (and
    the null's rows) by ONE common scale, the reference fit's winsorized standard
    deviation, which restores that premise without refitting anything: the two
    fits stay independent and only the yardstick is shared. It is a second
    reading, not a replacement; its own Hungarian assignment is reported beside
    the primary's, and so is the occupancy of its partner.
    """
    K = reference_fit.K
    match = match_states(reference_fit.centroids_destandardized, sub_fit.centroids_destandardized)
    ref_scale = reference_fit.params["scale"].reindex(reference_fit.columns)
    ref_z = reference_fit.centroids_destandardized / ref_scale.to_numpy()
    sub_z = sub_fit.centroids_destandardized.reindex(columns=reference_fit.columns) / ref_scale.to_numpy()
    match_z = match_states(ref_z, sub_z)
    rows_z = sub_fit.rows_destandardized.reindex(columns=reference_fit.columns) / ref_scale.to_numpy()
    sub_states = sub_fit.states.to_numpy(dtype=int)
    ref_in_sub = np.asarray(reference_states_in_subsample, dtype=int)
    if ref_in_sub.shape != sub_states.shape:
        raise ValueError("reference_states_in_subsample must align with the subsample fit's rows")
    episodes = state_episodes(sub_fit.states, n_states=K)
    n_sub = len(sub_states)
    extra = dict(extra or {})
    rows: list[dict] = []
    for state in range(K):
        partner = int(match["assignment"][state])
        partner_mask = sub_states == partner
        months = int(sub_fit.occupancy[partner])
        null = split_half_null(sub_fit.rows_destandardized.loc[partner_mask], n_reps=null_reps, seed=null_seed)
        row = stability_row(
            classifier=str(classifier),
            scheme=scheme,
            state=state,
            subsample_occupancy_months=months,
            subsample_occupancy_pct=(months / n_sub) if n_sub else float("nan"),
            matched_partner=partner,
            is_identity=match["is_identity"],
            matched_distance=match["matched_distance"][state],
            margin=match["margin"][state],
            split_half_null_median=null["median"],
            split_half_null_p10=null["p10"],
            split_half_null_p90=null["p90"],
            split_half_null_n=null["n"],
            n_episodes=episodes[partner]["n_episodes"],
            longest_episode=episodes[partner]["longest_episode"],
            block_length=extra.get("block_length"),
            n_seams=extra.get("n_seams"),
            seed=extra.get("seed"),
            degenerate=extra.get("degenerate"),
        )
        row["split_half_null_n_reps"] = int(null["n_reps"])
        null_z = split_half_null(rows_z.loc[partner_mask], n_reps=null_reps, seed=null_seed)
        partner_z = int(match_z["assignment"][state])
        months_z = int(sub_fit.occupancy[partner_z])
        row["refscaled_matched_distance"] = float(match_z["cost_matrix"][state, partner])
        row["refscaled_split_half_null_median"] = float(null_z["median"])
        row["refscaled_split_half_null_p10"] = float(null_z["p10"])
        row["refscaled_split_half_null_p90"] = float(null_z["p90"])
        row["refscaled_own_partner"] = partner_z
        row["refscaled_own_partner_agrees"] = bool(partner_z == partner)
        row["refscaled_own_is_identity"] = bool(match_z["is_identity"])
        row["refscaled_own_matched_distance"] = float(match_z["matched_distance"][state])
        row["refscaled_own_partner_occupancy_months"] = months_z
        row["refscaled_own_partner_evaporated"] = bool(months_z <= EVAPORATED_OCCUPANCY_MONTHS)
        row["n_subsample_months"] = int(n_sub)
        row["reference_months_in_subsample"] = int((ref_in_sub == state).sum())
        row["partner_overlap_months"] = int(((ref_in_sub == state) & partner_mask).sum())
        for key in ("scheme_family", "replicate", "dropped_state", "n_months_dropped",
                    "n_episodes_before", "dropped_start", "dropped_end"):
            if key in extra:
                row[key] = extra[key]
        rows.append(row)
    costs = {
        "winsorized": np.asarray(match["cost_matrix"], dtype=float),
        "reference_sd": np.asarray(match_z["cost_matrix"], dtype=float),
    }
    return rows, costs


# ── scheme construction ──


def _replicate_seed(seed: int, block_length: int, replicate: int) -> int:
    """Deterministic, distinct per (block length, replicate); carried on every row."""
    return int(np.random.SeedSequence([int(seed), int(block_length), int(replicate)]).generate_state(1)[0])


def build_tasks(
    reference: dict[str, Any], schemes: Sequence[str], *, seed: int, n_bootstrap: int
) -> list[dict[str, Any]]:
    """Every subsample to fit for one classifier: positions plus the row extras."""
    X = reference["X"]
    ref_states = reference["fit"].states
    index = X.index
    tasks: list[dict[str, Any]] = []
    unknown = [s for s in schemes if s not in SCHEME_FAMILIES]
    if unknown:
        raise ValueError(f"unknown scheme(s) {unknown}; choose from {SCHEME_FAMILIES}")
    for family in SCHEME_FAMILIES:  # fixed order regardless of how --schemes was spelled
        if family not in schemes:
            continue
        if family == "drop_first_decade":
            tasks.append({"scheme": family, "positions": scheme_drop_first_decade(index),
                          "extra": {"scheme_family": family}})
        elif family == "drop_last_decade":
            tasks.append({"scheme": family, "positions": scheme_drop_last_decade(index),
                          "extra": {"scheme_family": family}})
        elif family == "circular_block_bootstrap":
            for block_length in BLOCK_LENGTH_LADDER:
                for replicate in range(n_bootstrap):
                    rep_seed = _replicate_seed(seed, block_length, replicate)
                    positions, n_seams = scheme_circular_block_bootstrap(index, block_length, seed=rep_seed)
                    tasks.append({
                        "scheme": f"{family}_L{block_length}",
                        "positions": positions,
                        "extra": {"scheme_family": family, "block_length": int(block_length),
                                  "n_seams": int(n_seams), "seed": rep_seed, "replicate": replicate},
                    })
        elif family == "leave_one_episode_out":
            for state in range(reference["K"]):
                loo = scheme_leave_one_episode_out(ref_states, state, n_states=reference["K"])
                kept = loo["mask"]
                dropped = np.flatnonzero(~kept)
                tasks.append({
                    "scheme": f"{family}_s{state}",
                    "positions": loo["positions"],
                    "extra": {
                        "scheme_family": family, "dropped_state": state,
                        "degenerate": bool(loo["degenerate"]),
                        "n_months_dropped": int(loo["n_months_dropped"]),
                        "n_episodes_before": int(loo["n_episodes_before"]),
                        "dropped_start": str(index[dropped[0]].strftime("%Y-%m")),
                        "dropped_end": str(index[dropped[-1]].strftime("%Y-%m")),
                    },
                })
    return tasks


# ── execution (worker pool; results are order-deterministic) ──

_WORKER: dict[str, Any] = {}


def _init_worker(payload: dict[str, Any]) -> None:
    _WORKER.clear()
    _WORKER.update(payload)


def _run_task(task: dict[str, Any]) -> tuple[list[dict], dict[str, np.ndarray]]:
    ref = _WORKER["reference"]
    X = ref["X"]
    expected = list(ref["fit"].columns)
    positions = np.asarray(task["positions"], dtype=int)
    X_sub = X.iloc[positions]
    # Trap C: the frozen column list is held fixed at the full-sample list.
    _check_frozen_columns(list(X_sub.columns), expected, scheme=task["scheme"])
    _check_frozen_columns([c for c in X_sub.columns if X_sub[c].notna().any()], expected, scheme=task["scheme"])
    sub_fit = fit_for_stability(
        X_sub, K=ref["K"], lam=ref["lam"], n_restarts=ref["n_restarts"],
        sort_column=ref["sort_column"], random_state=FIT_RANDOM_STATE,
    )
    _check_frozen_columns(sub_fit.columns, expected, scheme=task["scheme"])
    if len(sub_fit.states) != len(positions):
        raise RuntimeError(f"{task['scheme']}: subsample fit dropped rows from a complete frame")
    return summarize_subsample(
        ref["fit"], sub_fit,
        reference_states_in_subsample=ref["fit"].states.to_numpy(dtype=int)[positions],
        classifier=ref["classifier"], scheme=task["scheme"],
        null_reps=_WORKER["null_reps"], null_seed=_WORKER["null_seed"], extra=task["extra"],
    )


def run_tasks(
    reference: dict[str, Any], tasks: list[dict[str, Any]], *, null_reps: int, null_seed: int, workers: int
) -> list[tuple[list[dict], dict[str, np.ndarray]]]:
    payload = {"reference": reference, "null_reps": null_reps, "null_seed": null_seed}
    if workers <= 1 or len(tasks) <= 1:
        _init_worker(payload)
        return [_run_task(t) for t in tasks]
    # SPAWN, not fork: forking after BLAS/OpenMP threads exist deadlocks the
    # workers (observed: 0% CPU, no progress). Children inherit the environment,
    # so pin their BLAS to one thread to avoid 4 x 4 oversubscription.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(payload,)) as pool:
        return list(pool.map(_run_task, tasks, chunksize=8))


# ── summary rows (the JSON record) ──


def _finite_or_none(x: Any) -> Any:
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _summary_from_single(row: dict) -> dict:
    """A contiguous-scheme row (one refit) in the record's schema."""
    out = {
        "classifier": int(row["classifier"]),
        "scheme": row["scheme_family"],
        "scheme_instance": row["scheme"],
        "reference_state": row["state"],
        "evaporated": row["evaporated"],
        "subsample_occupancy_months": row["subsample_occupancy_months"],
        "subsample_occupancy_pct": row["subsample_occupancy_pct"],
        "n_subsample_months": row["n_subsample_months"],
        "matched_partner": row["matched_partner"],
        "is_identity": row["is_identity"],
        "matched_distance": row["matched_distance"],
        "split_half_null": {
            "median": row["split_half_null_median"], "p10": row["split_half_null_p10"],
            "p90": row["split_half_null_p90"], "n": row["split_half_null_n"],
            "n_reps": row["split_half_null_n_reps"],
        },
        "margin": row["margin"],
        "subsample_episode_count": row["n_episodes"],
        "subsample_longest_episode": row["longest_episode"],
        "reference_months_in_subsample": row["reference_months_in_subsample"],
        "partner_overlap_months": row["partner_overlap_months"],
        "refscaled": {
            "matched_distance": row["refscaled_matched_distance"],
            "split_half_null": {
                "median": row["refscaled_split_half_null_median"], "p10": row["refscaled_split_half_null_p10"],
                "p90": row["refscaled_split_half_null_p90"], "n": row["split_half_null_n"],
            },
            "own_partner": row["refscaled_own_partner"],
            "own_partner_agrees": row["refscaled_own_partner_agrees"],
            "own_is_identity": row["refscaled_own_is_identity"],
            "own_matched_distance": row["refscaled_own_matched_distance"],
            "own_partner_occupancy_months": row["refscaled_own_partner_occupancy_months"],
            "own_partner_evaporated": row["refscaled_own_partner_evaporated"],
        },
    }
    if row["scheme_family"] == "leave_one_episode_out":
        out.update({
            "degenerate": row["degenerate"],
            "n_months_dropped": row["n_months_dropped"],
            "n_episodes_before": row["n_episodes_before"],
            "dropped_span": [row["dropped_start"], row["dropped_end"]],
        })
    return out


def _summary_from_bootstrap(rows: list[dict]) -> dict:
    """Aggregate B replicate rows for one (classifier, block length, state).

    ``evaporated`` is True if the state evaporated in ANY replicate; the count is
    carried. Distance, margin and null summaries are taken over the replicates in
    which the state did NOT evaporate, because an evaporated replicate's distance
    is a frozen centroid's and would flatter the median. Occupancy and episode
    count are taken over ALL replicates, zeros included. The null's ``n`` is the
    same median occupancy, so the null is at the subsample's own n by
    construction; per-replicate nulls, each at its own n, are in the parquet.
    """
    first = rows[0]
    occ = np.array([r["subsample_occupancy_months"] for r in rows], dtype=float)
    evap = np.array([r["evaporated"] for r in rows], dtype=bool)
    live = [r for r in rows if not r["evaporated"]]
    dist = np.array([r["matched_distance"] for r in live], dtype=float)
    marg = np.array([r["margin"] for r in live], dtype=float)
    null_med = np.array([r["split_half_null_median"] for r in live], dtype=float)
    seams = np.array([r["n_seams"] for r in rows], dtype=float)
    partners = [r["matched_partner"] for r in rows]
    identity = np.array([r["is_identity"] for r in rows], dtype=bool)
    episodes = np.array([r["n_episodes"] for r in rows], dtype=float)
    dist_z = np.array([r["refscaled_matched_distance"] for r in live], dtype=float)
    null_z = np.array([r["refscaled_split_half_null_median"] for r in live], dtype=float)
    agree_z = np.array([r["refscaled_own_partner_agrees"] for r in rows], dtype=bool)
    evap_z = np.array([r["refscaled_own_partner_evaporated"] for r in rows], dtype=bool)

    def q(a: np.ndarray, p: float) -> float:
        a = a[np.isfinite(a)]
        return float(np.quantile(a, p)) if a.size else float("nan")

    median_occ = float(np.median(occ))
    partner_mode = int(pd.Series(partners).mode().iloc[0])
    return {
        "classifier": int(first["classifier"]),
        "scheme": "circular_block_bootstrap",
        "scheme_instance": first["scheme"],
        "block_length": int(first["block_length"]),
        "n_replicates": len(rows),
        "reference_state": first["state"],
        "evaporated": bool(evap.any()),
        "n_replicates_evaporated": int(evap.sum()),
        "subsample_occupancy_months": median_occ,
        "subsample_occupancy_months_p10": q(occ, 0.10),
        "subsample_occupancy_months_p90": q(occ, 0.90),
        "subsample_occupancy_months_min": float(occ.min()),
        "subsample_occupancy_pct": median_occ / float(first["n_subsample_months"]),
        "n_subsample_months": int(first["n_subsample_months"]),
        "matched_partner": partner_mode,
        "matched_partner_mode_fraction": float(np.mean(np.array(partners) == partner_mode)),
        "is_identity": bool(identity.all()),
        "identity_fraction": float(identity.mean()),
        "matched_distance": q(dist, 0.5),
        "matched_distance_p10": q(dist, 0.10),
        "matched_distance_p90": q(dist, 0.90),
        "split_half_null": {
            "median": q(null_med, 0.5), "p10": q(null_med, 0.10), "p90": q(null_med, 0.90),
            "n": median_occ, "n_reps": int(first["split_half_null_n_reps"]),
            "aggregation": "quantiles across non-evaporated replicates of each replicate's own-n null median",
        },
        "margin": q(marg, 0.5),
        "subsample_episode_count": float(np.median(episodes)),
        "n_seams": {"median": float(np.median(seams)), "min": int(seams.min()), "max": int(seams.max())},
        "reference_months_in_subsample": float(np.median([r["reference_months_in_subsample"] for r in rows])),
        "partner_overlap_months": float(np.median([r["partner_overlap_months"] for r in rows])),
        "refscaled": {
            "matched_distance": q(dist_z, 0.5),
            "matched_distance_p10": q(dist_z, 0.10),
            "matched_distance_p90": q(dist_z, 0.90),
            "split_half_null": {"median": q(null_z, 0.5), "p10": q(null_z, 0.10), "p90": q(null_z, 0.90),
                                "n": median_occ},
            "own_partner_agrees_fraction": float(agree_z.mean()),
            "own_partner_evaporated_replicates": int(evap_z.sum()),
        },
    }


def _apply_condition_i(summary: dict) -> None:
    """Attach §4.4 AMENDMENT condition (i) to classifier #1 state 0 rows ONLY."""
    if (summary["classifier"], summary["reference_state"]) != (
        AMENDMENT_CONDITION_I_SCOPE["classifier"], AMENDMENT_CONDITION_I_SCOPE["reference_state"]
    ):
        return
    if summary["scheme"] in TEMPORAL_ORDER_SCHEMES:
        holds = summary["subsample_episode_count"] >= AMENDMENT_CONDITION_I_MIN_EPISODES
        summary["amendment_condition_i"] = {
            "holds": bool(holds),
            "subsample_episode_count": summary["subsample_episode_count"],
            "min_episodes_quoted": AMENDMENT_CONDITION_I_MIN_EPISODES,
            "condition": AMENDMENT_CONDITION_I_TEXT,
        }
    else:
        summary["amendment_condition_i"] = {
            "holds": None,
            "condition": AMENDMENT_CONDITION_I_TEXT,
            "not_evaluated_because": (
                "a block-bootstrapped series is not in calendar order, so 'temporally separated "
                "episodes' is undefined on it; its episode count counts resampling seams"
            ),
        }


def build_summary_rows(detail_rows: list[dict]) -> list[dict]:
    """One record row per (classifier, scheme, reference state) — the JSON ``rows``."""
    summaries: list[dict] = []
    boot: dict[tuple, list[dict]] = {}
    for row in detail_rows:
        family = row["scheme_family"]
        if family in ("drop_first_decade", "drop_last_decade"):
            summaries.append(_summary_from_single(row))
        elif family == "leave_one_episode_out":
            if row["state"] == row["dropped_state"]:  # the row that scheme exists for
                summaries.append(_summary_from_single(row))
        elif family == "circular_block_bootstrap":
            boot.setdefault((row["classifier"], row["block_length"], row["state"]), []).append(row)
    for key in sorted(boot):
        summaries.append(_summary_from_bootstrap(boot[key]))
    order = {f: i for i, f in enumerate(SCHEME_FAMILIES)}
    summaries.sort(key=lambda s: (s["classifier"], order[s["scheme"]], s.get("block_length") or 0, s["reference_state"]))
    for s in summaries:
        _apply_condition_i(s)
    return summaries


def distance_scale_dominance(fit: StabilityFit) -> dict[str, Any]:
    """Each column's share of the reference's total squared between-centroid distance.

    In winsorized units (the primary distance) and in reference-SD units (the
    companion). A column with ~0 share in winsorized units is one the primary
    distance cannot see.
    """
    cols = list(fit.columns)
    scale = fit.params["scale"].reindex(cols).to_numpy()
    out: dict[str, Any] = {"reference_winsorized_sd": dict(zip(cols, scale.tolist()))}
    for units, C in (("winsorized", fit.centroids_destandardized.to_numpy()),
                     ("reference_sd", fit.centroids_destandardized.to_numpy() / scale)):
        d2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=(0, 1))
        out[f"share_{units}"] = dict(zip(cols, (d2 / d2.sum()).tolist()))
    return out


def full_sample_table(fit: StabilityFit) -> dict[str, Any]:
    """Per-state occupancy and episode spans of the reference labeling."""
    eps = state_episodes(fit.states, n_states=fit.K)
    n = len(fit.states)
    return {
        str(s): {
            "occupancy_months": int(fit.occupancy[s]),
            "occupancy_pct": float(fit.occupancy[s] / n),
            "n_episodes": eps[s]["n_episodes"],
            "longest_episode": eps[s]["longest_episode"],
            "episodes": [
                {"start": e["start_label"].strftime("%Y-%m"), "end": e["end_label"].strftime("%Y-%m"),
                 "length": e["length"]}
                for e in eps[s]["episodes"]
            ],
        }
        for s in range(fit.K)
    }


# ── artifacts ──


def _detail_frame(detail_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(detail_rows).rename(columns={"state": "reference_state"})
    df["classifier"] = df["classifier"].astype(int)
    for col in ("block_length", "n_seams", "seed", "replicate", "dropped_state",
                "n_months_dropped", "n_episodes_before"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    if "degenerate" in df.columns:
        df["degenerate"] = df["degenerate"].astype("boolean")
    return df


def _cost_frame(entries: list[tuple[int, dict, dict[str, np.ndarray]]]) -> pd.DataFrame:
    recs = []
    for classifier, task, costs in entries:
        extra = task["extra"]
        for units, cost in costs.items():
            K = cost.shape[0]
            for i in range(K):
                for j in range(K):
                    recs.append({
                        "classifier": classifier, "scheme_family": extra["scheme_family"],
                        "scheme": task["scheme"], "block_length": extra.get("block_length"),
                        "replicate": extra.get("replicate"), "units": units, "reference_state": i,
                        "subsample_state": j, "distance": float(cost[i, j]),
                    })
    df = pd.DataFrame(recs)
    for col in ("block_length", "replicate"):
        df[col] = df[col].astype("Int64")
    return df


def _json_clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return _finite_or_none(o)


def _print_rows(summaries: list[dict]) -> None:
    header = (f"{'c':>1} {'scheme':<28} {'st':>2} {'ptr':>3} {'id':>2} {'dist':>7} {'null':>7} "
              f"{'margin':>6} {'occ':>6} {'eps':>5} {'evap':>5} {'rsd':>6} {'rnull':>6}")
    print(header)  # noqa: T201 — CLI run output
    for s in summaries:
        null = s["split_half_null"]["median"]
        print(  # noqa: T201
            f"{s['classifier']:>1} {s['scheme_instance']:<28} {s['reference_state']:>2} {s['matched_partner']:>3} "
            f"{'Y' if s['is_identity'] else 'N':>2} {s['matched_distance']:>7.3f} "
            f"{(null if null is not None else float('nan')):>7.3f} {s['margin']:>6.3f} "
            f"{s['subsample_occupancy_months']:>6.1f} {s['subsample_episode_count']:>5.1f} "
            f"{str(s['evaporated']):>5} {s['refscaled']['matched_distance']:>6.3f} "
            f"{(s['refscaled']['split_half_null']['median'] or float('nan')):>6.3f}"
        )


# ── main ──


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--classifier", choices=("1", "2", "both"), default="both")
    parser.add_argument("--schemes", nargs="+", default=list(SCHEME_FAMILIES), choices=SCHEME_FAMILIES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_STABILITY_SEED)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--null-reps", type=int, default=DEFAULT_NULL_REPS)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    t0 = time.monotonic()
    trials_before = total_trial_count()
    cfg = load_platform_config()
    frames = build_frames(cfg)
    classifiers = (1, 2) if args.classifier == "both" else (int(args.classifier),)

    # Both references are reproduced and asserted BEFORE any subsample is fit.
    references: dict[int, dict[str, Any]] = {}
    identities: dict[int, dict[str, Any]] = {}
    for c in (1, 2):
        references[c] = build_reference(cfg, c, frames)
        identities[c] = assert_reference_matches_checkpoint(references[c], references[c]["checkpoint"])

    detail_rows: list[dict] = []
    cost_entries: list[tuple[int, dict, dict[str, np.ndarray]]] = []
    fit_counts: dict[int, int] = {}
    for c in classifiers:
        ref = references[c]
        tasks = build_tasks(ref, args.schemes, seed=args.seed, n_bootstrap=args.n_bootstrap)
        fit_counts[c] = len(tasks)
        log.info("classifier #%d: %d subsample fits across %s", c, len(tasks), args.schemes)
        results = run_tasks(ref, tasks, null_reps=args.null_reps, null_seed=args.seed, workers=args.workers)
        for task, (rows, cost) in zip(tasks, results):
            detail_rows.extend(rows)
            cost_entries.append((c, task, cost))

    summaries = build_summary_rows(detail_rows)
    trials_after = total_trial_count()
    if trials_after != trials_before:
        raise RuntimeError(
            f"trial registry changed during a criterion-3 run ({trials_before} -> {trials_after}); "
            "re-fitting at pinned (K, lambda) must cost zero rows"
        )
    runtime = time.monotonic() - t0

    record = {
        "plan": "08-07",
        "criterion": "design §4.4 criterion 3 (subsample stability), PER-06",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout_cutoff": str(DEFAULT_HOLDOUT_CUTOFF),
        "seed": args.seed,
        "fit_random_state": FIT_RANDOM_STATE,
        "n_bootstrap": args.n_bootstrap,
        "n_null_reps": args.null_reps,
        "block_length_ladder": list(BLOCK_LENGTH_LADDER),
        "politis_white_anchor_months": {
            "value": round(float(len(references[1]["X"]) ** (1 / 3)), 2),
            "role": "quoted as the variance-estimation anchor n^(1/3); NOT obeyed",
        },
        "replicate_seed_rule": "np.random.SeedSequence([seed, block_length, replicate]).generate_state(1)[0]",
        "schemes_run": [s for s in SCHEME_FAMILIES if s in args.schemes],
        "runtime_seconds": round(runtime, 1),
        "workers": args.workers,
        "registry_trial_count": {"before": trials_before, "after": trials_after},
        "distance": "Euclidean distance between de-standardized (winsorized-unit) centroids",
        "keyed_on": "occupancy, null and episodes are read off the Hungarian-matched partner",
        "scope_note": (
            "No persistence cutoff exists in this record and none was invented. The only "
            "pass/fail is §4.4 AMENDMENT condition (i), attached to classifier #1 state 0 rows only."
        ),
        "classifiers": {},
        "rows": summaries,
    }
    for c in (1, 2):
        ref = references[c]
        record["classifiers"][str(c)] = {
            "frozen_columns": ref["frozen_columns"],
            "K": ref["K"], "lambda": ref["lam"], "n_restarts": ref["n_restarts"],
            "sort_column": ref["sort_column"],
            "reference_identity": identities[c],
            "full_sample": full_sample_table(ref["fit"]),
            "distance_scale_dominance": distance_scale_dominance(ref["fit"]),
            "reference_centroids_winsorized": {
                str(k): dict(zip(ref["fit"].columns, ref["fit"].centroids_destandardized.iloc[k].tolist()))
                for k in range(ref["K"])
            },
            "n_subsample_fits": fit_counts.get(c, 0),
        }
    record["classifiers"]["1"]["feature_set_note"] = (
        f"fit on the {len(frames['frozen_1'])} frozen columns from _reference_label_columns (first_decision "
        f"{frames['first_decision'].date()}), not lean_feature_set's {len(frames['lean_cols'])}: the ten are what "
        "reproduce the tracked regime_labels checkpoint elementwise"
    )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _detail_frame(detail_rows).to_parquet(out_dir / "stability_rows.parquet", index=False)
    _cost_frame(cost_entries).to_parquet(out_dir / "stability_cost_matrices.parquet", index=False)
    (out_dir / "stability_record.json").write_text(json.dumps(_json_clean(record), indent=2) + "\n")

    _print_rows(summaries)
    print(  # noqa: T201
        f"\n{len(detail_rows)} detail rows, {len(summaries)} record rows, {sum(fit_counts.values())} subsample "
        f"fits, {runtime:.1f}s; registry {trials_before} -> {trials_after}; artifacts in {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
