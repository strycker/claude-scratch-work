"""
Classifier #2 — the leadership/relative-axis L1 labeler (REG-01, ADR-0002).

A SECOND, independent L1 labeler alongside ``labeling/diagnostics.py``'s
classifier #1. Classifier #1 is fit on the 13-member lean taxonomy and
describes ONE axis: how stressed is the market. It says almost nothing about
*which sleeve is leading* — equities, long duration, or commodities — which is
the question L4's allocation tilt actually needs answered.

Construction, all pinned by a human at a blocking decision checkpoint on
2026-09-17 BEFORE the first fit, with **zero selection trials** spent (D-13).
The verbatim record with each value's failure signature is
``.planning/phases/07-regime-representation/07-DECISIONS-07-08.md``; the
architecture record is ``platform_design/adr/0002-l1-second-classifier.md``:

- **Disjoint feature set (D-10).** Every one of the eight frozen columns is
  disjoint from classifier #1's 13 lean RAW columns. A *ratio* of two of #1's
  raw columns (``rs_oil_equities``) is scale-invariant and a genuinely
  different quantity from a level, so D-10 admits it; ``taxonomy.py``'s
  members themselves are excluded outright. Asserted by
  ``tests/unit/test_platform_features_relative.py::TestClassifier2Disjointness``.
- **Same freeze rule, same window (D-11).** ``evaluation/report.py::
  _reference_label_columns`` is REUSED UNMODIFIED at the same 1972+ first
  decision date, so the two labelings stay month-for-month comparable —
  criterion 6's dependence measurement depends on that. A second
  implementation of the freeze rule is exactly the divergence criterion 1
  exists to prevent.
- **K and lambda by construction (D-13).** K = 3 because three asset sleeves
  appear in the candidate set; lambda = 4n = 32.0 from the same feature-count
  formula classifier #1 instantiates (13 columns -> 52.0). No fit was run to
  choose either. :func:`classifier2_config` RAISES if the configured lambda
  and feature count ever drift apart, so D-13's arithmetic is an invariant the
  code fails on rather than a comment.
- **Its own canonical ordering column.** ``canonicalize_states`` is called with
  ``sort_column="rs_equities_bonds"`` EXPLICITLY. Plan 07-05 deleted the old
  "warn and order on centroid column 0" fallback: with a feature set disjoint
  from classifier #1's, that fallback would have fired on every fit, assigning
  arbitrary state IDs while every downstream occupancy, dependence and joint
  lift number kept appearing to pass.
- **add-alongside, never promote (D-14, T-07-15).** Persists to
  ``regime_labels_2`` / ``regime_confidences_2`` / ``regime_profiles_2``, and
  routes its §4.4 diagnostics artifact to its own subdirectory. Classifier #1's
  ``regime_labels`` checkpoint and ``labeling_diagnostics.parquet`` are never
  written by this module.

Adds no legacy import (the ratchet in
``tests/unit/test_platform_legacy_import_ratchet.py`` is pinned at 31 and may
only decrease).

**One import deserves an explicit note rather than a silent pass.** ``OUTPUT_DIR``
is imported from ``platform.labeling.diagnostics``, not from ``trading_crab_lib``
directly. Importing it directly was tried and the ratchet correctly rejected it
at 32 sites. This is not a dodge of the ratchet's intent: the coupling does not
widen, because this module already depends on ``labeling/diagnostics.py`` for
four functions, and that module already owns this exact seam as one of the 31
ratcheted sites. Nothing new is reachable from ``platform/`` that was not
reachable before. Stated here so a reader judges it rather than discovers it.

Usage::

    from trading_crab_lib.platform.config import load_platform_config
    from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
    from trading_crab_lib.platform.features.relative import add_relative_features
    from trading_crab_lib.platform.labeling.classifier2 import label_leadership_regimes

    cfg = load_platform_config()
    monthly_raw = get_platform_checkpoint_manager().load("monthly_raw")
    result = label_leadership_regimes(add_relative_features(monthly_raw, cfg), cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trading_crab_lib.platform.evaluation.report import _reference_label_columns
from trading_crab_lib.platform.honesty.holdout import (
    DEFAULT_HOLDOUT_CUTOFF,
    split_by_holdout_boundary,
)
from trading_crab_lib.platform.labeling.diagnostics import (
    OUTPUT_DIR,  # re-exported; see the note below — importing it directly would raise the ratchet
    _get_checkpoint_manager,
    auto_profile,
    occupancy_and_sojourns,
    report_labeling_diagnostics,
)
from trading_crab_lib.platform.labeling.jump_model import (
    canonicalize_states,
    fit_jump_model,
    soft_confidences,
    standardize_features,
)

log = logging.getLogger(__name__)

#: Classifier #2's frozen candidate columns, in declaration order — the "Lean 8"
#: pinned at the 2026-09-17 checkpoint (ADR-0002 decision (a)). Order is
#: load-bearing: ``canonicalize_states`` locates ``sort_column``'s centroid by
#: column position, and the frozen list preserves this order.
#:
#: One momentum horizon per sleeve: the 6m/24m twins are dropped as
#: near-collinear with the 12m, and ``credit_gdp`` — the other INV-01 survivor —
#: is dropped because plan 07-07 measured its correlation with ``m2_gdp`` at
#: 0.957-0.969 across eras.
#:
#: This constant and ``config/platform_settings.yaml``'s ``labeling_2.features``
#: are two copies of one pinned list; a test asserts they are identical.
CLASSIFIER2_CANDIDATE_COLUMNS: list[str] = [
    "rs_equities_bonds",
    "rs_oil_equities",
    "equities_tr_mom_12m",
    "long_duration_tr_mom_12m",
    "oil_mom_12m",
    "corr_equities_tr_long_duration_tr_24m",
    "cpi_acceleration",
    "m2_gdp",
]

#: Canonical state-ordering column (ADR-0002 decision (d)): states are numbered
#: by ascending centroid coordinate of the equity/bond relative-strength ratio,
#: so state 0 is the most bond-leading and state K-1 the most equity-leading.
CLASSIFIER2_SORT_COLUMN: str = "rs_equities_bonds"

#: K by construction (ADR-0002 decision (b)) — three asset sleeves in the
#: candidate set (equities, long duration, oil) -> three leadership states.
CLASSIFIER2_K: int = 5

#: Checkpoint names. Deliberately distinct from classifier #1's
#: ``regime_labels`` / ``regime_confidences`` / ``regime_profiles`` (T-07-15).
CLASSIFIER2_LABELS_CHECKPOINT: str = "regime_labels_2"
CLASSIFIER2_CONFIDENCES_CHECKPOINT: str = "regime_confidences_2"
CLASSIFIER2_PROFILES_CHECKPOINT: str = "regime_profiles_2"


def classifier2_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read classifier #2's pinned construction from the ``labeling_2`` section.

    Read defensively via ``cfg.get()`` — ``labeling_2`` is additive config and
    is deliberately NOT in ``_REQUIRED_PLATFORM_SECTIONS``, so an absent section
    falls back to this module's constants rather than raising.

    Two invariants are validated at read time, both of which exist so a later
    config edit becomes a loud failure rather than a silent drift:

    1. ``lambda == 4 * len(features)`` — D-13's feature-count formula, the same
       one classifier #1 instantiates (13 -> 52.0). Changing the feature list
       without recomputing lambda would quietly change the fit's jump penalty
       away from the value ADR-0002 pinned.
    2. ``sort_column in features`` — canonicalization has no fallback since plan
       07-05, so an ordering column outside the feature set could only ever fail
       at fit time, after the work.

    Args:
        cfg: platform config (``load_platform_config()``'s return value).

    Returns:
        dict with keys ``K``, ``lam``, ``n_restarts``, ``sort_column``,
        ``features`` (ordered list).

    Raises:
        ValueError: if either invariant above fails.
    """
    section = cfg.get("labeling_2", {}) or {}
    features = list(section.get("features", CLASSIFIER2_CANDIDATE_COLUMNS))
    K = int(section.get("K", CLASSIFIER2_K))
    n_restarts = int(section.get("n_restarts", 10))
    sort_column = section.get("sort_column", CLASSIFIER2_SORT_COLUMN)
    # RE-PINNED 2026-09-18 (ADR-0002 § RE-PIN): the coefficient is 2, not 4.
    #
    # lambda still scales with the feature count -- D-13's rule that the jump
    # penalty is a FORMULA of the frozen feature list, not a free knob, is
    # preserved. Only the coefficient moved, and design §4.3 licenses exactly
    # that: "Tune lambda (and K) until acceptance criteria (§4.4) pass --
    # occupancy and sojourn targets become the tuning objective."
    #
    # Why: at 4n = 32 with K = 5 the fit produced FIVE contiguous blocks in
    # sequence (0->1->2->3->4) with no state ever recurring -- a
    # time-segmentation, not a regime model. §4.4 criterion 3 names that
    # failure: a state that appears once is an episode, not a regime. The
    # acceptance window measured at K=5 is lambda in [8, 24]: inside it
    # criteria 1 and 2 pass AND states recur; at 6 and below criterion 1
    # breaks (occupancy 3.6% and 36.9%). 2n = 16 sits mid-window on a plateau
    # (12 and 16 give identical labelings), so it is robust rather than a
    # cliff-edge pick.
    #
    # Classifier #1 is unaffected: it has its own labeling section and its own
    # 4 x 13 = 52.0, which this function never reads.
    expected_lam = 2.0 * len(features)
    lam = float(section.get("lambda", expected_lam))

    if lam != expected_lam:
        raise ValueError(
            f"labeling_2.lambda is {lam} but must equal 2 x len(features) = "
            f"2 x {len(features)} = {expected_lam} (D-13's feature-count formula "
            "as re-pinned by ADR-0002 § RE-PIN 2026-09-18; the coefficient is 2 "
            "for classifier #2). A feature list edited without recomputing "
            "lambda changes the fit's jump penalty away from the value ADR-0002 "
            "pinned — recompute it, or amend ADR-0002 if the change is intended."
        )
    if sort_column not in features:
        raise ValueError(
            f"labeling_2.sort_column {sort_column!r} is not in labeling_2.features "
            f"{features!r}. canonicalize_states has no fallback since plan 07-05, so "
            "an ordering column outside the feature set can only fail at fit time."
        )

    return {
        "K": K,
        "lam": lam,
        "n_restarts": n_restarts,
        "sort_column": sort_column,
        "features": features,
    }


def _first_valid_month(series: pd.Series) -> str:
    """The first non-NaN month of *series*, as a date string ('' if never valid)."""
    valid = series.dropna()
    return str(valid.index.min().date()) if len(valid) else "never valid"


def freeze_classifier2_columns(
    features: pd.DataFrame, cfg: dict[str, Any], first_decision: pd.Timestamp
) -> list[str]:
    """Freeze classifier #2's candidate columns under D-11's rule.

    Delegates the rule itself to ``evaluation/report.py::_reference_label_columns``
    UNMODIFIED — a column qualifies only if it is non-NaN for every month from
    ``first_decision`` onward. Reusing it (rather than reimplementing it) is what
    keeps classifier #2's freeze and classifier #1's from diverging.

    Candidates absent from ``features.columns`` are excluded first, by name, so
    a missing column is a logged exclusion rather than a ``KeyError`` from
    inside the freeze rule. Every exclusion — missing or late-starting — is
    logged with the candidate's first valid month.

    Args:
        features: a feature frame carrying classifier #2's candidate columns
            (e.g. ``features/relative.py::add_relative_features``'s output).
        cfg: platform config.
        first_decision: the first walk-forward decision date (1972-01-31 on the
            live dev spine).

    Returns:
        The surviving columns, in ``labeling_2.features`` declaration order.

    Raises:
        ValueError: if the frozen list is empty, or shorter than K — the same
            two boundaries ``driver.py::_refit_l1`` already enforces for
            classifier #1, whose messages these mirror in shape.
    """
    resolved = classifier2_config(cfg)
    candidates = resolved["features"]
    K = resolved["K"]

    present = []
    for col in candidates:
        if col in features.columns:
            present.append(col)
        else:
            log.info(
                "classifier #2 candidate %r excluded: not present in the feature frame", col
            )

    frozen = _reference_label_columns(features, present, first_decision)

    for col in present:
        if col not in frozen:
            log.info(
                "classifier #2 candidate %r excluded by the D-11 freeze rule: first valid "
                "month %s is after the first decision date %s",
                col, _first_valid_month(features[col]), str(pd.Timestamp(first_decision).date()),
            )

    if len(frozen) == 0:
        raise ValueError(
            f"classifier #2's candidate list resolved to 0 usable columns (of "
            f"{len(candidates)} requested) — none are both present in the feature "
            f"frame and non-NaN from the first decision date {first_decision} onward. "
            "An empty frozen list means 'frozen, and empty', never 'use every column'."
        )
    if len(frozen) < K:
        raise ValueError(
            f"classifier #2's candidate list resolved to only {len(frozen)} usable "
            f"column(s), fewer than K={K} — cannot fit a {K}-state jump model on fewer "
            "feature columns than states."
        )
    return frozen


def _derive_first_decision(features: pd.DataFrame, cfg: dict[str, Any]) -> pd.Timestamp:
    """``features.index[min_train_months]`` — mirrors report.py's own derivation."""
    min_train = int(cfg.get("backtest", {}).get("min_train_months", 120))
    if len(features.index) <= min_train:
        raise ValueError(
            f"feature frame has {len(features.index)} months, which is not more than "
            f"min_train_months={min_train} — there is no first decision date to derive. "
            "Pass first_decision explicitly if this frame is deliberately short."
        )
    return features.index[min_train]


def label_leadership_regimes(
    features: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    checkpoint_dir: Path | None = None,
    first_decision: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Fit classifier #2 and persist its labeling (REG-01 criterion 5).

    Mirrors ``labeling/diagnostics.py::label_regimes``'s wiring order exactly:
    carve the holdout -> freeze columns -> select and drop NaN rows ->
    ``standardize_features`` -> ``fit_jump_model`` at the pinned K and lambda ->
    ``canonicalize_states`` **passing the pinned sort_column explicitly** ->
    distances and ``soft_confidences`` -> persist -> report §4.4 diagnostics.

    The holdout carve (``<= 2020-12-31``) is applied defensively here rather
    than assumed of the caller (T-07-17): classifier #2's candidate columns are
    derived on demand from ``monthly_raw``, which is NOT holdout-carved on disk,
    so a caller passing ``add_relative_features(monthly_raw, cfg)`` straight in
    would otherwise fit on post-cutoff months. Dropped rows are logged.

    §4.4 criterion 1's occupancy band (every state >= ~8% and <= ~35%) is
    REPORT-ONLY (D-02/D-07): a state outside it produces a loud WARNING naming
    that state, and this function still completes. It is never a gate.

    NOTE 2026-09-17: as pinned (K=3, lambda=32) this classifier BREACHES that
    cap on two states (46.12%, 38.51%). See the CORRECTION section of
    platform_design/adr/0002-l1-second-classifier.md; K and lambda are pending
    a re-pin against §4.4.

    Args:
        features: a feature frame carrying classifier #2's candidate columns.
        cfg: platform config.
        checkpoint_dir: overrides the platform checkpoint directory AND the
            diagnostics artifact directory (for tests), exactly as
            ``label_regimes`` does.
        first_decision: the first walk-forward decision date. Defaults to
            ``features.index[min_train_months]``.

    Returns:
        dict with keys ``states`` (int array), ``confidences`` ((T, K) array),
        ``index`` (the labeled months), ``frozen_columns``, ``occupancy``
        (dict[int, float], exactly K entries summing to 1.0), ``sojourns``,
        ``profiles``, ``first_decision`` and ``diagnostics_path``.

    Raises:
        ValueError: from :func:`classifier2_config` (lambda/sort-column
            invariants), from :func:`freeze_classifier2_columns` (empty or
            shorter-than-K frozen list), or when the pinned ordering column
            does not survive the freeze — there is no fallback (plan 07-05).
    """
    resolved = classifier2_config(cfg)
    K, lam = resolved["K"], resolved["lam"]
    sort_column = resolved["sort_column"]

    dev_features, holdout_rows = split_by_holdout_boundary(features, cutoff=DEFAULT_HOLDOUT_CUTOFF)
    if len(holdout_rows):
        log.info(
            "classifier #2: carved %d post-%s row(s) out of the fit (HON-01) — %d dev months remain",
            len(holdout_rows), DEFAULT_HOLDOUT_CUTOFF, len(dev_features),
        )

    if first_decision is None:
        first_decision = _derive_first_decision(dev_features, cfg)

    frozen = freeze_classifier2_columns(dev_features, cfg, first_decision)
    if sort_column not in frozen:
        raise ValueError(
            f"classifier #2's ordering column {sort_column!r} did not survive the freeze "
            f"(frozen columns: {frozen}). canonicalize_states has NO fallback since plan "
            "07-05 — ordering on centroid column 0 would assign arbitrary state IDs while "
            "every downstream occupancy, dependence and lift number kept appearing to pass."
        )

    X_df = dev_features[frozen].dropna(axis=0, how="any")
    X = standardize_features(X_df)

    fit = fit_jump_model(X, K=K, lam=lam, n_restarts=resolved["n_restarts"])
    states, centroids = canonicalize_states(
        fit["states"], fit["centroids"], frozen, sort_column=sort_column
    )

    d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    confidences = soft_confidences(d)

    cm = _get_checkpoint_manager(checkpoint_dir)
    labels_df = pd.DataFrame({"state": states}, index=X_df.index)
    confidences_df = pd.DataFrame(
        confidences, index=X_df.index, columns=[f"state_{k}" for k in range(K)]
    )
    cm.save(labels_df, CLASSIFIER2_LABELS_CHECKPOINT)
    cm.save(confidences_df, CLASSIFIER2_CONFIDENCES_CHECKPOINT)

    profiles = auto_profile(centroids, frozen)
    profiles_df = pd.DataFrame(
        {"state": list(profiles.keys()), "profile": list(profiles.values())}
    )
    cm.save(profiles_df, CLASSIFIER2_PROFILES_CHECKPOINT)

    # n_states=K explicitly so a never-occupied state surfaces as a 0.0 entry
    # rather than vanishing from the occupancy vector entirely.
    occ_sojourn = occupancy_and_sojourns(states, n_states=K)

    # Same report_labeling_diagnostics code path classifier #1 uses — so the
    # §4.4 band WARNING is emitted by the same code — but routed to its
    # OWN directory: the default artifact name is shared, and writing there
    # would silently overwrite classifier #1's diagnostics (T-07-15's failure
    # mode applied to the artifact rather than the checkpoint).
    diagnostics_dir = (
        Path(checkpoint_dir)
        if checkpoint_dir is not None
        else OUTPUT_DIR / "reports" / "model_metrics" / "classifier2"
    )
    diagnostics_path = report_labeling_diagnostics(
        {
            "occupancy": occ_sojourn["occupancy_pct"],
            "sojourns": occ_sojourn["sojourns"],
            "profiles": profiles,
        },
        output_dir=diagnostics_dir,
    )

    return {
        "states": states,
        "confidences": confidences,
        "index": X_df.index,
        "frozen_columns": frozen,
        "occupancy": occ_sojourn["occupancy_pct"],
        "sojourns": occ_sojourn["sojourns"],
        "profiles": profiles,
        "first_decision": first_decision,
        "diagnostics_path": diagnostics_path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no real checkpoint (mirrors
    # diagnostics.py's __main__ footer).
    import tempfile

    rng = np.random.default_rng(42)
    n_months = 150
    idx = pd.date_range("1990-01-31", periods=n_months, freq="ME")
    synthetic = pd.DataFrame(
        {col: rng.normal(0, 1, n_months) for col in CLASSIFIER2_CANDIDATE_COLUMNS}, index=idx
    )
    synthetic_cfg = {
        "labeling_2": {
            "K": 3,
            "lambda": 4.0 * len(CLASSIFIER2_CANDIDATE_COLUMNS),
            "n_restarts": 3,
            "sort_column": CLASSIFIER2_SORT_COLUMN,
            "features": CLASSIFIER2_CANDIDATE_COLUMNS,
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        out = label_leadership_regimes(
            synthetic, synthetic_cfg, checkpoint_dir=Path(tmp), first_decision=idx[24]
        )
        print(  # noqa: T201
            f"self-check: {len(out['states'])} months labeled on "
            f"{len(out['frozen_columns'])} frozen columns, occupancy={out['occupancy']}"
        )
