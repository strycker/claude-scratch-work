"""
INV-01 invariant screening — PCA as a discovery tool, never a feature source (R4).

Design decision R4 requires that any feature admitted into the platform be a **named**
quantity, never an anonymous principal component. This module screens INV-01's named
invariant candidates (currently ``m2_gdp`` and ``credit_gdp`` from
``platform/features/relative.py::compute_invariant_ratios``) by USING dimensional
reduction (PCA) to *look at* how those named candidates relate to one another — never
to *produce* a candidate. The distinction is structural, not a comment:
:func:`compute_candidate_loadings` is only ever able to hand a caller back a
candidate's LOADING on a retained component (a coefficient describing how much that
named candidate contributes to a discovered axis of covariation), never the
component's SCORE (the transformed values that would be usable, if handed back, as a
drop-in anonymous feature). A caller cannot admit as a feature what this module
structurally never returns.

Market-cap/GDP (the "Buffett indicator") is deliberately absent from
:data:`INVARIANT_CANDIDATES` — no free 1962+ market-cap source exists in current
ingestion (FRED's Wilshire series starts around 1970). This is D-12's recorded
rejection, restated here rather than worked around; see
``config/platform_settings.yaml``'s ``buffett_indicator`` comment. The two credit
aggregates considered and rejected BEFORE ingestion (``BCNSDODNS`` — quarterly-native;
``TOTBKCR`` — starts 1973, after the 1962 spine) are recorded in
``.planning/phases/07-regime-representation/07-INV01-SCREENING.md``, not here — they
never reached ingestion, so they have no ``monthly_raw`` column to screen.

Era-stability is assessed walk-forward via
``honesty/walkforward.py::expanding_steps`` (:func:`loading_stability_across_eras`)
— the same generator ``run_backtest``'s per-step L1 refits use — never a hand-rolled
slice loop, so "assessed walk-forward" is a property of the code, not a claim in
prose. :func:`screen_invariant_candidates` ties everything together: it applies the
2021+ holdout carve BEFORE reading anything, applies D-11's freeze rule via
``evaluation/report.py::_reference_label_columns`` (unmodified), and logs every
candidate — survivor and reject alike — to the trial registry under an explicit tag.

Usage::

    from trading_crab_lib.platform.config import load_platform_config
    from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
    from trading_crab_lib.platform.features.invariants import (
        screen_invariant_candidates, named_survivors,
    )

    cfg = load_platform_config()
    monthly_raw = get_platform_checkpoint_manager().load("monthly_raw")
    results = screen_invariant_candidates(monthly_raw, cfg, trial_tag="07-07-inv01-screen")
    survivors = named_survivors(results)  # e.g. ["m2_gdp", "credit_gdp"]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.decomposition import PCA

from trading_crab_lib.platform.evaluation.report import _reference_label_columns
from trading_crab_lib.platform.features.relative import compute_invariant_ratios
from trading_crab_lib.platform.honesty import registry
from trading_crab_lib.platform.honesty.holdout import split_by_holdout_boundary
from trading_crab_lib.platform.honesty.walkforward import expanding_steps
from trading_crab_lib.platform.labeling.jump_model import standardize_features

log = logging.getLogger(__name__)


# ── Named candidates (R4: names, never component indices) ──────────────


@dataclass(frozen=True)
class InvariantCandidateSpec:
    """One INV-01 candidate: a NAME, its ``monthly_raw`` source columns, and a
    one-line description. Never a bare component label or index."""

    name: str
    source_columns: tuple[str, ...]
    description: str


#: INV-01's named invariant candidates, in a fixed declaration order. This order is
#: LOAD-BEARING downstream: every list this module returns is built by iterating this
#: constant, never by set/dict iteration, because ``canonicalize_states`` (classifier
#: #2's own canonicalization, plan 07-08) locates its sort column by POSITION in the
#: fitted column list.
#:
#: Market-cap/GDP is deliberately ABSENT (D-12, restated from
#: ``config/platform_settings.yaml``'s ``buffett_indicator`` comment): no free 1962+
#: market-cap source exists in current ingestion (FRED's Wilshire series starts
#: ~1970). This is a recorded rejection, not a silent omission.
INVARIANT_CANDIDATES: list[InvariantCandidateSpec] = [
    InvariantCandidateSpec(
        name="m2_gdp",
        source_columns=("fred_m2sl", "fred_gdp"),
        description="M2 money supply relative to GDP — the classic monetary-expansion invariant.",
    ),
    InvariantCandidateSpec(
        name="credit_gdp",
        source_columns=("fred_totalsl", "fred_gdp"),
        description="Total consumer credit outstanding relative to GDP — a credit-cycle invariant.",
    ),
]


#: INV-01's precision edge, named explicitly (a stability claim that does not name
#: the tolerance it would fail at is not a claim). Two loadings (or one loading
#: across two eras) are considered equal if they differ by no more than this many
#: loading units. Loadings are PCA eigenvector coefficients (unit-norm across the
#: retained feature set), so this is a fraction of a unit-norm vector's own scale —
#: chosen wide enough to tolerate ordinary five-decade drift in a two/three-candidate
#: covariance structure without being so wide it could never detect genuine
#: instability. Comparisons are made against the UNROUNDED value; rounding is for
#: display only.
LOADING_STABILITY_TOLERANCE: float = 0.15

#: Default number of PCA components to retain for discovery. Capped at the number of
#: computable candidates when fewer are available.
DEFAULT_N_COMPONENTS: int = 2

#: Era-stability step, in MONTHS. A COARSE step so the screen produces a HANDFUL of
#: eras rather than one per month — see :func:`loading_stability_across_eras`.
DEFAULT_ERA_STEP_MONTHS: int = 60

#: Minimum expanding-window training length, in MONTHS, before the first era is
#: assessed. Matches ``config/platform_settings.yaml``'s
#: ``backtest.min_train_months`` default (120 -> the 1972-01 first decision from the
#: 1962-01 spine start), so classifier #2's era assessment shares the SAME window
#: discipline classifier #1's walk-forward driver uses (D-11).
DEFAULT_ERA_MIN_TRAIN_MONTHS: int = 120


# ── PCA as a discovery tool only (R4) ───────────────────────────────────


def compute_candidate_loadings(
    frame: pd.DataFrame,
    candidates: list[InvariantCandidateSpec] | None = None,
    *,
    n_components: int = DEFAULT_N_COMPONENTS,
) -> pd.DataFrame:
    """PCA loadings for INV-01's named candidates — discovery only, never features.

    Selects the named candidate columns present (and not entirely NaN) in *frame*,
    drops rows with any NaN among them, standardizes through
    ``labeling/jump_model.py::standardize_features`` (reused, never a second
    hand-rolled scaler), fits a PCA retaining ``min(n_components, n_computable)``
    components, and returns a DataFrame indexed by candidate NAME whose columns are
    ``pc1``, ``pc2``, ... holding that candidate's loading (its eigenvector
    coefficient) on each retained component.

    **This function can never hand back a component SCORE.** Only
    ``pca.components_`` (the loadings matrix) is read; ``pca.transform()`` is never
    called and no transformed value is computed, persisted, or returned anywhere in
    this module. A caller that received scores could admit them as an anonymous
    feature, defeating R4 — this function is structurally unable to do that.

    The returned index preserves *candidates*' declared order (defaulting to
    :data:`INVARIANT_CANDIDATES`'s own order) restricted to the computable subset —
    never sorted alphabetically, never derived from set/dict iteration. This
    ordering is load-bearing downstream (see :data:`INVARIANT_CANDIDATES`'s
    docstring).

    Args:
        frame: a DataFrame already carrying the named candidate columns (e.g. the
            output of ``compute_invariant_ratios``), over whatever range the caller
            considers the relevant decision range.
        candidates: candidates to consider, in the order to preserve. Defaults to
            :data:`INVARIANT_CANDIDATES`.
        n_components: PCA components to retain (discovery only). Capped at the
            number of computable candidates.

    Returns:
        DataFrame indexed by candidate name, one column per retained component.

    Raises:
        ValueError: if NO candidate in *candidates* is computable (present and not
            entirely NaN) over *frame* — never returns an empty result a caller
            could misread as "screen passed, nothing to add".
    """
    if candidates is None:
        candidates = INVARIANT_CANDIDATES

    ordered_names = [c.name for c in candidates]
    computable: list[str] = []
    for name in ordered_names:
        if name not in frame.columns:
            log.warning(
                "Excluding INV-01 candidate %r from PCA discovery: column not "
                "present in the supplied frame.",
                name,
            )
            continue
        if frame[name].isna().all():
            log.warning(
                "Excluding INV-01 candidate %r from PCA discovery: entirely NaN "
                "over the supplied frame.",
                name,
            )
            continue
        computable.append(name)

    if not computable:
        raise ValueError(
            "compute_candidate_loadings: no INV-01 candidate is computable over the "
            f"supplied frame (candidates considered: {ordered_names}). A screen "
            "with zero computable candidates must raise, not return an empty "
            "result a caller could read as 'screen passed, nothing to add'."
        )

    subset = frame[computable].dropna(how="any")
    if subset.empty:
        raise ValueError(
            "compute_candidate_loadings: candidates "
            f"{computable} share no jointly non-NaN row after dropping rows with "
            "any NaN among them."
        )

    n_comp = min(n_components, len(computable))
    scaled = standardize_features(subset)  # (T, d) array; column order == `computable`
    pca = PCA(n_components=n_comp)
    pca.fit(scaled)
    # pca.components_: (n_comp, d) eigenvectors -- READ ONLY. pca.transform() is
    # never called anywhere in this module (that would be the R4 seam).
    loadings = pca.components_.T  # (d, n_comp): rows==features(computable order), cols==components
    return pd.DataFrame(
        loadings,
        index=computable,
        columns=[f"pc{i + 1}" for i in range(n_comp)],
    )


# ── Era-stability, assessed walk-forward (never a hand-rolled loop) ─────────────


def loading_stability_across_eras(
    frame: pd.DataFrame,
    candidates: list[InvariantCandidateSpec] | None = None,
    decision_index: pd.Index | None = None,
    *,
    min_train: int = DEFAULT_ERA_MIN_TRAIN_MONTHS,
    step: int = DEFAULT_ERA_STEP_MONTHS,
) -> dict[str, dict[str, Any]]:
    """Assess each candidate's PCA loading stability across expanding-window eras.

    Builds era windows via ``honesty/walkforward.py::expanding_steps`` over
    *decision_index* — the SAME generator ``run_backtest``'s per-step L1 refits use
    (``backtest/driver.py`` calls ``expanding_steps(dev_features.index,
    min_train=min_train)``) — rather than a hand-rolled slice loop, so "assessed
    walk-forward" is a property of the code, not a claim in prose.

    Takes a COARSE ``step`` (default :data:`DEFAULT_ERA_STEP_MONTHS`, 5 years) so
    the screen produces a HANDFUL of eras rather than one per month: with the
    default ``min_train`` (:data:`DEFAULT_ERA_MIN_TRAIN_MONTHS`, 120 months) over a
    ~700-month dev-side monthly index, this yields roughly
    ``(len(decision_index) - min_train) // step + 1`` eras — about 10 for a
    1962-2020 dev span.

    Within each era, calls :func:`compute_candidate_loadings` on
    ``frame.loc[train_index]`` — the TRAINING SLICE ONLY, strictly before that
    era's own decision date (``expanding_steps``'s own guarantee) — so no era's
    result is ever computed from data dated later than that era's own window end.
    An era in which no candidate is computable (e.g. too few valid rows) is
    skipped with a logged WARNING rather than fabricating a loading.

    A candidate whose loading sign flips between eras, OR whose loading range
    (max - min) across eras exceeds :data:`LOADING_STABILITY_TOLERANCE`, is
    reported ``"unstable"``. A candidate with a consistent sign and a loading
    range within the tolerance is reported ``"stable"``. A candidate with zero
    assessed eras is reported ``"not_assessed"``. **Both outcomes are reported;
    neither raises** — this is a plausibility band (D-07's posture), not a gate.

    Args:
        frame: the full (dev-side) frame carrying the named candidate columns.
        candidates: candidates to consider. Defaults to :data:`INVARIANT_CANDIDATES`.
        decision_index: the index to build eras over (typically ``frame.index``).
        min_train: months of history required before the first era (named, not a
            magic number inline).
        step: months between consecutive eras (named, not a magic number inline).

    Returns:
        dict keyed by candidate name, each value
        ``{"per_era_loadings": [(era_end_iso_date, loading), ...], "stability_verdict": str}``.
        ``loading`` is the candidate's PC1 loading for that era (the first retained
        component — the dominant discovered axis).
    """
    if candidates is None:
        candidates = INVARIANT_CANDIDATES
    if decision_index is None:
        decision_index = frame.index

    ordered_names = [c.name for c in candidates]
    per_candidate: dict[str, list[tuple[str, float]]] = {name: [] for name in ordered_names}

    for t, train_index, _test_index in expanding_steps(decision_index, min_train=min_train, step=step):
        era_frame = frame.loc[train_index]
        try:
            era_loadings = compute_candidate_loadings(era_frame, candidates, n_components=1)
        except ValueError:
            log.warning(
                "loading_stability_across_eras: no candidate computable for era "
                "ending %s; skipping this era rather than fabricating a loading.",
                t,
            )
            continue
        pc1_col = era_loadings.columns[0]
        era_end = str(pd.Timestamp(t).date())
        for name in era_loadings.index:
            per_candidate[name].append((era_end, float(era_loadings.loc[name, pc1_col])))

    result: dict[str, dict[str, Any]] = {}
    for name, series in per_candidate.items():
        if not series:
            result[name] = {"per_era_loadings": [], "stability_verdict": "not_assessed"}
            continue
        values = [v for _era, v in series]
        result[name] = {"per_era_loadings": series, "stability_verdict": _classify_stability(values)}
    return result


def _classify_stability(values: list[float]) -> str:
    """Classify a candidate's per-era loading values as ``"stable"``/``"unstable"``.

    A sign flip between ANY two eras, or a loading range (``max - min``) exceeding
    :data:`LOADING_STABILITY_TOLERANCE`, marks the candidate ``"unstable"``.
    Otherwise ``"stable"``. Compares the UNROUNDED values against the tolerance;
    rounding is for display only. Never raises — a plausibility-band
    classification (D-07's posture), not a gate.
    """
    signs = {1 if v >= 0 else -1 for v in values}
    sign_stable = len(signs) == 1
    magnitude_stable = (max(values) - min(values)) <= LOADING_STABILITY_TOLERANCE
    return "stable" if (sign_stable and magnitude_stable) else "unstable"


# ── The screen: every candidate, survivors and rejects, logged to the registry ──


@dataclass
class InvariantScreenResult:
    """One INV-01 candidate's full screening record — survivor or reject, always
    with evidence. Never omitted from the returned list, whichever verdict it got."""

    name: str
    source_columns: tuple[str, ...]
    verdict: str  # "survive" | "reject"
    stability_verdict: str  # "stable" | "unstable" | "not_assessed"
    per_era_loadings: list[tuple[str, float]] = field(default_factory=list)
    first_admissible_month: str | None = None
    rejection_reason: str = ""


def screen_invariant_candidates(
    monthly_raw: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    registry_path: Any = None,
    trial_tag: str | None = None,
) -> list[InvariantScreenResult]:
    """Screen every INV-01 candidate and log each one to the trial registry.

    (1) Applies ``honesty/holdout.py::split_by_holdout_boundary`` to *monthly_raw*
        FIRST, so nothing dated after the 2021+ cutoff is ever read by this
        development-time selection activity (T-07-09).
    (2) Computes the named ratios via
        ``platform/features/relative.py::compute_invariant_ratios``.
    (3) Derives ``first_decision`` the SAME way ``evaluation/report.py``'s own
        full-sample fit does — ``dev_raw.index[min_train]`` where ``min_train =
        cfg["backtest"]["min_train_months"]`` (default 120, yielding the 1972-01
        decision date from a 1962-01 spine) — and applies D-11's freeze rule by
        calling ``evaluation/report.py::_reference_label_columns`` UNMODIFIED
        against that date, exactly mirroring classifier #1's own admissibility
        test.
    (4) Runs :func:`loading_stability_across_eras` over the dev-side index.
    (5) Assembles one :class:`InvariantScreenResult` per candidate in
        :data:`INVARIANT_CANDIDATES`'s declared order — survivors AND rejects
        alike, each with a non-empty ``rejection_reason`` when rejected.

    **Registry arithmetic (stated as a formula, not a number to discover later):**
    exactly ONE ``append_trial`` row is appended per candidate in
    :data:`INVARIANT_CANDIDATES`, so::

        rows_added = len(INVARIANT_CANDIDATES)

    This count feeds D-16's deflated-Sharpe denominator
    (``honesty/registry.py::total_trial_count()``) and must appear in ADR-0002's
    trial ceiling ALONGSIDE the evaluation-run formula, never folded into it.
    Passing ``registry_path=honesty.registry.NO_REGISTRY`` builds every result but
    writes ZERO rows (smoke/wiring runs). Passing neither a *trial_tag* nor the
    sentinel surfaces ``append_trial``'s own refusal (an untagged row is never
    silently logged).

    Args:
        monthly_raw: the platform's raw monthly checkpoint (``monthly_raw``
            shape) — may physically extend past the holdout cutoff; this
            function carves it before reading anything else.
        cfg: platform config (``load_platform_config()`` output).
        registry_path: ledger path override, or
            ``honesty.registry.NO_REGISTRY`` for a smoke run that writes nothing.
        trial_tag: the tag every appended row carries. Required unless
            ``registry_path`` is the no-registry sentinel.

    Returns:
        One :class:`InvariantScreenResult` per member of
        :data:`INVARIANT_CANDIDATES`, in that constant's declared order.

    Raises:
        ValueError: if NO candidate in :data:`INVARIANT_CANDIDATES` is computable
            at all over the supplied *monthly_raw* (never returns an empty list a
            caller could read as success).
    """
    dev_raw, _holdout_raw = split_by_holdout_boundary(monthly_raw)

    max_visited = dev_raw.index.max() if len(dev_raw) else None
    if max_visited is not None:
        log.debug("screen_invariant_candidates: dev-side index ends %s (holdout-bounded).", max_visited)

    working = compute_invariant_ratios(dev_raw)

    candidate_names = [c.name for c in INVARIANT_CANDIDATES]
    present_names = [n for n in candidate_names if n in working.columns and not working[n].isna().all()]
    if not present_names:
        raise ValueError(
            "screen_invariant_candidates: none of INV-01's named candidates "
            f"({candidate_names}) is computable over the supplied monthly_raw — a "
            "screen that returns no result would be indistinguishable from success."
        )

    backtest_cfg = cfg.get("backtest", {})
    min_train = backtest_cfg.get("min_train_months", DEFAULT_ERA_MIN_TRAIN_MONTHS)
    first_decision = working.index[min_train]

    admissible_names = _reference_label_columns(working, present_names, first_decision)

    present_candidates = [c for c in INVARIANT_CANDIDATES if c.name in present_names]
    stability = loading_stability_across_eras(
        working, present_candidates, working.index, min_train=min_train, step=DEFAULT_ERA_STEP_MONTHS
    )

    results: list[InvariantScreenResult] = []
    for spec in INVARIANT_CANDIDATES:
        name = spec.name
        if name not in present_names:
            result = InvariantScreenResult(
                name=name,
                source_columns=spec.source_columns,
                verdict="reject",
                stability_verdict="not_assessed",
                per_era_loadings=[],
                first_admissible_month=None,
                rejection_reason=(
                    f"not computable over the supplied monthly_raw: source columns "
                    f"{spec.source_columns} are absent or entirely NaN"
                ),
            )
        else:
            first_valid = working[name].first_valid_index()
            first_admissible_month = str(pd.Timestamp(first_valid).date()) if first_valid is not None else None
            era_info = stability.get(name, {"per_era_loadings": [], "stability_verdict": "not_assessed"})
            if name in admissible_names:
                result = InvariantScreenResult(
                    name=name,
                    source_columns=spec.source_columns,
                    verdict="survive",
                    stability_verdict=era_info["stability_verdict"],
                    per_era_loadings=era_info["per_era_loadings"],
                    first_admissible_month=first_admissible_month,
                    rejection_reason="",
                )
            else:
                result = InvariantScreenResult(
                    name=name,
                    source_columns=spec.source_columns,
                    verdict="reject",
                    stability_verdict=era_info["stability_verdict"],
                    per_era_loadings=era_info["per_era_loadings"],
                    first_admissible_month=first_admissible_month,
                    rejection_reason=(
                        "fails D-11's common-support freeze: not non-NaN for every "
                        f"month from {pd.Timestamp(first_decision).date()} onward"
                    ),
                )
        results.append(result)

        row_config: dict[str, Any] = {
            "candidate_name": name,
            "source_columns": list(spec.source_columns),
            "verdict": result.verdict,
            "stability_verdict": result.stability_verdict,
            "trial_tag": trial_tag,
        }
        registry.append_trial(
            config=row_config,
            features=[name],
            metrics={
                "first_admissible_month": result.first_admissible_month,
                "n_eras_assessed": len(result.per_era_loadings),
            },
            path=registry_path,
        )

    return results


def named_survivors(results: list[InvariantScreenResult]) -> list[str]:
    """Ordered survivor names — the exact list classifier #2's candidate set may
    draw from. Deterministic: derived by filtering *results* in the order it was
    built (itself :data:`INVARIANT_CANDIDATES`'s own declared order), never from
    set or dict iteration. Two calls on the same input return the identical list,
    element for element.
    """
    return [r.name for r in results if r.verdict == "survive"]
