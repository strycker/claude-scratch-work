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

Era-stability assessment and the full trial-registry-logged screen
(``loading_stability_across_eras`` / ``screen_invariant_candidates``) are added in a
later commit on this same file (plan 07-07 Task 2) — this module currently provides
the named-candidate registry and the PCA-as-discovery-tool primitive those build on.

Usage::

    from trading_crab_lib.platform.features.invariants import (
        INVARIANT_CANDIDATES, compute_candidate_loadings,
    )

    loadings = compute_candidate_loadings(working_frame, INVARIANT_CANDIDATES)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.decomposition import PCA

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


# ── The screening record shape (populated by screen_invariant_candidates, Task 2) ──


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


def named_survivors(results: list[InvariantScreenResult]) -> list[str]:
    """Ordered survivor names — the exact list classifier #2's candidate set may
    draw from. Deterministic: derived by filtering *results* in the order it was
    built (itself :data:`INVARIANT_CANDIDATES`'s own declared order), never from
    set or dict iteration. Two calls on the same input return the identical list,
    element for element.
    """
    return [r.name for r in results if r.verdict == "survive"]
