"""
Subsample stability machinery for design §4.4 criterion 3 (PER-06) — pure
functions of a fit; the fit itself is never touched.

**Read the next two paragraphs before reading any number this module produces.**
They name two properties of ``jump_model.py`` that make a naive stability table
look fine while measuring nothing.

*Trap A — the discarded scaler.* ``standardize_features``
(``jump_model.py:112-124``) winsorizes to the per-column [1%, 99%] quantiles and
returns ``StandardScaler().fit_transform(winsorized)``, keeping **no handle on
the scaler**. Both the clip bounds and the scaler are fit on whatever rows are
passed in, so a subsample-fitted centroid lives in a *different* standardized
space than the full-sample centroid, and comparing the two coordinate-wise
compares apples to oranges. Every distance in this module is therefore computed
on **de-standardized** centroids, recovered via :func:`standardization_params`.

*Trap B — the frozen zero-occupancy centroid.* ``_recompute_centroids``
(``jump_model.py:126-138``) freezes "any zero-occupancy state at its previous
centroid", so a subsample fit at fixed K **always** returns K centroids —
including for a state that captured zero months. That frozen centroid matches its
reference partner at distance ~0 and the state would score *stable* while having
**evaporated**, which is the precise failure criterion 3 exists to catch. Every
row emitted here therefore carries the subsample occupancy, and ``evaporated`` is
constructed from occupancy **alone**, before any other field.

Residual caveat, stated rather than hidden: de-standardizing recovers
**winsorized** units, and the 1%/99% clip bounds also differ per subsample. For
centroids — means over forty or more months — that is second-order, but it is not
zero.

**Which distance, and why not the other two.** The jump model has no emission
distribution: it has centroids and squared-Euclidean distance to them
(``jump_model.py:179``), which is a spherical Gaussian with common variance.
``08-RESEARCH.md`` §5.2 benchmarked three candidates at this project's actual
dimension (d=10) and actual state sizes, 20 replications each. Signal-to-null
separation at n=40: **centroid distance 1.70x**, Gaussian closed-form W2 1.22x,
empirical multivariate Wasserstein 1.05x. ``scipy.stats.wasserstein_distance_nd``
is **rejected**: at n=40, d=10 its sampling bias (2.88) is three times the true
signal (0.949) — the known ``n^(-1/d)`` convergence of empirical optimal
transport, arriving exactly where this project lives. It is named here so the
rejection travels with the code; it is never imported and never called. Centroid
distance is the primary and the only distance implemented.

**The null is not zero.** At n=40 the split-half null centroid distance measured
in that benchmark was 0.706 against a 0.949 true signal. That figure is a
*synthetic* benchmark and is deliberately **not** a constant anywhere in this
module: :func:`split_half_null` recomputes the yardstick from this project's own
data at each state's own n.

**No threshold, no verdict.** This module reports quantities; a human reads them.
It defines no persistence threshold and emits no verdict field. The one existing,
human-authorised pass/fail is §4.4's AMENDMENT condition (i) — "it recurs in at
least three temporally separated episodes" — and it binds only on the sub-floor
state invoking the recurrence exemption. Adding a threshold after seeing the
numbers is the tie-break shape the pre-registration at ``298b1bc`` forbids.

**Registry cost: zero.** Re-fitting the labeler on a subsample at the *pinned*
(K, lambda) is a diagnostic, not an evaluated configuration. Nothing here writes
to the trial ledger and nothing here may re-pin K or lambda.

Usage::

    from trading_crab_lib.platform.labeling.stability import (
        fit_for_stability, run_stability, scheme_drop_first_decade,
    )
    ref = fit_for_stability(X_df, K=6, lam=10.0, n_restarts=10,
                            sort_column="trailing_return_1m")
    rows = run_stability(
        X_df, K=6, lam=10.0, n_restarts=10, sort_column="trailing_return_1m",
        reference_fit=ref,
        schemes={"drop_first_decade": scheme_drop_first_decade(X_df.index)},
    )
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from trading_crab_lib.platform.labeling.diagnostics import occupancy_and_sojourns
from trading_crab_lib.platform.labeling.jump_model import (
    canonicalize_states,
    fit_jump_model,
    standardize_features,
)

log = logging.getLogger(__name__)

#: A state is EVAPORATED at **zero months exactly**, not at a fraction of them.
#: "Near-zero" is a judgement and this module does not make judgements: a state
#: holding 1 to 3 months of a subsample is NOT flagged — its occupancy is carried
#: on the row and the reader sees it. Raising this to a non-zero count would be a
#: persistence threshold invented after the fact, which §4.4 gives no warrant for.
EVAPORATED_OCCUPANCY_MONTHS = 0
#: Default seed for every stochastic scheme and for the split-half null. Carried
#: on every bootstrap row so a re-run is reproducible without guessing.
DEFAULT_STABILITY_SEED = 20260921

# ── de-standardization: the common unit space ──


def standardization_params(X: pd.DataFrame) -> dict[str, pd.Series]:
    """Recompute the three quantities ``standardize_features`` uses internally.

    This is a **recomputation**, not a re-implementation. ``standardize_features``
    (``jump_model.py:122-123``) is, verbatim::

        winsorized = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
        return StandardScaler().fit_transform(winsorized)

    and it keeps no handle on the scaler, so the (center, scale) pair has to be
    rebuilt here. What keeps the two in step is the pin test
    ``test_standardization_params_inverts_standardize_features``, which asserts
    elementwise that ``standardize_features(X) * scale + center`` equals the
    winsorized frame. That test is the tripwire on any future edit to
    ``standardize_features``; it fails on a changed ``ddof``, a changed clip
    order, or a changed composition.

    ``standardize_features`` is deliberately **not** modified to return its
    scaler: its signature is used by ``label_regimes``, ``_refit_l1``,
    ``_refit_classifier2`` and ``label_leadership_regimes``, and changing it would
    ripple into four call sites for no benefit this module needs.

    Args:
        X: the feature frame exactly as it would be handed to
            ``standardize_features`` (columns in the fitted order).

    Returns:
        dict with ``"winsor_lower"``, ``"winsor_upper"``, ``"center"`` and
        ``"scale"``, each a pd.Series indexed by ``X.columns``. ``scale`` is the
        **population** standard deviation (``ddof=0``, matching
        ``StandardScaler``) of the *winsorized* frame, with zero-variance columns
        set to 1.0 exactly as ``sklearn``'s ``_handle_zeros_in_scale`` does.
    """
    lower = X.quantile(0.01)
    upper = X.quantile(0.99)
    winsorized = X.clip(lower=lower, upper=upper, axis=1)
    center = winsorized.mean()
    scale = winsorized.std(ddof=0)
    # sklearn's _handle_zeros_in_scale: a constant column is left unscaled rather
    # than divided by zero. Mirrored so the inverse stays exact on such a column.
    scale = scale.mask(scale <= 10.0 * np.finfo(np.float64).eps, 1.0)
    return {"winsor_lower": lower, "winsor_upper": upper, "center": center, "scale": scale}


def destandardize_centroids(
    centroids: np.ndarray, params: Mapping[str, pd.Series], columns: Sequence[str]
) -> pd.DataFrame:
    """Map standardized centroids back into winsorized feature units.

    Args:
        centroids: (K, d) standardized centroid array, column order matching
            *columns*.
        params: output of :func:`standardization_params` for the SAME fit.
        columns: the fitted feature names, in order.

    Returns:
        (K, d) pd.DataFrame in winsorized feature units, columns = *columns*.

    Raises:
        ValueError: if *centroids*' second axis does not match *columns*.
    """
    arr = np.asarray(centroids, dtype=float)
    cols = list(columns)
    if arr.ndim != 2 or arr.shape[1] != len(cols):
        raise ValueError(f"centroids shape {arr.shape} does not match {len(cols)} columns")
    center = params["center"].reindex(cols).to_numpy()
    scale = params["scale"].reindex(cols).to_numpy()
    return pd.DataFrame(arr * scale + center, columns=cols)


def winsorized_frame(X: pd.DataFrame, params: Mapping[str, pd.Series]) -> pd.DataFrame:
    """The de-standardized row space: *X* clipped at *params*' own 1%/99% bounds.

    This is the space :func:`destandardize_centroids` maps into, so a centroid
    and the rows it is the mean of are directly comparable.
    """
    return X.clip(lower=params["winsor_lower"], upper=params["winsor_upper"], axis=1)


@dataclass
class StabilityFit:
    """One labeler fit, carried in both centroid spaces plus its occupancy.

    ``occupancy`` is length K **including zeros** — a state that captured no
    months still has an entry, because ``_recompute_centroids`` still returns a
    centroid for it (Trap B). Read ``occupancy`` before reading any distance.
    """

    states: pd.Series
    centroids_standardized: np.ndarray
    centroids_destandardized: pd.DataFrame
    columns: list[str]
    params: dict[str, pd.Series]
    occupancy: np.ndarray
    rows_destandardized: pd.DataFrame = field(repr=False)
    K: int = 0


def fit_for_stability(
    X_df: pd.DataFrame,
    *,
    K: int,
    lam: float,
    n_restarts: int = 10,
    sort_column: str = "trailing_return_1m",
    random_state: int = 42,
) -> StabilityFit:
    """Fit the labeler in ``label_regimes``' own order and carry both centroid spaces.

    Composes the public helpers in exactly the order every fit in this project
    mirrors (``diagnostics.py:274-315``): select columns -> dropna -> standardize
    -> fit -> canonicalize. It adds nothing to that pipeline; the only extra work
    is recomputing the (center, scale) pair ``standardize_features`` discards.

    Args:
        X_df: feature frame with the columns already selected, in fitted order.
        K: number of states (PINNED by the caller — never selected here).
        lam: per-jump penalty (PINNED by the caller — never selected here).
        n_restarts: k-means warm starts.
        sort_column: canonicalization key; must be present in ``X_df.columns``
            (``canonicalize_states`` raises otherwise, with no fallback).
        random_state: base seed passed to ``fit_jump_model``.

    Returns:
        :class:`StabilityFit`.

    Raises:
        ValueError: propagated from ``canonicalize_states`` if *sort_column* is
            absent from the fitted feature set (there is deliberately no
            fallback: a silent fallback to centroid column 0 assigns arbitrary
            state ids while every downstream number keeps appearing to pass).
    """
    columns = list(X_df.columns)
    clean = X_df.dropna()
    if clean.empty:
        raise ValueError("X_df has no complete rows after dropna()")
    params = standardization_params(clean)
    X = standardize_features(clean)
    fit = fit_jump_model(X, K=K, lam=lam, n_restarts=n_restarts, random_state=random_state)
    states, centroids = canonicalize_states(
        fit["states"], fit["centroids"], columns, sort_column=sort_column
    )
    occupancy = np.bincount(np.asarray(states, dtype=int), minlength=K)[:K]
    return StabilityFit(
        states=pd.Series(states, index=clean.index, name="state"),
        centroids_standardized=centroids,
        centroids_destandardized=destandardize_centroids(centroids, params, columns),
        columns=columns,
        params=params,
        occupancy=occupancy,
        rows_destandardized=winsorized_frame(clean, params),
        K=K,
    )


# ── Hungarian matching, the null, and episodes ──


def match_states(
    ref_centroids_destd: pd.DataFrame, sub_centroids_destd: pd.DataFrame
) -> dict:
    """Hungarian assignment of subsample states to reference states.

    The cost is plain Euclidean distance between **de-standardized** centroids
    (§5.2: faithful to the model's own squared-Euclidean objective, zero free
    parameters, best signal-to-null ratio at every n that matters here). Solved
    with ``scipy.optimize.linear_sum_assignment``.

    **What an identity assignment does and does not tell you.** ``canonicalize_states``
    already defeats most label switching by sorting on the ascending
    ``sort_column`` centroid, so an identity assignment says the canonical
    ordering held — it says **nothing** about whether the states persisted. The
    informative output is ``matched_distance``, not ``assignment``. A NON-identity
    assignment is a finding in its own right: the canonical ordering itself
    flipped, and every downstream occupancy, profile and lift number is keyed on
    those ids.

    Args:
        ref_centroids_destd: (K, d) reference centroids in winsorized units.
        sub_centroids_destd: (K, d) subsample centroids in winsorized units.

    Returns:
        dict with ``"assignment"`` (dict ref state id -> sub state id),
        ``"cost_matrix"`` (the full K x K distance matrix, persisted so a later
        reader can re-judge without a re-run), ``"matched_distance"`` (length-K
        array, indexed by ref state), ``"margin"`` (length-K array: matched
        distance divided by the next-best alternative in that row; near 1.0 means
        the assignment is arbitrary regardless of how small the distance is), and
        ``"is_identity"`` (bool).

    Raises:
        ValueError: on a shape mismatch or a column set the two do not share.
    """
    ref = pd.DataFrame(ref_centroids_destd)
    sub = pd.DataFrame(sub_centroids_destd)
    if ref.shape != sub.shape:
        raise ValueError(f"centroid shape mismatch: ref {ref.shape} vs sub {sub.shape}")
    missing = [c for c in ref.columns if c not in sub.columns]
    if missing:
        raise ValueError(f"subsample centroids are missing reference columns: {missing}")
    sub = sub.reindex(columns=list(ref.columns))

    ref_arr = ref.to_numpy(dtype=float)
    sub_arr = sub.to_numpy(dtype=float)
    K = ref_arr.shape[0]
    cost = np.linalg.norm(ref_arr[:, None, :] - sub_arr[None, :, :], axis=2)

    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.argsort(row_ind)
    partners = col_ind[order]
    assignment = {int(k): int(partners[k]) for k in range(K)}

    matched = np.array([cost[k, partners[k]] for k in range(K)], dtype=float)
    margin = np.empty(K, dtype=float)
    for k in range(K):
        row = cost[k].copy()
        row[partners[k]] = np.inf
        next_best = float(row.min())
        if not np.isfinite(next_best):  # K == 1: there is no alternative
            margin[k] = float("nan")
        elif next_best == 0.0:
            margin[k] = float("inf") if matched[k] > 0.0 else 1.0
        else:
            margin[k] = float(matched[k]) / next_best

    return {
        "assignment": assignment,
        "cost_matrix": cost,
        "matched_distance": matched,
        "margin": margin,
        "is_identity": bool(np.array_equal(partners, np.arange(K))),
    }

def split_half_null(
    X_destd_rows, *, n_reps: int = 200, seed: int = DEFAULT_STABILITY_SEED
) -> dict:
    """Within-state split-half null centroid distance at this state's own n.

    Splits ONE state's own months at random into two halves, takes the centroid
    distance between them, repeats, and reports the quantiles. This is the
    data-derived answer to "how far apart do two samples of *the same* state land
    at this n", with zero invented parameters — and it is indispensable, because
    the null is **not zero**: ``08-RESEARCH.md`` §5.2 measured 0.706 at n=40, d=10
    against a true signal of 0.949. That figure is a *synthetic* benchmark and is
    deliberately not hard-coded here or anywhere in this module.

    Args:
        X_destd_rows: (n, d) array-like of one state's own rows, in the SAME
            de-standardized (winsorized) units as the centroids being compared.
        n_reps: number of random splits.
        seed: RNG seed; the RNG is created per call, never global.

    Returns:
        dict with ``"median"``, ``"p10"``, ``"p90"``, ``"n"`` and ``"n_reps"``.
        ``median``/``p10``/``p90`` are ``nan`` when n < 2 (no split exists).
    """
    arr = np.asarray(pd.DataFrame(X_destd_rows).to_numpy(), dtype=float)
    n = int(arr.shape[0])
    if n < 2:
        return {"median": float("nan"), "p10": float("nan"), "p90": float("nan"), "n": n, "n_reps": 0}
    rng = np.random.default_rng(seed)
    half = n // 2
    dists = np.empty(n_reps, dtype=float)
    for rep in range(n_reps):
        perm = rng.permutation(n)
        a = arr[perm[:half]].mean(axis=0)
        b = arr[perm[half:]].mean(axis=0)
        dists[rep] = float(np.linalg.norm(a - b))
    return {
        "median": float(np.median(dists)),
        "p10": float(np.quantile(dists, 0.10)),
        "p90": float(np.quantile(dists, 0.90)),
        "n": n,
        "n_reps": int(n_reps),
    }


def state_episodes(states, *, n_states: int | None = None) -> dict[int, dict]:
    """Contiguous runs ("episodes") per state, with spans, count and longest.

    The run-length arithmetic is cross-checked against
    ``diagnostics.occupancy_and_sojourns`` — the project's existing run-length
    machinery — rather than trusted on its own: a mismatch in ``n_runs`` raises.
    This module needs the *spans* (which ``occupancy_and_sojourns`` does not
    return) for leave-one-episode-out, so the scan is unavoidable; pinning it
    against the existing scanner is what keeps it from being a second, silently
    divergent implementation.

    Args:
        states: array-like or pd.Series of canonicalized state labels.
        n_states: total number of possible states; defaults to ``max+1``. Pass K
            to surface never-occupied states (they get ``n_episodes == 0``).

    Returns:
        dict state id -> ``{"episodes": [{"start", "end", "length", ...}, ...],
        "n_episodes": int, "longest_episode": int, "months": int}``. ``start`` and
        ``end`` are **positional** (0-based, end inclusive); when *states* is a
        pd.Series the index labels are carried as ``start_label``/``end_label``.

    Raises:
        RuntimeError: if the scan disagrees with ``occupancy_and_sojourns``.
    """
    labels = states.index if isinstance(states, pd.Series) else None
    arr = np.asarray(states, dtype=int)
    if arr.size == 0:
        raise ValueError("states must be non-empty")
    total = int(n_states) if n_states is not None else int(arr.max()) + 1

    out: dict[int, dict] = {state: {"episodes": [], "n_episodes": 0, "longest_episode": 0, "months": 0}
                            for state in range(total)}
    start = 0
    for pos in range(1, arr.size + 1):
        if pos == arr.size or arr[pos] != arr[start]:
            state = int(arr[start])
            episode = {"start": start, "end": pos - 1, "length": pos - start}
            if labels is not None:
                episode["start_label"] = labels[start]
                episode["end_label"] = labels[pos - 1]
            if state in out:
                out[state]["episodes"].append(episode)
            start = pos
    for state, info in out.items():
        info["n_episodes"] = len(info["episodes"])
        info["longest_episode"] = max((e["length"] for e in info["episodes"]), default=0)
        info["months"] = int((arr == state).sum())

    # Pin against the existing run-length scanner rather than duplicating it silently.
    reference = occupancy_and_sojourns(arr, n_states=total)["sojourns"]
    for state, info in out.items():
        expected = int(reference[state]["n_runs"])
        if info["n_episodes"] != expected:
            raise RuntimeError(
                f"episode scan disagrees with occupancy_and_sojourns for state {state}: "
                f"{info['n_episodes']} vs {expected}"
            )
    return out


def stability_row(
    *,
    classifier: str,
    scheme: str,
    state: int,
    subsample_occupancy_months: int,
    subsample_occupancy_pct: float,
    matched_partner: int,
    is_identity: bool,
    matched_distance: float,
    margin: float,
    split_half_null_median: float,
    split_half_null_p10: float = float("nan"),
    split_half_null_p90: float = float("nan"),
    split_half_null_n: int = 0,
    n_episodes: int = 0,
    longest_episode: int = 0,
    block_length: int | None = None,
    n_seams: int | None = None,
    seed: int | None = None,
    degenerate: bool | None = None,
) -> dict:
    """One (classifier, scheme, state) record — §5.5's seven quantities plus ``evaporated``.

    ``evaporated`` is constructed **first**, from occupancy alone, and every
    consumer can read it without reading the distance. That ordering is the whole
    point: ``_recompute_centroids`` freezes a zero-occupancy state at its previous
    centroid, so an evaporated state's ``matched_distance`` is ~0 and a
    distance-only reader scores it *stable* (Trap B).

    Logs a WARNING per evaporated state, naming the classifier, the scheme and the
    state id — a flagged field in a parquet is easy to miss; a WARNING in the run
    log is not.

    Returns:
        dict; ``"evaporated"`` is bool, never None.
    """
    evaporated = bool(int(subsample_occupancy_months) <= EVAPORATED_OCCUPANCY_MONTHS)
    if evaporated:
        log.warning(
            "EVAPORATED: classifier=%s scheme=%s state=%d captured 0 months in the "
            "subsample. Its centroid was FROZEN by _recompute_centroids, so its "
            "matched distance (%.6g) means the centroid did not move — NOT that the "
            "state persisted.",
            classifier, scheme, int(state), float(matched_distance),
        )
    return {
        "classifier": classifier,
        "scheme": scheme,
        "state": int(state),
        "evaporated": evaporated,
        "subsample_occupancy_months": int(subsample_occupancy_months),
        "subsample_occupancy_pct": float(subsample_occupancy_pct),
        "matched_partner": int(matched_partner),
        "is_identity": bool(is_identity),
        "matched_distance": float(matched_distance),
        "margin": float(margin),
        "split_half_null_median": float(split_half_null_median),
        "split_half_null_p10": float(split_half_null_p10),
        "split_half_null_p90": float(split_half_null_p90),
        "split_half_null_n": int(split_half_null_n),
        "n_episodes": int(n_episodes),
        "longest_episode": int(longest_episode),
        "block_length": block_length,
        "n_seams": n_seams,
        "seed": seed,
        "degenerate": degenerate,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Synthetic self-check — no network, no checkpoint (mirrors diagnostics.py's
    # __main__ footer). Two well-separated clusters, fit, de-standardize, print
    # the two centroids and the gap between them.
    _rng = np.random.default_rng(7)
    _cols = [
        "curve_10y3m", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
        "trailing_return_1m", "realized_vol_3m", "cape_shiller", "div_yield", "real_rate_level",
    ]
    _n = 120
    _shift = np.concatenate([np.zeros(_n // 2), np.full(_n - _n // 2, 3.0)])
    _frame = pd.DataFrame(
        {c: _rng.normal(0, 1, _n) + _shift for c in _cols},
        index=pd.date_range("2010-01-31", periods=_n, freq="ME"),
    )
    _fit = fit_for_stability(_frame, K=2, lam=5.0, n_restarts=3, sort_column="trailing_return_1m")
    _gap = float(np.linalg.norm(
        _fit.centroids_destandardized.iloc[0].to_numpy()
        - _fit.centroids_destandardized.iloc[1].to_numpy()
    ))
    print("self-check: de-standardized centroids (winsorized units)")  # noqa: T201
    print(_fit.centroids_destandardized.round(3).to_string())  # noqa: T201
    print(f"self-check: occupancy={_fit.occupancy.tolist()}  centroid gap={_gap:.4f}")  # noqa: T201
