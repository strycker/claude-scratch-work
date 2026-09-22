"""Unit tests for trading_crab_lib.platform.labeling.stability (PER-06, design §4.4 criterion 3).

Every test here is written so that it FAILS on the trap it names. That is the
whole point of the file: this project has six recorded instances of a check that
can only confirm and never fail, and a stability table is exactly the artifact
that produces plausible-looking numbers on a broken implementation.

Three traps, and the value each test would see if the implementation were wrong:

- **Trap A (the discarded scaler).** ``standardize_features`` refits its
  winsorization bounds and its ``StandardScaler`` on whatever rows it is given,
  so two fits on differently-scaled data have IDENTICAL standardized centroids. A
  naive implementation comparing ``centroids_standardized`` across fits reports a
  matched distance of **0.0** — "perfectly stable" — when the true de-standardized
  distance is half the centroid gap. ``TestTrapAUnitSpace`` asserts the correct
  value AND that the naive one is wrong.
- **Trap B (the frozen zero-occupancy centroid).** ``_recompute_centroids``
  freezes a zero-occupancy state at its previous centroid, so an evaporated state
  matches its reference partner at distance **0.0** and a distance-only reader
  scores it stable. ``TestTrapBEvaporation`` asserts distance ~0 AND
  ``evaporated is True``.
- **Trap C (feature-set churn).** Re-deriving the frozen column list per
  subsample makes criterion 3 measure the feature set rather than the states.
  ``TestTrapCFrozenColumns`` asserts ``run_stability`` raises.

Synthetic frames only — no network, no checkpoints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.labeling.jump_model import (
    standardize_features,
)
from trading_crab_lib.platform.labeling.stability import (
    StabilityFit,
    destandardize_centroids,
    fit_for_stability,
    match_states,
    standardization_params,
)

COLUMNS = [
    "curve_10y3m",
    "credit_spread_baa_aaa",
    "fred_vix",
    "gold",
    "oil",
    "trailing_return_1m",
    "realized_vol_3m",
    "cape_shiller",
    "div_yield",
    "real_rate_level",
]
D = len(COLUMNS)
SORT_COLUMN = "trailing_return_1m"


def _random_frame(n_rows: int = 200, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {c: rng.normal(0.0, 1.0, n_rows) for c in COLUMNS},
        index=pd.date_range("1990-01-31", periods=n_rows, freq="ME"),
    )


def _clustered_frame(
    shifts: list[float], n_per: int = 60, noise: float = 0.3, seed: int = 3
) -> pd.DataFrame:
    """Contiguous blocks, each a spherical Gaussian shifted by ``shift`` on every column."""
    rng = np.random.default_rng(seed)
    n = n_per * len(shifts)
    offsets = np.concatenate([np.full(n_per, s) for s in shifts])
    return pd.DataFrame(
        {c: rng.normal(0.0, noise, n) + offsets for c in COLUMNS},
        index=pd.date_range("1960-01-31", periods=n, freq="ME"),
    )

# ── Task 1: a common unit space ──


class TestStandardizationInversePin:
    """The pin that keeps ``standardization_params`` and ``standardize_features`` in step."""

    def test_standardization_params_inverts_standardize_features(self):
        """The de-standardization inverse is the ACTUAL inverse, elementwise.

        This is a round-trip identity against the function it inverts, not a
        shape check: it fails if ``ddof`` is wrong, if the winsorization is
        applied in the wrong order, or if a future edit changes
        ``standardize_features``' composition. That tripwire is the point.
        """
        X = _random_frame(200)
        params = standardization_params(X)
        Z = standardize_features(X)

        recovered = Z * params["scale"].to_numpy() + params["center"].to_numpy()
        expected = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1).to_numpy()

        max_err = float(np.abs(recovered - expected).max())
        assert max_err < 1e-10, (
            f"de-standardization is not the inverse of standardize_features "
            f"(max elementwise error {max_err:.3e}); the (center, scale) pair has "
            f"drifted from the function it recomputes"
        )

    def test_inverse_pin_discriminates_a_wrong_ddof(self):
        """The pin above would FAIL on ddof=1 — i.e. it can fail at all."""
        X = _random_frame(200)
        params = standardization_params(X)
        Z = standardize_features(X)
        winsorized = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)

        wrong = Z * winsorized.std(ddof=1).to_numpy() + params["center"].to_numpy()
        max_err = float(np.abs(wrong - winsorized.to_numpy()).max())
        assert max_err > 1e-6, (
            "a sample-sd (ddof=1) inverse is indistinguishable from the population-sd "
            "one at this tolerance, so the pin test above cannot fail — fix the fixture"
        )

    def test_standardization_params_leaves_a_constant_column_unscaled(self):
        """A zero-variance column gets scale 1.0, mirroring sklearn's _handle_zeros_in_scale."""
        X = _random_frame(80)
        X["gold"] = 4.0
        params = standardization_params(X)
        assert params["scale"]["gold"] == 1.0
        Z = standardize_features(X)
        recovered = Z * params["scale"].to_numpy() + params["center"].to_numpy()
        assert np.abs(recovered[:, COLUMNS.index("gold")] - 4.0).max() < 1e-10


class TestDestandardizeCentroids:
    def test_destandardize_centroids_round_trip_matches_the_row_space(self):
        """A de-standardized centroid equals the mean of that state's winsorized rows."""
        X = _clustered_frame([0.0, 3.0])
        fit = fit_for_stability(X, K=2, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)
        states = fit.states.to_numpy()
        for k in range(2):
            rows_mean = fit.rows_destandardized.loc[states == k].mean().to_numpy()
            centroid = fit.centroids_destandardized.iloc[k].to_numpy()
            assert np.abs(rows_mean - centroid).max() < 1e-9

    def test_destandardize_centroids_rejects_a_column_mismatch(self):
        X = _random_frame(60)
        params = standardization_params(X)
        with pytest.raises(ValueError, match="does not match"):
            destandardize_centroids(np.zeros((2, D - 1)), params, COLUMNS)


class TestTrapAUnitSpace:
    """Trap A: centroids compared across differently-standardized fits."""

    def test_trap_a_destandardized_distance_is_right_and_the_naive_one_is_wrong(self):
        """Both halves matter. The naive standardized comparison reports 0.0 — "stable" —
        on two fits whose true centroid geometry differs by a factor of two.
        """
        shift = 3.0
        X_ref = _clustered_frame([0.0, shift])
        X_sub = X_ref * 0.5  # every column halved: the SAME data at half the scale
        true_gap = shift * np.sqrt(D)

        ref = fit_for_stability(X_ref, K=2, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)
        sub = fit_for_stability(X_sub, K=2, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)

        # 1. The de-standardized geometry is recovered on each side.
        ref_gap = float(np.linalg.norm(
            ref.centroids_destandardized.iloc[1].to_numpy()
            - ref.centroids_destandardized.iloc[0].to_numpy()
        ))
        sub_gap = float(np.linalg.norm(
            sub.centroids_destandardized.iloc[1].to_numpy()
            - sub.centroids_destandardized.iloc[0].to_numpy()
        ))
        assert abs(ref_gap - true_gap) < 0.3, f"ref gap {ref_gap:.4f} != true {true_gap:.4f}"
        assert abs(sub_gap - 0.5 * true_gap) < 0.3, (
            f"sub gap {sub_gap:.4f} != true {0.5 * true_gap:.4f}"
        )

        # 2. The de-standardized matched distance sees the change.
        destd = match_states(ref.centroids_destandardized, sub.centroids_destandardized)
        high_state_distance = float(destd["matched_distance"][1])
        assert abs(high_state_distance - 0.5 * true_gap) < 0.3, (
            f"de-standardized matched distance for the high state is "
            f"{high_state_distance:.4f}; the true displacement is {0.5 * true_gap:.4f}"
        )

        # 3. The NAIVE standardized comparison does not. This is the trap.
        naive = match_states(
            pd.DataFrame(ref.centroids_standardized, columns=COLUMNS),
            pd.DataFrame(sub.centroids_standardized, columns=COLUMNS),
        )
        naive_distance = float(naive["matched_distance"].max())
        assert naive_distance < 1e-6, (
            "fixture broken: the two fits were supposed to have identical STANDARDIZED "
            "centroids so the trap is demonstrable"
        )
        assert naive_distance < 0.5 * high_state_distance, (
            f"TRAP A: comparing centroids_standardized across two independently-fitted "
            f"scalers reports a matched distance of {naive_distance:.3e} — i.e. "
            f"'perfectly stable' — when the true de-standardized displacement is "
            f"{high_state_distance:.4f}. standardize_features refits and DISCARDS its "
            f"scaler (jump_model.py:112-124), so the two centroid sets live in "
            f"different spaces. Always de-standardize before computing any distance."
        )


class TestFitForStability:
    def test_occupancy_is_length_k_and_sums_to_the_row_count(self):
        X = _clustered_frame([0.0, 2.0, 4.0], n_per=40)
        fit = fit_for_stability(X, K=3, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)
        assert fit.occupancy.shape == (3,)
        assert int(fit.occupancy.sum()) == len(X)

    def test_occupancy_keeps_a_k_length_array_when_a_state_captures_zero_months(self):
        """A huge lambda collapses the decode to one state; the other two still have entries.

        This is Trap B's precondition: a fit at fixed K ALWAYS returns K
        centroids and K occupancy slots, zeros included.
        """
        X = _clustered_frame([0.0, 2.0, 4.0], n_per=40)
        fit = fit_for_stability(X, K=3, lam=1e9, n_restarts=2, sort_column=SORT_COLUMN)
        assert fit.occupancy.shape == (3,)
        assert int(fit.occupancy.sum()) == len(X)
        assert int((fit.occupancy == 0).sum()) == 2
        assert fit.centroids_destandardized.shape == (3, D)

    def test_fit_for_stability_propagates_the_sort_column_error(self):
        X = _clustered_frame([0.0, 3.0])
        with pytest.raises(ValueError, match="sort_column"):
            fit_for_stability(X, K=2, lam=5.0, n_restarts=2, sort_column="not_a_column")

    def test_fit_for_stability_returns_a_stability_fit(self):
        X = _clustered_frame([0.0, 3.0])
        fit = fit_for_stability(X, K=2, lam=5.0, n_restarts=2, sort_column=SORT_COLUMN)
        assert isinstance(fit, StabilityFit)
        assert fit.columns == COLUMNS
        assert list(fit.states.index) == list(X.index)
