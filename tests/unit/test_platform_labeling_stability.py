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
    _recompute_centroids,
    standardize_features,
)
from trading_crab_lib.platform.labeling.stability import (
    BLOCK_LENGTH_LADDER,
    DEFAULT_STABILITY_SEED,
    EVAPORATED_OCCUPANCY_MONTHS,
    StabilityFit,
    destandardize_centroids,
    fit_for_stability,
    match_states,
    run_stability,
    scheme_circular_block_bootstrap,
    scheme_drop_first_decade,
    scheme_drop_last_decade,
    scheme_leave_one_episode_out,
    split_half_null,
    stability_row,
    standardization_params,
    state_episodes,
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


# ── Task 2: matching, the null, evaporation, margins ──


class TestHungarianMatching:
    def test_match_states_recovers_a_known_non_identity_permutation(self):
        X = _clustered_frame([0.0, 2.0, 4.0, 6.0], n_per=40)
        ref = fit_for_stability(X, K=4, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)

        perm = np.array([2, 0, 3, 1])
        sub = ref.centroids_destandardized.iloc[perm].reset_index(drop=True)
        result = match_states(ref.centroids_destandardized, sub)

        inverse = np.argsort(perm)
        assert result["assignment"] == {k: int(inverse[k]) for k in range(4)}
        assert float(result["matched_distance"].max()) < 1e-9
        assert result["is_identity"] is False
        assert result["cost_matrix"].shape == (4, 4)

    def test_match_states_reports_identity_when_nothing_moved(self):
        X = _clustered_frame([0.0, 3.0])
        ref = fit_for_stability(X, K=2, lam=5.0, n_restarts=2, sort_column=SORT_COLUMN)
        result = match_states(ref.centroids_destandardized, ref.centroids_destandardized)
        assert result["is_identity"] is True
        assert float(result["matched_distance"].max()) == 0.0

    def test_margin_near_one_flags_an_arbitrary_assignment_at_a_tiny_distance(self):
        """Margins near 1.0 say the assignment is arbitrary REGARDLESS of how small
        the matched distance is — the construction below has both."""
        K, radius = 4, 1e-3
        ref = np.zeros((K, D))
        for k in range(K):
            ref[k, k] = radius
        centre = ref.mean(axis=0)
        rng = np.random.default_rng(5)
        sub = centre[None, :] + rng.normal(0.0, radius * 1e-3, (K, D))

        result = match_states(
            pd.DataFrame(ref, columns=COLUMNS), pd.DataFrame(sub, columns=COLUMNS)
        )
        assert float(result["matched_distance"].max()) < 1e-2, "distances should be small"
        assert np.abs(result["margin"] - 1.0).max() < 0.2, (
            f"margins {np.round(result['margin'], 4).tolist()} should sit near 1.0 when "
            "every subsample centroid is equidistant from every reference centroid"
        )


class TestSplitHalfNull:
    def test_split_half_null_is_materially_non_zero_at_n_40(self):
        """The null is not zero. A distance-to-self implementation would return ~0."""
        rng = np.random.default_rng(1)
        rows = pd.DataFrame(rng.normal(0.0, 1.0, (40, D)), columns=COLUMNS)
        null = split_half_null(rows, n_reps=200, seed=DEFAULT_STABILITY_SEED)
        assert null["n"] == 40
        assert null["n_reps"] == 200
        assert null["median"] > 0.3, (
            f"split-half null median {null['median']:.4f} at n=40, d=10 is implausibly "
            "small — a null computed as a distance-to-self is ~0 and would make every "
            "matched distance look significant"
        )
        assert null["p10"] <= null["median"] <= null["p90"]

    def test_split_half_null_falls_as_n_rises(self):
        rng = np.random.default_rng(2)
        big = rng.normal(0.0, 1.0, (400, D))
        small = split_half_null(pd.DataFrame(big[:40], columns=COLUMNS), n_reps=200)
        large = split_half_null(pd.DataFrame(big, columns=COLUMNS), n_reps=200)
        assert large["median"] < small["median"], (
            f"null at n=400 ({large['median']:.4f}) should be below the null at n=40 "
            f"({small['median']:.4f}) on the same distribution"
        )

    def test_split_half_null_is_nan_when_no_split_exists(self):
        null = split_half_null(pd.DataFrame(np.zeros((1, D)), columns=COLUMNS))
        assert np.isnan(null["median"])
        assert null["n"] == 1


class TestTrapBEvaporation:
    """Trap B: a state that vanishes between halves is a stability failure even
    when the surviving states match well."""

    def test_evaporated_state_is_flagged_true_despite_a_zero_matched_distance(self):
        X = _clustered_frame([0.0, 2.0, 4.0], n_per=40)
        ref = fit_for_stability(X, K=3, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)

        # A subsample decode in which state 2 captured ZERO months.
        sub_states = ref.states.to_numpy().copy()
        sub_states[sub_states == 2] = 1
        assert int((sub_states == 2).sum()) == 0

        frozen = _recompute_centroids(
            standardize_features(X), sub_states, 3, ref.centroids_standardized
        )
        assert np.array_equal(frozen[2], ref.centroids_standardized[2]), (
            "_recompute_centroids' freeze-on-empty rule is what makes this trap real"
        )

        sub_destd = destandardize_centroids(frozen, ref.params, ref.columns)
        result = match_states(ref.centroids_destandardized, sub_destd)
        distance = float(result["matched_distance"][2])

        row = stability_row(
            classifier="test", scheme="trap_b", state=2,
            subsample_occupancy_months=0, subsample_occupancy_pct=0.0,
            matched_partner=result["assignment"][2], is_identity=result["is_identity"],
            matched_distance=distance, margin=result["margin"][2],
            split_half_null_median=float("nan"),
        )

        assert distance < 1e-12, f"expected a frozen (distance-0) centroid, got {distance:.3e}"
        assert row["subsample_occupancy_months"] == 0
        assert row["evaporated"] is True, (
            f"TRAP B: state 2 captured ZERO months yet its matched distance is "
            f"{distance:.3e}. That zero means _recompute_centroids FROZE the centroid "
            f"(jump_model.py:126-138), not that the state persisted. A reader scoring "
            f"'small distance => stable' marks an evaporated state stable, which is the "
            f"precise failure criterion 3 exists to catch. evaporated must outrank the "
            f"distance."
        )

    def test_a_state_with_a_few_months_is_not_flagged_evaporated(self):
        """Near-zero is a judgement. EVAPORATED is zero months exactly."""
        assert EVAPORATED_OCCUPANCY_MONTHS == 0
        row = stability_row(
            classifier="test", scheme="s", state=1,
            subsample_occupancy_months=3, subsample_occupancy_pct=0.01,
            matched_partner=1, is_identity=True, matched_distance=0.4, margin=0.3,
            split_half_null_median=0.7,
        )
        assert row["evaporated"] is False
        assert row["subsample_occupancy_months"] == 3

    def test_evaporation_is_logged_at_warning(self, caplog):
        with caplog.at_level("WARNING"):
            stability_row(
                classifier="classifier1", scheme="drop_last_decade", state=4,
                subsample_occupancy_months=0, subsample_occupancy_pct=0.0,
                matched_partner=4, is_identity=True, matched_distance=0.0, margin=1.0,
                split_half_null_median=float("nan"),
            )
        assert "EVAPORATED" in caplog.text
        assert "classifier1" in caplog.text
        assert "drop_last_decade" in caplog.text


class TestStateEpisodes:
    def test_state_episodes_reports_spans_counts_and_the_longest(self):
        states = pd.Series([0] * 15 + [1] * 10 + [2] * 20 + [1] * 10 + [0] * 6 + [1] * 10 + [0] * 4)
        episodes = state_episodes(states, n_states=3)
        assert episodes[0]["n_episodes"] == 3
        assert [e["length"] for e in episodes[0]["episodes"]] == [15, 6, 4]
        assert episodes[0]["longest_episode"] == 15
        assert episodes[0]["months"] == 25
        assert episodes[2]["n_episodes"] == 1
        span = episodes[2]["episodes"][0]
        assert (span["start"], span["end"], span["length"]) == (25, 44, 20)
        assert (span["start_label"], span["end_label"]) == (25, 44)  # Series index labels

    def test_state_episodes_surfaces_a_never_occupied_state(self):
        episodes = state_episodes(np.array([0] * 10 + [1] * 10), n_states=4)
        assert episodes[3]["n_episodes"] == 0
        assert episodes[3]["longest_episode"] == 0
        assert episodes[3]["months"] == 0


# ── Task 3: the four subsample schemes ──


class TestContiguousSchemes:
    def test_drop_first_decade_drops_the_leading_months(self):
        idx = pd.date_range("1963-02-28", periods=695, freq="ME")
        positions = scheme_drop_first_decade(idx)
        assert positions[0] == 120
        assert len(positions) == 695 - 120
        assert np.array_equal(positions, np.arange(120, 695))

    def test_drop_last_decade_drops_the_trailing_months(self):
        idx = pd.date_range("1963-02-28", periods=695, freq="ME")
        positions = scheme_drop_last_decade(idx)
        assert positions[-1] == 695 - 121
        assert len(positions) == 695 - 120

    def test_decade_drops_refuse_a_too_short_index(self):
        idx = pd.date_range("2015-01-31", periods=60, freq="ME")
        with pytest.raises(ValueError, match="cannot drop"):
            scheme_drop_first_decade(idx)
        with pytest.raises(ValueError, match="cannot drop"):
            scheme_drop_last_decade(idx)


class TestCircularBlockBootstrap:
    def test_length_is_preserved_for_every_ladder_value(self):
        idx = pd.date_range("1963-02-28", periods=695, freq="ME")
        for block in BLOCK_LENGTH_LADDER:
            positions, _ = scheme_circular_block_bootstrap(idx, block, seed=DEFAULT_STABILITY_SEED)
            assert len(positions) == 695, f"block_length={block} did not preserve n"
            assert positions.min() >= 0 and positions.max() < 695
        assert 695 % 48 != 0, "fixture must include a block length that does not divide n"

    def test_one_block_of_full_length_produces_exactly_zero_or_one_seam(self):
        """Both outcomes must occur: 1 at the wrap point, and 0 when the draw starts at 0.

        Asserting only ``n_seams in {0, 1}`` would be satisfied by the broken
        ``n_seams = n_blocks`` (one block => always 1). Pinning the zero case to
        the draw that starts at position 0 is what makes seams distinguishable
        from blocks.
        """
        idx = pd.date_range("1963-02-28", periods=101, freq="ME")
        seen_zero_start = seen_wrapped = False
        for seed in range(400):
            positions, n_seams = scheme_circular_block_bootstrap(idx, len(idx), seed=seed)
            assert len(positions) == len(idx)
            assert n_seams in {0, 1}, f"seed={seed} gave {n_seams} seams for a single block"
            if positions[0] == 0:
                assert n_seams == 0, (
                    f"seed={seed} starts at position 0 so the block is the original series "
                    f"in order and contains NO synthetic seam, but n_seams={n_seams} — "
                    "seams are being counted as blocks"
                )
                seen_zero_start = True
            else:
                assert n_seams == 1, f"seed={seed} wraps once but n_seams={n_seams}"
                seen_wrapped = True
        assert seen_zero_start and seen_wrapped, "fixture never exercised both cases"

    def test_unit_blocks_are_almost_all_seams_but_never_more_than_n_minus_one(self):
        idx = pd.date_range("1963-02-28", periods=300, freq="ME")
        _, n_seams = scheme_circular_block_bootstrap(idx, 1, seed=DEFAULT_STABILITY_SEED)
        assert n_seams >= len(idx) - 6, (
            f"block_length=1 should give ~{len(idx) - 1} seams, got {n_seams} — seams are "
            "not being counted"
        )
        assert n_seams <= len(idx) - 1, (
            f"n_seams={n_seams} exceeds the {len(idx) - 1} adjacent pairs that exist — "
            f"blocks ({len(idx)} of them here) are being counted instead of seams"
        )

    def test_seam_count_matches_the_calendar_adjacency_of_the_resampled_labels(self):
        """Independent formulation: a seam is any consecutive pair of resampled
        months that are not consecutive months in the original calendar."""
        idx = pd.date_range("1963-02-28", periods=120, freq="ME")
        for block in (3, 7, 24):
            positions, n_seams = scheme_circular_block_bootstrap(idx, block, seed=99)
            labels = idx[positions]
            expected = sum(
                1 for i in range(1, len(labels))
                if labels[i] != (labels[i - 1] + pd.offsets.MonthEnd(1))
            )
            assert n_seams == expected, (
                f"block_length={block}: reported {n_seams} seams but the resampled "
                f"calendar contains {expected} non-consecutive adjacencies"
            )

    def test_bootstrap_is_seeded_per_call(self):
        idx = pd.date_range("1963-02-28", periods=200, freq="ME")
        a, sa = scheme_circular_block_bootstrap(idx, 12, seed=DEFAULT_STABILITY_SEED)
        b, sb = scheme_circular_block_bootstrap(idx, 12, seed=DEFAULT_STABILITY_SEED)
        c, _ = scheme_circular_block_bootstrap(idx, 12, seed=DEFAULT_STABILITY_SEED + 1)
        assert np.array_equal(a, b) and sa == sb
        assert not np.array_equal(a, c)

    def test_block_length_is_validated(self):
        idx = pd.date_range("1963-02-28", periods=50, freq="ME")
        with pytest.raises(ValueError, match="block_length"):
            scheme_circular_block_bootstrap(idx, 51)
        with pytest.raises(ValueError, match="block_length"):
            scheme_circular_block_bootstrap(idx, 0)


class TestLeaveOneEpisodeOut:
    STATES = pd.Series([0] * 15 + [1] * 10 + [2] * 20 + [1] * 10 + [0] * 6 + [1] * 10 + [0] * 4)

    def test_a_one_episode_state_is_degenerate_and_that_is_the_answer(self):
        occupancy = np.bincount(self.STATES.to_numpy(), minlength=3)
        result = scheme_leave_one_episode_out(self.STATES, 2, n_states=3)
        assert result["degenerate"] is True, (
            "a state with exactly one episode cannot survive leave-one-episode-out; "
            "that degeneracy IS the finding (the design's own definition of an episode), "
            "reached with no invented threshold"
        )
        assert result["n_episodes_before"] == 1
        assert result["n_months_dropped"] == int(occupancy[2]) == 20
        assert int((self.STATES.to_numpy()[result["positions"]] == 2).sum()) == 0

    def test_a_three_episode_state_survives_on_its_two_shorter_episodes(self):
        result = scheme_leave_one_episode_out(self.STATES, 0, n_states=3)
        assert result["degenerate"] is False
        assert result["n_episodes_before"] == 3
        assert result["n_months_dropped"] == 15
        survivors = self.STATES.to_numpy()[result["positions"]]
        assert int((survivors == 0).sum()) == 10  # the 6-month and 4-month episodes
        assert state_episodes(survivors, n_states=3)[0]["n_episodes"] == 2

    def test_an_unoccupied_state_raises_rather_than_silently_surviving(self):
        """Returning the full sample for a never-occupied state would report it as
        having survived the scheme. It has already evaporated; say so loudly."""
        never_occupied = pd.Series([0] * 20)
        with pytest.raises(ValueError, match="zero episodes"):
            scheme_leave_one_episode_out(never_occupied, 1, n_states=3)


# ── Task 3: the runner and Trap C ──


class TestTrapCFrozenColumns:
    def test_run_stability_raises_when_the_subsample_column_set_differs(self):
        X = _clustered_frame([0.0, 2.0, 4.0], n_per=40)
        ref = fit_for_stability(X, K=3, lam=5.0, n_restarts=2, sort_column=SORT_COLUMN)
        X_missing = X.drop(columns=["gold"])

        with pytest.raises(ValueError) as excinfo:
            run_stability(
                X_missing, K=3, lam=5.0, n_restarts=2, sort_column=SORT_COLUMN,
                reference_fit=ref,
                schemes={"drop_first": scheme_drop_first_decade(X_missing.index, months=20)},
            )
        message = str(excinfo.value)
        assert "Trap C" in message
        assert f"{D - 1} surviving vs {D} expected" in message, message
        assert "gold" in message

    def test_run_stability_raises_when_a_column_is_all_nan_within_the_subsample(self):
        """Trap C's realistic channel: the freeze rule admits a column only if it is
        non-NaN over the window, so a subsample can silently shrink the feature set."""
        X = _clustered_frame([0.0, 2.0, 4.0], n_per=40)
        X.loc[X.index[:50], "gold"] = np.nan
        ref = fit_for_stability(X, K=3, lam=5.0, n_restarts=2, sort_column=SORT_COLUMN)

        with pytest.raises(ValueError, match="Trap C"):
            run_stability(
                X, K=3, lam=5.0, n_restarts=2, sort_column=SORT_COLUMN,
                reference_fit=ref, schemes={"first_50": np.arange(50)},
            )


class TestRunStability:
    def test_run_stability_emits_one_row_per_scheme_and_state(self):
        X = _clustered_frame([0.0, 2.0, 4.0], n_per=40)
        ref = fit_for_stability(X, K=3, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)
        block_positions, n_seams = scheme_circular_block_bootstrap(
            X.index, 12, seed=DEFAULT_STABILITY_SEED
        )
        rows = run_stability(
            X, K=3, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN,
            reference_fit=ref, classifier="synthetic", null_reps=25,
            schemes=[
                ("drop_first_decade", scheme_drop_first_decade(X.index, months=20), {}),
                ("drop_last_decade", scheme_drop_last_decade(X.index, months=20), {}),
                ("block_12", block_positions,
                 {"block_length": 12, "n_seams": n_seams, "seed": DEFAULT_STABILITY_SEED}),
            ],
        )
        assert len(rows) == 3 * 3
        required = {
            "classifier", "scheme", "state", "evaporated", "subsample_occupancy_months",
            "subsample_occupancy_pct", "matched_partner", "is_identity", "matched_distance",
            "margin", "split_half_null_median", "split_half_null_n", "n_episodes",
            "longest_episode",
        }
        for row in rows:
            assert required <= set(row)
            assert isinstance(row["evaporated"], bool)
        block_rows = [r for r in rows if r["scheme"] == "block_12"]
        assert all(r["n_seams"] == n_seams and r["block_length"] == 12 for r in block_rows)
        assert all(r["seed"] == DEFAULT_STABILITY_SEED for r in block_rows)

    def test_run_stability_accepts_a_leave_one_episode_out_dict_and_carries_degeneracy(self):
        X = _clustered_frame([0.0, 2.0, 4.0], n_per=40)
        ref = fit_for_stability(X, K=3, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)
        loo = scheme_leave_one_episode_out(ref.states, 0, n_states=3)
        rows = run_stability(
            X, K=3, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN,
            reference_fit=ref, classifier="synthetic", null_reps=25,
            schemes=[("loo_state0", loo, {})],
        )
        assert len(rows) == 3
        assert all(r["degenerate"] is loo["degenerate"] for r in rows)

    def test_run_stability_carries_the_null_beside_every_matched_distance(self):
        X = _clustered_frame([0.0, 2.0, 4.0], n_per=40)
        ref = fit_for_stability(X, K=3, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN)
        rows = run_stability(
            X, K=3, lam=5.0, n_restarts=3, sort_column=SORT_COLUMN,
            reference_fit=ref, classifier="synthetic", null_reps=50,
            schemes={"drop_first_decade": scheme_drop_first_decade(X.index, months=20)},
        )
        for row in rows:
            if row["subsample_occupancy_months"] >= 2:
                assert row["split_half_null_median"] > 0.0, (
                    "a matched distance without its n-matched null is unreadable; the "
                    "null is not zero (08-RESEARCH.md §5.2)"
                )
                assert row["split_half_null_n"] == row["subsample_occupancy_months"]


class TestNoThresholdAndNoVerdict:
    def test_the_module_defines_no_persistence_threshold_and_emits_no_verdict(self):
        from trading_crab_lib.platform.labeling import stability

        row = stability_row(
            classifier="c", scheme="s", state=0, subsample_occupancy_months=40,
            subsample_occupancy_pct=0.1, matched_partner=0, is_identity=True,
            matched_distance=0.5, margin=0.4, split_half_null_median=0.7,
        )
        for banned in ("verdict", "passed", "stable", "persists", "is_regime"):
            assert banned not in row, f"stability_row emits a verdict field {banned!r}"
        assert stability.EVAPORATED_OCCUPANCY_MONTHS == 0
        names = [n for n in vars(stability) if n.isupper()]
        assert set(names) == {
            "EVAPORATED_OCCUPANCY_MONTHS", "BLOCK_LENGTH_LADDER",
            "DEFAULT_STABILITY_SEED", "DECADE_MONTHS",
        }, f"unexpected module constant (a persistence threshold?): {names}"

    def test_block_length_ladder_brackets_both_median_sojourns(self):
        assert BLOCK_LENGTH_LADDER == (6, 12, 24, 48)
        assert min(BLOCK_LENGTH_LADDER) < 9.5 < max(BLOCK_LENGTH_LADDER)   # classifier #1
        assert min(BLOCK_LENGTH_LADDER) < 29.0 < max(BLOCK_LENGTH_LADDER)  # classifier #2
