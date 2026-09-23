"""Guards for §4.4 criterion 3 as RUN by ``scripts/run_subsample_stability.py`` (plan 08-07).

Two layers, each written to be able to FAIL:

1. **The reference.** Both full-sample labelings are refit in process and must
   equal the tracked checkpoints elementwise (695 months / K=6; 696 months /
   K=5). The assertion's failure path is itself exercised on a perturbed series,
   so it cannot quietly be a log line. The frozen lists are pinned by name AND
   order, because ``canonicalize_states`` locates ``sort_column`` by position.

2. **Partner keying.** Occupancy, the split-half null and the episode count are
   read off the Hungarian-matched partner. A synthetic permuted fit proves the
   runner keys on the partner (and that same-id keying would read the wrong
   state's occupancy); a synthetic identity fit proves the runner reproduces
   ``stability.run_stability``'s rows exactly when the two conventions agree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from run_subsample_stability import (
    assert_reference_matches_checkpoint,
    build_frames,
    build_reference,
    summarize_subsample,
)

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.labeling.classifier2 import classifier2_config
from trading_crab_lib.platform.labeling.stability import (
    DEFAULT_STABILITY_SEED,
    StabilityFit,
    fit_for_stability,
    run_stability,
    scheme_drop_first_decade,
)

#: The ten columns classifier #1's tracked checkpoint reproduces from, in order.
CLASSIFIER1_FROZEN = [
    "cape_shiller", "credit_spread_baa_aaa", "curve_10y3m", "div_yield", "oil",
    "real_rate_level", "realized_vol_1m", "realized_vol_3m", "trailing_return_1m",
    "trailing_return_3m",
]


@pytest.fixture(scope="module")
def cfg():
    return load_platform_config()


@pytest.fixture(scope="module")
def frames(cfg):
    return build_frames(cfg)


@pytest.fixture(scope="module")
def references(cfg, frames):
    return {c: build_reference(cfg, c, frames) for c in (1, 2)}


# ── 1. the reference ──


class TestReferenceIdentity:
    @pytest.mark.parametrize(
        "classifier,checkpoint,n_months,n_states,start",
        [
            (1, "regime_labels", 695, 6, "1963-02-28"),
            (2, "regime_labels_2", 696, 5, "1963-01-31"),
        ],
    )
    def test_reference_reproduces_checkpoint_elementwise(
        self, references, classifier, checkpoint, n_months, n_states, start
    ):
        ref = references[classifier]
        assert ref["checkpoint"] == checkpoint
        record = assert_reference_matches_checkpoint(ref, checkpoint)
        assert record["mismatches"] == 0 and record["identical"] is True
        assert record["n_months"] == n_months
        assert record["n_states"] == n_states == ref["K"]
        assert record["start"] == start
        assert record["end"] == "2020-12-31", "the reference must end at the holdout boundary"

    def test_reference_pinned_constants(self, references):
        assert (references[1]["K"], references[1]["lam"], references[1]["n_restarts"]) == (6, 10.0, 10)
        assert references[1]["sort_column"] == "trailing_return_1m"
        assert (references[2]["K"], references[2]["lam"], references[2]["n_restarts"]) == (5, 16.0, 10)
        assert references[2]["sort_column"] == "rs_equities_bonds"

    def test_no_reference_row_is_post_holdout(self, frames, references):
        for c in (1, 2):
            assert frames[f"features_{c}"].index.max() <= pd.Timestamp("2020-12-31")
            assert references[c]["X"].index.max() <= pd.Timestamp("2020-12-31")


class TestMismatchPathRaises:
    def test_a_single_perturbed_month_raises_with_the_count(self, references):
        ref = references[1]
        perturbed = ref["fit"].states.copy()
        perturbed.iloc[[10, 300]] = (perturbed.iloc[[10, 300]] + 1) % ref["K"]
        with pytest.raises(AssertionError, match=r"2 mismatches of 695 months") as excinfo:
            assert_reference_matches_checkpoint(ref, "regime_labels", checkpoint_states=perturbed)
        assert str(ref["fit"].states.index[10].date()) in str(excinfo.value)

    def test_a_shifted_index_raises(self, references):
        ref = references[2]
        shifted = ref["fit"].states.iloc[1:]
        with pytest.raises(AssertionError, match="index differs"):
            assert_reference_matches_checkpoint(ref, "regime_labels_2", checkpoint_states=shifted)


class TestFrozenListIdentity:
    def test_classifier1_frozen_list_is_the_ten_in_order(self, frames, references):
        assert frames["frozen_1"] == CLASSIFIER1_FROZEN
        assert references[1]["frozen_columns"] == CLASSIFIER1_FROZEN
        assert references[1]["fit"].columns == CLASSIFIER1_FROZEN
        assert str(frames["first_decision"].date()) == "1972-01-31"

    def test_classifier1_is_not_the_lean_set(self, frames):
        # label_regimes selects the lean set; the checkpoint reproduces from the ten.
        assert len(frames["lean_cols"]) == 13
        assert set(CLASSIFIER1_FROZEN) < set(frames["lean_cols"])

    def test_classifier2_frozen_list_is_the_config_list_in_order(self, cfg, frames, references):
        expected = classifier2_config(cfg)["features"]
        assert frames["frozen_2"] == expected
        assert references[2]["fit"].columns == expected


# ── 2. partner keying ──


def _synthetic_frame(sizes: tuple[int, ...] = (30, 45, 60), seed: int = 3) -> pd.DataFrame:
    """Three well-separated blocks of DISTINCT sizes, so a same-id occupancy read is detectable."""
    rng = np.random.default_rng(seed)
    cols = ["a", "b", "trailing_return_1m"]
    blocks = []
    for size, shift in zip(sizes, (0.0, 4.0, 8.0)):
        blocks.append(rng.normal(0, 1, (size, len(cols))) + shift)
    arr = np.vstack(blocks)
    return pd.DataFrame(arr, columns=cols, index=pd.date_range("1990-01-31", periods=len(arr), freq="ME"))


def _permuted(fit: StabilityFit, perm: np.ndarray) -> StabilityFit:
    """The same fit with state ids relabeled old -> perm[old]."""
    inv = np.argsort(perm)
    return StabilityFit(
        states=pd.Series(perm[fit.states.to_numpy()], index=fit.states.index, name="state"),
        centroids_standardized=fit.centroids_standardized[inv],
        centroids_destandardized=fit.centroids_destandardized.iloc[inv].reset_index(drop=True),
        columns=fit.columns,
        params=fit.params,
        occupancy=fit.occupancy[inv],
        rows_destandardized=fit.rows_destandardized,
        K=fit.K,
    )


class TestPartnerKeying:
    def test_permuted_fit_is_read_off_the_partner(self):
        X = _synthetic_frame()
        ref = fit_for_stability(X, K=3, lam=2.0, n_restarts=3, sort_column="trailing_return_1m")
        assert ref.occupancy.tolist() == [30, 45, 60], "fixture must give distinct occupancies"
        perm = np.array([2, 0, 1])
        sub = _permuted(ref, perm)
        rows, cost = summarize_subsample(
            ref, sub, reference_states_in_subsample=ref.states.to_numpy(),
            classifier=9, scheme="synthetic", null_reps=50, null_seed=1,
        )
        assert cost.shape == (3, 3)
        for s, row in enumerate(rows):
            assert row["matched_partner"] == perm[s]
            assert row["is_identity"] is False
            assert row["matched_distance"] == pytest.approx(0.0, abs=1e-12)
            assert row["subsample_occupancy_months"] == int(ref.occupancy[s]), (
                "occupancy must be the PARTNER's; reading sub.occupancy[s] would report "
                f"{int(sub.occupancy[s])} for reference state {s}"
            )
            assert row["split_half_null_n"] == int(ref.occupancy[s])
            assert row["partner_overlap_months"] == int(ref.occupancy[s])

    def test_an_evaporated_partner_is_flagged_under_a_non_identity_assignment(self):
        X = _synthetic_frame()
        ref = fit_for_stability(X, K=3, lam=2.0, n_restarts=3, sort_column="trailing_return_1m")
        perm = np.array([1, 2, 0])
        sub = _permuted(ref, perm)
        # Reference state 0's partner is sub state 1. Empty it: its months go to
        # sub state 2, its centroid stays FROZEN where it was (Trap B).
        states = sub.states.to_numpy().copy()
        states[states == 1] = 2
        occupancy = np.bincount(states, minlength=3)
        sub = StabilityFit(
            states=pd.Series(states, index=sub.states.index, name="state"),
            centroids_standardized=sub.centroids_standardized,
            centroids_destandardized=sub.centroids_destandardized,
            columns=sub.columns, params=sub.params, occupancy=occupancy,
            rows_destandardized=sub.rows_destandardized, K=3,
        )
        rows, _ = summarize_subsample(
            ref, sub, reference_states_in_subsample=ref.states.to_numpy(),
            classifier=9, scheme="synthetic", null_reps=50, null_seed=1,
        )
        assert rows[0]["matched_partner"] == 1
        assert rows[0]["matched_distance"] == pytest.approx(0.0, abs=1e-12)
        assert rows[0]["evaporated"] is True, (
            "reference state 0's partner captured 0 months; a same-id read would report "
            f"sub state 0's {int(occupancy[0])} months and call it alive"
        )
        assert rows[0]["subsample_occupancy_months"] == 0
        assert occupancy[0] > 0  # the same-id read really would have been non-zero

    def test_runner_reproduces_run_stability_when_the_assignment_is_identity(self):
        X = _synthetic_frame(sizes=(60, 60, 60))
        kw = dict(K=3, lam=2.0, n_restarts=3, sort_column="trailing_return_1m")
        ref = fit_for_stability(X, **kw)
        positions = scheme_drop_first_decade(X.index, months=30)
        expected = run_stability(
            X, **kw, reference_fit=ref, schemes={"drop": positions}, classifier="9",
            null_reps=50, seed=DEFAULT_STABILITY_SEED,
        )
        sub = fit_for_stability(X.iloc[positions], **kw)
        ours, _ = summarize_subsample(
            ref, sub, reference_states_in_subsample=ref.states.to_numpy()[positions],
            classifier=9, scheme="drop", null_reps=50, null_seed=DEFAULT_STABILITY_SEED,
        )
        assert all(r["is_identity"] for r in expected), "fixture must produce an identity assignment"
        for e, o in zip(expected, ours):
            for key, value in e.items():
                if isinstance(value, float) and np.isnan(value):
                    assert np.isnan(o[key]), key
                else:
                    assert o[key] == value, key
