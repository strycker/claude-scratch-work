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

import json
from pathlib import Path

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
        rows, costs = summarize_subsample(
            ref, sub, reference_states_in_subsample=ref.states.to_numpy(),
            classifier=9, scheme="synthetic", null_reps=50, null_seed=1,
        )
        assert set(costs) == {"winsorized", "reference_sd"}
        assert all(c.shape == (3, 3) for c in costs.values())
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


class TestReferenceScaledCompanion:
    def test_primary_distance_is_blind_to_a_small_scale_column_and_the_companion_is_not(self):
        """The primary (winsorized-unit) distance is dominated by large-scale columns.

        Shift ONLY the small-scale column's centroid by a full reference SD: the
        primary distance barely moves (the large column dominates it), while the
        reference-SD companion reads ~1.0. Fails if the companion is computed in
        the same units as the primary, or not computed at all.
        """
        rng = np.random.default_rng(11)
        n = 120
        big = np.concatenate([rng.normal(0, 1000, n // 2), rng.normal(5000, 1000, n // 2)])
        small = np.concatenate([rng.normal(0, 0.01, n // 2), rng.normal(0.05, 0.01, n // 2)])
        X = pd.DataFrame({"big": big, "trailing_return_1m": small},
                         index=pd.date_range("2000-01-31", periods=n, freq="ME"))
        ref = fit_for_stability(X, K=2, lam=1.0, n_restarts=3, sort_column="trailing_return_1m")
        sd_small = float(ref.params["scale"]["trailing_return_1m"])
        moved = ref.centroids_destandardized.copy()
        moved["trailing_return_1m"] = moved["trailing_return_1m"] + sd_small
        sub = StabilityFit(
            states=ref.states, centroids_standardized=ref.centroids_standardized,
            centroids_destandardized=moved, columns=ref.columns, params=ref.params,
            occupancy=ref.occupancy, rows_destandardized=ref.rows_destandardized, K=2,
        )
        rows, costs = summarize_subsample(
            ref, sub, reference_states_in_subsample=ref.states.to_numpy(),
            classifier=9, scheme="synthetic", null_reps=50, null_seed=1,
        )
        for row in rows:
            assert row["matched_distance"] == pytest.approx(sd_small, rel=1e-9)
            assert row["matched_distance"] < 0.1, "primary: a full SD of the small column is ~invisible"
            assert row["refscaled_matched_distance"] == pytest.approx(1.0, rel=1e-9)
        assert costs["reference_sd"][0, 0] == pytest.approx(1.0, rel=1e-9)


# ── 3. the persisted artifacts ──

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "reports" / "platform" / "stability"
LADDER = (6, 12, 24, 48)
FAMILIES = {"drop_first_decade", "drop_last_decade", "circular_block_bootstrap", "leave_one_episode_out"}
K_BY_CLASSIFIER = {1: 6, 2: 5}


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.fail(f"{path} is missing — the criterion-3 artifacts are committed, not optional")
    return path


@pytest.fixture(scope="module")
def record():
    return json.loads(_require(ARTIFACT_DIR / "stability_record.json").read_text())


@pytest.fixture(scope="module")
def detail():
    return pd.read_parquet(_require(ARTIFACT_DIR / "stability_rows.parquet"))


@pytest.fixture(scope="module")
def costs():
    return pd.read_parquet(_require(ARTIFACT_DIR / "stability_cost_matrices.parquet"))


class TestArtifactCoverage:
    def test_exact_row_count_and_every_combination(self, record):
        rows = record["rows"]
        per_classifier_schemes = 2 + len(LADDER) + 1  # two decade drops, the ladder, LOO (own row)
        expected = sum(K * per_classifier_schemes for K in K_BY_CLASSIFIER.values())
        assert len(rows) == expected == 77
        assert {r["scheme"] for r in rows} == FAMILIES
        seen = {(r["classifier"], r["scheme"], r.get("block_length"), r["reference_state"]) for r in rows}
        assert len(seen) == len(rows), "duplicate (classifier, scheme, state) rows"
        for c, K in K_BY_CLASSIFIER.items():
            for state in range(K):
                for fam in ("drop_first_decade", "drop_last_decade", "leave_one_episode_out"):
                    assert (c, fam, None, state) in seen, (c, fam, state)
                for L in LADDER:
                    assert (c, "circular_block_bootstrap", L, state) in seen, (c, L, state)

    def test_detail_table_has_every_refit(self, record, detail):
        B = record["n_bootstrap"]
        for c, K in K_BY_CLASSIFIER.items():
            sub = detail[detail["classifier"] == c]
            n_fits = 2 + len(LADDER) * B + K
            assert len(sub) == n_fits * K
            assert record["classifiers"][str(c)]["n_subsample_fits"] == n_fits
            boot = sub[sub["scheme_family"] == "circular_block_bootstrap"]
            assert sorted(boot["block_length"].unique().tolist()) == list(LADDER)
            assert (boot.groupby("block_length")["replicate"].nunique() == B).all()


class TestArtifactOccupancyAndEvaporation:
    def test_every_row_carries_occupancy(self, record, detail):
        assert all(r.get("subsample_occupancy_months") is not None for r in record["rows"])
        assert detail["subsample_occupancy_months"].notna().all()

    def test_evaporated_is_exactly_zero_occupancy_on_every_refit(self, detail):
        # Derived from occupancy ALONE: equality, both directions, on every refit row.
        assert (detail["evaporated"] == (detail["subsample_occupancy_months"] == 0)).all()

    def test_record_evaporation_is_derived_from_occupancy(self, record):
        for r in record["rows"]:
            if r["scheme"] == "circular_block_bootstrap":
                assert r["evaporated"] is (r["n_replicates_evaporated"] > 0)
                if r["subsample_occupancy_months"] == 0:
                    assert r["evaporated"] is True
            else:
                assert r["evaporated"] is (r["subsample_occupancy_months"] == 0)


class TestArtifactNull:
    def test_null_n_is_the_rows_own_subsample_count(self, detail):
        assert (detail["split_half_null_n"] == detail["subsample_occupancy_months"]).all()

    def test_null_is_not_the_full_sample_count(self, record, detail):
        # Discriminating: the subsample count must differ from the full-sample count
        # on many rows, so the equality above cannot hold by coincidence.
        full = {(int(c), int(s)): v["occupancy_months"]
                for c, cl in record["classifiers"].items() for s, v in cl["full_sample"].items()}
        full_n = detail.apply(lambda r: full[(int(r["classifier"]), int(r["reference_state"]))], axis=1)
        assert (detail["split_half_null_n"] != full_n).mean() > 0.5

    def test_null_is_present_and_positive_wherever_a_split_exists(self, record, detail):
        live = detail[detail["subsample_occupancy_months"] >= 2]
        assert (live["split_half_null_median"] > 0).all()
        assert (live["refscaled_split_half_null_median"] > 0).all()
        dead = detail[detail["subsample_occupancy_months"] < 2]
        assert dead["split_half_null_median"].isna().all(), "a null with no split must be NaN, not 0"
        for r in record["rows"]:
            if r["subsample_occupancy_months"] >= 2 or r["scheme"] == "circular_block_bootstrap":
                assert r["split_half_null"]["median"] is not None and r["split_half_null"]["median"] > 0
                assert r["split_half_null"]["n"] == r["subsample_occupancy_months"]


class TestArtifactSchemes:
    def test_classifier1_state2_leave_one_episode_out_is_degenerate(self, record):
        loo = [r for r in record["rows"] if r["scheme"] == "leave_one_episode_out"
               and r["classifier"] == 1 and r["reference_state"] == 2]
        assert len(loo) == 1
        row = loo[0]
        assert row["degenerate"] is True
        assert row["n_months_dropped"] == 71
        assert row["dropped_span"] == ["1996-07", "2002-05"]
        assert row["n_episodes_before"] == 1
        assert row["reference_months_in_subsample"] == 0
        assert row["partner_overlap_months"] == 0

    def test_full_sample_episode_table_reproduces_research_5_1(self, record):
        fs = record["classifiers"]["1"]["full_sample"]
        assert {int(s): v["n_episodes"] for s, v in fs.items()} == {0: 9, 1: 5, 2: 1, 3: 4, 4: 3, 5: 4}
        assert {int(s): v["occupancy_months"] for s, v in fs.items()} == {0: 40, 1: 228, 2: 71, 3: 200, 4: 84, 5: 72}
        assert fs["2"]["episodes"] == [{"start": "1996-07", "end": "2002-05", "length": 71}]

    def test_only_one_leave_one_episode_out_is_degenerate_per_one_episode_state(self, record):
        for r in record["rows"]:
            if r["scheme"] == "leave_one_episode_out":
                fs = record["classifiers"][str(r["classifier"])]["full_sample"][str(r["reference_state"])]
                assert r["degenerate"] is (fs["n_episodes"] == 1)
                assert r["n_months_dropped"] == fs["longest_episode"]

    def test_bootstrap_rows_carry_positive_seam_counts(self, record, detail):
        boot = detail[detail["scheme_family"] == "circular_block_bootstrap"]
        assert boot["n_seams"].notna().all() and (boot["n_seams"] > 0).all()
        for r in record["rows"]:
            if r["scheme"] == "circular_block_bootstrap":
                assert r["n_seams"]["min"] > 0

    def test_decade_drops_remove_exactly_120_months(self, record):
        for r in record["rows"]:
            if r["scheme"] in ("drop_first_decade", "drop_last_decade"):
                ident = record["classifiers"][str(r["classifier"])]["reference_identity"]
                assert r["n_subsample_months"] == ident["n_months"] - 120


class TestArtifactCostMatrices:
    def test_every_refit_has_a_full_matrix_in_both_units(self, record, detail, costs):
        assert set(costs["units"].unique()) == {"winsorized", "reference_sd"}
        for c, K in K_BY_CLASSIFIER.items():
            sub = costs[costs["classifier"] == c]
            n_fits = record["classifiers"][str(c)]["n_subsample_fits"]
            assert len(sub) == n_fits * K * K * 2
            sizes = sub.groupby(["scheme", "replicate", "units"], dropna=False).size()
            assert (sizes == K * K).all()

    def test_persisted_matrix_is_the_one_that_produced_each_row(self, detail, costs):
        w = costs[costs["units"] == "winsorized"]
        key = ["classifier", "scheme", "replicate", "reference_state"]
        merged = detail.merge(
            w.rename(columns={"subsample_state": "matched_partner", "distance": "cost_at_partner"}),
            on=key + ["matched_partner"], how="left",
        )
        assert len(merged) == len(detail) and merged["cost_at_partner"].notna().all()
        assert np.allclose(merged["cost_at_partner"], merged["matched_distance"], rtol=0, atol=1e-9)


class TestArtifactNoThresholdNoSelection:
    def test_no_threshold_shaped_key(self, record):
        def keys(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield k
                    yield from keys(v)
            elif isinstance(o, list):
                for v in o:
                    yield from keys(v)
        bad = [k for k in keys(record) if any(s in k for s in ("threshold", "stable_if", "passes", "verdict"))]
        assert not bad, bad

    def test_condition_i_appears_on_classifier1_state0_only(self, record):
        for r in record["rows"]:
            scoped = r["classifier"] == 1 and r["reference_state"] == 0
            assert ("amendment_condition_i" in r) is scoped, (r["classifier"], r["scheme"], r["reference_state"])
            if scoped and r["scheme"] != "circular_block_bootstrap":
                cond = r["amendment_condition_i"]
                assert cond["holds"] is (r["subsample_episode_count"] >= 3)
                assert "three temporally separated episodes" in cond["condition"]

    def test_registry_unchanged_and_references_identical(self, record):
        assert record["registry_trial_count"]["before"] == record["registry_trial_count"]["after"] == 42
        for c, n in ((1, 695), (2, 696)):
            ident = record["classifiers"][str(c)]["reference_identity"]
            assert ident["mismatches"] == 0 and ident["n_months"] == n and ident["end"] == "2020-12-31"
