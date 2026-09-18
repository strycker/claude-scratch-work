"""Unit tests for trading_crab_lib.platform.features.invariants (INV-01, R4).

Synthetic monthly DataFrames, no network — mirrors
tests/unit/test_platform_features_relative.py's fixture-construction convention
and tests/unit/test_platform_labeling.py's synthetic-frame/caplog shape.

Every registry-touching test passes an EXPLICIT tmp_path-based `registry_path` (or
the NO_REGISTRY sentinel) — never the default, which would write to the real
git-tracked `registry/trials.jsonl` ledger.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.features.invariants import (
    DEFAULT_ERA_MIN_TRAIN_MONTHS,
    DEFAULT_ERA_STEP_MONTHS,
    INVARIANT_CANDIDATES,
    LOADING_STABILITY_TOLERANCE,
    InvariantCandidateSpec,
    InvariantScreenResult,
    _classify_stability,
    compute_candidate_loadings,
    loading_stability_across_eras,
    named_survivors,
    screen_invariant_candidates,
)
from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
from trading_crab_lib.platform.honesty.registry import NO_REGISTRY, read_trials


def _spec(name: str) -> InvariantCandidateSpec:
    return InvariantCandidateSpec(name=name, source_columns=(name,), description=f"synthetic {name}")


def _monthly_index(n_months: int, start: str = "1990-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_months, freq="ME")


# ── compute_candidate_loadings (Task 1) ─────────────────────────────────────


class TestComputeCandidateLoadings:
    def test_returns_frame_indexed_by_ordered_candidate_names(self):
        idx = _monthly_index(60)
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "credit_gdp": rng.normal(0, 1, 60).cumsum(),
                "m2_gdp": rng.normal(0, 1, 60).cumsum(),
            },
            index=idx,
        )
        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        loadings = compute_candidate_loadings(df, candidates)
        # index must equal the ORDERED candidate list, element for element -- not
        # sorted alphabetically (which would put credit_gdp first) and not a set.
        assert list(loadings.index) == ["m2_gdp", "credit_gdp"]

    def test_rejects_integer_or_component_label_index(self):
        """A frame indexed by integers or component labels is exactly the
        anonymous-component shape R4 forbids -- pin the NAMED-index contract."""
        idx = _monthly_index(60)
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {"m2_gdp": rng.normal(0, 1, 60).cumsum(), "credit_gdp": rng.normal(0, 1, 60).cumsum()},
            index=idx,
        )
        loadings = compute_candidate_loadings(df, [_spec("m2_gdp"), _spec("credit_gdp")])
        assert not all(isinstance(i, int) for i in loadings.index)
        assert set(loadings.index) == {"m2_gdp", "credit_gdp"}

    def test_never_exposes_component_scores(self):
        """The returned frame's shape is (n_candidates, n_components) -- never
        (n_rows, n_components), which would be the transformed SCORES (the R4
        seam this function must structurally be unable to hand back)."""
        idx = _monthly_index(60)
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {"m2_gdp": rng.normal(0, 1, 60).cumsum(), "credit_gdp": rng.normal(0, 1, 60).cumsum()},
            index=idx,
        )
        loadings = compute_candidate_loadings(df, [_spec("m2_gdp"), _spec("credit_gdp")])
        assert len(loadings) == 2  # one row per candidate, never one row per timestamp (60)

    def test_duplicate_candidate_shares_loading_sign_and_magnitude(self):
        """A pure duplicate is a discovery SIGNAL, reported, never an automatic
        rejection -- the two candidates' loadings must be equal (same sign AND
        magnitude) within LOADING_STABILITY_TOLERANCE."""
        idx = _monthly_index(80)
        rng = np.random.default_rng(4)
        a = rng.normal(0, 1, 80).cumsum()
        df = pd.DataFrame({"m2_gdp": a, "credit_gdp": a.copy()}, index=idx)
        loadings = compute_candidate_loadings(df, [_spec("m2_gdp"), _spec("credit_gdp")], n_components=1)
        v_a = loadings.loc["m2_gdp", "pc1"]
        v_b = loadings.loc["credit_gdp", "pc1"]
        assert v_a == pytest.approx(v_b, abs=1e-9)
        assert abs(v_a - v_b) <= LOADING_STABILITY_TOLERANCE

    def test_all_nan_candidate_excluded_with_warning(self, caplog):
        idx = _monthly_index(60)
        rng = np.random.default_rng(5)
        df = pd.DataFrame(
            {
                "m2_gdp": rng.normal(0, 1, 60).cumsum(),
                "credit_gdp": np.nan,
            },
            index=idx,
        )
        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        with caplog.at_level(logging.WARNING):
            loadings = compute_candidate_loadings(df, candidates)
        excluded = {c.name for c in candidates} - set(loadings.index)
        assert "credit_gdp" in excluded
        assert any("credit_gdp" in r.message for r in caplog.records)

    def test_missing_column_excluded_with_warning(self, caplog):
        idx = _monthly_index(60)
        rng = np.random.default_rng(6)
        df = pd.DataFrame({"m2_gdp": rng.normal(0, 1, 60).cumsum()}, index=idx)
        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        with caplog.at_level(logging.WARNING):
            loadings = compute_candidate_loadings(df, candidates)
        excluded = {c.name for c in candidates} - set(loadings.index)
        assert "credit_gdp" in excluded
        assert any("credit_gdp" in r.message for r in caplog.records)

    def test_no_computable_candidate_raises(self):
        idx = _monthly_index(60)
        df = pd.DataFrame({"unrelated_col": np.arange(60.0)}, index=idx)
        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        with pytest.raises(ValueError, match="no INV-01 candidate is computable"):
            compute_candidate_loadings(df, candidates)


class TestNamedSurvivorsDeterminism:
    def test_named_survivors_determinism(self):
        """Two named_survivors() calls on the same results input return the
        identical list, element for element -- never merely equal sets/lengths."""
        results = [
            InvariantScreenResult(
                name="m2_gdp",
                source_columns=("fred_m2sl", "fred_gdp"),
                verdict="survive",
                stability_verdict="stable",
            ),
            InvariantScreenResult(
                name="credit_gdp",
                source_columns=("fred_totalsl", "fred_gdp"),
                verdict="reject",
                stability_verdict="not_assessed",
                rejection_reason="x",
            ),
        ]
        first = named_survivors(results)
        second = named_survivors(results)
        assert first == second == ["m2_gdp"]


# ── _classify_stability (the per-era classification logic) ─────────────────


class TestClassifyStability:
    def test_stable_within_tolerance_no_sign_flip(self):
        assert _classify_stability([0.5, 0.55, 0.52]) == "stable"

    def test_unstable_on_sign_flip(self):
        assert _classify_stability([0.5, -0.5]) == "unstable"

    def test_unstable_on_magnitude_range_exceeding_tolerance(self):
        assert _classify_stability([0.1, 0.1 + LOADING_STABILITY_TOLERANCE + 0.01]) == "unstable"

    def test_stable_at_exact_tolerance_boundary(self):
        assert _classify_stability([0.1, 0.1 + LOADING_STABILITY_TOLERANCE]) == "stable"


# ── loading_stability_across_eras (Task 2: era-stability, walk-forward) ────


class TestLoadingStabilityAcrossEras:
    def _consistent_pair_frame(self, n_months: int = 300, seed: int = 7) -> pd.DataFrame:
        idx = _monthly_index(n_months)
        rng = np.random.default_rng(seed)
        base = rng.normal(0, 1, n_months).cumsum()
        return pd.DataFrame(
            {"m2_gdp": base + rng.normal(0, 0.05, n_months), "credit_gdp": base + rng.normal(0, 0.05, n_months)},
            index=idx,
        )

    def test_era_count_is_a_handful_not_one_per_month(self):
        df = self._consistent_pair_frame(n_months=300)
        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        result = loading_stability_across_eras(df, candidates, df.index, min_train=100, step=50)
        n_eras = len(result["m2_gdp"]["per_era_loadings"])
        assert 2 <= n_eras <= 10  # a handful, not one of the ~200 eligible months

    def test_consistently_correlated_pair_reports_stable(self):
        df = self._consistent_pair_frame(n_months=300)
        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        result = loading_stability_across_eras(df, candidates, df.index, min_train=100, step=50)
        assert result["m2_gdp"]["stability_verdict"] == "stable"
        assert result["credit_gdp"]["stability_verdict"] == "stable"

    def test_not_assessed_when_zero_eras_fit(self):
        df = self._consistent_pair_frame(n_months=50)
        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        # min_train exceeds the frame length -> expanding_steps yields nothing.
        result = loading_stability_across_eras(df, candidates, df.index, min_train=100, step=50)
        assert result["m2_gdp"]["stability_verdict"] == "not_assessed"
        assert result["m2_gdp"]["per_era_loadings"] == []

    def test_era_windows_never_leak_future_values_into_an_early_era(self):
        """Structural leak test (T-07-10): a later era's loadings must never
        differ from a run truncated at that later date's own window end, for
        the EARLIER era shared by both runs -- if a future value leaked in,
        the two would differ."""
        n_months = 300
        leak_start = 250
        idx = _monthly_index(n_months)
        rng = np.random.default_rng(8)
        base = rng.normal(0, 1, n_months).cumsum()
        b = base.copy()
        # Deliberately extreme, divergent values AFTER leak_start -- if a
        # future row leaked into an earlier era's PCA fit, its loadings would
        # be measurably different from the truncated run's.
        base_full = base.copy()
        b_full = b.copy()
        base_full[leak_start:] += 1_000_000.0
        b_full[leak_start:] -= 1_000_000.0
        full_frame = pd.DataFrame({"m2_gdp": base_full, "credit_gdp": b_full}, index=idx)
        truncated_frame = full_frame.iloc[:leak_start]

        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        full_result = loading_stability_across_eras(
            full_frame, candidates, full_frame.index, min_train=100, step=50
        )
        truncated_result = loading_stability_across_eras(
            truncated_frame, candidates, truncated_frame.index, min_train=100, step=50
        )

        # Era 1 (train_index = index[:100], i.e. strictly before leak_start=250
        # in BOTH runs) must be byte-for-byte identical between the two runs.
        full_era1 = full_result["m2_gdp"]["per_era_loadings"][0]
        truncated_era1 = truncated_result["m2_gdp"]["per_era_loadings"][0]
        assert full_era1[0] == truncated_era1[0]  # same era-end date
        assert full_era1[1] == pytest.approx(truncated_era1[1], abs=1e-12)

    def test_holdout_never_visits_a_date_past_the_cutoff(self):
        """No era timestamp this function produces exceeds DEFAULT_HOLDOUT_CUTOFF
        when the caller restricts decision_index to the dev side."""
        n_months = 800  # 1990-01 + 800 months extends well past 2020-12-31
        idx = _monthly_index(n_months)
        rng = np.random.default_rng(9)
        base = rng.normal(0, 1, n_months).cumsum()
        df = pd.DataFrame({"m2_gdp": base, "credit_gdp": base.copy()}, index=idx)
        dev_df, _holdout_df = split_by_holdout_boundary(df)
        candidates = [_spec("m2_gdp"), _spec("credit_gdp")]
        result = loading_stability_across_eras(
            dev_df, candidates, dev_df.index, min_train=DEFAULT_ERA_MIN_TRAIN_MONTHS, step=DEFAULT_ERA_STEP_MONTHS
        )
        cutoff = pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)
        max_visited = max(
            pd.Timestamp(era_end) for era_end, _loading in result["m2_gdp"]["per_era_loadings"]
        )
        assert max_visited <= cutoff


# ── screen_invariant_candidates (Task 2/3: the full screen) ─────────────────


def _full_monthly_raw(n_months: int = 750, seed: int = 42) -> pd.DataFrame:
    """A monthly_raw-shaped synthetic frame with INV-01's source columns fully
    valid over the whole span (mirrors the real checkpoint's GDP-limited-start
    behavior per 07-05-SUMMARY.md), extending well past the 2020-12-31 holdout
    cutoff so screen_invariant_candidates' own carve is exercised for real."""
    idx = _monthly_index(n_months, start="1962-01-31")
    rng = np.random.default_rng(seed)
    fred_gdp = 500.0 + np.cumsum(rng.normal(2.0, 0.3, n_months))
    fred_m2sl = 300.0 + np.cumsum(rng.normal(1.0, 0.2, n_months))
    fred_totalsl = 200.0 + np.cumsum(rng.normal(0.8, 0.2, n_months))
    return pd.DataFrame(
        {"fred_gdp": fred_gdp, "fred_m2sl": fred_m2sl, "fred_totalsl": fred_totalsl}, index=idx
    )


CFG = {"backtest": {"min_train_months": DEFAULT_ERA_MIN_TRAIN_MONTHS}}


class TestScreenInvariantCandidatesHoldout:
    def test_holdout_boundary_applied_before_any_ratio_is_computed(self, tmp_path):
        monthly_raw = _full_monthly_raw()
        registry_path = tmp_path / "trials.jsonl"
        results = screen_invariant_candidates(
            monthly_raw, CFG, registry_path=registry_path, trial_tag="test-holdout"
        )
        cutoff = pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)
        for result in results:
            for era_end, _loading in result.per_era_loadings:
                assert pd.Timestamp(era_end) <= cutoff, (
                    f"candidate {result.name} has an era dated {era_end}, after the "
                    f"holdout cutoff {cutoff}"
                )

    def test_no_result_admissible_month_exceeds_the_holdout_cutoff(self, tmp_path):
        monthly_raw = _full_monthly_raw()
        registry_path = tmp_path / "trials.jsonl"
        results = screen_invariant_candidates(
            monthly_raw, CFG, registry_path=registry_path, trial_tag="test-holdout-2"
        )
        cutoff = pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)
        for result in results:
            if result.first_admissible_month is not None:
                assert pd.Timestamp(result.first_admissible_month) <= cutoff


class TestScreenInvariantCandidatesRegistry:
    def test_registry_exact_row_count_per_candidate_with_tag(self, tmp_path):
        monthly_raw = _full_monthly_raw()
        registry_path = tmp_path / "trials.jsonl"
        results = screen_invariant_candidates(
            monthly_raw, CFG, registry_path=registry_path, trial_tag="07-07-test-tag"
        )
        rows = read_trials(registry_path)
        assert len(rows) == len(INVARIANT_CANDIDATES) == len(results)

    def test_registry_sentinel_writes_zero_rows_but_full_result_set(self, tmp_path):
        monthly_raw = _full_monthly_raw()
        registry_path = tmp_path / "trials.jsonl"  # never created if NO_REGISTRY is honored
        results = screen_invariant_candidates(monthly_raw, CFG, registry_path=NO_REGISTRY)
        rows = read_trials(registry_path)
        assert len(rows) == 0
        assert len(results) == len(INVARIANT_CANDIDATES)

    def test_missing_trial_tag_surfaces_append_trial_refusal(self, tmp_path):
        monthly_raw = _full_monthly_raw()
        registry_path = tmp_path / "trials.jsonl"
        with pytest.raises(ValueError, match="trial_tag"):
            screen_invariant_candidates(monthly_raw, CFG, registry_path=registry_path, trial_tag=None)


class TestScreenInvariantCandidatesResults:
    def test_returns_a_result_for_every_candidate_including_rejects(self, tmp_path):
        # Missing fred_totalsl -> credit_gdp cannot be computed at all -> reject,
        # while m2_gdp (fred_m2sl/fred_gdp both present) survives. Exercises
        # BOTH verdicts in one screen without touching INVARIANT_CANDIDATES.
        monthly_raw = _full_monthly_raw().drop(columns=["fred_totalsl"])
        registry_path = tmp_path / "trials.jsonl"
        results = screen_invariant_candidates(
            monthly_raw, CFG, registry_path=registry_path, trial_tag="test-mixed"
        )
        names_by_verdict = {r.name: r.verdict for r in results}
        assert names_by_verdict == {"m2_gdp": "survive", "credit_gdp": "reject"}

    def test_every_rejected_candidate_has_a_nonempty_reason(self, tmp_path):
        monthly_raw = _full_monthly_raw().drop(columns=["fred_totalsl"])
        registry_path = tmp_path / "trials.jsonl"
        results = screen_invariant_candidates(
            monthly_raw, CFG, registry_path=registry_path, trial_tag="test-reasons"
        )
        for result in results:
            if result.verdict == "reject":
                assert result.rejection_reason.strip() != ""

    def test_freeze_rule_rejection_has_its_own_distinct_reason(self, tmp_path):
        """A candidate present but failing D-11's common-support freeze (a NaN
        gap inside the decision range) gets a DIFFERENT reason than a candidate
        that was never computable at all."""
        monthly_raw = _full_monthly_raw()
        # Punch a NaN gap into fred_m2sl well inside the post-1972 decision
        # range so m2_gdp is "present" but fails the freeze -- credit_gdp is
        # untouched and should still survive.
        monthly_raw = monthly_raw.copy()
        gap_start = monthly_raw.index[400]
        gap_end = monthly_raw.index[410]
        monthly_raw.loc[gap_start:gap_end, "fred_m2sl"] = float("nan")
        registry_path = tmp_path / "trials.jsonl"
        results = screen_invariant_candidates(
            monthly_raw, CFG, registry_path=registry_path, trial_tag="test-freeze-reject"
        )
        by_name = {r.name: r for r in results}
        assert by_name["m2_gdp"].verdict == "reject"
        assert "freeze" in by_name["m2_gdp"].rejection_reason
        assert by_name["credit_gdp"].verdict == "survive"

    def test_all_candidates_survive_on_the_realistic_fully_valid_fixture(self, tmp_path):
        monthly_raw = _full_monthly_raw()
        registry_path = tmp_path / "trials.jsonl"
        results = screen_invariant_candidates(
            monthly_raw, CFG, registry_path=registry_path, trial_tag="test-both-survive"
        )
        assert named_survivors(results) == [c.name for c in INVARIANT_CANDIDATES]

    def test_no_candidate_computable_raises(self, tmp_path):
        monthly_raw = _full_monthly_raw().drop(columns=["fred_m2sl", "fred_totalsl", "fred_gdp"])
        registry_path = tmp_path / "trials.jsonl"
        with pytest.raises(ValueError, match="none of INV-01's named candidates"):
            screen_invariant_candidates(monthly_raw, CFG, registry_path=registry_path, trial_tag="test-empty")

    def test_results_are_in_invariant_candidates_declared_order(self, tmp_path):
        monthly_raw = _full_monthly_raw()
        registry_path = tmp_path / "trials.jsonl"
        results = screen_invariant_candidates(
            monthly_raw, CFG, registry_path=registry_path, trial_tag="test-order"
        )
        assert [r.name for r in results] == [c.name for c in INVARIANT_CANDIDATES]
