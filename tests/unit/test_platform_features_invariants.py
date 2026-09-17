"""Unit tests for trading_crab_lib.platform.features.invariants (INV-01, R4).

Synthetic monthly DataFrames, no network — mirrors
tests/unit/test_platform_features_relative.py's fixture-construction convention
and tests/unit/test_platform_labeling.py's synthetic-frame/caplog shape.

Task 2 (era-stability assessment + the trial-registry-logged screen) extends this
file with its own test classes in a later commit; every registry-touching test
added then passes an EXPLICIT tmp_path-based `registry_path` (or the NO_REGISTRY
sentinel) — never the default, which would write to the real git-tracked
`registry/trials.jsonl` ledger.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.features.invariants import (
    LOADING_STABILITY_TOLERANCE,
    InvariantCandidateSpec,
    InvariantScreenResult,
    compute_candidate_loadings,
    named_survivors,
)


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
