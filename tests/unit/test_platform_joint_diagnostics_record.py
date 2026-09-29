"""Guards for the three wave-2 quantities that were REPORTED but not GOVERNED.

Phase 7 closed with several numbers recorded in prose and in a JSON diagnostics
record, with nothing in the suite able to notice if they changed. This file is
that notice. It is deliberately *pin*-shaped, not band-shaped: no band exists
for any of these quantities, and inventing one would repeat the phase's own
signature defect (a bound wider than the quantity's reachable range, which can
only confirm). A pin can fail; a made-up band cannot.

Covered here:

1. **Filtered-labeling churn** (07-12 open item 5). Classifier #1's walk-forward
   *filtered* labeling changes state across 246 of the 587 adjacent pairs in 588
   decision months (41.91% = 246/587).
   (Superseded: the 41.84% recorded before plan 08-01 divided by months; see F-4.)
   against a 3.60% full-sample rate. That churn feeds the tilt directly and no
   band governs it. The tests below RE-DERIVE the churn from the persisted
   per-step ``state_1`` / ``state_2`` columns and assert it equals the recorded
   diagnostics -- so the record cannot drift away from the data it describes,
   and a change in the churn itself breaks a test instead of passing silently.

2. **Ablation validity.** The joint and baseline legs must walk the same
   classifier paths; only the blend differs. If ``state_1`` ever diverged
   between legs, the measured lift would not be an ablation at all. Nothing
   asserted this.

3. **`n_resolved` as a live quantity** (07-12 open item 6, and the defect class
   where ``compute_sojourn_lag_headline`` returned ``n_resolved = 0,
   median_lag = NaN`` silently on a wrong matrix shape). Classifier #2 resolved
   5 of 12 transitions. A regression to 0 would look like "detection never
   happened" and would read as a NaN ratio rather than as an error -- pinned
   here, plus a direct unit-level demonstration of the silent-zero signature.

4. **ADR-0002's probe-edge table** (07-12 open item 8), which names tests but is
   documentation. The test below asserts every test file and test function the
   table names still exists. That guards NAMES ONLY -- it cannot notice a named
   test whose body drifted away from the edge it is cited for. That limitation
   is the finding, and it is recorded in 07-VALIDATION.md rather than papered
   over here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation.churn import argmax_churn, read_probability_matrix
from trading_crab_lib.platform.evaluation.sojourn_lag import compute_sojourn_lag_headline

_ROOT = Path(__file__).resolve().parents[2]
_JOINT = _ROOT / "outputs" / "reports" / "platform" / "joint_lift"
_ADR = _ROOT / "platform_design" / "adr" / "0002-l1-second-classifier.md"
_TESTS_UNIT = Path(__file__).resolve().parent

#: The two routings run in plan 07-11. Both are persisted; only the L1-only one
#: is decision-bearing (ADR-0002 decision (e)), but both must be internally
#: consistent, so both are checked.
_SUFFIXES = ("l1only", "l2")

#: 07-11-SUMMARY / 07-12-SUMMARY open item 5, as measured 2026-09-21.
_PINNED_FILTERED_TRANSITIONS = {"classifier_1": 246, "classifier_2": 24}
_PINNED_N_STEPS = 588
#: Track B's window under the firewalled L2 routing: the driver accumulates a
#: per-step row only on NON-degraded steps (joint_driver.py:508-510), so the
#: probability matrix is 100 rows shorter than the curve. Pitfall 6: a churn
#: rate quoted without this count is not quotable.
_PINNED_L2_DEGRADED = 100
_PINNED_L2_NOWCAST_ROWS = _PINNED_N_STEPS - _PINNED_L2_DEGRADED
#: Open item 6: 5 of 12 -- a drop to 0 is the "detection never happened" shape.
_PINNED_C2_RESOLVED = (5, 12)


def _diagnostics(suffix: str) -> dict:
    name = "diagnostics_l1only.json" if suffix == "l1only" else "diagnostics_l2_observational.json"
    path = _JOINT / name
    assert path.is_file(), f"{path} missing -- it is a git-tracked wave-2 artifact"
    return json.loads(path.read_text())


def _measurement(suffix: str) -> dict:
    name = "measurement_l1only.json" if suffix == "l1only" else "measurement_l2_observational.json"
    path = _JOINT / name
    assert path.is_file(), f"{path} missing -- it is a git-tracked wave-2 artifact"
    payload = json.loads(path.read_text())
    for key in ("joint", "leg_joint", "legs"):
        if key in payload and isinstance(payload[key], dict) and "n_degraded" in payload[key]:
            return payload[key]
    if "n_degraded" in payload:
        return payload
    for value in payload.values():
        if isinstance(value, dict) and "n_degraded" in value:
            return value
    raise AssertionError(f"no n_degraded in {path}")


def _curve(suffix: str, leg: str) -> pd.DataFrame:
    path = _JOINT / f"joint_lift_{leg}_{suffix}.parquet"
    assert path.is_file(), f"{path} missing -- it is a git-tracked wave-2 artifact"
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def table() -> str:
    """ADR-0002's eight-row probe-edge section, isolated from its amendment."""
    assert _ADR.is_file(), f"{_ADR} missing"
    text = _ADR.read_text()
    start = text.index("The eight spec-less probe edges")
    end = text.index("AMENDMENT 2026-09-21", start)
    section = text[start:end]
    assert section.count("**resolved**") >= 8
    return section


def _n_state_changes(states: pd.Series) -> int:
    """Adjacent-month state changes, the same count the driver records."""
    return int((states != states.shift()).iloc[1:].sum())


# ── 1. Filtered-labeling churn: the record must match the data ──────────────


class TestFilteredChurnIsReDerivedFromTheCurve:
    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("clf,col", [("classifier_1", "state_1"), ("classifier_2", "state_2")])
    def test_recorded_transition_count_equals_the_count_in_the_curve(self, suffix, clf, col):
        """The JSON diagnostics are a *claim*; the parquet is the evidence."""
        recorded = _diagnostics(suffix)[clf]["walk_forward_filtered"]["n_transitions"]
        measured = _n_state_changes(_curve(suffix, "joint")[col])
        assert measured == recorded, (
            f"{suffix}/{clf}: diagnostics record {recorded} filtered transitions but the "
            f"persisted curve's {col} column contains {measured}"
        )

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_recorded_rate_equals_count_over_steps(self, suffix, clf):
        """F-4 (plan 08-01): the rate is denominated in adjacent PAIRS.

        The name is kept for traceability to the plan that moved this pin; the
        body no longer divides by steps. Before 08-01 this asserted
        ``n_transitions / n_steps`` -- a correct assertion about an incorrect
        quantity, since a change is a property of a pair and 588 months carry
        587 pairs.
        """
        wf = _diagnostics(suffix)[clf]["walk_forward_filtered"]
        assert wf["n_steps"] == _PINNED_N_STEPS
        assert wf["n_pairs"] == wf["n_steps"] - 1
        assert wf["transition_rate"] == pytest.approx(wf["n_transitions"] / wf["n_pairs"], abs=1e-12)

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_recorded_rate_is_not_the_month_denominated_value(self, suffix, clf):
        """Rejects F-4's old denominator explicitly.

        Without this, the pair-denominated assertion above passes under EITHER
        denominator whenever the record and the test are changed together -- the
        failure mode this project has recorded three times. 246/587 and 246/588
        differ by ~6e-4, far outside the 1e-9 tolerance.
        """
        wf = _diagnostics(suffix)[clf]["walk_forward_filtered"]
        assert wf["transition_rate"] != pytest.approx(wf["n_transitions"] / wf["n_steps"], abs=1e-9), (
            f"{suffix}/{clf}: the recorded filtered rate divides by MONTHS again (F-4 regression)"
        )

    @pytest.mark.parametrize("clf,pinned", sorted(_PINNED_FILTERED_TRANSITIONS.items()))
    def test_churn_is_pinned_at_its_reported_value(self, clf, pinned):
        """Open item 5's number, pinned. No band governs it -- so it is pinned.

        A change here is not necessarily a bug, but it MUST be a deliberate,
        visible act: the reported 41.91% / 4.09% churn (246/587, 24/587) is what the tilt trades
        on, and it must not move unnoticed.
        """
        measured = _diagnostics("l1only")[clf]["walk_forward_filtered"]["n_transitions"]
        assert measured == pinned

    def test_filtered_churn_is_an_order_of_magnitude_above_full_sample(self):
        """The asymmetry itself -- the thing open item 5 is about.

        Fails if the filtered path were ever smoothed into agreement with the
        full-sample labeling (which would silently erase the finding) or if the
        full-sample rate blew up to meet it.
        """
        c1 = _diagnostics("l1only")["classifier_1"]
        ratio = c1["walk_forward_filtered"]["transition_rate"] / c1["full_sample"]["transition_rate"]
        assert ratio == pytest.approx(11.63, rel=0.02), (
            "classifier #1's filtered-vs-full-sample churn ratio moved off its "
            f"recorded ~11.6x (measured {ratio:.3f})"
        )

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("col", ["state_1", "state_2"])
    def test_the_churn_measurement_is_not_degenerate(self, suffix, col):
        """A churn number computed off a constant or NaN column proves nothing.

        This is the counterpart of the ``n_resolved = 0`` failure: a quantity
        that LOOKS measured because the arithmetic completed.
        """
        states = _curve(suffix, "joint")[col]
        assert len(states) == _PINNED_N_STEPS
        assert states.notna().all(), f"{col} carries NaN -- churn over it is not a measurement"
        assert states.nunique() >= 2, f"{col} is constant -- a 0% churn would be an artefact"


# ── 1b. The two churn series: Track A and Track B, never interchangeable ────


class TestTheTwoChurnSeriesAreSeparatelyDenominated:
    """08-RESEARCH.md § F-1's split, pinned so neither series can wear the other's name.

    Track A is ``state_N`` — the L1 jump model's terminal-month label
    (``joint_driver.py:502``). Track B is the argmax of the persisted per-step
    probability matrix, the object design §5.1 changes. Under the
    decision-bearing ``ROUTING_L1_ONLY`` they are the SAME SERIES by
    construction, and that degeneracy is what these tests record.
    """

    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_under_l1only_the_argmax_IS_the_state_column(self, clf):
        """The identity pin — a pin of a DEGENERACY, not of a success.

        Under ``ROUTING_L1_ONLY`` the tilt is fed
        ``_last_state_one_hot(states_1)`` (``joint_driver.py:431``), so the
        "nowcast churn" is the label churn wearing a different name and every
        criterion-2 number measured on Track A is unmovable by an L2 change.

        If this ever flips to False, something has begun feeding the tilt a
        NON-degenerate probability vector under the decision-bearing routing.
        That is not necessarily wrong — but it means every criterion-7 number
        recorded before the change was measured against a different input and is
        NO LONGER COMPARABLE to any number measured after it.
        """
        ident = _diagnostics("l1only")["series_identity"][clf]
        assert ident["argmax_equals_state_elementwise"] is True, (
            f"l1only/{clf}: the one-hot degeneracy at joint_driver.py:431 no longer "
            "holds. Pre-change criterion-7 numbers are not comparable to post-change ones."
        )
        assert ident["n_mismatched_months"] == 0
        assert ident["n_compared"] == _PINNED_N_STEPS

    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_under_l2_the_argmax_is_NOT_the_state_column(self, clf):
        """The falsifier of "the split is cosmetic".

        If Track A and Track B were one object read twice, this assertion could
        not fail under any routing. It fails here only because the L2 routing
        actually calls ``_refit_l2`` and the nowcaster's argmax departs from the
        jump model's terminal label.
        """
        ident = _diagnostics("l2")["series_identity"][clf]
        assert ident["argmax_equals_state_elementwise"] is False, (
            f"l2/{clf}: the nowcaster's argmax is elementwise identical to {ident['state_column']} "
            "— the two 'series' are one object and the split records nothing."
        )
        assert ident["n_mismatched_months"] > 0

    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_the_two_blocks_carry_different_denominators_under_l2(self, clf):
        """Track A spans every step; Track B spans only the non-degraded ones.

        A test that asserted only "both blocks exist" could not fail on a
        copy-paste. These three numbers can only agree if the two blocks were
        computed from two different objects over two different windows.
        """
        rec = _diagnostics("l2")[clf]
        assert rec["walk_forward_filtered"]["n_steps"] == _PINNED_N_STEPS
        assert rec["walk_forward_nowcast"]["n_rows"] == _PINNED_L2_NOWCAST_ROWS
        assert rec["walk_forward_nowcast"]["n_degraded"] == _PINNED_L2_DEGRADED
        assert (
            rec["walk_forward_nowcast"]["n_rows"]
            + rec["walk_forward_nowcast"]["n_degraded"]
            == _PINNED_N_STEPS
        )

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_each_block_names_its_own_track_in_the_record(self, suffix, clf):
        """The strings are part of the artifact, not a comment in the source."""
        rec = _diagnostics(suffix)[clf]
        assert rec["walk_forward_filtered"]["track"].startswith("A —")
        assert rec["walk_forward_nowcast"]["track"].startswith("B —")
        assert "state_N" in rec["walk_forward_filtered"]["track"]
        assert "§5.1" in rec["walk_forward_nowcast"]["track"]

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_track_b_is_re_derivable_from_its_own_named_artifact(self, suffix, clf):
        """The record names a parquet; the parquet must reproduce the number.

        This is what makes Track B falsifiable rather than merely reported: the
        rate is recomputed here from the persisted matrix, not trusted.
        """
        block = _diagnostics(suffix)[clf]["walk_forward_nowcast"]
        source = Path(block["source"])
        matrix = read_probability_matrix(source if source.is_absolute() else _ROOT / source)
        measured = argmax_churn(matrix)
        assert measured["n_changes"] == block["n_changes"]
        assert measured["n_rows"] == block["n_rows"]
        assert measured["rate"] == pytest.approx(block["rate"], abs=1e-12)

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("clf,pinned", sorted(_PINNED_FILTERED_TRANSITIONS.items()))
    def test_track_a_count_is_unchanged_by_the_split(self, suffix, clf, pinned):
        """Adding Track B must not have moved Track A's own number."""
        assert _diagnostics(suffix)[clf]["walk_forward_filtered"]["n_transitions"] == pinned


# ── 2. Ablation validity: the legs must share the classifier paths ──────────


class TestBothLegsWalkTheSameClassifierPaths:
    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("col", ["state_1", "state_2"])
    def test_joint_and_baseline_legs_share_the_state_path_exactly(self, suffix, col):
        joint, baseline = _curve(suffix, "joint"), _curve(suffix, "baseline")
        assert joint.index.equals(baseline.index)
        mismatches = int((joint[col].to_numpy() != baseline[col].to_numpy()).sum())
        assert mismatches == 0, (
            f"{suffix}: {col} differs between the joint and baseline legs in {mismatches} "
            "months -- the measured lift is then not an ablation of the blend alone"
        )

    @pytest.mark.parametrize("col", ["state_1", "state_2"])
    def test_the_two_routings_share_the_state_path_exactly(self, col):
        """Routing changes what is DONE with the labels, never the labels."""
        a, b = _curve("l1only", "joint")[col], _curve("l2", "joint")[col]
        assert int((a.to_numpy() != b.to_numpy()).sum()) == 0

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    def test_degraded_step_count_matches_the_measurement_record(self, suffix):
        """Degraded steps hold the previous weights, so they dampen measured
        churn. The count is recorded (0 under L1-only, 100 of 588 under the
        firewalled L2 routing) and is re-derived here from the curve itself."""
        recorded = _measurement(suffix)["n_degraded"]
        assert int(_curve(suffix, "joint")["degraded"].sum()) == recorded
        assert int(_curve(suffix, "baseline")["degraded"].sum()) == recorded


# ── 3. n_resolved: the live pin, and the silent-zero signature ──────────────


class TestResolvedTransitionCountIsLive:
    def test_classifier_2_resolved_count_is_pinned(self):
        sl = _diagnostics("l1only")["classifier_2"]["sojourn_lag"]
        assert (sl["n_resolved"], sl["n_transitions"]) == _PINNED_C2_RESOLVED

    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_a_reported_ratio_is_never_backed_by_zero_resolved_transitions(self, clf):
        """``n_resolved = 0`` with a finite ratio is the shape that once read as
        'detection never happened' while looking like a completed measurement."""
        sl = _diagnostics("l1only")[clf]["sojourn_lag"]
        assert sl["n_transitions"] > 0
        if np.isfinite(sl["ratio"]):
            assert sl["n_resolved"] > 0, f"{clf}: finite ratio on 0 resolved transitions"
            assert np.isfinite(sl["median_lag"])

    def test_wrong_column_labels_now_RAISE_instead_of_a_silent_zero(self):
        """T0.12 CLOSED 2026-09-21 — this test previously asserted the defect.

        It was written by the wave-2 validation audit to pin that the silent zero
        was *still live* at the function boundary: a probability matrix keyed by
        ``state_{k}`` strings returned a clean-looking ``n_resolved = 0 /
        median_lag = NaN`` with no exception and no warning, and only the caller
        stood between that and a published number.

        Glenn marked it blocking and it was fixed. The pin is inverted rather than
        deleted, so the history stays visible and a regression that restores the
        silent zero fails here — which is exactly what the original pin promised:
        "any future hardening of this boundary is a visible, test-breaking change".
        """
        index = pd.date_range("1972-01-31", periods=12, freq="ME")
        states = pd.Series([0] * 4 + [1] * 4 + [0] * 4, index=index)
        good = pd.DataFrame({0: 0.0, 1: 0.0}, index=index)
        good.loc[index[6:8], 1] = 0.95   # state 1 detected 2 months late
        good.loc[index[10:], 0] = 0.95   # state 0 detected 2 months late

        # Positive control: the integer-keyed path still measures.
        resolved = compute_sojourn_lag_headline(states, good)
        assert resolved["n_resolved"] > 0, "positive control failed -- detection is broken"
        assert np.isfinite(resolved["median_lag"])

        # The defect's own input now raises rather than returning a zero.
        mislabelled = good.rename(columns={0: "state_0", 1: "state_1"})
        with pytest.raises(ValueError, match="CANONICAL INTEGER state labels"):
            compute_sojourn_lag_headline(states, mislabelled)


# ── 4. ADR-0002's probe-edge table: names only, and says so ─────────────────


class TestProbeEdgeTableNamesRealTests:
    """Turns eight rows of prose into something that can fail.

    Scope limit, stated rather than hidden: this checks that every test file and
    test function the table cites EXISTS. It cannot check that a cited test
    still tests the edge it is cited for.
    """


    def test_the_table_cites_a_nontrivial_number_of_tests(self, table):
        names = set(re.findall(r"\btest_[a-z0-9_]+\b", table))
        functions = {n for n in names if not n.endswith("_py")}
        assert len(functions) >= 20, f"probe-edge table cites only {len(functions)} tests"

    def test_every_cited_test_file_exists(self, table):
        files = sorted(set(re.findall(r"\b(test_[a-z0-9_]+\.py)\b", table)))
        assert files
        missing = [f for f in files if not (_TESTS_UNIT / f).is_file()]
        assert not missing, f"probe-edge table cites nonexistent test files: {missing}"

    def test_every_cited_test_function_is_defined_somewhere_in_the_suite(self, table):
        cited = set(re.findall(r"\btest_[a-z0-9_]+\b", table))
        cited = {n for n in cited if f"{n}.py" not in table or not (_TESTS_UNIT / f"{n}.py").is_file()}
        defined = set()
        for path in _TESTS_UNIT.glob("test_*.py"):
            defined.update(re.findall(r"^\s*def (test_[a-z0-9_]+)\(", path.read_text(), re.M))
        missing = sorted(cited - defined)
        assert not missing, (
            "ADR-0002's probe-edge table names tests that no longer exist: " + ", ".join(missing)
        )


# ── 5. Plan 08-08: three churn numbers — A (pinned), B0 (the control), B1 (reported) ──

#: Track B's pre-filter value, QUOTED from 08-01-SUMMARY.md ("221 / 487 = 45.38%",
#: "66 / 487 = 13.55%", 488 rows, 100 degraded), not re-derived after the change —
#: re-deriving it afterwards would compare against a number nobody can reproduce.
#: The nowcaster is untouched by plan 08-08, so B0 must not move; that invariance is
#: the CONTROL that makes any movement in B1 attributable to the filter.
_PINNED_B0_FROM_0801 = {"classifier_1": (221, 487), "classifier_2": (66, 487)}


def _artifact(block: dict) -> pd.DataFrame:
    source = Path(block["source"])
    return read_probability_matrix(source if source.is_absolute() else _ROOT / source)


class TestThreeChurnNumbers:
    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("clf,pinned", sorted(_PINNED_FILTERED_TRANSITIONS.items()))
    def test_track_a_is_unchanged_in_both_routings(self, suffix, clf, pinned):
        got = _diagnostics(suffix)[clf]["walk_forward_filtered"]["n_transitions"]
        assert got == pinned, (
            f"{suffix}/{clf}: Track A moved {pinned} -> {got}. Plan 08-08 changes only what is "
            "done with the L2 posterior; if state_N moved, something touched L1. That is a BUG, not a result."
        )

    @pytest.mark.parametrize("clf,pinned", sorted(_PINNED_B0_FROM_0801.items()))
    def test_b0_the_raw_posterior_churn_is_unchanged_from_0801(self, clf, pinned):
        block = _diagnostics("l2")[clf]["walk_forward_nowcast"]
        assert (block["n_changes"], block["n_pairs"]) == pinned, (
            f"{clf}: B0 moved from 08-01's {pinned} — the nowcaster's output changed, so B1's "
            "movement can no longer be attributed to the filter."
        )

    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_the_degraded_count_is_unchanged_at_100_of_588_for_b0_and_b1(self, clf):
        """No feature column was added, so _cv_safe_active_features cannot degrade more
        often. A churn change bought by more degraded steps is not an improvement."""
        rec = _diagnostics("l2")[clf]
        assert rec["walk_forward_nowcast"]["n_degraded"] == _PINNED_L2_DEGRADED
        assert rec["walk_forward_belief"]["n_degraded"] == _PINNED_L2_DEGRADED
        assert rec["walk_forward_belief"]["n_rows"] + _PINNED_L2_DEGRADED == _PINNED_N_STEPS
        assert rec["walk_forward_belief"]["index_equals_nowcast_index"] is True

    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_b1_is_re_derivable_from_its_own_named_artifact(self, clf):
        block = _diagnostics("l2")[clf]["walk_forward_belief"]
        assert block["track"].startswith("B1 —")
        assert "joint_lift_belief_" in block["source"]
        measured = argmax_churn(_artifact(block))
        assert measured["n_changes"] == block["n_changes"]
        assert measured["n_pairs"] == block["n_pairs"]
        assert measured["rate"] == pytest.approx(block["rate"], abs=1e-12)

    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_b1_is_not_trivially_b0(self, clf):
        """A no-op filter would otherwise report a 'result'. Recomputed from the two
        artifacts, and checked against the record."""
        rec = _diagnostics("l2")[clf]
        belief, posterior = _artifact(rec["walk_forward_belief"]), _artifact(rec["walk_forward_nowcast"])
        assert belief.index.equals(posterior.index)
        n_mismatched = int((belief.idxmax(axis=1) != posterior.idxmax(axis=1)).sum())
        assert n_mismatched > 0
        assert n_mismatched == rec["walk_forward_belief"]["n_mismatched_months_vs_posterior"]

    @pytest.mark.parametrize("clf", ["classifier_1", "classifier_2"])
    def test_under_l1only_there_is_no_belief_by_design(self, clf):
        # diagnostics_l1only.json is the decision-bearing record and is not regenerated
        # by plan 08-08; when it is, its belief block must say "not applicable".
        block = _diagnostics("l1only")[clf].get("walk_forward_belief", {"applicable": False})
        assert block["applicable"] is False
        assert not (_JOINT / f"joint_lift_belief_{clf[-1]}_l1only.parquet").exists()


def _churn_target_assertions(source: str) -> list[str]:
    """Assertions that compare B1's churn against a fixed number or against B0.

    B1 = any expression mentioning ``walk_forward_belief`` together with
    ``n_changes`` or ``rate``. A violation is a comparison whose other side is a
    numeric literal or mentions ``walk_forward_nowcast`` (a direction against B0).
    Comparisons against a value RE-DERIVED from an artifact are not targets.
    """
    import ast

    tree = ast.parse(source)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        for cmp in [n for n in ast.walk(node.test) if isinstance(n, ast.Compare)]:
            sides = [cmp.left, *cmp.comparators]
            texts = [ast.unparse(side) for side in sides]
            is_b1 = [("walk_forward_belief" in t and ("n_changes" in t or "rate" in t)) for t in texts]
            if not any(is_b1):
                continue
            for side, text, b1 in zip(sides, texts, is_b1):
                if b1:
                    continue
                if (isinstance(side, ast.Constant) and isinstance(side.value, (int, float))) or (
                    "walk_forward_nowcast" in text
                ):
                    found.append(ast.unparse(cmp))
    return found


class TestNoChurnTargetIsAsserted:
    def test_the_detector_fires_on_a_target_and_on_a_direction(self):
        """The check must be able to fail: shown on two violating snippets."""
        target = "assert rec['walk_forward_belief']['n_changes'] < 200\n"
        direction = "assert rec['walk_forward_belief']['rate'] < rec['walk_forward_nowcast']['rate']\n"
        rederived = "assert measured['n_changes'] == block_walk_forward_belief_n_changes\n"
        assert _churn_target_assertions(target)
        assert _churn_target_assertions(direction)
        assert not _churn_target_assertions(rederived)

    @pytest.mark.parametrize("name", ["test_platform_joint_diagnostics_record.py", "test_platform_nowcaster_recursion.py"])
    def test_no_module_asserts_a_b1_target_or_direction(self, name):
        text = (_TESTS_UNIT / name).read_text()
        # The detector's own demonstration snippets live in string literals, which
        # ast.parse does not treat as assertions.
        assert _churn_target_assertions(text) == [], name


# ── 6. Plan 08-10: the A11 quality tier inside joint_lift_table ─────────────
#
# 08-A11.md (Glenn, 2026-09-21, option b-promote-dsr) promoted the deflated-Sharpe
# hurdle to the one gate that can fail a bad-but-working model. Its code consequence
# was handed to plan 08-10, to land BEFORE criterion 7 is re-measured. These arms are
# synthetic on purpose: they must hold whatever the re-measured record says.


def _gate_curve(mean: float, sd: float, n: int = 240, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    return pd.DataFrame({"return": rng.normal(mean, sd, n)}, index=idx)


class TestTheA11QualityTierIsInTheLiftTable:
    def test_a_sharpe_above_the_hurdle_passes_and_one_below_fails(self):
        """Both sides of the gate, in one table: without the passing arm the gate could
        be a constant False; without the failing arm, a constant True."""
        from trading_crab_lib.platform.backtest.joint_driver import joint_lift_table

        strong, weak = _gate_curve(0.05, 0.02), _gate_curve(0.005, 0.04)
        out = joint_lift_table(strong, weak, n_trials=44, sharpe_variance=1.0)
        assert out["joint_observed_sharpe"] > out["quality_tier_hurdle"] > out["baseline_observed_sharpe"]
        assert out["joint_quality_tier_ok"] is True
        assert out["baseline_quality_tier_ok"] is False
        assert out["joint_dsr"] > 0.5 > out["baseline_dsr"]

    @pytest.mark.parametrize("n_trials", [2, 42, 44, 1000])
    def test_the_hurdle_is_expected_max_sharpe_at_the_count_passed(self, n_trials):
        from trading_crab_lib.platform.backtest.joint_driver import joint_lift_table
        from trading_crab_lib.platform.evaluation.deflated_sharpe import expected_max_sharpe

        out = joint_lift_table(_gate_curve(0.01, 0.03), _gate_curve(0.01, 0.03, seed=8), n_trials=n_trials,
                               sharpe_variance=1.0)
        assert out["quality_tier_hurdle"] == expected_max_sharpe(n_trials, 1.0)
        assert out["quality_tier_n_trials"] == n_trials
        for leg in ("joint", "baseline"):
            assert out[f"{leg}_quality_tier_ok"] is (
                out[f"{leg}_observed_sharpe"] > out["quality_tier_hurdle"]
            ), "dsr > 0.5 and observed_sharpe > hurdle are one statement (08-A11.md §3.1)"

    def test_without_counts_the_hurdle_is_read_live_not_from_a_literal(self, monkeypatch):
        """08-A11.md §5.3: 'The hurdle may not be written as a literal'. A patched
        registry count must move the hurdle; a frozen 2.208694 would not."""
        from trading_crab_lib.platform.backtest import joint_driver as jd
        from trading_crab_lib.platform.evaluation.deflated_sharpe import expected_max_sharpe

        curve = _gate_curve(0.01, 0.03)
        monkeypatch.setattr(jd.registry, "total_trial_count", lambda *a, **k: 1000)
        monkeypatch.setattr(jd, "registry_sharpe_variance", lambda *a, **k: 1.0)
        out = jd.joint_lift_table(curve, curve)
        assert out["quality_tier_n_trials"] == 1000
        assert out["quality_tier_hurdle"] == expected_max_sharpe(1000, 1.0)
        assert out["quality_tier_hurdle"] > expected_max_sharpe(44, 1.0)

    def test_the_boundary_is_exclusive(self, monkeypatch):
        """A DSR of exactly 0.5 is a Sharpe EQUAL to the hurdle: equalling the bar is
        not clearing it. Mutating ``>`` to ``>=`` turns this red."""
        from trading_crab_lib.platform.backtest import joint_driver as jd

        monkeypatch.setattr(jd, "deflated_sharpe_ratio", lambda **kw: 0.5)
        assert jd.quality_tier(_gate_curve(0.05, 0.02)["return"], n_trials=44, sharpe_variance=1.0)["ok"] is False

    def test_a_leg_with_no_sharpe_does_not_pass(self):
        """A constant leg has no Sharpe; it is reported undefined, and a hurdle cannot
        be cleared by a number that does not exist."""
        from trading_crab_lib.platform.backtest.joint_driver import quality_tier

        idx = pd.date_range("1990-01-31", periods=24, freq="ME")
        q = quality_tier(pd.Series(0.01, index=idx), n_trials=44, sharpe_variance=1.0)
        assert q["defined"] is False and q["ok"] is False
        assert np.isnan(q["dsr"])

    def test_the_four_plausibility_bands_were_not_retuned(self):
        """A11 promoted the DSR hurdle and ruled NO band change (08-A11.md §5.3)."""
        from trading_crab_lib.platform.backtest import joint_driver as jd

        assert (jd.WEALTH_DELTA_UNIVERSAL, jd.WEALTH_DELTA_DOMAIN) == (15.0, 5.0)
        assert (jd.DD_DELTA_UNIVERSAL, jd.DD_DELTA_DOMAIN) == ((-1.0, 1.0), 0.5)


# ── 7. Plan 08-10: criterion 7 re-measured — both legs, one harness, window inline ──
#
# The prior decision-bearing value (07-11, band off) was wealth_delta -0.12343826162064975,
# dd_delta +0.02408401236666291 over 588 steps 1972-01-31 -> 2020-12-31, at 42 trials. It is
# a COMPARISON POINT, not a target, and no assertion below compares against it: the band
# (08-A7.md, b-bounded-turnover) is not inert on l1only, so the number is expected to move.

_RECORDS = {"l1only": "measurement_l1only.json", "l2": "measurement_l2_observational.json"}
_WINDOW = "588 steps, 1972-01-31 -> 2020-12-31"
#: 08-A7.md / 08-A11.md budget: 42 before 08-10; A7 authorises exactly 2 rows, A11 spends 0.
_REGISTRY_BEFORE, _REGISTRY_AFTER = 42, 44
_TAGS_0810 = ("08-10-c1-alone-L1only-notrade5pp", "08-10-joint-c1xc2-L1only-notrade5pp")


def _record(suffix: str) -> dict:
    return json.loads((_JOINT / _RECORDS[suffix]).read_text())


# ── 7a. Plan 08-11 (G-08-1): one comparison routine, portable across platforms ──
#
# The record is a claim; the parquet curves are the evidence. The re-derivation recomputes the
# claim from the evidence, and that recomputation is a floating-point REDUCTION
# (``np.log1p(r).sum()`` over 588 months, then norm.cdf / exp / log for the DSR and hurdle).
# Reductions are not bit-portable: on Apple Silicon (NEON + macOS libm) terminal log wealth lands
# 32 ULP away from the x86-glibc value that wrote the record -- relative 7.1e-15 (UAT test 1:
# -0.12530657740828932 recomputed vs -0.1253065774082902 recorded). A GENUINE mismatch (the
# band-off 07-11 record against the band-on 08-10 curves) differs at ~1.5e-2 relative. The
# tolerance sits between them, twelve orders of magnitude from each, and the arms in
# ``TestTheReDerivationToleranceDiscriminates`` prove it does.
#
# ``abs=0.0`` is MANDATORY. ``pytest.approx(x, rel=1e-9)`` with no ``abs`` keeps the default
# absolute floor of 1e-12, and the committed DSRs are ~1.8e-13 and ~4.1e-12 (l1only) and ~1e-35 /
# ~1e-45 (l2): a rel-only approx accepts ``0.0 == approx(1.8e-13, rel=1e-9)`` (measured True, pytest
# 9.1.1). That would be a check that can only confirm.
_PORTABLE_REL = 1e-9

_FLOAT_KEYS = ("wealth_delta", "dd_delta", "joint_dsr", "baseline_dsr", "quality_tier_hurdle",
               "joint_mean_turnover", "baseline_mean_turnover")
_BOOL_KEYS = ("joint_quality_tier_ok", "baseline_quality_tier_ok")

#: The band-off l1only record: commit d5c3ac9 (plan 07-11, before the 08-A7 5pp no-trade band),
#: ``outputs/reports/platform/joint_lift/measurement_l1only.json``. Pinned rather than read from
#: git at test time -- a shallow clone or an sdist has no history. It predates A11, so it has no
#: DSR or hurdle fields. 08-11 Task 1's verify re-read git once and asserted this pin equals it.
_BAND_OFF_L1ONLY = {
    "wealth_delta": -0.12343826162064975,
    "dd_delta": 0.02408401236666291,
    "joint_mean_turnover": 0.11978772920573971,
    "baseline_mean_turnover": 0.16325097728262197,
}


def _rederived(suffix: str) -> tuple[dict, dict]:
    """``(recomputed, recorded)`` for the nine criterion-7 keys of one routing.

    ``recomputed`` is the lift + quality tier re-run from the committed curves at the record's own
    trial count and variance, plus each leg's mean turnover from its own ``turnover`` column.
    ``recorded`` is the same nine keys read from the committed record.
    """
    from trading_crab_lib.platform.backtest.joint_driver import joint_lift_table

    rec = _record(suffix)
    joint, base = _curve(suffix, "joint"), _curve(suffix, "baseline")
    again = joint_lift_table(
        joint, base,
        n_trials=rec["lift"]["quality_tier_n_trials"],
        sharpe_variance=rec["lift"]["quality_tier_sharpe_variance"],
    )
    lift_keys = [k for k in (*_FLOAT_KEYS, *_BOOL_KEYS) if not k.endswith("_mean_turnover")]
    recomputed = {k: again[k] for k in lift_keys}
    recomputed["joint_mean_turnover"] = float(joint["turnover"].mean())
    recomputed["baseline_mean_turnover"] = float(base["turnover"].mean())
    recorded = {k: rec["lift"][k] for k in lift_keys}
    recorded["joint_mean_turnover"] = rec["joint_leg"]["mean_turnover"]
    recorded["baseline_mean_turnover"] = rec["baseline_leg"]["mean_turnover"]
    return recomputed, recorded


def _record_mismatches(recomputed: dict, recorded: dict) -> list[str]:
    """Sorted keys of ``recorded`` whose recomputed value does not match. THE comparison routine:
    the gating re-derivation and every discrimination arm go through it. A key missing from
    ``recomputed`` is a mismatch, never a skip."""
    bad = []
    for key, want in recorded.items():
        if key not in recomputed:
            bad.append(key)
            continue
        got = recomputed[key]
        if isinstance(want, bool):
            ok = got is want
        elif isinstance(want, (int, str)):
            ok = got == want
        else:
            ok = got == want
        if not ok:
            bad.append(key)
    return sorted(bad)


class TestCriterion7ReMeasuredIn0810:
    @pytest.mark.parametrize("suffix", _SUFFIXES)
    def test_every_lift_cell_carries_its_window_and_both_legs_share_it(self, suffix):
        rec = _record(suffix)
        lift = rec["lift"]
        assert lift["n_steps"] == lift["n_steps_joint"] == lift["n_steps_baseline"] == _PINNED_N_STEPS
        assert lift["indexes_identical"] is True
        assert str(lift["first_date"]).startswith("1972-01-31") and str(lift["last_date"]).startswith("2020-12-31")
        for leg in ("baseline_leg", "joint_leg"):
            k = rec[leg]
            assert (k["n_steps"], k["first_date"], k["last_date"]) == (_PINNED_N_STEPS, "1972-01-31", "2020-12-31")
        for name in ("baseline", "joint"):
            assert rec["deflated_sharpe"][name]["window"] == _WINDOW

    def test_the_two_routings_were_measured_on_one_window(self):
        a, b = _record("l1only")["lift"], _record("l2")["lift"]
        assert (a["n_steps"], a["first_date"], a["last_date"]) == (b["n_steps"], b["first_date"], b["last_date"])

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    def test_both_records_measure_the_band_on_configuration(self, suffix):
        """The re-measurement is of the live configuration: the 5pp band (08-A7.md), and
        the Bayes filter under l2 only (08-08). A record that did not say so could not be
        told apart from the pre-band one."""
        rec = _record(suffix)
        assert rec["no_trade_band"] == 0.05
        assert rec["use_regime_filter"] is (suffix == "l2")

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    def test_the_record_is_re_derivable_from_its_own_curves(self, suffix):
        """The committed curves and the committed record must be one run: recompute the
        lift and the quality tier from the parquet at the record's own trial count.
        Floats are compared relatively (1e-9) with a zero absolute floor: the recomputation is a
        588-month reduction that is not bit-portable across libm/SIMD (G-08-1), while a record
        from a different run is off by ~1e-2 and a zero floor keeps a ~1e-13 DSR from matching 0."""
        assert _record_mismatches(*_rederived(suffix)) == [], suffix

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    def test_the_four_plausibility_band_flags_are_recorded(self, suffix):
        lift = _record(suffix)["lift"]
        for flag in ("wealth_delta_universal_ok", "wealth_delta_domain_note",
                     "dd_delta_universal_ok", "dd_delta_domain_note"):
            assert isinstance(lift[flag], bool), flag
        assert lift["wealth_delta_universal_ok"] and lift["dd_delta_universal_ok"], (
            "a universal breach means the measurement is broken, not that the lift is bad"
        )


class TestTheReDerivationToleranceDiscriminates:
    """Plan 08-11 (G-08-1): the tolerance must ABSORB last-bit platform noise and still REJECT a
    record from another run. Every arm goes through ``_record_mismatches`` -- the routine the
    gating re-derivation uses -- so an arm proves something about the check that gates."""

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("key", _FLOAT_KEYS)
    def test_cross_platform_last_bit_noise_is_absorbed(self, suffix, key):
        """Catches a float rule that reverted to exact equality (or was tightened below ~1e-13):
        1e-13 relative is 14x the 7.1e-15 macOS divergence G-08-1 measured."""
        recomputed, recorded = _rederived(suffix)
        recomputed[key] = recomputed[key] * (1 + 1e-13)
        assert _record_mismatches(recomputed, recorded) == []

    def test_the_uat_macos_pair_is_not_a_mismatch(self):
        """Catches exact equality: the verbatim UAT test 1 pair (08-UAT.md, G-08-1), recomputed on
        Apple Silicon vs recorded on Linux. RED under ``==`` -- the Linux suite reproduces G-08-1."""
        assert _record_mismatches(
            {"wealth_delta": -0.12530657740828932}, {"wealth_delta": -0.1253065774082902}
        ) == []

    def test_the_band_off_record_fails_against_the_band_on_curves(self):
        """Catches a tolerance loosened far enough (e.g. rel 1e-2, or an absolute floor) to accept
        the 07-11 record against the 08-10 curves, and any of the four fields dropping out of the
        comparison: the EXACT four-key list is asserted, not "non-empty"."""
        recomputed, recorded = _rederived("l1only")
        recorded.update(_BAND_OFF_L1ONLY)
        assert _record_mismatches(recomputed, recorded) == [
            "baseline_mean_turnover", "dd_delta", "joint_mean_turnover", "wealth_delta",
        ]

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("key", _FLOAT_KEYS)
    def test_a_one_part_per_million_error_is_caught(self, suffix, key):
        """Catches any single field's tolerance widened above 1e-6, a field dropping out of the
        comparison, and (on the DSR keys) a dropped ``abs=0.0``: the default 1e-12 floor would
        swallow a 1.8e-19 difference on a 1.8e-13 DSR."""
        recomputed, recorded = _rederived(suffix)
        recorded[key] = recorded[key] * (1 + 1e-6)
        assert _record_mismatches(recomputed, recorded) == [key]

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("key", ("joint_dsr", "baseline_dsr"))
    def test_a_dsr_of_zero_is_not_accepted(self, suffix, key):
        """Catches a dropped ``abs=0.0``: pytest.approx's default floor of 1e-12 accepts 0.0 for a
        ~1e-13 DSR (the trap measured at plan time)."""
        recomputed, recorded = _rederived(suffix)
        recorded[key] = 0.0
        assert _record_mismatches(recomputed, recorded) == [key]

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("key", _BOOL_KEYS)
    def test_a_flipped_quality_gate_is_caught(self, suffix, key):
        """Catches booleans compared by approx or truthiness instead of identity."""
        recomputed, recorded = _rederived(suffix)
        recorded[key] = not recorded[key]
        assert _record_mismatches(recomputed, recorded) == [key]


class TestF4SecondSiteInTheMeasurementRecords:
    """F-4's second site (Glenn, 2026-09-23): the measurement records now divide by
    adjacent PAIRS, like the diagnostics records, for the same 246 and 24 changes."""

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    @pytest.mark.parametrize("leg", ["baseline_leg", "joint_leg"])
    @pytest.mark.parametrize("clf,n", [("1", 246), ("2", 24)])
    def test_rate_is_pair_denominated_and_rejects_the_month_denominator(self, suffix, leg, clf, n):
        k = _record(suffix)[leg]
        assert k[f"n_state_{clf}_transitions"] == n
        rate = k[f"state_{clf}_transition_rate"]
        assert rate == pytest.approx(n / 587, abs=1e-12)
        assert rate != pytest.approx(n / 588, abs=1e-9), "still divides by n_steps (F-4 regression)"

    @pytest.mark.parametrize("suffix", _SUFFIXES)
    def test_measurement_and_diagnostics_records_agree(self, suffix):
        k = _record(suffix)["joint_leg"]
        d = _diagnostics(suffix)
        assert k["state_1_transition_rate"] == d["classifier_1"]["walk_forward_filtered"]["transition_rate"]
        assert k["state_2_transition_rate"] == d["classifier_2"]["walk_forward_filtered"]["transition_rate"]


class TestTheA11GateInTheRecord:
    @pytest.mark.parametrize("suffix", _SUFFIXES)
    def test_the_dsr_block_and_the_lift_agree_and_the_gate_is_the_hurdle(self, suffix):
        from trading_crab_lib.platform.evaluation.deflated_sharpe import expected_max_sharpe

        rec = _record(suffix)
        lift, q = rec["lift"], rec["quality_tier"]
        hurdle = expected_max_sharpe(q["n_trials"], q["sharpe_variance"])
        assert lift["quality_tier_hurdle"] == q["hurdle"] == hurdle
        assert lift["quality_tier_n_trials"] == q["n_trials"] == rec["registry"]["count_after"]
        for name in ("baseline", "joint"):
            block = rec["deflated_sharpe"][name]
            assert block["dsr"] == lift[f"{name}_dsr"]
            assert block["observed_sharpe"] == lift[f"{name}_observed_sharpe"] == rec[f"{name}_leg"]["sharpe_annualized"]
            assert block["quality_tier_ok"] is lift[f"{name}_quality_tier_ok"] is q[f"{name}_ok"]
            assert block["quality_tier_ok"] is (block["observed_sharpe"] > hurdle) is (block["dsr"] > 0.5)

    def test_the_gate_governs_the_decision_bearing_leg_only(self):
        l1, l2 = _record("l1only"), _record("l2")
        assert (l1["decision_bearing"], l1["quality_tier"]["governs"]) == (True, True)
        assert (l2["decision_bearing"], l2["quality_tier"]["governs"]) == (False, False)
        assert l2["quality_tier"]["verdict"].startswith("NOT GOVERNING")
        failing = [n for n in ("baseline", "joint") if not l1["quality_tier"][f"{n}_ok"]]
        expected = (
            f"FAILED on {len(failing)} of 2 legs ({', '.join(failing)})" if failing else "PASSED on both legs"
        )
        assert l1["quality_tier"]["verdict"] == expected

    def test_the_hurdle_rests_on_the_declared_placeholder(self):
        """ADR-0002 open item 8, now load-bearing on a gate (08-A11.md §4)."""
        for suffix in _SUFFIXES:
            q = _record(suffix)["quality_tier"]
            assert q["sharpe_variance"] == 1.0 and q["sharpe_variance_is_placeholder"] is True


class TestTheDecisionBearingRunSpentExactlyTwoRows:
    """08-A7.md authorised 2 rows; 08-10 consumed them, once. 42 -> 44, the ceiling."""

    def test_the_l1only_record_moved_the_registry_by_exactly_two(self):
        reg = _record("l1only")["registry"]
        assert (reg["count_before"], reg["count_after"], reg["rows_added"]) == (_REGISTRY_BEFORE, _REGISTRY_AFTER, 2)
        assert reg["count_after"] <= reg["adr_0002_ceiling"] == 44
        assert (reg["baseline_tag"], reg["joint_tag"]) == _TAGS_0810

    def test_the_l2_record_moved_it_by_zero(self):
        reg = _record("l2")["registry"]
        assert reg["rows_added"] == 0
        assert reg["baseline_tag"] == reg["joint_tag"] == "(NO_REGISTRY)"

    def test_the_tracked_ledger_carries_the_two_rows_the_record_claims(self):
        """No commit may claim 44 while the ledger reads 42: the ledger is read here."""
        from trading_crab_lib.platform.honesty.registry import total_trial_count

        assert total_trial_count() == _REGISTRY_AFTER
        rows = [json.loads(line) for line in (_ROOT / "registry" / "trials.jsonl").read_text().splitlines() if line]
        rec = _record("l1only")
        tagged = {r["config"]["trial_tag"]: r for r in rows if r["config"].get("trial_tag") in _TAGS_0810}
        assert sorted(tagged) == sorted(_TAGS_0810) and [r["config"]["trial_tag"] for r in rows[-2:]] == list(_TAGS_0810)
        for tag, leg, weight in zip(_TAGS_0810, ("baseline_leg", "joint_leg"), (1.0, 0.5)):
            row = tagged[tag]
            assert row["config"]["no_trade_band"] == 0.05
            assert row["config"]["routing"] == "L1_ONLY_LAST_FILTERED_STATE"
            assert row["config"]["blend_weight_1"] == weight
            assert row["metrics"]["n_steps"] == _PINNED_N_STEPS
            assert row["metrics"]["terminal_log_wealth"] == pytest.approx(rec[leg]["terminal_log_wealth"], abs=1e-12)
