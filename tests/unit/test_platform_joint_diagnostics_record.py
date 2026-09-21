"""Guards for the three wave-2 quantities that were REPORTED but not GOVERNED.

Phase 7 closed with several numbers recorded in prose and in a JSON diagnostics
record, with nothing in the suite able to notice if they changed. This file is
that notice. It is deliberately *pin*-shaped, not band-shaped: no band exists
for any of these quantities, and inventing one would repeat the phase's own
signature defect (a bound wider than the quantity's reachable range, which can
only confirm). A pin can fail; a made-up band cannot.

Covered here:

1. **Filtered-labeling churn** (07-12 open item 5). Classifier #1's walk-forward
   *filtered* labeling changes state in 246 of 588 decision months (41.84%)
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
        wf = _diagnostics(suffix)[clf]["walk_forward_filtered"]
        assert wf["n_steps"] == _PINNED_N_STEPS
        assert wf["transition_rate"] == pytest.approx(wf["n_transitions"] / wf["n_steps"], abs=1e-12)

    @pytest.mark.parametrize("clf,pinned", sorted(_PINNED_FILTERED_TRANSITIONS.items()))
    def test_churn_is_pinned_at_its_reported_value(self, clf, pinned):
        """Open item 5's number, pinned. No band governs it -- so it is pinned.

        A change here is not necessarily a bug, but it MUST be a deliberate,
        visible act: the reported 41.84% / 4.08% churn is what the tilt trades
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
