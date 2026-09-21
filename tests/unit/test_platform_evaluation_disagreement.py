"""Unit tests for trading_crab_lib.platform.evaluation.disagreement
(criterion 3, 07-CONTEXT.md D-05, 07-VALIDATION.md).

TestMeasureLabelDisagreement is the load-bearing suite for the silent-zero
trap ``07-PREFIX-EVIDENCE.md`` documents: the persisted
``backtest_filtered_state_probs.parquet`` artifact's columns are ``state_N``
STRINGS, and ``label_disagreement`` (the delegate) silently returns
``n_compared == 0`` if handed those strings uncoerced — no exception, reads
as "0% disagreement, fully resolved" while comparing nothing.

``test_reproduces_the_prefix_baseline`` proves methodological equality with
the located 389/470 = 82.8% baseline by reading the pre-fix artifacts at
their PINNED commit via ``git show`` — robust regardless of whether the
working tree has since been overwritten by this phase's own policy-trial
runs (07-03-PLAN.md Task 2 regenerates these exact paths).
"""

from __future__ import annotations

import io
import logging
import subprocess

import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement

# ── pinned pre-fix artifact fixture (git show, not the live working tree) ──

_PREFIX_ARTIFACT_SHA = "27fe529f7d155ab177b002afd1bfc07b43e4af6b"
_PREFIX_FULL_SAMPLE_STATES_PATH = "outputs/reports/platform/backtest_full_sample_states.parquet"
_PREFIX_FILTERED_STATE_PROBS_PATH = "outputs/reports/platform/backtest_filtered_state_probs.parquet"


def _read_pinned_parquet(path: str) -> pd.DataFrame | None:
    """Read *path* as it stood at the pinned pre-fix commit, or None if the
    blob/commit is unreachable (no git, shallow clone, etc.) — never raises.
    """
    try:
        result = subprocess.run(
            ["git", "show", f"{_PREFIX_ARTIFACT_SHA}:{path}"],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    try:
        return pd.read_parquet(io.BytesIO(result.stdout))
    except Exception:  # noqa: BLE001 — any parse failure means "not reachable"
        return None


_PREFIX_FULL = _read_pinned_parquet(_PREFIX_FULL_SAMPLE_STATES_PATH)
_PREFIX_PROBS = _read_pinned_parquet(_PREFIX_FILTERED_STATE_PROBS_PATH)
_PREFIX_AVAILABLE = _PREFIX_FULL is not None and _PREFIX_PROBS is not None


def _one_hot_probs(reference: pd.Series, states: list[int]) -> pd.DataFrame:
    """Build a filtered_state_probs-shaped frame (``state_{k}`` string
    columns) that argmaxes to exactly *reference* at every date."""
    columns = [f"state_{k}" for k in states]
    probs = pd.DataFrame(0.0, index=reference.index, columns=columns)
    for t, s in reference.items():
        probs.loc[t, f"state_{s}"] = 1.0
    return probs


# ── TestMeasureLabelDisagreement ──────────────────────────────────────────────


class TestMeasureLabelDisagreement:
    def test_identical_labelings_disagree_on_nothing(self):
        index = pd.date_range("2000-01-31", periods=10, freq="ME")
        reference = pd.Series([0, 0, 1, 1, 2, 2, 0, 0, 1, 1], index=index, name="state")
        probs = _one_hot_probs(reference, [0, 1, 2])

        result = measure_label_disagreement(reference.to_frame(), probs)

        assert result["n_compared"] == 10
        assert result["n_disagree"] == 0
        assert result["pct_disagree"] == 0.0

    def test_state_prefixed_columns_are_coerced_not_dropped(self):
        """Pins the silent-zero trap: state_N STRING columns must be coerced,
        never dropped. The rejected value is n_compared == 0 on a frame with
        genuinely overlapping dates."""
        index = pd.date_range("2000-01-31", periods=6, freq="ME")
        reference = pd.Series([0, 1, 2, 3, 4, 0], index=index, name="state")
        probs = _one_hot_probs(reference, [0, 1, 2, 3, 4])

        result = measure_label_disagreement(reference.to_frame(), probs)

        assert result["n_compared"] == 6, (
            f"n_compared must equal the 6 overlapping dates, got {result['n_compared']} "
            "— this is the silent-zero coercion trap (state_N strings passed through "
            "uncoerced) reproduced as a failing assertion, not a passing one."
        )
        assert result["n_compared"] != 0

    def test_disjoint_indexes_report_zero_compared_not_an_error(self, caplog):
        ref_index = pd.date_range("2000-01-31", periods=5, freq="ME")
        probs_index = pd.date_range("2010-01-31", periods=5, freq="ME")
        reference = pd.Series([0, 1, 2, 3, 4], index=ref_index, name="state")
        probs = _one_hot_probs(pd.Series([0, 1, 2, 3, 4], index=probs_index), [0, 1, 2, 3, 4])

        with caplog.at_level(logging.WARNING):
            result = measure_label_disagreement(reference.to_frame(), probs)

        assert result["n_compared"] == 0
        assert result["n_disagree"] == 0
        assert result["pct_disagree"] == 0.0
        assert any("n_compared == 0" in rec.message for rec in caplog.records), (
            "a disjoint span must log a WARNING — it is a finding, not a silent no-op"
        )

    def test_result_carries_its_own_denominator(self):
        index = pd.date_range("2000-01-31", periods=8, freq="ME")
        reference = pd.Series([0, 1, 0, 1, 2, 2, 1, 0], index=index, name="state")
        probs = _one_hot_probs(reference, [0, 1, 2])

        result = measure_label_disagreement(reference.to_frame(), probs)

        for key in (
            "n_compared",
            "n_disagree",
            "pct_disagree",
            "first_common_date",
            "last_common_date",
            "per_state_confusion",
            "suspicious",
            "suspicious_reason",
        ):
            assert key in result, f"result is missing {key!r} — a percentage without its own denominator is not a result"

    def test_suspiciously_low_disagreement_is_flagged(self):
        index = pd.date_range("2000-01-31", periods=100, freq="ME")

        # 1/100 disagreement (1%) -> below the 2% default threshold.
        reference_low = pd.Series([0] * 100, index=index, name="state")
        probs_low = _one_hot_probs(reference_low, [0, 1])
        probs_low.loc[index[0], "state_0"] = 0.0
        probs_low.loc[index[0], "state_1"] = 1.0

        result_low = measure_label_disagreement(reference_low.to_frame(), probs_low)
        assert result_low["pct_disagree"] == pytest.approx(0.01)
        assert result_low["suspicious"] is True
        assert result_low["suspicious_reason"] != ""

        # 50/100 disagreement -> comfortably above the threshold, not suspicious.
        reference_mid = pd.Series([0, 1] * 50, index=index, name="state")
        probs_mid = pd.DataFrame(0.0, index=index, columns=["state_0", "state_1"])
        probs_mid["state_0"] = 1.0  # comparison always predicts state 0

        result_mid = measure_label_disagreement(reference_mid.to_frame(), probs_mid)
        assert result_mid["pct_disagree"] == pytest.approx(0.5)
        assert result_mid["suspicious"] is False
        assert result_mid["suspicious_reason"] == ""

    @pytest.mark.skipif(
        not _PREFIX_AVAILABLE,
        reason="pre-fix artifacts not reachable via `git show` at the pinned commit",
    )
    def test_reproduces_the_prefix_baseline(self):
        """The methodological-equality proof: this module's computation on
        the pre-fix artifacts must reproduce 07-PREFIX-EVIDENCE.md's exact
        triple. Any other pct_disagree means this is not the same computation
        as the baseline, and criterion 3 is not satisfiable."""
        result = measure_label_disagreement(_PREFIX_FULL, _PREFIX_PROBS)

        assert result["n_compared"] == 470
        assert result["n_disagree"] == 389
        assert result["pct_disagree"] == pytest.approx(0.8276595744680851, abs=1e-12)


# ── Band 4 as revised 2026-09-18 (07-BANDS.md, Glenn's disposition) ─────────
#
# The band is now: suspicious if pct_disagree < threshold OR n_compared == 0 OR
# n_compared is materially below expectation with no recorded reason.
#
# The zero-denominator clause is the load-bearing one: before this revision the
# code WARNED on n_compared == 0 but set suspicious = False, so a caller reading
# the flag rather than the log saw "not suspicious" on the single case the band
# exists to catch.


class TestBand4RevisedDenominatorClauses:
    @staticmethod
    def _probs(idx, state: int, k: int = 5) -> pd.DataFrame:
        data = {f"state_{i}": [1.0 if i == state else 0.0] * len(idx) for i in range(k)}
        return pd.DataFrame(data, index=idx)

    def test_zero_denominator_is_now_suspicious_not_merely_logged(self):
        """Regression on the exact defect: disjoint spans returned
        suspicious=False while pct_disagree read 0.0 — 'perfect agreement'
        over nothing. Flipping this back to False fails here."""
        from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement

        a = pd.date_range("1970-01-31", periods=12, freq="ME")
        b = pd.date_range("2000-01-31", periods=12, freq="ME")
        ref = pd.Series([0] * 12, index=a, name="state")
        res = measure_label_disagreement(ref, self._probs(b, 0))

        assert res["n_compared"] == 0
        assert res["suspicious"] is True
        assert "n_compared == 0" in res["suspicious_reason"]

    def test_short_window_without_a_recorded_reason_is_suspicious(self):
        """ADR-0001's 232-of-588 narrowing is this shape and was found by hand."""
        from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement

        idx = pd.date_range("1972-01-31", periods=100, freq="ME")
        ref = pd.Series([0] * 50 + [1] * 50, index=idx, name="state")
        comp = self._probs(idx[:60], 0)  # only 60 of an expected 100 months
        res = measure_label_disagreement(ref, comp, expected_n_compared=100)

        assert res["n_compared"] == 60
        assert res["suspicious"] is True
        assert "materially below" in res["suspicious_reason"]

    def test_a_RECORDED_reason_suppresses_the_coverage_clause(self):
        """'Recorded' is the operative word — an explained shortfall is not
        suspicious, an unexplained one is. Without this the band would fire on
        every legitimately narrowed window."""
        from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement

        idx = pd.date_range("1972-01-31", periods=100, freq="ME")
        ref = pd.Series([0] * 50 + [1] * 50, index=idx, name="state")
        comp = self._probs(idx[:60], 0)
        res = measure_label_disagreement(
            ref, comp, expected_n_compared=100,
            coverage_reason="L2 degraded 40 steps; recorded in ADR-0001",
        )
        assert res["n_compared"] == 60
        assert "materially below" not in res["suspicious_reason"]

    def test_full_coverage_and_normal_disagreement_is_not_suspicious(self):
        """The band must still be able to say 'fine' — otherwise it only confirms
        in the other direction."""
        from trading_crab_lib.platform.evaluation.disagreement import measure_label_disagreement

        idx = pd.date_range("1972-01-31", periods=100, freq="ME")
        ref = pd.Series([0] * 50 + [1] * 50, index=idx, name="state")
        comp = self._probs(idx, 0)   # disagrees on the last 50 -> 0.50
        res = measure_label_disagreement(ref, comp, expected_n_compared=100)
        assert res["n_compared"] == 100
        assert res["suspicious"] is False
        assert res["suspicious_reason"] == ""

    def test_threshold_itself_is_unchanged_at_0_02(self):
        from trading_crab_lib.platform.evaluation.disagreement import DEFAULT_SUSPICIOUS_THRESHOLD

        assert DEFAULT_SUSPICIOUS_THRESHOLD == 0.02
