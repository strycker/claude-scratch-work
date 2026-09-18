"""Unit tests for trading_crab_lib.platform.evaluation.deflated_sharpe (D-16, criterion 7).

Follows the incumbent test_platform_labeling.py::TestDPDecodeExact precedent — the
strongest verification pattern in this project: prove a formula against an
INDEPENDENTLY computed reference, not against its own shape or a number copied from
the implementation. The normal-case oracle below builds its expected value with a
separate, inline call to ``scipy.stats.norm`` in this test file — never by calling
``deflated_sharpe_ratio`` and comparing it to itself.

VALIDATION.md's own "Evidence-Shape Requirement" names two prior burns from
existence/shape-only checks in this project (a terminal log wealth of 111.06,
e^111 ~= 10^48, tabulated as an improvement; a decoupling grep ending
`| grep -v platform` that discarded every match line). A bare `0 <= dsr <= 1` check
is the same failure mode here — every possible implementation ending in
`norm.cdf(...)` satisfies it, right or wrong. None of the tests below rely on that
range alone.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.stats import norm

from trading_crab_lib.platform.evaluation.deflated_sharpe import (
    DEGENERATE_SHARPE_VARIANCE,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    format_dsr_verdict,
    registry_sharpe_variance,
)
from trading_crab_lib.platform.honesty.registry import PROVENANCE_RECORD_TYPE

_EULER_MASCHERONI = 0.5772156649


def _hand_expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Independent re-derivation of Eq. 5/6 (paper's own Python snippet), typed
    fresh here rather than imported — the oracle for expected_max_sharpe."""
    max_z = (1 - _EULER_MASCHERONI) * norm.ppf(1 - 1.0 / n_trials) + _EULER_MASCHERONI * norm.ppf(
        1 - 1.0 / (n_trials * np.e)
    )
    return float(np.sqrt(sharpe_variance) * max_z)


def _write_ledger(path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _header_row(prior: int) -> dict:
    return {
        "config_hash": "RESET",
        "config": {
            "trial_tag": "SYNTHETIC-RESET",
            "record_type": PROVENANCE_RECORD_TYPE,
            "prior_genuine_trials": prior,
        },
        "features": [],
        "metrics": {"prior_genuine_trials": prior},
        "git_sha": None,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


def _sharpe_row(tag: str, sharpe: float) -> dict:
    return {
        "config_hash": tag,
        "config": {"trial_tag": tag},
        "features": ["f1"],
        "metrics": {"sharpe": sharpe},
        "git_sha": "deadbeef",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


# ── expected_max_sharpe ──────────────────────────────────────────────────────────


class TestExpectedMaxSharpe:
    def test_one_trial_returns_exactly_zero(self):
        """With one trial there is no selection to correct for — exactly 0.0."""
        assert expected_max_sharpe(1, 2.5) == 0.0

    def test_zero_trials_returns_exactly_zero(self):
        """A trial count of zero is also '<= 1' — exactly 0.0, not a division error."""
        assert expected_max_sharpe(0, 2.5) == 0.0

    def test_negative_trials_returns_exactly_zero(self):
        """A negative trial count is nonsensical but must not raise — exactly 0.0."""
        assert expected_max_sharpe(-3, 2.5) == 0.0

    def test_zero_variance_returns_exactly_zero_regardless_of_trials(self):
        """A degenerate (zero-dispersion) null has no maximum to select — 0.0 at any N."""
        assert expected_max_sharpe(2, 0.0) == 0.0
        assert expected_max_sharpe(1000, 0.0) == 0.0

    def test_matches_hand_derived_oracle_at_moderate_n(self):
        """Exact match (1e-12) against the paper's own Eq. 5/6, typed independently."""
        n, var = 25, 0.8
        got = expected_max_sharpe(n, var)
        expected = _hand_expected_max_sharpe(n, var)
        assert got == pytest.approx(expected, abs=1e-12)

    def test_strictly_increasing_across_three_trial_counts(self):
        """Monotonicity: the value at 100 trials > at 10 > at 2, fixed positive variance.

        A strict inequality CHAIN across three points, not merely two, and not a
        non-strict (>=) comparison — either weakening would pass a broken
        implementation that plateaus or ties.
        """
        variance = 1.0
        v2 = expected_max_sharpe(2, variance)
        v10 = expected_max_sharpe(10, variance)
        v100 = expected_max_sharpe(100, variance)
        assert v2 < v10 < v100

    def test_increasing_holds_for_a_different_fixed_variance(self):
        """Same monotonicity chain at a different variance — not a coincidence of variance=1."""
        variance = 4.0
        v2 = expected_max_sharpe(2, variance)
        v10 = expected_max_sharpe(10, variance)
        v100 = expected_max_sharpe(100, variance)
        assert v2 < v10 < v100


# ── deflated_sharpe_ratio ────────────────────────────────────────────────────────


class TestDeflatedSharpeRatio:
    def test_normal_case_oracle_against_independent_scipy_computation(self):
        """The full DSR formula, independently re-derived with scipy.stats.norm in
        this test (not delegated to the implementation), matches to 1e-12.

        skew=0.0, kurtosis=3.0 (RAW, i.e. Normal), n_trials=1 (so
        expected_max_sharpe == 0.0 regardless of variance, per the test above) —
        the DSR reduces to the PSR of observed_sharpe against a null of exactly
        zero. The oracle constructs the full non-normality-adjusted z-score by
        hand and calls norm.cdf directly, exactly mirroring the paper's own
        formula components (07-DSR-ESTIMATOR-NOTE.md §2) rather than copying a
        number out of the implementation.
        """
        observed_sharpe = 0.15
        n_trials = 1
        sharpe_variance = 3.7  # irrelevant at n_trials=1; exercises that irrelevance too
        skew = 0.0
        kurtosis = 3.0
        n_obs = 61

        sr0 = 0.0  # expected_max_sharpe(1, *) == 0.0, independently asserted above
        denom = np.sqrt(1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2)
        z = (observed_sharpe - sr0) * np.sqrt(n_obs - 1) / denom
        expected = float(norm.cdf(z))

        got = deflated_sharpe_ratio(observed_sharpe, n_trials, sharpe_variance, skew, kurtosis, n_obs)
        assert got == pytest.approx(expected, abs=1e-12)

    def test_second_oracle_with_multiple_trials_and_nonzero_skew(self):
        """A second independent oracle point, this time with N > 1 (so sr0 != 0)
        and nonzero skew — guards against an implementation that only handles the
        n_trials=1 special case correctly."""
        observed_sharpe = 1.2
        n_trials = 50
        sharpe_variance = 0.5
        skew = -0.3
        kurtosis = 5.0
        n_obs = 240

        sr0 = _hand_expected_max_sharpe(n_trials, sharpe_variance)
        denom = np.sqrt(1.0 - skew * observed_sharpe + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2)
        z = (observed_sharpe - sr0) * np.sqrt(n_obs - 1) / denom
        expected = float(norm.cdf(z))

        got = deflated_sharpe_ratio(observed_sharpe, n_trials, sharpe_variance, skew, kurtosis, n_obs)
        assert got == pytest.approx(expected, abs=1e-12)

    def test_result_strictly_inside_open_unit_interval(self):
        """A finite-input DSR is strictly inside (0, 1), never exactly 0 or 1."""
        dsr = deflated_sharpe_ratio(0.9, 10, 1.0, 0.0, 3.0, 120)
        assert 0.0 < dsr < 1.0

    def test_strictly_decreasing_in_trial_count(self):
        """Monotonicity: DSR at 100 trials is strictly below DSR at 2 trials, all else fixed."""
        kwargs = dict(observed_sharpe=1.0, sharpe_variance=1.0, skew=0.0, kurtosis=3.0, n_obs=200)
        dsr_2 = deflated_sharpe_ratio(n_trials=2, **kwargs)
        dsr_10 = deflated_sharpe_ratio(n_trials=10, **kwargs)
        dsr_100 = deflated_sharpe_ratio(n_trials=100, **kwargs)
        assert dsr_100 < dsr_10 < dsr_2

    def test_negative_observed_sharpe_yields_dsr_strictly_below_half(self):
        """A negative observed Sharpe can never clear the 0.5 hurdle, at any trial count.

        Not merely 'low' — strictly below the exact 0.5 threshold format_dsr_verdict
        keys off of.
        """
        dsr_few_trials = deflated_sharpe_ratio(-0.5, 2, 1.0, 0.0, 3.0, 100)
        dsr_many_trials = deflated_sharpe_ratio(-0.5, 500, 1.0, 0.0, 3.0, 100)
        assert dsr_few_trials < 0.5
        assert dsr_many_trials < 0.5

    def test_non_positive_denominator_raises_named_value_error(self):
        """An invalid skew/kurtosis/SR combination raises ValueError, never a silent NaN."""
        # kurtosis=1.0 -> the SR^2 term vanishes; a large positive skew * large SR
        # drives (1 - skew*sr) negative.
        with pytest.raises(ValueError, match="non-normality denominator"):
            deflated_sharpe_ratio(observed_sharpe=10.0, n_trials=5, sharpe_variance=1.0, skew=5.0, kurtosis=1.0, n_obs=50)

    def test_non_positive_denominator_never_returns_nan(self):
        """Same malformed input: confirm it raises rather than returning float('nan')."""
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(observed_sharpe=10.0, n_trials=5, sharpe_variance=1.0, skew=5.0, kurtosis=1.0, n_obs=50)


# ── format_dsr_verdict ───────────────────────────────────────────────────────────


class TestFormatDsrVerdict:
    _DOES_NOT_CLEAR = "does not clear the multiple-testing hurdle"

    def test_at_hurdle_reports_does_not_clear(self):
        assert self._DOES_NOT_CLEAR in format_dsr_verdict(0.5)

    def test_just_below_hurdle_reports_does_not_clear(self):
        """0.49 is not softened into a near-miss — same wording as 0.5."""
        assert self._DOES_NOT_CLEAR in format_dsr_verdict(0.49)

    def test_above_hurdle_does_not_contain_does_not_clear_phrase(self):
        assert self._DOES_NOT_CLEAR not in format_dsr_verdict(0.9)

    def test_no_softening_qualifiers_near_the_hurdle(self):
        """A value just below 0.5 must not read differently from one far below it —
        no 'close', 'nearly', 'borderline', 'almost' language at any distance."""
        near = format_dsr_verdict(0.499).lower()
        far = format_dsr_verdict(0.05).lower()
        for qualifier in ("close", "nearly", "borderline", "almost", "just missed"):
            assert qualifier not in near
            assert qualifier not in far


# ── registry_sharpe_variance ─────────────────────────────────────────────────────


class TestRegistrySharpeVariance:
    def test_degenerate_on_missing_ledger(self, tmp_path):
        """No usable observations at all -> the degenerate placeholder, no raise."""
        result = registry_sharpe_variance(path=tmp_path / "does_not_exist.jsonl")
        assert result == DEGENERATE_SHARPE_VARIANCE

    def test_degenerate_on_header_only_ledger(self, tmp_path):
        """A ledger holding only the provenance header has zero usable Sharpe rows."""
        path = tmp_path / "trials.jsonl"
        _write_ledger(path, [_header_row(38)])
        assert registry_sharpe_variance(path=path) == DEGENERATE_SHARPE_VARIANCE

    def test_degenerate_on_single_sharpe_observation(self, tmp_path):
        """Exactly one usable Sharpe row is still fewer than two -> degenerate."""
        path = tmp_path / "trials.jsonl"
        _write_ledger(path, [_header_row(38), _sharpe_row("a", 0.4)])
        assert registry_sharpe_variance(path=path) == DEGENERATE_SHARPE_VARIANCE

    def test_computes_sample_variance_with_two_or_more_observations(self, tmp_path):
        """>= 2 usable Sharpe rows -> real sample variance (ddof=1), not the placeholder."""
        path = tmp_path / "trials.jsonl"
        sharpes = [0.2, 0.5, 0.9, 1.1]
        _write_ledger(path, [_header_row(38)] + [_sharpe_row(str(i), s) for i, s in enumerate(sharpes)])
        result = registry_sharpe_variance(path=path)
        assert result == pytest.approx(float(np.var(sharpes, ddof=1)))
        assert result != DEGENERATE_SHARPE_VARIANCE

    def test_never_raises_on_malformed_metrics(self, tmp_path):
        """A row with a non-numeric sharpe value is skipped, not a crash."""
        path = tmp_path / "trials.jsonl"
        rows = [
            _header_row(38),
            {
                "config_hash": "bad",
                "config": {"trial_tag": "bad"},
                "features": [],
                "metrics": {"sharpe": "not-a-number"},
                "git_sha": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            _sharpe_row("good", 0.6),
        ]
        _write_ledger(path, rows)
        # Only 1 usable ("good") -> still degenerate, but must not raise.
        assert registry_sharpe_variance(path=path) == DEGENERATE_SHARPE_VARIANCE

    def test_live_registry_does_not_raise(self):
        """Sanity check against the REAL live ledger — must not raise regardless of content."""
        result = registry_sharpe_variance()
        assert isinstance(result, float)
        assert result > 0.0
