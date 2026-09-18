"""Unit tests for trading_crab_lib.platform.evaluation.dependence
(criterion 6, 07-CONTEXT.md D-15, 07-VALIDATION.md).

These tests are written to fail in **both** directions. A dependence
statistic is not self-validating: a *wrong* number must fail, and so must a
*perfect* one. ARI or NMI at exactly 1.0 almost certainly means both
classifiers are reading the same underlying labels — the wiring-bug class
``disagreement.py``'s ``suspicious_reason`` already guards for criterion 3 —
and all three statistics at exactly 0.0 simultaneously is unusually clean for
real financial data. Each of those is asserted as a *flag*, never as a pass.

The oracle values below were enumerated live in this environment
(``scikit-learn`` 1.9.1, ``scipy`` 1.17.1). See
``TestSixElementOracle``'s docstring for the correction to
``07-PATTERNS.md``'s recorded triple.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.evaluation import dependence as dependence_module
from trading_crab_lib.platform.evaluation.dependence import (
    DEPENDENCE_FLAG_ARI,
    DEPENDENCE_FLAG_CRAMERS_V,
    DEPENDENCE_FLAG_NMI,
    format_dependence_report,
    measure_labeling_dependence,
)
from trading_crab_lib.platform.plotting.regime import label_disagreement

# ── oracles, all live-verified in this environment ────────────────────────

#: Six elements, two three-block partitions -> a 3x3 crosstab.
#: ARI 1/6, NMI 0.5793801642856948, Cramer's V 0.7071067811865476.
_ORACLE_3X3_A = [0, 0, 1, 1, 2, 2]
_ORACLE_3X3_B = [0, 0, 1, 2, 1, 2]
_ORACLE_3X3_ARI = 0.16666666666666666
_ORACLE_3X3_NMI = 0.5793801642856948
_ORACLE_3X3_V = 0.7071067811865476

#: Six elements, two two-block partitions -> a 2x2 crosstab with Cramer's V
#: exactly 0.5 (the [[2, 0], [2, 2]] table).
_ORACLE_2X2_A = [0, 0, 1, 1, 1, 1]
_ORACLE_2X2_B = [0, 0, 0, 0, 1, 1]
_ORACLE_2X2_ARI = -0.07142857142857142
_ORACLE_2X2_NMI = 0.27401754212128127
_ORACLE_2X2_V = 0.5

#: The independence oracle's stated bound. Named here so the assertion
#: message can quote it rather than leaving the reader to infer it.
_INDEPENDENCE_ARI_BOUND = 0.02


def _months(n: int, start: str = "1972-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def _series(values: list[int], start: str = "1972-01-31") -> pd.Series:
    return pd.Series(values, index=_months(len(values), start=start), name="state")


def _independent_pair(n: int = 600) -> tuple[pd.Series, pd.Series]:
    """Two deterministically constructed, structurally unrelated labelings.

    ``states_1`` is a 37-month block sequence cycling through five states;
    ``states_2`` cycles through four states every month. Neither is random,
    so the test is reproducible byte-for-byte; the block length (37) is
    coprime with the cycle length (4) so the two carry no shared structure.
    """
    i = np.arange(n)
    idx = _months(n)
    return (
        pd.Series((i // 37) % 5, index=idx, name="state"),
        pd.Series(i % 4, index=idx, name="state"),
    )


# ── delegation: one alignment, not two ────────────────────────────────────


class TestDelegatesAlignmentAndCrosstab:
    def test_returns_every_label_disagreement_key_with_identical_values(self):
        """The shared keys must come from ``label_disagreement`` itself.

        A second alignment implementation is exactly the divergence
        criterion 1 exists to prevent, so this compares the module's output
        against the delegate's own output key by key.
        """
        s1 = _series([0, 0, 1, 1, 2, 2, 0, 1])
        s2 = _series([0, 1, 1, 1, 2, 0, 0, 1])

        delegate = label_disagreement(s1, s2)
        result = measure_labeling_dependence(s1, s2)

        for key in ("n_compared", "n_disagree", "pct_disagree", "first_common_date", "last_common_date"):
            assert result[key] == delegate[key], f"{key} diverged from label_disagreement's own value"
        pd.testing.assert_frame_equal(result["per_state_confusion"], delegate["per_state_confusion"])

    def test_crosstab_is_returned_with_both_margins_derivable(self):
        s1 = _series([0, 0, 1, 1, 2, 2])
        s2 = _series([0, 0, 1, 2, 1, 2])
        result = measure_labeling_dependence(s1, s2)
        ct = result["per_state_confusion"]
        assert ct.to_numpy().tolist() == [[2, 0, 0], [0, 1, 1], [0, 1, 1]]
        assert ct.sum(axis=1).tolist() == [2, 2, 2]
        assert ct.sum(axis=0).tolist() == [2, 2, 2]


# ── the perfect-statistic direction: a suspicion, never a pass ────────────


class TestIdenticalLabelingsAreASuspicion:
    def test_identical_labelings_flagged_not_reported_as_a_strong_finding(self):
        s = _series([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 20)
        result = measure_labeling_dependence(s, s)

        assert result["adjusted_rand"] == 1.0
        assert result["nmi"] == 1.0
        assert result["cramers_v"] == 1.0
        assert result["suspicious"] is True, (
            "ARI and NMI at exactly 1.0 must be flagged: the most likely cause is "
            "both classifiers reading the same underlying labels, not a strong finding."
        )
        assert result["suspicious_reason"], "a suspicion without a reason is not actionable"
        assert "same" in result["suspicious_reason"].lower()

    def test_permutation_of_the_same_partition_also_gives_ari_exactly_one(self):
        """Pinned so a future reader does not mistake a relabeling for independence.

        ARI and NMI are permutation-invariant by construction: the two
        classifiers number their states independently (#1 orders on
        ``trailing_return_1m``, #2 on ``rs_equities_bonds``), so a pure
        relabeling is the same partition and must score 1.0, not 0.0.
        """
        base = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 20
        perm = {0: 2, 1: 4, 2: 0, 3: 1, 4: 3}
        s1 = _series(base)
        s2 = _series([perm[x] for x in base])

        result = measure_labeling_dependence(s1, s2)
        assert result["adjusted_rand"] == 1.0
        assert result["nmi"] == 1.0
        assert result["suspicious"] is True

    def test_neither_public_function_raises_on_a_perfect_statistic(self):
        """D-15: nothing here gates. A high statistic is reported, never raised on."""
        s = _series([0, 1, 2, 3, 4] * 40)
        result = measure_labeling_dependence(s, s)
        text = format_dependence_report(result)
        assert isinstance(result, dict)
        assert isinstance(text, str) and text


# ── the independence direction ────────────────────────────────────────────


class TestIndependentLabelings:
    def test_structurally_unrelated_labelings_give_ari_near_zero(self):
        s1, s2 = _independent_pair()
        result = measure_labeling_dependence(s1, s2)
        ari = result["adjusted_rand"]
        assert abs(ari) < _INDEPENDENCE_ARI_BOUND, (
            f"adjusted_rand={ari!r} is not below the stated bound "
            f"{_INDEPENDENCE_ARI_BOUND} for two structurally unrelated labelings"
        )

    def test_independent_labelings_are_not_flagged_as_all_zero(self):
        """Near-zero is a result; *exactly* zero on all three is a suspicion."""
        s1, s2 = _independent_pair()
        result = measure_labeling_dependence(s1, s2)
        assert result["nmi"] > 0.0
        assert result["cramers_v"] > 0.0
        assert result["suspicious"] is False


class TestAllThreeExactlyZeroIsASuspicion:
    def test_degenerate_single_state_comparison_flags_an_alignment_bug(self):
        s1 = _series([(i // 37) % 5 for i in range(600)])
        s2 = _series([0] * 600)

        result = measure_labeling_dependence(s1, s2)
        assert result["adjusted_rand"] == 0.0
        assert result["nmi"] == 0.0
        assert result["cramers_v"] == 0.0
        assert result["suspicious"] is True
        assert "alignment" in result["suspicious_reason"].lower()


# ── the six-element oracles ───────────────────────────────────────────────


class TestSixElementOracle:
    """The live-verified six-element oracles.

    ``07-PATTERNS.md`` records the triple as "ARI=0.1667, NMI=0.5794,
    Cramer's V=0.5 for a 2x2 table". That triple is **not jointly
    realizable** and the record is corrected here: exhaustive enumeration of
    every six-element labeling pair in this environment shows

    - ARI 0.1667 with NMI 0.5794 arises only from a **3x3** table (two
      three-block partitions), where Cramer's V is 0.7071067811865476; and
    - the only six-element **2x2** table with Cramer's V exactly 0.5 is
      ``[[2, 0], [2, 2]]``, whose ARI is -0.0714 and NMI is 0.2740.

    Both are pinned below, so the correction is evidence rather than an
    assertion. See 07-09-SUMMARY.md's deviation record.
    """

    def test_three_by_three_oracle_matches_to_recorded_precision(self):
        result = measure_labeling_dependence(_series(_ORACLE_3X3_A), _series(_ORACLE_3X3_B))
        assert result["n_compared"] == 6
        assert result["adjusted_rand"] == pytest.approx(_ORACLE_3X3_ARI, abs=1e-12)
        assert result["nmi"] == pytest.approx(_ORACLE_3X3_NMI, abs=1e-12)
        assert result["cramers_v"] == pytest.approx(_ORACLE_3X3_V, abs=1e-12)

    def test_two_by_two_oracle_matches_to_recorded_precision(self):
        result = measure_labeling_dependence(_series(_ORACLE_2X2_A), _series(_ORACLE_2X2_B))
        assert result["n_compared"] == 6
        assert result["adjusted_rand"] == pytest.approx(_ORACLE_2X2_ARI, abs=1e-12)
        assert result["nmi"] == pytest.approx(_ORACLE_2X2_NMI, abs=1e-12)
        assert result["cramers_v"] == pytest.approx(_ORACLE_2X2_V, abs=1e-12)

    def test_patterns_recorded_triple_is_not_jointly_realizable(self):
        """Pins the correction itself, so it cannot be quietly reverted."""
        three = measure_labeling_dependence(_series(_ORACLE_3X3_A), _series(_ORACLE_3X3_B))
        assert three["cramers_v"] != pytest.approx(0.5, abs=1e-4), (
            "07-PATTERNS.md records Cramer's V = 0.5 alongside ARI 0.1667 / NMI 0.5794; "
            "the table that produces those two is 3x3 and its V is 0.7071."
        )


# ── the disjoint-span direction: NaN, never zero ──────────────────────────


class TestDisjointIndex:
    def test_disjoint_index_gives_nan_statistics_and_warns_without_raising(self, caplog):
        s1 = _series([0, 1, 0, 1, 2, 2], start="1972-01-31")
        s2 = _series([0, 1, 0, 1, 2, 2], start="2005-01-31")

        with caplog.at_level(logging.WARNING):
            result = measure_labeling_dependence(s1, s2)

        assert result["n_compared"] == 0
        assert math.isnan(result["adjusted_rand"])
        assert math.isnan(result["nmi"])
        assert math.isnan(result["cramers_v"])
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_disjoint_index_never_reports_zero(self):
        """Zero would read as independence; NaN reads as 'nothing was compared'."""
        s1 = _series([0, 1, 0, 1, 2, 2], start="1972-01-31")
        s2 = _series([0, 1, 0, 1, 2, 2], start="2005-01-31")
        result = measure_labeling_dependence(s1, s2)
        for key in ("adjusted_rand", "nmi", "cramers_v"):
            assert result[key] != 0.0, f"{key} reported 0.0 for a disjoint span — reads as independence"
        assert result["suspicious"] is False

    def test_raw_state_prefix_strings_are_reported_as_a_coercion_finding(self, caplog):
        """The silent-zero trap ``07-PREFIX-EVIDENCE.md`` documents.

        ``label_disagreement`` coerces non-numeric values to NaN and drops
        them, so raw ``"state_3"`` labels return ``n_compared == 0`` with no
        exception — which reads as 'totally resolved' while nothing was
        compared. The indexes here overlap fully, so the module must name
        the coercion, not the span.
        """
        idx = _months(6)
        s1 = pd.Series([f"state_{v}" for v in [0, 1, 0, 1, 2, 2]], index=idx, name="state")
        s2 = pd.Series([f"state_{v}" for v in [0, 0, 1, 1, 2, 2]], index=idx, name="state")

        with caplog.at_level(logging.WARNING):
            result = measure_labeling_dependence(s1, s2)

        assert result["n_compared"] == 0
        assert math.isnan(result["adjusted_rand"])
        messages = " ".join(rec.getMessage() for rec in caplog.records).lower()
        assert "coerc" in messages or "non-numeric" in messages, (
            "a full index overlap with n_compared == 0 is a coercion bug, not a "
            "disjoint span, and the warning must say so"
        )


# ── D-15: the flag levels are prose, not thresholds ───────────────────────


class TestFlagConstantsNeverGate:
    def test_flag_levels_hold_the_documented_values(self):
        assert DEPENDENCE_FLAG_ARI == 0.7
        assert DEPENDENCE_FLAG_NMI == 0.5
        assert DEPENDENCE_FLAG_CRAMERS_V == 0.5

    def test_measurement_source_never_references_a_flag_constant(self):
        source = inspect.getsource(measure_labeling_dependence)
        assert "DEPENDENCE_FLAG" not in source, (
            "measure_labeling_dependence must not reference a flag level: D-15 "
            "forbids a pre-declared pass/fail threshold, so nothing the measurement "
            "returns may depend on one."
        )

    def test_flag_constants_are_referenced_only_inside_the_report_formatter(self):
        module_path = Path(inspect.getfile(dependence_module))
        tree = ast.parse(module_path.read_text())
        offenders: list[str] = []
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name == "format_dependence_report":
                continue
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and sub.id.startswith("DEPENDENCE_FLAG"):
                    offenders.append(f"{node.name}:{sub.lineno}")
        assert offenders == [], f"flag levels referenced outside the report formatter: {offenders}"


# ── the rendered record ───────────────────────────────────────────────────


class TestFormatDependenceReport:
    def test_every_statistic_renders_with_its_denominator_and_window(self):
        s1 = _series(_ORACLE_3X3_A)
        s2 = _series(_ORACLE_3X3_B)
        text = format_dependence_report(measure_labeling_dependence(s1, s2))

        for line in text.splitlines():
            low = line.lower()
            if "adjusted rand" in low or "normalized mutual" in low or "cram" in low:
                assert "6" in line, f"statistic line carries no n_compared denominator: {line!r}"
                assert "1972-01-31" in line and str(s1.index[-1].date()) in line, (
                    f"statistic line carries no window: {line!r}"
                )

    def test_an_exceeded_flag_renders_the_failure_to_add_an_axis_wording(self):
        s = _series([0, 1, 2, 3, 4] * 40)
        text = format_dependence_report(measure_labeling_dependence(s, s)).lower()
        assert "failure to add an axis" in text

    def test_low_statistics_do_not_render_the_failure_wording(self):
        s1, s2 = _independent_pair()
        text = format_dependence_report(measure_labeling_dependence(s1, s2)).lower()
        assert "failure to add an axis" not in text

    def test_disjoint_span_renders_as_a_finding_not_as_a_number(self):
        s1 = _series([0, 1, 0, 1, 2, 2], start="1972-01-31")
        s2 = _series([0, 1, 0, 1, 2, 2], start="2005-01-31")
        text = format_dependence_report(measure_labeling_dependence(s1, s2)).lower()
        assert "n_compared" in text
        assert "nan" in text
        assert "failure to add an axis" not in text


# ── block_permutation_null: the pre-registered control ──────────────────────
#
# Oracles that fail in BOTH directions. A control that only ever says "inside
# the null" would rubber-stamp verdict (b); one that only ever says "above"
# would rubber-stamp (a). Both cases are pinned.


class TestBlockPermutationNull:
    @staticmethod
    def _blocky(pattern: list[tuple[int, int]]) -> pd.Series:
        vals = np.concatenate([np.full(n, s) for s, n in pattern])
        idx = pd.date_range("1963-01-31", periods=len(vals), freq="ME")
        return pd.Series(vals, index=idx, name="state")

    def test_shuffling_preserves_occupancy_exactly(self):
        from trading_crab_lib.platform.evaluation.dependence import _blocks, _shuffle_blocks

        s = self._blocky([(0, 30), (1, 50), (0, 20), (2, 40)]).to_numpy()
        rng = np.random.default_rng(1)
        for _ in range(25):
            out = _shuffle_blocks(s, rng)
            assert len(out) == len(s)
            assert sorted(pd.Series(out).value_counts().items()) == sorted(pd.Series(s).value_counts().items())

    def test_shuffling_never_increases_the_block_count(self):
        """Blocks may MERGE when the shuffle puts two same-state blocks
        adjacent, so the count can fall but must never rise — a rise would mean
        the shuffle invented structure. The merge bias is documented in
        _shuffle_blocks: it shifts the null UP, which can only push toward (b)."""
        from trading_crab_lib.platform.evaluation.dependence import _blocks, _shuffle_blocks

        s = self._blocky([(0, 30), (1, 50), (0, 20), (2, 40)]).to_numpy()
        rng = np.random.default_rng(2)
        n_orig = len(_blocks(s))
        for _ in range(25):
            assert len(_blocks(_shuffle_blocks(s, rng))) <= n_orig

    def test_identical_labelings_sit_far_ABOVE_the_null(self):
        """Direction 1. Two identical blocky labelings have observed NMI = 1.0,
        which block-shuffling cannot reproduce — the control MUST place it above
        the 99th percentile. A control that failed this could never return (a)."""
        from trading_crab_lib.platform.evaluation.dependence import block_permutation_null

        s = self._blocky([(0, 40), (1, 60), (2, 50), (0, 30), (1, 45)])
        res = block_permutation_null(s, s.copy(), n_resamples=300, random_state=7)
        assert res["observed"]["nmi"] == pytest.approx(1.0)
        assert res["observed"]["nmi"] > res["null"]["nmi"]["p99"]

    def test_independent_blocky_labelings_sit_INSIDE_the_null(self):
        """Direction 2. Two labelings whose block structure is real but whose
        alignment is arbitrary must NOT clear the 95th percentile. A control
        that failed this could never return (b)."""
        from trading_crab_lib.platform.evaluation.dependence import block_permutation_null

        rng = np.random.default_rng(11)
        a = self._blocky([(int(rng.integers(0, 3)), int(rng.integers(20, 60))) for _ in range(9)])
        b = self._blocky([(int(rng.integers(0, 3)), int(rng.integers(20, 60))) for _ in range(9)])
        n = min(len(a), len(b))
        a, b = a.iloc[:n], b.iloc[:n]
        res = block_permutation_null(a, b, n_resamples=300, random_state=8)
        assert res["observed"]["nmi"] <= res["null"]["nmi"]["p99"]

    def test_is_deterministic_under_a_fixed_seed(self):
        from trading_crab_lib.platform.evaluation.dependence import block_permutation_null

        a = self._blocky([(0, 30), (1, 40), (2, 35)])
        b = self._blocky([(1, 25), (0, 45), (2, 35)])
        r1 = block_permutation_null(a, b, n_resamples=120, random_state=99)
        r2 = block_permutation_null(a, b, n_resamples=120, random_state=99)
        assert r1["null"]["nmi"]["p95"] == r2["null"]["nmi"]["p95"]

    def test_reports_block_counts_it_actually_used(self):
        from trading_crab_lib.platform.evaluation.dependence import block_permutation_null

        a = self._blocky([(0, 30), (1, 40), (2, 35)])       # 3 blocks
        b = self._blocky([(1, 25), (0, 45), (2, 20), (1, 15)])  # 4 blocks
        n = min(len(a), len(b))
        res = block_permutation_null(a.iloc[:n], b.iloc[:n], n_resamples=50, random_state=5)
        assert res["n_blocks_1"] >= 3
        assert res["n_blocks_2"] >= 3
