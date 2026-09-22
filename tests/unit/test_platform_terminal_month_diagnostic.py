"""Guards for the terminal-month churn-vs-k diagnostic (plan 08-02).

Two layers, and the split matters:

1. **Synthetic-fixture tests** of the counting rules. Every one of them is built so
   that it FAILS on a specific, named implementation mistake — an off-by-one in the
   adjacent-pair comparison, comparing NaN with ``!=``, and silently assuming the
   fit's terminal month is the window's last month. No real data, no network, no fit.

2. **Artifact-level tests** against the persisted run, which re-derive every quoted
   number from the label matrix rather than trusting the JSON's own summary.

**WHY THERE IS NO MONOTONICITY TEST, AND IT IS A DECISION.**

The diagnostic exists to answer whether churn falls with k. Asserting that it does
would promote the hypothesis to a gate after the data had been seen, and would make
the diagnostic structurally incapable of returning its own refutation — precisely the
"check that can only confirm" this project has now recorded six times
(``08-CONTEXT.md`` AMENDMENT). A flat curve is a VALID and reportable outcome: it
refutes the terminal-month story and leaves λ/d as the whole explanation. So nothing
in this module asserts any relationship between ``by_lag[k].rate`` and ``k``, in
either direction. The omission is the point; do not "fix" it.

What IS asserted is the anchor: k=1 must reproduce the tracked curve elementwise
(246 / 24). That is a check that can fail, and if it fails no k>1 number means
anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from terminal_month_diagnostic import (  # noqa: E402
    AnchorError,
    assert_anchor,
    churn_by_lag,
    fixed_month_revision,
    lag_record,
    state_change_count,
)

_TRACK_A = _ROOT / "outputs" / "reports" / "platform" / "track_a"
_LABELS = _TRACK_A / "terminal_month_labels.parquet"
_CHURN = _TRACK_A / "terminal_month_churn.json"

#: The full-window run this phase measured. Artifact tests are pinned to it so a
#: truncated smoke artifact cannot masquerade as the measurement.
_EXPECTED_N_STEPS = 588
_EXPECTED_MAX_LAG = 6
_ANCHOR_COUNTS = {"classifier_1": 246, "classifier_2": 24}


def _months(n: int, start: str = "1990-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


# ── layer 1: the counting rules, on synthetic fixtures ─────────────────────────

class TestStateChangeCount:
    """The dropna-then-compare rule, pair-denominated."""

    def test_counts_adjacent_changes_not_rows(self):
        # 0 0 1 1 2 0 -> changes at positions 2, 4, 5 = 3 changes over 5 pairs.
        out = state_change_count(pd.Series([0, 0, 1, 1, 2, 0]))
        assert out == {"n_changes": 3, "n_rows": 6, "n_pairs": 5, "rate": 3 / 5}

    def test_constant_series_has_zero_changes(self):
        # Fails on the classic off-by-one where the leading `!= shift()` True is
        # not subtracted: that would report 1 change on a series that never moves.
        assert state_change_count(pd.Series([2, 2, 2, 2]))["n_changes"] == 0

    def test_single_row_has_no_pairs(self):
        out = state_change_count(pd.Series([3]))
        assert out["n_changes"] == 0 and out["n_pairs"] == 0

    def test_empty_series(self):
        out = state_change_count(pd.Series(dtype=float))
        assert out == {"n_changes": 0, "n_rows": 0, "n_pairs": 0, "rate": 0.0}

    def test_nan_is_dropped_not_compared(self):
        # 0, NaN, 0 -> after dropna: 0, 0 -> ZERO changes.
        # Fails if NaN is compared with `!=` (np.nan != np.nan is True), which
        # would report 2 changes here and inflate every short-window lag series.
        out = state_change_count(pd.Series([0.0, np.nan, 0.0]))
        assert out["n_changes"] == 0
        assert out["n_rows"] == 2 and out["n_pairs"] == 1


class TestChurnByLag:
    """A hand-built (step x lag) matrix whose per-lag counts are known by construction."""

    @staticmethod
    def _matrix() -> pd.DataFrame:
        # lag1: 0 0 1 0 2 2 -> 3 changes. lag4: 1 1 1 1 2 2 -> 1 change.
        lag1 = [0, 0, 1, 0, 2, 2]
        lag4 = [1, 1, 1, 1, 2, 2]
        idx = _months(6)
        cols = {"t": idx}
        for k in range(1, 7):
            cols[f"c1_lag{k}_state"] = [float(v) for v in (lag1 if k == 1 else lag4)]
            cols[f"c1_lag{k}_date"] = idx
            cols[f"c2_lag{k}_state"] = [0.0] * 6
            cols[f"c2_lag{k}_date"] = idx
        return pd.DataFrame(cols)

    def test_known_counts_per_lag(self):
        out = churn_by_lag(self._matrix())
        by_k = {e["k"]: e for e in out["classifier_1"]}
        assert by_k[1]["n_changes"] == 3
        assert by_k[4]["n_changes"] == 1
        assert by_k[1]["n_pairs"] == 5 and by_k[1]["rate"] == 3 / 5

    def test_both_classifiers_and_all_six_lags_present(self):
        out = churn_by_lag(self._matrix())
        assert set(out) == {"classifier_1", "classifier_2"}
        assert [e["k"] for e in out["classifier_2"]] == [1, 2, 3, 4, 5, 6]
        assert all(e["n_changes"] == 0 for e in out["classifier_2"])

    def test_nan_lag_cells_skip_the_pair(self):
        """A short-window step records NaN; the lag's churn must not count it."""
        df = self._matrix()
        df.loc[2, "c1_lag6_state"] = np.nan
        df.loc[2, "c1_lag6_date"] = pd.NaT
        # Remaining lag6 values: 1 1 1 2 2 -> 1 change over 4 pairs.
        entry = {e["k"]: e for e in churn_by_lag(df)["classifier_1"]}[6]
        assert entry["n_changes"] == 1
        assert entry["n_rows"] == 5 and entry["n_pairs"] == 4


class TestLagRecord:
    """Per-step recording: NaN for unavailable lags, and date alignment is CHECKED."""

    def test_short_window_records_nan_for_unavailable_lags(self):
        idx = _months(3)
        states = pd.Series([0, 1, 1], index=idx)
        cols, misaligned = lag_record(states, idx, max_lag=6, prefix="c1")
        assert cols["c1_lag3_state"] == 0.0
        assert pd.isna(cols["c1_lag4_state"]) and pd.isna(cols["c1_lag4_date"])
        assert pd.isna(cols["c1_lag6_state"])
        assert misaligned is False

    def test_dates_walk_backwards_from_the_terminal_month(self):
        idx = _months(6)
        states = pd.Series([0, 1, 2, 3, 4, 5], index=idx)
        cols, _ = lag_record(states, idx, max_lag=3, prefix="c2")
        assert cols["c2_lag1_date"] == idx[-1]
        assert cols["c2_lag2_date"] == idx[-2]
        assert cols["c2_lag3_date"] == idx[-3]
        assert cols["c2_lag1_state"] == 5.0

    def test_misalignment_is_detected_when_frozen_columns_end_early(self):
        """``_refit_l1`` returns post-dropna states: ``iloc[-1]`` need not be
        ``train_index[-1]``. Fails if the harness silently assumes alignment."""
        train_index = _months(6)
        states = pd.Series([0, 1, 1, 0, 0], index=train_index[:-1])  # trailing NaN row
        _cols, misaligned = lag_record(states, train_index, max_lag=6, prefix="c1")
        assert misaligned is True

    def test_empty_states_is_not_reported_as_misaligned(self):
        cols, misaligned = lag_record(pd.Series(dtype=float), _months(6), max_lag=2, prefix="c1")
        assert misaligned is False
        assert pd.isna(cols["c1_lag1_state"])


class TestCollectRecordsMisalignments:
    """The harness accumulates misalignments into a non-empty list for a bad fixture."""

    def test_misalignment_list_is_populated(self, monkeypatch):
        import terminal_month_diagnostic as tmd

        idx = _months(130)
        frame = pd.DataFrame({"a": np.arange(130, dtype=float)}, index=idx)
        inputs = {
            "features_1": frame, "features_2": frame,
            "frozen_1": ["a"], "frozen_2": ["a"], "min_train": 120,
        }

        def _short_fit(train_features, *_a, **_kw):
            # Always one month short of the window's end — the misalignment signature.
            sub = train_features.index[:-1]
            return pd.Series(np.zeros(len(sub)), index=sub, name="state")

        monkeypatch.setattr(tmd, "_refit_l1", lambda tf, cfg, **kw: _short_fit(tf))
        monkeypatch.setattr(tmd, "_refit_classifier2", lambda tf, **kw: _short_fit(tf))
        monkeypatch.setattr(
            tmd, "classifier2_config",
            lambda cfg: {"K": 2, "lam": 2.0, "n_restarts": 1, "sort_column": "a"},
        )
        monkeypatch.setattr(tmd, "split_by_holdout_boundary", lambda df, cutoff=None: (df, df.iloc[:0]))

        out = tmd.collect_terminal_labels({}, max_lag=3, inputs=inputs)
        assert len(out.attrs["date_misalignments"]) > 0
        assert out.attrs["date_misalignments"][0]["fit_terminal_month"] != \
            out.attrs["date_misalignments"][0]["train_index_last"]


class TestFixedMonthRevision:
    """The same matrix read down the other axis — a re-read, not a second experiment."""

    def test_revision_counts_disagreements_on_shared_months(self):
        idx = _months(4)
        df = pd.DataFrame(
            {
                "t": idx,
                "c1_lag1_state": [0.0, 1.0, 1.0, 2.0],
                "c1_lag1_date": idx,
                "c1_lag2_state": [np.nan, 0.0, 9.0, 1.0],   # month idx[1] revised 1 -> 9
                "c1_lag2_date": [pd.NaT, idx[0], idx[1], idx[2]],
                "c2_lag1_state": [0.0] * 4, "c2_lag1_date": idx,
                "c2_lag2_state": [np.nan, 0.0, 0.0, 0.0],
                "c2_lag2_date": [pd.NaT, idx[0], idx[1], idx[2]],
            }
        )
        out = fixed_month_revision(df, max_lag=2)
        entry = out["classifier_1"][0]
        assert entry["k"] == 2
        assert entry["n_compared"] == 3          # idx[0], idx[1], idx[2]
        assert entry["n_differs"] == 1           # only idx[1]: 1.0 vs 9.0
        assert out["classifier_2"][0]["n_differs"] == 0


class TestAnchorFailsLoudly:
    """The anchor must be able to FAIL — it is the plan's load-bearing discriminator."""

    @staticmethod
    def _tracked(tmp_path: Path, state_1, state_2, idx) -> Path:
        path = tmp_path / "tracked.parquet"
        pd.DataFrame({"state_1": state_1, "state_2": state_2}, index=idx).to_parquet(path)
        return path

    def test_matching_series_anchors(self, tmp_path):
        idx = pd.DatetimeIndex(_months(5), name="date")
        df = pd.DataFrame(
            {"t": idx, "c1_lag1_state": [0.0, 1, 1, 2, 2], "c2_lag1_state": [3.0, 3, 3, 3, 3]}
        )
        path = self._tracked(tmp_path, [0, 1, 1, 2, 2], [3, 3, 3, 3, 3], idx)
        out = assert_anchor(df, tracked_path=path, require_full_counts=False)
        assert out["classifier_1"]["k1_matches_tracked_curve"] is True
        assert out["classifier_1"]["n_mismatched"] == 0
        assert out["classifier_1"]["k1_n_changes"] == 2

    def test_one_differing_cell_raises(self, tmp_path):
        idx = pd.DatetimeIndex(_months(5), name="date")
        df = pd.DataFrame(
            {"t": idx, "c1_lag1_state": [0.0, 1, 1, 2, 2], "c2_lag1_state": [3.0, 3, 3, 3, 3]}
        )
        path = self._tracked(tmp_path, [0, 1, 1, 2, 5], [3, 3, 3, 3, 3], idx)
        with pytest.raises(AnchorError, match="NO k>1 number may be reported"):
            assert_anchor(df, tracked_path=path, require_full_counts=False)

    def test_shifted_index_raises(self, tmp_path):
        idx = pd.DatetimeIndex(_months(5), name="date")
        other = pd.DatetimeIndex(_months(5, start="1991-01-31"), name="date")
        df = pd.DataFrame(
            {"t": idx, "c1_lag1_state": [0.0] * 5, "c2_lag1_state": [0.0] * 5}
        )
        path = self._tracked(tmp_path, [0] * 5, [0] * 5, other)
        with pytest.raises(AnchorError):
            assert_anchor(df, tracked_path=path, require_full_counts=False)

    def test_wrong_full_window_count_raises(self, tmp_path):
        idx = pd.DatetimeIndex(_months(5), name="date")
        df = pd.DataFrame(
            {"t": idx, "c1_lag1_state": [0.0, 1, 1, 2, 2], "c2_lag1_state": [3.0, 3, 3, 3, 3]}
        )
        path = self._tracked(tmp_path, [0, 1, 1, 2, 2], [3, 3, 3, 3, 3], idx)
        # require_full_counts demands 246/24; this fixture has 2/0.
        with pytest.raises(AnchorError, match="k=1 churn is 2, expected 246"):
            assert_anchor(df, tracked_path=path, require_full_counts=True)


# ── layer 2: the persisted artifacts ───────────────────────────────────────────

_ARTIFACT_REASON = (
    f"terminal-month artifacts absent ({_LABELS} / {_CHURN}) — regenerate with "
    "`python scripts/terminal_month_diagnostic.py --out-dir outputs/reports/platform/track_a`. "
    "The phase baseline is 0 skipped, so a skip here means the artifacts were not committed."
)

pytestmark_artifacts = pytest.mark.skipif(
    not (_LABELS.exists() and _CHURN.exists()), reason=_ARTIFACT_REASON
)


@pytest.fixture(scope="module")
def record() -> dict:
    if not _CHURN.exists():
        pytest.skip(_ARTIFACT_REASON)
    return json.loads(_CHURN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def labels() -> pd.DataFrame:
    if not _LABELS.exists():
        pytest.skip(_ARTIFACT_REASON)
    return pd.read_parquet(_LABELS)


@pytestmark_artifacts
class TestPersistedRecord:
    def test_full_window_was_run(self, record, labels):
        assert record["full_window_run"] is True
        assert record["window"]["n_steps"] == _EXPECTED_N_STEPS
        assert record["window"]["first_date"] == "1972-01-31"
        assert record["window"]["last_date"] == "2020-12-31"
        assert len(labels) == _EXPECTED_N_STEPS

    @pytest.mark.parametrize("key", ["classifier_1", "classifier_2"])
    def test_k1_anchor_holds(self, record, key):
        block = record[key]
        assert block["anchor"]["k1_matches_tracked_curve"] is True
        assert block["anchor"]["n_mismatched"] == 0
        assert block["by_lag"][0]["k"] == 1
        assert block["by_lag"][0]["n_changes"] == _ANCHOR_COUNTS[key]

    @pytest.mark.parametrize("key", ["classifier_1", "classifier_2"])
    def test_six_strictly_increasing_lags_each_pair_denominated(self, record, key):
        by_lag = record[key]["by_lag"]
        assert len(by_lag) == _EXPECTED_MAX_LAG
        assert [e["k"] for e in by_lag] == list(range(1, _EXPECTED_MAX_LAG + 1))
        for entry in by_lag:
            assert entry["n_pairs"] == entry["n_rows"] - 1
            assert entry["rate"] == pytest.approx(entry["n_changes"] / entry["n_pairs"])

    @pytest.mark.parametrize(
        "key,expected_lambda_over_d", [("classifier_1", 1.0), ("classifier_2", 2.0)]
    )
    def test_lambda_over_d_from_live_config(self, record, key, expected_lambda_over_d):
        """Recomputed from the live config and the RESOLVED frozen list, not a literal.

        Fails loudly if either λ or a frozen feature list is re-pinned without this
        record being regenerated — the stale-recorded-number failure mode
        ``08-CONTEXT.md`` D-07 names.
        """
        from trading_crab_lib.platform.config import load_platform_config
        from trading_crab_lib.platform.labeling.classifier2 import classifier2_config

        cfg = load_platform_config()
        block = record[key]
        live_lambda = (
            float(cfg["labeling"]["lambda"]) if key == "classifier_1"
            else float(classifier2_config(cfg)["lam"])
        )
        assert block["lambda"] == live_lambda
        assert block["d"] == len(block["frozen_columns"])
        assert block["lambda_over_d"] == pytest.approx(block["lambda"] / block["d"])
        assert block["lambda_over_d"] == pytest.approx(expected_lambda_over_d)

    @pytest.mark.parametrize("key", ["classifier_1", "classifier_2"])
    def test_degraded_count_is_recorded(self, record, key):
        """A lag series quoted without its degraded count is Pitfall 6 in a new place."""
        assert "n_degraded" in record[key]
        assert isinstance(record[key]["n_degraded"], int)

    def test_no_lambda_was_swept(self, record):
        assert record["lambda_swept"] is False
        assert record["registry_trials_consumed"] == 0

    def test_churn_recomputes_from_the_matrix(self, record, labels):
        """Every quoted rate is re-derived from the label matrix, not trusted.

        The whole point of persisting the (step x k) matrix is that a later reader can
        recompute every derived number from it. If the JSON were hand-edited, this fails.
        """
        recomputed = churn_by_lag(labels, max_lag=_EXPECTED_MAX_LAG)
        for key in ("classifier_1", "classifier_2"):
            for stored, fresh in zip(record[key]["by_lag"], recomputed[key], strict=True):
                assert stored["k"] == fresh["k"]
                assert stored["n_changes"] == fresh["n_changes"]
                assert stored["n_rows"] == fresh["n_rows"]
                assert stored["rate"] == pytest.approx(fresh["rate"])

    def test_fixed_month_revision_is_a_separate_keyed_block(self, record):
        for key in ("classifier_1", "classifier_2"):
            rev = record[key]["fixed_month_revision"]
            assert [e["k"] for e in rev] == list(range(2, _EXPECTED_MAX_LAG + 1))
            for entry in rev:
                assert 0 <= entry["n_differs"] <= entry["n_compared"]

    def test_label_matrix_carries_a_date_per_cell(self, labels):
        for n in (1, 2):
            for k in range(1, _EXPECTED_MAX_LAG + 1):
                assert f"c{n}_lag{k}_state" in labels.columns
                assert f"c{n}_lag{k}_date" in labels.columns


class TestNoMonotonicityAssertionExists:
    """A structural guard on this module's own omission (see the module docstring).

    If a future edit adds "churn falls with k" as an assertion, the diagnostic stops
    being able to report its own refutation. This test makes that edit visible. It is
    deliberately NOT gated on the artifacts: the omission is a property of the source,
    not of any run.
    """

    #: Split so the token does not appear literally in the file it scans.
    _BANNED_TOKEN = "is_" + "monotonic"

    def test_no_assertion_orders_one_lag_against_another(self):
        src = Path(__file__).read_text(encoding="utf-8")
        assert self._BANNED_TOKEN not in src, (
            f"a {self._BANNED_TOKEN} assertion appeared in this module — see the "
            "docstring: asserting the hypothesis would make the diagnostic unable to "
            "return its own refutation."
        )
        offenders = [
            line.strip()
            for line in src.splitlines()
            if line.strip().startswith("assert")
            and "rate" in line
            and any(op in line for op in ("<", ">"))
        ]
        assert not offenders, (
            f"assertion(s) ordering lag rates found: {offenders}. The churn-vs-k "
            "relationship is the diagnostic's OUTPUT and must not be a gate."
        )

    def test_the_diagnostic_script_asserts_no_ordering_either(self):
        script = (_ROOT / "scripts" / "terminal_month_diagnostic.py").read_text(encoding="utf-8")
        assert self._BANNED_TOKEN not in script
