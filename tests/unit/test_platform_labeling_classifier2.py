"""Unit tests for trading_crab_lib.platform.labeling.classifier2 (REG-01 criterion 5).

Classifier #2 is the leadership/relative-axis L1 labeler, fit unsupervised on a
feature set disjoint from classifier #1's 13 lean columns (D-10), frozen under
the same rule at the same 1972+ decision window (D-11), with K and lambda set by
construction and ZERO selection trials (D-13, ADR-0002).

Synthetic frames with hand-reasoned outcomes only — no network, no live
checkpoint reads. Mirrors tests/unit/test_platform_labeling.py's fixture shape.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.evaluation.report import _reference_label_columns
from trading_crab_lib.platform.labeling.classifier2 import (
    CLASSIFIER2_CANDIDATE_COLUMNS,
    CLASSIFIER2_LABELS_CHECKPOINT,
    classifier2_config,
    freeze_classifier2_columns,
    label_leadership_regimes,
)
from trading_crab_lib.platform.labeling.diagnostics import (
    _MAX_OCCUPANCY_THRESHOLD,
    _MIN_OCCUPANCY_THRESHOLD,
)


def _monthly_index(n_months: int, start: str = "1990-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_months, freq="ME")


def _candidate_frame(n_months: int = 120, seed: int = 3, start: str = "1990-01-31") -> pd.DataFrame:
    """A feature frame carrying every one of classifier #2's eight candidates."""
    rng = np.random.default_rng(seed)
    idx = _monthly_index(n_months, start=start)
    return pd.DataFrame(
        {col: rng.normal(0.0, 1.0, n_months) for col in CLASSIFIER2_CANDIDATE_COLUMNS},
        index=idx,
    )


def _degenerate_frame(n_months: int = 150) -> pd.DataFrame:
    """Every candidate column constant, so the fit collapses onto ONE state.

    After ``standardize_features`` every row is the zero vector, so K=3 cannot
    find three occupied clusters: two states end up never occupied (occupancy
    0.0, below §4.4 criterion 1's ~8% floor). No lambda override is needed — the
    collapse comes from the data, so the pinned lambda = 4n stays honest.
    """
    idx = _monthly_index(n_months)
    return pd.DataFrame(
        {col: np.full(n_months, float(i + 1)) for i, col in enumerate(CLASSIFIER2_CANDIDATE_COLUMNS)},
        index=idx,
    )


def _cfg(**overrides) -> dict:
    """A minimal platform-config-shaped dict for classifier #2."""
    features = overrides.pop("features", list(CLASSIFIER2_CANDIDATE_COLUMNS))
    section = {
        "K": 3,
        "lambda": 4.0 * len(features),
        "n_restarts": 2,
        "sort_column": "rs_equities_bonds",
        "features": features,
    }
    section.update(overrides)
    return {"labeling_2": section}


# ── classifier2_config: D-13's lambda = 4n made unfakeable ──────────────────


class TestClassifier2Config:
    def test_live_config_carries_the_six_pinned_values(self):
        """ADR-0002's decision (a)-(d), read off the real settings.yaml."""
        resolved = classifier2_config(load_platform_config())
        assert resolved["K"] == 3
        assert resolved["lam"] == 32.0
        assert resolved["sort_column"] == "rs_equities_bonds"
        assert resolved["features"] == list(CLASSIFIER2_CANDIDATE_COLUMNS)
        assert len(resolved["features"]) == 8

    def test_module_constant_matches_the_configured_feature_list(self):
        """The constant and the config are two copies of one pinned list; if they
        ever drift, the ADR and the code disagree about what was fit."""
        configured = load_platform_config()["labeling_2"]["features"]
        assert list(CLASSIFIER2_CANDIDATE_COLUMNS) == list(configured)

    def test_lambda_not_four_times_feature_count_raises(self):
        """D-13's formula is an invariant, not a comment: an edit that changes
        the feature list without recomputing lambda must fail loudly."""
        cfg = _cfg()
        cfg["labeling_2"]["lambda"] = 52.0  # classifier #1's value, 4 x 13, not 4 x 8
        with pytest.raises(ValueError, match="lambda"):
            classifier2_config(cfg)

    def test_lambda_off_by_one_column_raises(self):
        """Drop a column but keep lambda: the exact drift this guard exists for."""
        cfg = _cfg()
        cfg["labeling_2"]["features"] = list(CLASSIFIER2_CANDIDATE_COLUMNS)[:-1]  # n = 7
        assert cfg["labeling_2"]["lambda"] == 32.0  # still 4 x 8
        with pytest.raises(ValueError, match="lambda"):
            classifier2_config(cfg)

    def test_lambda_exactly_four_times_feature_count_accepted(self):
        cfg = _cfg(features=["a", "b", "c"], sort_column="a")
        assert classifier2_config(cfg)["lam"] == 12.0

    def test_sort_column_absent_from_feature_list_raises(self):
        cfg = _cfg()
        cfg["labeling_2"]["sort_column"] = "trailing_return_1m"  # classifier #1's key
        with pytest.raises(ValueError, match="sort_column"):
            classifier2_config(cfg)

    def test_missing_section_falls_back_to_the_module_constants(self):
        """Read defensively via cfg.get() — an absent labeling_2 section must not
        KeyError, per the additive-config convention."""
        resolved = classifier2_config({})
        assert resolved["features"] == list(CLASSIFIER2_CANDIDATE_COLUMNS)
        assert resolved["lam"] == 4.0 * len(CLASSIFIER2_CANDIDATE_COLUMNS)


# ── freeze_classifier2_columns: D-11's freeze rule, reused not reimplemented ─


class TestFreezeClassifier2Columns:
    def test_returns_every_candidate_in_declaration_order(self):
        df = _candidate_frame()
        frozen = freeze_classifier2_columns(df, _cfg(), df.index[24])
        assert frozen == list(CLASSIFIER2_CANDIDATE_COLUMNS)

    def test_matches_reference_label_columns_called_directly(self):
        """The freeze rule is reused, never reimplemented — a second
        implementation is exactly the divergence criterion 1 exists to prevent."""
        df = _candidate_frame()
        df.loc[df.index[:40], "oil_mom_12m"] = np.nan
        first_decision = df.index[24]
        frozen = freeze_classifier2_columns(df, _cfg(), first_decision)
        oracle = _reference_label_columns(df, list(CLASSIFIER2_CANDIDATE_COLUMNS), first_decision)
        assert frozen == oracle

    def test_late_starting_candidate_excluded_by_name_with_its_first_valid_month(self, caplog):
        df = _candidate_frame()
        df.loc[df.index[:40], "oil_mom_12m"] = np.nan  # starts after the decision date
        with caplog.at_level(logging.INFO):
            frozen = freeze_classifier2_columns(df, _cfg(), df.index[24])
        assert "oil_mom_12m" not in frozen
        assert len(frozen) == 7
        text = caplog.text
        assert "oil_mom_12m" in text
        assert str(df.index[40].date()) in text  # its first valid month, by name

    def test_candidate_absent_from_the_frame_is_excluded_not_a_keyerror(self, caplog):
        df = _candidate_frame().drop(columns=["m2_gdp"])
        with caplog.at_level(logging.INFO):
            frozen = freeze_classifier2_columns(df, _cfg(), df.index[24])
        assert "m2_gdp" not in frozen
        assert len(frozen) == 7
        assert "m2_gdp" in caplog.text

    def test_empty_frozen_list_raises_naming_the_count(self):
        df = pd.DataFrame({"unrelated": np.arange(60.0)}, index=_monthly_index(60))
        with pytest.raises(ValueError, match="0 usable column"):
            freeze_classifier2_columns(df, _cfg(), df.index[24])

    def test_shorter_than_K_raises_naming_the_count_and_K(self):
        df = _candidate_frame()[["rs_equities_bonds", "rs_oil_equities"]]
        with pytest.raises(ValueError, match="fewer than K=3"):
            freeze_classifier2_columns(df, _cfg(), df.index[24])

    def test_all_candidates_nan_after_the_decision_date_raises(self):
        df = _candidate_frame()
        df.loc[df.index[30:], :] = np.nan
        with pytest.raises(ValueError, match="0 usable column"):
            freeze_classifier2_columns(df, _cfg(), df.index[24])


# ── label_leadership_regimes: the fit, its ordering, occupancy and floor ─────


class TestLabelLeadershipRegimes:
    def test_occupancy_sums_to_one_and_has_exactly_K_entries(self, tmp_path):
        df = _candidate_frame(n_months=150)
        result = label_leadership_regimes(
            df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24]
        )
        occupancy = result["occupancy"]
        assert len(occupancy) == 3
        assert abs(sum(occupancy.values()) - 1.0) < 1e-12

    def test_never_occupied_state_is_a_zero_entry_not_a_missing_key(self, tmp_path):
        """A huge jump penalty collapses the fit to one state; K=3 must still
        report three entries, two of them 0.0 — passing n_states=K explicitly is
        what makes a never-occupied state surface instead of vanishing."""
        df = _degenerate_frame()
        result = label_leadership_regimes(
            df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24]
        )
        occupancy = result["occupancy"]
        assert len(occupancy) == 3
        assert sorted(occupancy.values()) == pytest.approx([0.0, 0.0, 1.0])
        assert abs(sum(occupancy.values()) - 1.0) < 1e-12

    def test_below_floor_state_warns_naming_that_state_and_still_returns(self, tmp_path, caplog):
        """§4.4 criterion 1's ~8% floor is report-only (D-02/D-07): a loud
        WARNING naming the offending state index, and the labeler still
        completes. (Corrected 2026-09-17: this asserted a 5% floor, a number
        that appears nowhere in design §4.4.)"""
        df = _degenerate_frame()
        with caplog.at_level(logging.WARNING):
            result = label_leadership_regimes(
                df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24]
            )
        below = [s for s, occ in result["occupancy"].items() if occ < _MIN_OCCUPANCY_THRESHOLD]
        assert below, "fixture did not produce a sub-floor state — test cannot fail"
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        for state in below:
            assert any(f"State {state}" in msg for msg in warnings), (
                f"no WARNING named state {state}; warnings were {warnings}"
            )
        assert result["states"] is not None  # returned normally, never raised

    def test_missing_ordering_column_raises(self, tmp_path):
        """Plan 07-05 deleted canonicalize_states' centroid-column-0 fallback
        precisely so this cannot pass silently with arbitrary state IDs."""
        df = _candidate_frame().drop(columns=["rs_equities_bonds"])
        with pytest.raises(ValueError, match="rs_equities_bonds"):
            label_leadership_regimes(
                df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24]
            )

    def test_states_are_numbered_by_ascending_sort_column_centroid(self, tmp_path):
        """The canonical-ordering oracle. ``decoy`` is column 0 and is the exact
        NEGATIVE of the sort column, so an implementation that ignores
        sort_column and falls back to centroid column 0 produces the REVERSED
        numbering and fails here."""
        n = 120
        idx = _monthly_index(n)
        ramp = np.concatenate([np.full(n // 2, -3.0), np.full(n - n // 2, 3.0)])
        df = pd.DataFrame(
            {"decoy": -ramp, "rs_equities_bonds": ramp, "third": np.zeros(n)}, index=idx
        )
        cfg = _cfg(features=["decoy", "rs_equities_bonds", "third"], K=2)
        cfg["labeling_2"]["lambda"] = 12.0
        result = label_leadership_regimes(
            df, cfg, checkpoint_dir=tmp_path, first_decision=idx[10]
        )
        states = result["states"]
        assert states[0] == 0, "first (low rs_equities_bonds) months must be state 0"
        assert states[-1] == 1, "last (high rs_equities_bonds) months must be state K-1"

    def test_persists_under_classifier2_names_and_never_touches_regime_labels(self, tmp_path):
        """T-07-15: overwriting classifier #1's checkpoint would silently
        re-point §5.4, the Brier tables and the nowcaster at the wrong labeling."""
        df = _candidate_frame(n_months=150)
        label_leadership_regimes(df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24])
        written = {p.stem for p in tmp_path.glob("*.parquet")}
        assert "regime_labels_2" in written
        assert "regime_confidences_2" in written
        assert "regime_labels" not in written
        assert "regime_confidences" not in written

    def test_returns_frozen_columns_states_and_confidences_aligned(self, tmp_path):
        df = _candidate_frame(n_months=150)
        result = label_leadership_regimes(
            df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24]
        )
        assert result["frozen_columns"] == list(CLASSIFIER2_CANDIDATE_COLUMNS)
        assert len(result["states"]) == len(result["index"]) == 150
        assert result["confidences"].shape == (150, 3)
        row_sums = result["confidences"].sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-12)

    def test_post_holdout_months_are_carved_before_the_fit(self, tmp_path):
        """T-07-17: the fit is a development activity and must not read
        post-2020-12 data, whatever frame the caller hands it."""
        df = _candidate_frame(n_months=420, start="1995-01-31")  # runs past 2020-12
        assert df.index.max() > pd.Timestamp("2020-12-31")
        result = label_leadership_regimes(
            df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24]
        )
        assert result["index"].max() <= pd.Timestamp("2020-12-31")
        assert len(result["states"]) < len(df)

    def test_first_decision_defaults_to_the_min_train_months_index(self, tmp_path):
        """Mirrors report.py's own derivation: dev_features.index[min_train]."""
        df = _candidate_frame(n_months=200)
        cfg = _cfg()
        cfg["backtest"] = {"min_train_months": 120}
        df.loc[df.index[:100], "oil_mom_12m"] = np.nan  # valid before index[120]
        df.loc[df.index[:130], "m2_gdp"] = np.nan  # NOT valid at index[120]
        result = label_leadership_regimes(df, cfg, checkpoint_dir=tmp_path)
        assert result["first_decision"] == df.index[120]
        assert "oil_mom_12m" in result["frozen_columns"]
        assert "m2_gdp" not in result["frozen_columns"]

    def test_frame_shorter_than_min_train_months_raises_rather_than_guessing(self, tmp_path):
        df = _candidate_frame(n_months=60)
        cfg = _cfg()
        cfg["backtest"] = {"min_train_months": 120}
        with pytest.raises(ValueError, match="min_train_months"):
            label_leadership_regimes(df, cfg, checkpoint_dir=tmp_path)

    def test_two_calls_on_the_same_frame_return_identical_states(self, tmp_path):
        df = _candidate_frame(n_months=150)
        first = label_leadership_regimes(
            df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24]
        )
        second = label_leadership_regimes(
            df, _cfg(), checkpoint_dir=tmp_path, first_decision=df.index[24]
        )
        assert np.array_equal(first["states"], second["states"])


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])


# ── Design §4.4 criterion 1 against the LIVE fit ────────────────────────────
#
# Added 2026-09-17 after the correction described in diagnostics.py: the whole
# project had been citing a "§4.4 five-percent floor" that does not exist, and
# the cap ("<= ~35%") had never been implemented or checked for ANY classifier.
# These two tests check the real criterion against the committed live fit.
#
# They are deliberately split. A single strict-xfail test would also "xfail" if
# the checkpoint were simply missing — passing for the wrong reason, which is
# exactly the evidence-shape failure this phase exists to catch. So presence is
# a plain, loud test and only the band assertion carries the marker.

_LIVE_LABELS_PARQUET = (
    Path(__file__).resolve().parents[2]
    / "data" / "checkpoints" / "platform" / f"{CLASSIFIER2_LABELS_CHECKPOINT}.parquet"
)


class TestClassifier2LiveOccupancyAgainstDesign44:
    def test_live_labels_checkpoint_is_present(self):
        """Guards the xfail below: it must fail on the BAND, never on absence.

        ``data/checkpoints/platform/`` is a git-tracked namespace, so a missing
        file here is a real defect, not an environment quirk.
        """
        assert _LIVE_LABELS_PARQUET.is_file(), (
            f"{_LIVE_LABELS_PARQUET} missing — the occupancy band test below would "
            "xfail for the wrong reason and read as expected"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "classifier #2 (K=3, lambda=32) breaches design §4.4 criterion 1: "
            "states 1 and 2 occupy 46.12% and 38.51% against a ~35% cap. "
            "K=3 is near-infeasible against the 8-35% band at all — three states "
            "summing to 100% under a 35% cap must each sit in [30%, 35%], which "
            "is forced balance, the thing §4.3 set out to replace. Pending the "
            "K/lambda re-pin; design §4.3 licenses tuning both until §4.4 passes. "
            "strict=True: an xpass FAILS, so this marker cannot outlive the fix."
        ),
    )
    def test_live_occupancy_within_design_44_band(self):
        states = pd.read_parquet(_LIVE_LABELS_PARQUET)["state"]
        occupancy = states.value_counts(normalize=True).sort_index()
        breaches = {
            int(s): float(occ)
            for s, occ in occupancy.items()
            if occ < _MIN_OCCUPANCY_THRESHOLD or occ > _MAX_OCCUPANCY_THRESHOLD
        }
        assert not breaches, (
            "states outside design §4.4 criterion 1's "
            f"[{_MIN_OCCUPANCY_THRESHOLD:.0%}, {_MAX_OCCUPANCY_THRESHOLD:.0%}] band: "
            + ", ".join(f"state {s} at {occ:.2%}" for s, occ in sorted(breaches.items()))
        )
