"""monthly_raw <-> monthly_features consistency (the merge-on-save trap).

``monthly_raw`` is a merge-on-save checkpoint: ``CheckpointManager.save()``
merges the in-memory frame with whatever is already on disk so a degraded
fetch cannot silently truncate history. It returns a ``Path``, not the merged
frame. ``build_monthly_spine`` therefore used to derive ``monthly_features``
from the PRE-MERGE local variable, which meant the two checkpoints were
structurally free to disagree — and nothing checked.

They did disagree, by 23 years, on ``oil``:

  - the 2026-09-18 build resolved the oil splice to macrotrends ``wti_crude``
    (1985-02+), so monthly_features got oil from 1985-02;
  - merge-on-save preserved an older 1962-01+ oil column in monthly_raw.

Each artifact looked internally consistent, so the staleness was diagnosed
BACKWARDS for a week: monthly_features was called stale and "recomputed" from
the stale raw column, and both design decision D-02-A and the A13 change-point
pinning in test_platform_plotting_regime.py were pinned to that bad recompute.
The error only surfaced when a human ran a clean rebuild and the suite went
red.

These tests are the guard that would have caught it without a human. They run
against the committed checkpoints in the git-tracked platform namespace, so a
missing file is a real defect and is asserted, never skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CKPT = _ROOT / "data" / "checkpoints" / "platform"
_RAW = _CKPT / "monthly_raw.parquet"
_FEATURES = _CKPT / "monthly_features.parquet"
_PROVENANCE = _CKPT / "splice_provenance.json"
_SETTINGS = _ROOT / "config" / "platform_settings.yaml"

# Lean columns that compute_lean_features assigns as a bare passthrough of a
# monthly_raw column of the SAME name (transforms_monthly.py). For these,
# features[col] must equal raw[col] exactly -- any divergence means the two
# checkpoints were built from different frames.
_PASSTHROUGH_COLUMNS = ("gold", "oil", "fred_vix")


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    assert _RAW.is_file(), f"{_RAW} missing from the git-tracked platform namespace"
    return pd.read_parquet(_RAW)


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    assert _FEATURES.is_file(), f"{_FEATURES} missing from the git-tracked platform namespace"
    return pd.read_parquet(_FEATURES)


class TestPassthroughColumnsAgree:
    """A passthrough lean column must be identical to its raw source."""

    @pytest.mark.parametrize("col", _PASSTHROUGH_COLUMNS)
    def test_passthrough_column_matches_raw(self, col, raw, features):
        if col not in raw.columns or col not in features.columns:
            pytest.fail(f"{col!r} absent from raw ({col in raw.columns}) / features ({col in features.columns})")

        # features is dev-carved (<= holdout cutoff); compare on its own span.
        common = features.index.intersection(raw.index)
        assert len(common) > 0, "raw and features share no index — one was built from a different spine"

        lhs = features.loc[common, col]
        rhs = raw.loc[common, col]

        lhs_first, rhs_first = lhs.first_valid_index(), rhs.first_valid_index()
        assert lhs_first == rhs_first, (
            f"{col!r} first-valid disagrees: features={lhs_first}, raw={rhs_first}. "
            "monthly_features was derived from a different frame than the monthly_raw "
            "on disk — the merge-on-save trap this module documents."
        )
        assert int(lhs.notna().sum()) == int(rhs.notna().sum()), (
            f"{col!r} non-null count disagrees: features={int(lhs.notna().sum())}, "
            f"raw={int(rhs.notna().sum())}"
        )
        pd.testing.assert_series_equal(lhs, rhs, check_names=False)


class TestSpliceProvenanceMatchesData:
    """The resolved splice source must actually explain the column on disk."""

    def test_oil_resolves_to_the_configured_primary(self):
        assert _PROVENANCE.is_file(), f"{_PROVENANCE} missing"
        resolved = (
            json.loads(_PROVENANCE.read_text())["provenance"]["oil"]["sources"]["source_col"]["resolved"]
        )
        configured = yaml.safe_load(_SETTINGS.read_text())["splice"]["oil"]["source_col"]
        assert resolved == configured[0], (
            f"oil resolved to {resolved!r} but the configured primary is {configured[0]!r}. "
            "Either the primary source failed to fetch (check the build log for a fallback "
            "WARNING) or the checkpoint predates a config change and needs a rebuild."
        )

    def test_oil_column_matches_its_resolved_source(self, raw):
        assert _PROVENANCE.is_file(), f"{_PROVENANCE} missing"
        resolved = (
            json.loads(_PROVENANCE.read_text())["provenance"]["oil"]["sources"]["source_col"]["resolved"]
        )
        assert resolved in raw.columns, f"resolved oil source {resolved!r} is not a monthly_raw column"
        assert raw["oil"].first_valid_index() == raw[resolved].first_valid_index(), (
            f"monthly_raw['oil'] starts {raw['oil'].first_valid_index()} but its resolved source "
            f"{resolved!r} starts {raw[resolved].first_valid_index()} — 'oil' is a merge-preserved "
            "leftover from an earlier build, not the output of the splice that last ran."
        )
