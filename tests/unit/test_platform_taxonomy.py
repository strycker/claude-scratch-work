"""Unit tests for trading_crab_lib.platform.taxonomy (DATA-04).

Follows the incumbent tests/unit/test_config.py structure: small in-memory
cfg dicts for behavior tests, plus one test against the real
config/platform_settings.yaml to guarantee the DATA-04 guarantee holds for
the actual declarative config the downstream plans consume.
"""

from __future__ import annotations

import pytest

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.taxonomy import (
    check_columns_tagged,
    classify_feature,
    lean_feature_set,
    validate_taxonomy,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _well_formed_cfg() -> dict:
    """A taxonomy where every feature appears in exactly one tier."""
    return {
        "taxonomy": {
            "fast": ["curve_10y3m", "fred_vix"],
            "slow": ["cape_shiller", "div_yield"],
            "agency": ["fred_unrate", "fred_gdp"],
        }
    }


def _double_tagged_cfg() -> dict:
    """A taxonomy where 'fred_vix' appears in both fast and slow."""
    return {
        "taxonomy": {
            "fast": ["curve_10y3m", "fred_vix"],
            "slow": ["cape_shiller", "fred_vix"],
            "agency": ["fred_unrate"],
        }
    }


# ── validate_taxonomy ────────────────────────────────────────────────────────────


class TestValidateTaxonomy:
    def test_double_tagged_feature_raises_with_feature_and_tiers_named(self):
        """A feature listed in two tiers raises ValueError naming the feature and both tiers."""
        with pytest.raises(ValueError) as exc_info:
            validate_taxonomy(_double_tagged_cfg())
        msg = str(exc_info.value)
        assert "fred_vix" in msg
        assert "fast" in msg
        assert "slow" in msg

    def test_well_formed_taxonomy_does_not_raise(self):
        """Each feature in exactly one tier → returns None, no raise."""
        assert validate_taxonomy(_well_formed_cfg()) is None


# ── classify_feature ────────────────────────────────────────────────────────────


class TestClassifyFeature:
    def test_agency_feature_classified_agency(self):
        assert classify_feature("fred_unrate", _well_formed_cfg()) == "agency"

    def test_slow_feature_classified_slow(self):
        assert classify_feature("cape_shiller", _well_formed_cfg()) == "slow"

    def test_unknown_feature_returns_none(self):
        assert classify_feature("unknown_x", _well_formed_cfg()) is None


# ── lean_feature_set ────────────────────────────────────────────────────────────


class TestLeanFeatureSet:
    def test_lean_set_is_fast_union_slow(self):
        cfg = _well_formed_cfg()
        result = lean_feature_set(cfg)
        expected = set(cfg["taxonomy"]["fast"]) | set(cfg["taxonomy"]["slow"])
        assert result == expected

    def test_lean_set_excludes_agency_only_features(self):
        cfg = _well_formed_cfg()
        result = lean_feature_set(cfg)
        for feature in cfg["taxonomy"]["agency"]:
            assert feature not in result


# ── check_columns_tagged ────────────────────────────────────────────────────────────


class TestCheckColumnsTagged:
    def test_all_tagged_returns_empty_list(self):
        cfg = _well_formed_cfg()
        columns = ["curve_10y3m", "fred_vix", "cape_shiller", "div_yield", "fred_unrate", "fred_gdp"]
        assert check_columns_tagged(columns, cfg) == []

    def test_untagged_columns_returned(self):
        cfg = _well_formed_cfg()
        columns = ["curve_10y3m", "mystery_col", "another_mystery"]
        result = check_columns_tagged(columns, cfg)
        assert set(result) == {"mystery_col", "another_mystery"}


# ── Real config integration ────────────────────────────────────────────────────────────


class TestRealPlatformConfig:
    def test_real_config_passes_validation_and_has_nonempty_lean_set(self):
        """The actual config/platform_settings.yaml has zero double-tagged features."""
        cfg = load_platform_config()
        assert validate_taxonomy(cfg) is None
        lean = lean_feature_set(cfg)
        assert len(lean) > 0
