"""Unit tests for trading_crab_lib.platform.honesty.gating (HON-06).

Follows the incumbent tests/unit/test_platform_taxonomy.py structure: a
class per behavior, docstring per test. Verifies the RESOLVED Open Question 1
guard shape (02-PATTERNS.md): monthly_features is causal-by-construction, so
gating.py is an assertion guard — not a two-file selector like the
incumbent's features.parquet / features_supervised.parquet split.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from trading_crab_lib.platform.honesty.gating import (
    assert_causal_features,
    select_platform_feature_path,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _clean_monthly_features_df() -> pd.DataFrame:
    """An all-causal monthly_features frame (no forbidden centered-window columns)."""
    index = pd.date_range("2020-01-31", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "fred_vix": range(6),
            "trailing_return_3m": [0.1] * 6,
            "realized_vol_3m": [0.02] * 6,
        },
        index=index,
    )


def _noncausal_monthly_features_df() -> pd.DataFrame:
    """A monthly_features frame with an injected forbidden centered-window column."""
    df = _clean_monthly_features_df()
    df["credit_spread_centered"] = 0.5
    return df


# ── assert_causal_features ──────────────────────────────────────────────────────


class TestAssertCausalFeatures:
    def test_clean_columns_return_false_no_raise(self):
        """No forbidden suffix present -> returns False, does not raise."""
        assert assert_causal_features(_clean_monthly_features_df().columns) is False

    def test_forbidden_column_blocked_by_default(self):
        """A '_centered' column raises with the offending column and 'allow_noncausal' named."""
        with pytest.raises(ValueError) as exc_info:
            assert_causal_features(_noncausal_monthly_features_df().columns)
        msg = str(exc_info.value)
        assert "credit_spread_centered" in msg
        assert "allow_noncausal" in msg

    def test_forbidden_column_opt_out_warns_and_returns_true(self, caplog):
        """allow_noncausal=True returns True and logs a NONCAUSAL_USED WARNING."""
        with caplog.at_level(logging.WARNING, logger="trading_crab_lib.platform.honesty.gating"):
            result = assert_causal_features(_noncausal_monthly_features_df().columns, allow_noncausal=True)
        assert result is True
        assert any("NONCAUSAL_USED" in r.message for r in caplog.records)


# ── select_platform_feature_path ────────────────────────────────────────────────


class TestSelectPlatformFeaturePath:
    def test_clean_features_pass(self, tmp_path):
        """An all-causal monthly_features checkpoint returns (<path>, "monthly_features", False)."""
        parquet_path = tmp_path / "monthly_features.parquet"
        _clean_monthly_features_df().to_parquet(parquet_path)

        path, source, noncausal_used = select_platform_feature_path(checkpoint_dir=tmp_path)

        assert path == parquet_path
        assert source == "monthly_features"
        assert noncausal_used is False

    def test_noncausal_blocked_by_default(self, tmp_path):
        """A monthly_features checkpoint with a forbidden column raises, naming it and allow_noncausal."""
        _noncausal_monthly_features_df().to_parquet(tmp_path / "monthly_features.parquet")

        with pytest.raises(ValueError) as exc_info:
            select_platform_feature_path(checkpoint_dir=tmp_path)
        msg = str(exc_info.value)
        assert "credit_spread_centered" in msg
        assert "allow_noncausal" in msg

    def test_noncausal_opt_out_warns(self, tmp_path, caplog):
        """allow_noncausal=True returns noncausal_used=True and logs a WARNING."""
        _noncausal_monthly_features_df().to_parquet(tmp_path / "monthly_features.parquet")

        with caplog.at_level(logging.WARNING, logger="trading_crab_lib.platform.honesty.gating"):
            path, source, noncausal_used = select_platform_feature_path(
                checkpoint_dir=tmp_path, allow_noncausal=True
            )
        assert noncausal_used is True
        assert any("NONCAUSAL_USED" in r.message for r in caplog.records)

    def test_missing_checkpoint_raises_file_not_found(self, tmp_path):
        """No monthly_features.parquet present at all -> FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            select_platform_feature_path(checkpoint_dir=tmp_path)
