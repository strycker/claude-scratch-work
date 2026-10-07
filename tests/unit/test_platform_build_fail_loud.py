"""Fail-loud build (phase 08.4, DECISIONS D-08).

A lost source stops the build before anything is written, and a derived column
is never refilled from an old disk copy (the 221defc incident: gold_spot lost,
IAU spliced in from 2005-01, the old monthly_raw filling the pre-IAU months).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from test_platform_point_in_time import (
    _build,
    _pit_cfg,
    _synthetic_macro,
    _synthetic_prices,
    _synthetic_vintages,
)

from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.transforms_monthly import BuildFailed

# The synthetic world spans 1962-1976; this month stands in for IAU's real 2005-01 start.
IAU_START = pd.Timestamp("1970-01-31")


def _world(seed: int = 11):
    cfg = _pit_cfg()
    rng = np.random.default_rng(seed)
    return cfg, _synthetic_macro(cfg, rng), _synthetic_prices(cfg, rng), _synthetic_vintages(rng)


def _221defc_inputs(macro: pd.DataFrame, prices: pd.DataFrame):
    """gold_spot lost; IAU only from IAU_START and on a ~30 scale instead of gold's ~100."""
    iau = prices["IAU"].where(prices.index >= IAU_START) * 0.03
    return macro.drop(columns=["gold_spot"]), prices.assign(IAU=iau)


def test_the_live_config_is_fail_loud_with_no_allowed_gaps():
    build = load_platform_config()["build"]
    assert build["fail_loud"] is True
    assert build["allow_missing_sources"] == []


def test_a_lost_source_stops_the_build_before_any_write(tmp_path):
    cfg, macro, prices, vintages = _world()
    _build(cfg, macro, prices, vintages, tmp_path)  # the "old disk" (gate off, whole world)
    raw_path = tmp_path / "platform" / "monthly_raw.parquet"
    before = raw_path.read_bytes()

    cfg["build"] = {"fail_loud": True, "allow_missing_sources": []}
    lost_macro, lost_prices = _221defc_inputs(macro, prices)
    with pytest.raises(BuildFailed) as err:
        _build(cfg, lost_macro, lost_prices, vintages, tmp_path)

    text = str(err.value)
    assert "gold_spot" in text and "macrotrends_monthly" in text
    assert "scripts/build_platform_data.py" in text and "allow_missing_sources" in text
    assert "corporate VPNs" in text  # a macrotrends loss names the likely cause (08.4 UAT, 2026-10-07)
    assert raw_path.read_bytes() == before


def test_a_source_column_that_arrives_all_nan_counts_as_not_delivered(tmp_path):
    """A silent parse failure (the column is there, every value NaN) stops the build like a lost
    column does, also for a series no splice class watches (08.4 review)."""
    from trading_crab_lib.platform.transforms_monthly import expected_source_columns

    cfg, macro, prices, vintages = _world()
    cfg["build"] = {"fail_loud": True, "allow_missing_sources": []}
    column = next(c for c, s in expected_source_columns(cfg).items() if s == "fred_monthly" and c in macro.columns)
    with pytest.raises(BuildFailed) as err:
        _build(cfg, macro.assign(**{column: np.nan}), prices, vintages, tmp_path)
    assert f"source column '{column}' (fred_monthly) was not delivered" in str(err.value)


def test_an_allowed_missing_column_goes_on_with_a_warning_and_lets_its_class_fall_back(tmp_path, caplog):
    cfg, macro, prices, vintages = _world()
    cfg["build"] = {"fail_loud": True, "allow_missing_sources": ["gold_spot"]}
    lost_macro, lost_prices = _221defc_inputs(macro, prices)
    with caplog.at_level(logging.WARNING):
        _build(cfg, lost_macro, lost_prices, vintages, tmp_path)
    assert any("gold_spot" in r.getMessage() and "allows" in r.getMessage() for r in caplog.records)


def test_main_returns_1_when_the_build_fails(monkeypatch):
    import build_platform_data as script
    import dotenv

    import trading_crab_lib.platform.transforms_monthly as transforms_monthly

    def boom(cfg):
        raise BuildFailed("source column 'gold_spot' (macrotrends_monthly) was not delivered")

    monkeypatch.setenv("FRED_API_KEY", "x")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(transforms_monthly, "build_monthly_spine", boom)
    assert script.main() == 1


# ── merge-on-save: derived columns are replaced, raw columns still fill from disk ───


def test_replace_columns_take_the_new_value_while_raw_columns_still_fill_from_disk():
    from trading_crab_lib.checkpoints import merge_preserving

    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    old = pd.DataFrame({"a": [1.0, 2.0, 3.0], "d": [10.0, 20.0, 30.0], "only_disk_d": [7.0, 8.0, 9.0]}, index=idx)
    new = pd.DataFrame({"a": [np.nan, 5.0, np.nan], "d": [np.nan, 50.0, np.nan]}, index=idx)

    merged, stats = merge_preserving(old, new, replace_columns=["d", "only_disk_d"])

    assert merged["a"].tolist() == [1.0, 5.0, 3.0]  # raw: NaN cells still filled from disk
    assert merged["d"].isna().tolist() == [True, False, True]  # derived: new verbatim
    assert merged["d"].iloc[1] == 50.0
    assert "only_disk_d" not in merged.columns  # a derived column only on disk is dropped
    assert sorted(stats["cols_replaced"]) == ["d", "only_disk_d"]

    plain, _ = merge_preserving(old, new)
    assert plain["d"].tolist() == [10.0, 50.0, 30.0]  # without replace_columns: today's merge


def test_221defc_gold_spot_lost_leaves_no_pre_iau_gold_and_no_month_below_minus_half(tmp_path):
    """The 2026-10 incident: gold_spot lost, IAU spliced from 2005-01, the old monthly_raw
    refilling the pre-IAU months of the derived `gold` column (a -98% month)."""
    cfg, macro, prices, vintages = _world()
    _build(cfg, macro, prices, vintages, tmp_path)  # the old disk: gold from gold_spot, whole span

    cfg["build"] = {"fail_loud": True, "allow_missing_sources": ["gold_spot"]}
    lost_macro, lost_prices = _221defc_inputs(macro, prices)
    _build(cfg, lost_macro, lost_prices, vintages, tmp_path)

    gold = pd.read_parquet(tmp_path / "platform" / "monthly_raw.parquet")["gold"]
    assert gold.loc[gold.index < IAU_START].isna().all()
    assert gold.pct_change(fill_method=None).min() > -0.5


def test_network_hint_names_a_blocked_or_throttled_fetch():
    """08.4 UAT (2026-10-07): behind a corporate VPN, macrotrends answered 403 and Yahoo rate-limited.
    The fetch WARNING now says which, in plain words; any other failure gets no guess."""
    from trading_crab_lib.platform.ingestion.macro_monthly import network_hint

    class YFRateLimitError(Exception):
        pass

    assert "blocked (HTTP 403)" in network_hint(RuntimeError("HTTP Error 403: Forbidden"))
    assert "rate-limited" in network_hint(YFRateLimitError("Too Many Requests. Rate limited."))
    assert network_hint(ValueError("no table found")) == ""
