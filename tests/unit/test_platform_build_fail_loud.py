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
    assert raw_path.read_bytes() == before


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
