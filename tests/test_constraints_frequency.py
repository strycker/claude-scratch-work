from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from trading_crab_lib.checkpoints import CheckpointManager


@pytest.fixture(scope="module")
def checkpoints() -> CheckpointManager:
    return CheckpointManager()


def _require_checkpoint(name: str, cm: CheckpointManager) -> pd.DataFrame:
    try:
        return cm.load(name)
    except FileNotFoundError:
        pytest.skip(
            f"{name} checkpoint not found; run the corresponding pipeline steps to "
            f"materialise data/checkpoints/{name}.parquet before enforcing cadence constraints."
        )


def _assert_quarterly_index(index: pd.Index) -> None:
    if isinstance(index, pd.PeriodIndex):
        base = index.asfreq("Q")
        if not base.equals(index.asfreq("Q")):
            raise AssertionError("Expected quarterly PeriodIndex for macro data")
        return

    if not isinstance(index, pd.DatetimeIndex):
        raise AssertionError(f"Expected DatetimeIndex/PeriodIndex, got {type(index)!r}")

    freq = pd.infer_freq(index)
    if not (freq and freq.upper().startswith("Q")):
        raise AssertionError(f"Expected quarterly frequency, inferred {freq!r}")


def _assert_no_intraday(index: pd.Index) -> None:
    if isinstance(index, pd.DatetimeIndex):
        times = index.time
        # Allow midnight or quarter/month-end timestamps; disallow intra-day times
        has_intraday = any(t not in (dt.time(0, 0),) for t in times)
        if has_intraday:
            raise AssertionError("Found intraday timestamps in core artifacts")


def test_macro_raw_is_quarterly(checkpoints: CheckpointManager) -> None:
    macro = _require_checkpoint("macro_raw", checkpoints)
    assert not macro.empty, "macro_raw checkpoint is empty"
    _assert_quarterly_index(macro.index)


def test_asset_prices_are_not_intraday(checkpoints: CheckpointManager) -> None:
    prices = _require_checkpoint("asset_prices", checkpoints)
    assert not prices.empty, "asset_prices checkpoint is empty"
    _assert_no_intraday(prices.index)


@pytest.mark.parametrize("name", ["features_noncausal", "features_causal"])
def test_feature_artifacts_are_quarterly(name: str, checkpoints: CheckpointManager) -> None:
    features = _require_checkpoint(name, checkpoints)
    assert not features.empty, f"{name} checkpoint is empty"
    _assert_quarterly_index(features.index)
    _assert_no_intraday(features.index)

