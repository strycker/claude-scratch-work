from __future__ import annotations

from typing import Iterable

import pandas as pd
import pytest

from market_regime.config import load
from market_regime.checkpoints import CheckpointManager


def _assert_columns_within_universe(
    columns: Iterable[str],
    universe: Iterable[str],
) -> None:
    universe_set = {ticker.upper() for ticker in universe}
    cols_set = {c.upper() for c in columns}
    extras = sorted(cols_set - universe_set)
    if extras:
        raise AssertionError(
            f"Found tickers outside configured ETF universe: {extras}"
        )


@pytest.fixture(scope="module")
def etf_universe() -> list[str]:
    cfg = load()
    etfs = cfg["assets"]["etfs"]
    # Normalise to upper-case strings
    return [str(t).upper() for t in etfs]


@pytest.fixture(scope="module")
def checkpoints() -> CheckpointManager:
    return CheckpointManager()


def test_asset_prices_columns_subset_of_etf_universe(
    etf_universe: list[str],
    checkpoints: CheckpointManager,
) -> None:
    """All asset price tickers must come from the configured ETF universe."""
    try:
        prices = checkpoints.load("asset_prices")
    except FileNotFoundError:
        pytest.skip(
            "asset_prices checkpoint not found; run ingestion/assets pipeline "
            "to materialise ETF price checkpoints before enforcing constraints."
        )

    assert not prices.empty, "asset_prices checkpoint is empty"
    _assert_columns_within_universe(prices.columns, etf_universe)


def test_helper_rejects_out_of_universe_ticker(etf_universe: list[str]) -> None:
    """Negative case: helper should fail when an unknown ticker appears."""
    known = etf_universe[0] if etf_universe else "SPY"
    df = pd.DataFrame(
        {
            known: [100.0, 101.0],
            "NOT_AN_ETF": [10.0, 11.0],
        }
    )

    with pytest.raises(AssertionError):
        _assert_columns_within_universe(df.columns, etf_universe)

