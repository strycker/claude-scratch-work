"""Shared fixtures for all tests."""

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Checkpoint isolation
# ---------------------------------------------------------------------------
# All checkpoint I/O is redirected to a session-scoped temporary directory so
# that pytest NEVER reads from or writes to the production data/checkpoints/
# directory.  This prevents:
#   • tests corrupting checkpoints that a production pipeline run relies on
#   • non-deterministic behaviour when pytest is run between pipeline runs
#
# Strategy:
#   1. At session start, copy every file from data/checkpoints/ (if it exists)
#      into a fresh tmp dir.  Tests that need a real checkpoint as a read-only
#      fixture will find it there ("read fallback").
#   2. Patch trading_crab_lib.checkpoints.CHECKPOINT_DIR to point to the tmp
#      dir so that every CheckpointManager() instantiated during the session
#      uses it.  Because __init__ reads the module-level variable at call time
#      (not class-definition time) the patch takes effect for all new instances.
#   3. Set TC_CHECKPOINT_DIR env var so any subprocess or late-import also
#      picks up the override.
#   4. If the asset_prices checkpoint was not copied from production (e.g. both
#      data/raw/ and data/checkpoints/ were cleared, or yfinance is unavailable),
#      synthesise a minimal one from the configured ETF tickers so that
#      structural constraint tests always run rather than skip.
#   5. Restore everything on session teardown.
# ---------------------------------------------------------------------------


def _synthesize_asset_prices(session_dir: Path) -> None:
    """Write a minimal synthetic asset_prices checkpoint to *session_dir*.

    Uses real ETF tickers from the project config so the column-universe
    constraint is exercised against the actual configured list.
    The DatetimeIndex is quarterly so the no-intraday constraint holds too.
    Only called when no production copy exists — real data always takes priority.
    """
    import trading_crab_lib.checkpoints as ckpt_mod

    try:
        from trading_crab_lib.config import load as _load_cfg
        cfg = _load_cfg()
        tickers = [str(t).upper() for t in cfg["assets"]["etfs"]][:8]
    except Exception:
        tickers = ["SPY", "GLD", "TLT", "QQQ", "IWM", "VNQ", "AGG", "USO"]

    index = pd.date_range("2000-03-31", periods=40, freq="QE")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        {ticker: rng.uniform(50, 400, len(index)) for ticker in tickers},
        index=index,
    )
    # Save via CheckpointManager so the manifest is also written
    cm = ckpt_mod.CheckpointManager(checkpoint_dir=session_dir)
    cm.save(prices, "asset_prices")


@pytest.fixture(autouse=True, scope="session")
def _isolated_checkpoint_dir(tmp_path_factory: pytest.TempPathFactory):
    """Route all checkpoint I/O to a session-scoped tmp dir.

    Production data/checkpoints/ is never written to during pytest.
    Structural constraint tests (asset_prices columns/frequency) always run —
    never skip — because a minimal synthetic checkpoint is generated when no
    production copy is available (e.g. yfinance absent or data dirs cleared).
    """
    import trading_crab_lib.checkpoints as ckpt_mod

    session_dir = tmp_path_factory.mktemp("checkpoints", numbered=False)

    # Copy production checkpoints into session dir so read-based tests work
    prod_dir = ckpt_mod.CHECKPOINT_DIR
    if prod_dir.exists():
        for src in prod_dir.iterdir():
            shutil.copy2(src, session_dir / src.name)

    # Redirect all CheckpointManager() instances to session_dir
    original_checkpoint_dir = ckpt_mod.CHECKPOINT_DIR
    ckpt_mod.CHECKPOINT_DIR = session_dir
    os.environ["TC_CHECKPOINT_DIR"] = str(session_dir)

    # Synthesise asset_prices if absent (no production data or data dirs cleared).
    # Real production data always takes priority — this is only a safety net.
    if not (session_dir / "asset_prices.parquet").exists():
        _synthesize_asset_prices(session_dir)

    yield session_dir

    # Restore — production directory is never touched
    ckpt_mod.CHECKPOINT_DIR = original_checkpoint_dir
    os.environ.pop("TC_CHECKPOINT_DIR", None)


@pytest.fixture
def quarterly_index():
    """20 quarter-end dates starting 2000-Q1."""
    return pd.date_range("2000-03-31", periods=20, freq="QE")


@pytest.fixture
def raw_macro_df(quarterly_index):
    """Minimal macro DataFrame with all columns needed by add_cross_ratios."""
    rng = np.random.default_rng(0)
    n = len(quarterly_index)
    return pd.DataFrame(
        {
            "sp500":     rng.uniform(800, 4000, n),
            "sp500_adj": rng.uniform(800, 4000, n),
            "dividend":  rng.uniform(10, 60, n),
            "div_yield": rng.uniform(0.01, 0.05, n),
            "gdp":       rng.uniform(8000, 22000, n),
            "cpi":       rng.uniform(150, 280, n),
            "fred_gdp":  rng.uniform(8000, 22000, n),
            "fred_gnp":  rng.uniform(7500, 21000, n),
            "fred_baa":  rng.uniform(3.0, 9.0, n),
            "fred_aaa":  rng.uniform(2.5, 8.0, n),
            "fred_cpi":  rng.uniform(150, 280, n),
        },
        index=quarterly_index,
    )


@pytest.fixture
def cluster_labels(quarterly_index):
    """Integer cluster labels cycling 0–4."""
    return pd.Series(
        np.tile(np.arange(5), len(quarterly_index) // 5 + 1)[: len(quarterly_index)],
        index=quarterly_index,
        name="balanced_cluster",
    )


@pytest.fixture
def asset_prices(quarterly_index):
    """Synthetic quarterly asset prices for 3 tickers."""
    rng = np.random.default_rng(1)
    n = len(quarterly_index)
    return pd.DataFrame(
        {
            "SPY": rng.uniform(80, 400, n),
            "GLD": rng.uniform(50, 200, n),
            "TLT": rng.uniform(70, 150, n),
        },
        index=quarterly_index,
    )
