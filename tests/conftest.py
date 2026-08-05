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


def _synthesize_macro_raw(session_dir: Path) -> None:
    """Write a minimal synthetic macro_raw checkpoint to *session_dir*.

    Mirrors the real ingestion output structure so frequency/cadence
    constraint tests always run rather than skip.
    """
    import trading_crab_lib.checkpoints as ckpt_mod

    rng = np.random.default_rng(0)
    n = 100
    idx = pd.date_range("2000-03-31", periods=n, freq="QE")
    sp500 = np.abs(rng.uniform(300, 5000, n)) + 200
    data: dict[str, np.ndarray] = {
        "sp500":        sp500,
        "sp500_adj":    sp500 * rng.uniform(0.95, 1.05, n),
        "dividend":     rng.uniform(10, 80, n),
        "div_yield":    rng.uniform(0.01, 0.06, n),
        "earnings":     rng.uniform(20, 200, n),
        "cape_shiller": rng.uniform(8, 40, n),
        "gdp":          rng.uniform(5000, 25000, n),
        "cpi":          rng.uniform(100, 320, n),
        "10yr_ustreas":  rng.uniform(1.0, 16.0, n),
        "us_cpi":       rng.uniform(100, 320, n),
        "book_value":   rng.uniform(100, 800, n),
        "price_sales":  rng.uniform(0.5, 5.0, n),
        "price_book":   rng.uniform(1.0, 6.0, n),
        "fred_gdp":     rng.uniform(5000, 25000, n),
        "fred_gnp":     rng.uniform(4800, 24000, n),
        "fred_baa":     rng.uniform(3.0, 12.0, n),
        "fred_aaa":     rng.uniform(2.5, 10.0, n),
        "fred_cpi":     rng.uniform(100, 320, n),
        "fred_gs10":    rng.uniform(1.0, 16.0, n),
        "fred_tb3ms":   rng.uniform(0.01, 10.0, n),
        "fred_vix":     rng.uniform(10, 80, n),
        "fred_unrate":  rng.uniform(3.0, 15.0, n),
        "fred_m2sl":    rng.uniform(3000, 25000, n),
        "fred_gs2":     rng.uniform(0.01, 10.0, n),
        "market_code":  np.zeros(n, dtype=float),
    }
    df = pd.DataFrame(data, index=idx)
    df.index.name = "date"
    cm = ckpt_mod.CheckpointManager(checkpoint_dir=session_dir)
    cm.save(df, "macro_raw")


def _synthesize_features(session_dir: Path) -> None:
    """Write synthetic features_noncausal and features_causal checkpoints.

    Runs engineer_all() on synthetic macro data so the feature structure
    exactly matches what the real pipeline produces.  Falls back to a
    simple quarterly DataFrame if engineer_all() fails (missing config/deps).
    """
    import trading_crab_lib.checkpoints as ckpt_mod

    cm = ckpt_mod.CheckpointManager(checkpoint_dir=session_dir)

    # Try to load macro_raw from the session dir; synthesise if absent.
    try:
        macro = cm.load("macro_raw")
    except FileNotFoundError:
        macro = None

    if macro is None or macro.empty:
        _synthesize_macro_raw(session_dir)
        try:
            macro = cm.load("macro_raw")
        except FileNotFoundError:
            macro = pd.DataFrame()

    try:
        from trading_crab_lib.config import load as _load_cfg
        from trading_crab_lib.transforms import engineer_all
        cfg = _load_cfg()
        features_centered = engineer_all(macro.copy(), cfg, causal=False)
        features_causal   = engineer_all(macro.copy(), cfg, causal=True)
        cm.save(features_centered, "features_noncausal")
        cm.save(features_causal,   "features_causal")
        return
    except Exception:
        pass

    # Fallback: minimal quarterly DataFrame (structure only, no real values)
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-03-31", periods=40, freq="QE")
    placeholder = pd.DataFrame(
        {"feature_a": rng.standard_normal(40), "feature_b": rng.standard_normal(40)},
        index=idx,
    )
    placeholder.index.name = "date"
    cm.save(placeholder, "features_noncausal")
    cm.save(placeholder.copy(), "features_causal")


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

    rng = np.random.default_rng(42)
    index = pd.date_range("2000-03-31", periods=40, freq="QE")
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
            dst = session_dir / src.name
            # Handle subdirectories (e.g. the platform/ namespace created by the
            # platform data build) recursively — a bare copy2 on a directory
            # raises IsADirectoryError and breaks the whole session.
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)

    # Redirect all CheckpointManager() instances to session_dir
    original_checkpoint_dir = ckpt_mod.CHECKPOINT_DIR
    ckpt_mod.CHECKPOINT_DIR = session_dir
    os.environ["TC_CHECKPOINT_DIR"] = str(session_dir)

    # Ensure all checkpoint-dependent constraint tests run rather than skip.
    # When data/raw/ and data/checkpoints/ are cleared, synthesise minimal
    # stand-ins with the correct structure.  Real production data always wins —
    # synthesis only fires when a file is absent.
    if not (session_dir / "macro_raw.parquet").exists():
        _synthesize_macro_raw(session_dir)
    if not (session_dir / "features_noncausal.parquet").exists() or \
            not (session_dir / "features_causal.parquet").exists():
        _synthesize_features(session_dir)
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


# ── no test may launch a real browser ───────────────────────────────────────


@pytest.fixture(autouse=True)
def _forbid_real_browser_launch(monkeypatch, request):
    """Fail loudly if any test actually starts a browser.

    Unit tests mock both engines, but a test that patches only ONE engine's
    availability flag silently falls through to the other and launches a real
    Chrome — slow, network-dependent, and green for the wrong reason. That
    happened once: an engine-unavailable test patched only the Playwright flag
    and passed solely because selenium was uninstalled at the time; installing
    selenium turned it into a live browser launch.

    Tests that genuinely need a real browser can opt out with
    @pytest.mark.real_browser.
    """
    if request.node.get_closest_marker("real_browser"):
        return

    def _blocked(*_args, **_kwargs):
        raise AssertionError(
            "A test tried to launch a REAL browser. Mock the engine, or patch BOTH "
            "_PLAYWRIGHT_AVAILABLE and _SELENIUM_AVAILABLE when asserting no engine is available. "
            "Opt out deliberately with @pytest.mark.real_browser."
        )

    try:
        from selenium import webdriver

        monkeypatch.setattr(webdriver, "Chrome", _blocked, raising=False)
    except ImportError:
        pass
