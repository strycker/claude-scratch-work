from __future__ import annotations

from pathlib import Path
import importlib.util
import types

import pandas as pd
import pytest

from trading_crab_lib import transforms as transforms_module
from trading_crab_lib.config import load


def _load_step_module(script_name: str) -> types.ModuleType:
    """
    Load a step script from the top-level `pipelines/` directory as a module.

    This avoids relying on non-standard module names like `pipelines01_ingest`
    while still giving tests a handle to call `main()`.
    """
    root = Path(__file__).parent.parent
    script_path = root / "pipelines" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load step module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


# Guard against missing optional dependencies (fredapi, yfinance, etc.).
# When ingestion deps are absent the step modules cannot be imported at
# collection time, which would cause an ERROR rather than a SKIP.
# Tests that require step01/step02 are marked accordingly below.
_INGESTION_DEPS_REASON = ""
try:
    step01 = _load_step_module("01_ingest.py")
    step02 = _load_step_module("02_features.py")
    _INGESTION_DEPS_AVAILABLE = True
except ImportError as _import_err:
    step01 = step02 = None  # type: ignore[assignment]
    _INGESTION_DEPS_AVAILABLE = False
    _INGESTION_DEPS_REASON = str(_import_err)

_requires_ingestion = pytest.mark.skipif(
    not _INGESTION_DEPS_AVAILABLE,
    reason=_INGESTION_DEPS_REASON if not _INGESTION_DEPS_AVAILABLE else "",
)


def _make_synthetic_macro() -> pd.DataFrame:
    dates = pd.date_range("2000-03-31", periods=4, freq="QE")
    return pd.DataFrame(
        {
            "fred_gdp": [1000.0, 1010.0, 1020.0, 1030.0],
            "fred_cpi": [200.0, 201.0, 202.0, 203.0],
        },
        index=dates,
    )


@pytest.fixture
def cfg():
    return load()


@_requires_ingestion
def test_step01_ingest_writes_macro_raw_without_network(monkeypatch, tmp_path, cfg) -> None:
    """
    Smoke test for pipelines/01_ingest.py.

    Network-dependent fetches are patched to return a tiny synthetic DataFrame.
    All I/O is redirected to tmp_path so no production checkpoints are touched.
    """
    from trading_crab_lib.ingestion import fred as fred_module
    from trading_crab_lib.ingestion import multpl as multpl_module

    synthetic = _make_synthetic_macro()

    monkeypatch.setattr(fred_module, "fetch_all", lambda _cfg: synthetic)
    monkeypatch.setattr(multpl_module, "fetch_all", lambda _cfg: pd.DataFrame(index=synthetic.index))
    monkeypatch.setattr(step01, "DATA_DIR", tmp_path)

    step01.main([])

    out_path = tmp_path / "raw" / "macro_raw.parquet"
    assert out_path.exists(), "01_ingest.main() did not write macro_raw.parquet"
    loaded = pd.read_parquet(out_path)
    pd.testing.assert_index_equal(loaded.index, synthetic.index)


@_requires_ingestion
def test_step02_features_writes_feature_artifacts_without_network(monkeypatch, tmp_path, cfg) -> None:
    """
    Smoke test for pipelines/02_features.py.

    Patches engineer_all to return a tiny dummy DataFrame and redirects all I/O
    to tmp_path so no production data files are touched.
    """
    # Provide the raw input the step needs inside tmp_path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    _make_synthetic_macro().to_parquet(raw_dir / "macro_raw.parquet")

    dummy_features = pd.DataFrame(
        {"feature1": [1.0, 2.0, 3.0, 4.0]},
        index=pd.date_range("2000-03-31", periods=4, freq="QE"),
    )

    def fake_engineer_all(raw, _cfg, causal: bool):
        return dummy_features

    monkeypatch.setattr(transforms_module, "engineer_all", fake_engineer_all)
    monkeypatch.setattr(step02, "DATA_DIR", tmp_path)

    step02.main()

    features_path = tmp_path / "processed" / "features.parquet"
    features_sup_path = tmp_path / "processed" / "features_supervised.parquet"

    assert features_path.exists(), "02_features.main() did not write features.parquet"
    assert features_sup_path.exists(), "02_features.main() did not write features_supervised.parquet"

    assert not pd.read_parquet(features_path).empty
    assert not pd.read_parquet(features_sup_path).empty


def test_fetch_and_cache_asset_prices_writes_checkpoint_from_raw_cache(
    monkeypatch, tmp_path, cfg
) -> None:
    """
    Regression test: _fetch_and_cache_asset_prices must write the asset_prices
    checkpoint even when loading from the raw-parquet cache (i.e. when
    refresh_asset_prices=False and data/raw/asset_prices.parquet already exists).

    Previously the checkpoint was only written inside the fresh-fetch branch,
    so a second pipeline run (without --refresh-assets) left the checkpoint
    absent, causing the constraint tests to skip.

    Note: CheckpointManager.CHECKPOINT_DIR is a module-level constant, so the
    save lands in the real data/checkpoints/ directory (not tmp_path).  We verify
    this by checking cm.load() succeeds after the call.
    """
    import trading_crab.pipeline as pipeline_module
    from trading_crab_lib.checkpoints import CheckpointManager
    from trading_crab_lib.runtime import RunConfig
    from unittest.mock import patch

    # Arrange: pre-populate data/raw/asset_prices.parquet in tmp_path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    prices_df = pd.DataFrame(
        {"SPY": [100.0, 101.0], "GLD": [50.0, 51.0]},
        index=pd.date_range("2023-03-31", periods=2, freq="QE"),
    )
    prices_df.to_parquet(raw_dir / "asset_prices.parquet")

    # Redirect DATA_DIR so the function picks up the raw parquet from tmp_path
    monkeypatch.setattr(pipeline_module, "DATA_DIR", tmp_path)

    # Track whether cm.save() is called with "asset_prices"
    saved_names: list[str] = []
    original_save = CheckpointManager.save

    def recording_save(self, df, name, *args, **kwargs):
        saved_names.append(name)
        return original_save(self, df, name, *args, **kwargs)

    # Act: call without refresh — should load from raw cache and still checkpoint
    run_cfg = RunConfig(refresh_asset_prices=False)
    with patch.object(CheckpointManager, "save", recording_save):
        result = pipeline_module._fetch_and_cache_asset_prices(cfg, run_cfg)

    # Assert: function returned data and cm.save was called for asset_prices
    assert not result.empty, "Expected non-empty prices from raw-cache fallback"
    assert "asset_prices" in saved_names, (
        "asset_prices checkpoint was not written when loading from raw-parquet cache. "
        "This causes constraint tests to skip on every non-refresh run."
    )
