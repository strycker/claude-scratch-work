from __future__ import annotations

from pathlib import Path
import importlib.util
import types

import pandas as pd
import pytest

from market_regime import transforms as transforms_module
from market_regime.config import load


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


step01 = _load_step_module("01_ingest.py")
step02 = _load_step_module("02_features.py")


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


def test_step01_ingest_writes_macro_raw_without_network(monkeypatch, tmp_path, cfg) -> None:
    """
    Smoke test for pipelines/01_ingest.py.

    Network-dependent fetches are patched to return a tiny synthetic DataFrame.
    All I/O is redirected to tmp_path so no production checkpoints are touched.
    """
    from market_regime.ingestion import fred as fred_module
    from market_regime.ingestion import multpl as multpl_module

    synthetic = _make_synthetic_macro()

    monkeypatch.setattr(fred_module, "fetch_all", lambda _cfg: synthetic)
    monkeypatch.setattr(multpl_module, "fetch_all", lambda _cfg: pd.DataFrame(index=synthetic.index))
    monkeypatch.setattr(step01, "DATA_DIR", tmp_path)

    step01.main([])

    out_path = tmp_path / "raw" / "macro_raw.parquet"
    assert out_path.exists(), "01_ingest.main() did not write macro_raw.parquet"
    loaded = pd.read_parquet(out_path)
    pd.testing.assert_index_equal(loaded.index, synthetic.index)


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
