"""Tests for trading_crab_lib.__init__ — env var path overrides and convenience imports."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


class TestEnvVarPathOverrides:
    """C6.1 — TC_ROOT_DIR, TC_CONFIG_DIR, TC_DATA_DIR, TC_OUTPUT_DIR."""

    def test_default_paths_relative_to_package(self):
        import trading_crab_lib

        # Default ROOT should be 3 levels up from __init__.py
        expected_root = Path(trading_crab_lib.__file__).parent.parent.parent
        assert trading_crab_lib._DEFAULT_ROOT == expected_root

    def test_tc_root_dir_overrides_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TC_ROOT_DIR", str(tmp_path))
        # Must reload to pick up env var
        import trading_crab_lib

        importlib.reload(trading_crab_lib)
        try:
            assert trading_crab_lib.ROOT == tmp_path
        finally:
            monkeypatch.delenv("TC_ROOT_DIR")
            importlib.reload(trading_crab_lib)

    def test_tc_config_dir_overrides_config(self, monkeypatch, tmp_path):
        custom = tmp_path / "my_config"
        monkeypatch.setenv("TC_CONFIG_DIR", str(custom))
        import trading_crab_lib

        importlib.reload(trading_crab_lib)
        try:
            assert trading_crab_lib.CONFIG_DIR == custom
        finally:
            monkeypatch.delenv("TC_CONFIG_DIR")
            importlib.reload(trading_crab_lib)

    def test_tc_data_dir_overrides_data(self, monkeypatch, tmp_path):
        custom = tmp_path / "my_data"
        monkeypatch.setenv("TC_DATA_DIR", str(custom))
        import trading_crab_lib

        importlib.reload(trading_crab_lib)
        try:
            assert trading_crab_lib.DATA_DIR == custom
        finally:
            monkeypatch.delenv("TC_DATA_DIR")
            importlib.reload(trading_crab_lib)

    def test_tc_output_dir_overrides_output(self, monkeypatch, tmp_path):
        custom = tmp_path / "my_outputs"
        monkeypatch.setenv("TC_OUTPUT_DIR", str(custom))
        import trading_crab_lib

        importlib.reload(trading_crab_lib)
        try:
            assert trading_crab_lib.OUTPUT_DIR == custom
        finally:
            monkeypatch.delenv("TC_OUTPUT_DIR")
            importlib.reload(trading_crab_lib)

    def test_root_override_does_not_affect_explicit_config_dir(self, monkeypatch, tmp_path):
        """TC_CONFIG_DIR takes precedence even when TC_ROOT_DIR is also set."""
        custom_root = tmp_path / "root"
        custom_cfg = tmp_path / "cfg"
        monkeypatch.setenv("TC_ROOT_DIR", str(custom_root))
        monkeypatch.setenv("TC_CONFIG_DIR", str(custom_cfg))
        import trading_crab_lib

        importlib.reload(trading_crab_lib)
        try:
            assert trading_crab_lib.ROOT == custom_root
            assert trading_crab_lib.CONFIG_DIR == custom_cfg  # not custom_root/config
        finally:
            monkeypatch.delenv("TC_ROOT_DIR")
            monkeypatch.delenv("TC_CONFIG_DIR")
            importlib.reload(trading_crab_lib)

    def test_root_override_cascades_to_unset_dirs(self, monkeypatch, tmp_path):
        """When only TC_ROOT_DIR is set, DATA_DIR and OUTPUT_DIR derive from it."""
        monkeypatch.setenv("TC_ROOT_DIR", str(tmp_path))
        import trading_crab_lib

        importlib.reload(trading_crab_lib)
        try:
            assert trading_crab_lib.DATA_DIR == tmp_path / "data"
            assert trading_crab_lib.OUTPUT_DIR == tmp_path / "outputs"
        finally:
            monkeypatch.delenv("TC_ROOT_DIR")
            importlib.reload(trading_crab_lib)


class TestResolveDir:
    """Unit tests for the _resolve_dir helper."""

    def test_returns_default_when_env_unset(self):
        from trading_crab_lib import _resolve_dir

        default = Path("/some/default")
        assert _resolve_dir("NONEXISTENT_VAR_12345", default) == default

    def test_returns_env_path_when_set(self, monkeypatch):
        from trading_crab_lib import _resolve_dir

        monkeypatch.setenv("TEST_DIR_OVERRIDE", "/custom/path")
        assert _resolve_dir("TEST_DIR_OVERRIDE", Path("/fallback")) == Path("/custom/path")


class TestConvenienceImports:
    """C6.2 — load(), load_portfolio(), RunConfig, CheckpointManager accessible from package root."""

    def test_load_callable(self):
        import trading_crab_lib

        assert callable(trading_crab_lib.load)

    def test_load_portfolio_callable(self):
        import trading_crab_lib

        assert callable(trading_crab_lib.load_portfolio)

    def test_runconfig_accessible(self):
        import trading_crab_lib

        RC = trading_crab_lib.RunConfig
        assert RC is not None
        # Verify it's the actual dataclass
        rc = RC()
        assert hasattr(rc, "verbose")
        assert hasattr(rc, "generate_plots")

    @pytest.mark.skipif(
        not importlib.util.find_spec("joblib"),
        reason="joblib not installed",
    )
    def test_checkpoint_manager_accessible(self):
        import trading_crab_lib

        CM = trading_crab_lib.CheckpointManager
        assert CM is not None
        assert hasattr(CM, "save")
        assert hasattr(CM, "load")

    def test_invalid_attr_raises(self):
        import trading_crab_lib

        with pytest.raises(AttributeError, match="no attribute"):
            _ = trading_crab_lib.nonexistent_thing

    def test_load_returns_dict(self):
        import trading_crab_lib

        cfg = trading_crab_lib.load()
        assert isinstance(cfg, dict)
        assert "data" in cfg or "features" in cfg
