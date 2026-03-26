"""Tests for src/trading_crab_lib/runtime.py — RunConfig dataclass."""
from __future__ import annotations

import logging
from argparse import Namespace

import pytest

from trading_crab_lib.runtime import RunConfig


class TestRunConfigDefaults:
    def test_all_defaults_are_false_or_none(self):
        cfg = RunConfig()
        assert cfg.verbose is False
        assert cfg.generate_plots is False
        assert cfg.generate_pairplot is False
        assert cfg.generate_scatter_matrix is False
        assert cfg.show_plots is False
        assert cfg.refresh_source_datasets is False
        assert cfg.recompute_derived_datasets is False
        assert cfg.refresh_asset_prices is False
        assert cfg.refresh_preservation_checkpoints is False
        assert cfg.market_code_source is None

    def test_save_plots_defaults_true(self):
        cfg = RunConfig()
        assert cfg.save_plots is True

    def test_use_constrained_kmeans_defaults_true(self):
        cfg = RunConfig()
        assert cfg.use_constrained_kmeans is True

    def test_drop_incomplete_tail_defaults_true(self):
        cfg = RunConfig()
        assert cfg.drop_incomplete_tail is True


class TestRunConfigFromArgs:
    def test_minimal_namespace(self):
        args = Namespace()
        cfg = RunConfig.from_args(args)
        assert cfg.verbose is False
        assert cfg.generate_plots is False

    def test_verbose_flag(self):
        args = Namespace(verbose=True)
        cfg = RunConfig.from_args(args)
        assert cfg.verbose is True

    def test_plots_flag(self):
        args = Namespace(plots=True)
        cfg = RunConfig.from_args(args)
        assert cfg.generate_plots is True

    def test_refresh_flag(self):
        args = Namespace(refresh=True)
        cfg = RunConfig.from_args(args)
        assert cfg.refresh_source_datasets is True

    def test_recompute_flag(self):
        args = Namespace(recompute=True)
        cfg = RunConfig.from_args(args)
        assert cfg.recompute_derived_datasets is True

    def test_no_constrained_flag(self):
        args = Namespace(no_constrained=True)
        cfg = RunConfig.from_args(args)
        assert cfg.use_constrained_kmeans is False

    def test_market_code_source(self):
        args = Namespace(market_code="grok")
        cfg = RunConfig.from_args(args)
        assert cfg.market_code_source == "grok"

    def test_no_save_plots(self):
        args = Namespace(no_save_plots=True)
        cfg = RunConfig.from_args(args)
        assert cfg.save_plots is False

    def test_show_plots(self):
        args = Namespace(show_plots=True)
        cfg = RunConfig.from_args(args)
        assert cfg.show_plots is True

    def test_no_drop_tail(self):
        args = Namespace(no_drop_tail=True)
        cfg = RunConfig.from_args(args)
        assert cfg.drop_incomplete_tail is False

    def test_pairplot_flag(self):
        args = Namespace(pairplot=True)
        cfg = RunConfig.from_args(args)
        assert cfg.generate_pairplot is True

    def test_scatter_matrix_flag(self):
        args = Namespace(scatter_matrix=True)
        cfg = RunConfig.from_args(args)
        assert cfg.generate_scatter_matrix is True

    def test_refresh_assets_flag(self):
        args = Namespace(refresh_assets=True)
        cfg = RunConfig.from_args(args)
        assert cfg.refresh_asset_prices is True

    def test_refresh_preservation_flag(self):
        args = Namespace(refresh_preservation=True)
        cfg = RunConfig.from_args(args)
        assert cfg.refresh_preservation_checkpoints is True


class TestRunConfigApplyLogging:
    def test_verbose_sets_debug(self):
        cfg = RunConfig(verbose=True)
        cfg.apply_logging()
        assert logging.getLogger().level == logging.DEBUG
        # Reset
        logging.getLogger().setLevel(logging.WARNING)

    def test_non_verbose_does_not_change_level(self):
        original = logging.getLogger().level
        cfg = RunConfig(verbose=False)
        cfg.apply_logging()
        assert logging.getLogger().level == original


class TestRunConfigStr:
    def test_defaults_str(self):
        cfg = RunConfig()
        assert str(cfg) == "RunConfig(defaults)"

    def test_verbose_in_str(self):
        cfg = RunConfig(verbose=True)
        assert "verbose" in str(cfg)

    def test_plots_in_str(self):
        cfg = RunConfig(generate_plots=True)
        assert "plots" in str(cfg)

    def test_refresh_in_str(self):
        cfg = RunConfig(refresh_source_datasets=True)
        assert "refresh" in str(cfg)

    def test_market_code_in_str(self):
        cfg = RunConfig(market_code_source="grok")
        assert "market_code=grok" in str(cfg)

    def test_refresh_preservation_in_str(self):
        cfg = RunConfig(refresh_preservation_checkpoints=True)
        assert "refresh-preservation" in str(cfg)

    def test_multiple_flags_in_str(self):
        cfg = RunConfig(verbose=True, generate_plots=True, refresh_source_datasets=True)
        s = str(cfg)
        assert "verbose" in s
        assert "plots" in s
        assert "refresh" in s
