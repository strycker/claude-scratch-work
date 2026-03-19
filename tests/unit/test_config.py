"""Unit tests for config loader and portfolio loader."""

from pathlib import Path

import pytest
import yaml

from trading_crab_lib.config import load_portfolio


class TestLoadPortfolio:
    def test_missing_file_returns_empty(self, tmp_path):
        assert load_portfolio(portfolio_path=tmp_path / "nonexistent.yaml") == {}

    def test_normalizes_weights_to_sum_one(self, tmp_path):
        path = tmp_path / "portfolio.yaml"
        path.write_text("SPY: 0.6\nIEF: 0.4\n")
        w = load_portfolio(portfolio_path=path)
        assert set(w.keys()) == {"SPY", "IEF"}
        assert abs(w["SPY"] - 0.6) < 1e-9
        assert abs(w["IEF"] - 0.4) < 1e-9
        assert abs(sum(w.values()) - 1.0) < 1e-9

    def test_unnormalized_weights_rescaled(self, tmp_path):
        path = tmp_path / "portfolio.yaml"
        path.write_text("SPY: 50\nIEF: 50\n")  # sum 100
        w = load_portfolio(portfolio_path=path)
        assert abs(w["SPY"] - 0.5) < 1e-9
        assert abs(w["IEF"] - 0.5) < 1e-9

    def test_skips_non_positive_and_comments(self, tmp_path):
        path = tmp_path / "portfolio.yaml"
        path.write_text("# comment\nSPY: 0.5\nIEF: 0\nQQQ: -0.1\n")
        w = load_portfolio(portfolio_path=path)
        assert list(w.keys()) == ["SPY"]
        assert abs(w["SPY"] - 1.0) < 1e-9
