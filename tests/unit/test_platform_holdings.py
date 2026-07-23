"""
Tests for the L4-03 per-account holdings YAML loader (04-CONTEXT.md D-01:
warn-don't-fail, cash-aware, no-silent-normalize).

Headline invariant: an out-of-tolerance weights+cash sum logs a WARNING but
NEVER normalizes the returned weights — the report must show Glenn exactly
what's on file, not a silently-corrected version (T-04-14).
"""

from __future__ import annotations

import logging

from trading_crab_lib.platform.report.holdings import load_account_weights

# ── well-formed account: loads silently ──────────────────────────────────────


class TestWellFormedAccount:
    def test_loads_silently_with_no_warning(self, tmp_path, caplog):
        (tmp_path / "glenn_ira.yaml").write_text(
            "weights:\n  SPY: 0.5\n  TLT: 0.3\ncash: 0.2\n", encoding="utf-8"
        )
        with caplog.at_level(logging.WARNING):
            result = load_account_weights("glenn_ira", accounts_dir=tmp_path)

        assert result == {"weights": {"SPY": 0.5, "TLT": 0.3}, "cash": 0.2}
        assert caplog.records == []


# ── out-of-tolerance sum: warns, does NOT normalize (T-04-14) ────────────────


class TestOutOfToleranceSum:
    def test_warns_but_returns_raw_unnormalized_weights(self, tmp_path, caplog):
        # weights + cash = 0.9 + 0.3 + 0.05 = 1.25, well beyond the 0.02 tolerance.
        (tmp_path / "glenn_ira.yaml").write_text(
            "weights:\n  SPY: 0.9\n  TLT: 0.3\ncash: 0.05\n", encoding="utf-8"
        )
        with caplog.at_level(logging.WARNING):
            result = load_account_weights("glenn_ira", accounts_dir=tmp_path)

        assert result["weights"] == {"SPY": 0.9, "TLT": 0.3}
        assert result["cash"] == 0.05
        assert any("sum to" in r.message for r in caplog.records)


# ── missing file: neutral dict, never a crash ─────────────────────────────────


class TestMissingFile:
    def test_returns_neutral_dict_with_warning_not_crash(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            result = load_account_weights("nonexistent_account", accounts_dir=tmp_path)

        assert result == {"weights": {}, "cash": 1.0}
        assert any("not found" in r.message for r in caplog.records)


# ── non-numeric weight: coerced via try/except, WARNING, skip (V5) ───────────


class TestNonNumericWeight:
    def test_non_numeric_weight_skipped_with_warning_not_crash(self, tmp_path, caplog):
        (tmp_path / "glenn_ira.yaml").write_text(
            "weights:\n  SPY: 0.5\n  TLT: not_a_number\ncash: 0.3\n", encoding="utf-8"
        )
        with caplog.at_level(logging.WARNING):
            result = load_account_weights("glenn_ira", accounts_dir=tmp_path)

        assert result["weights"] == {"SPY": 0.5}
        assert result["cash"] == 0.3
        assert any("non-numeric" in r.message for r in caplog.records)

    def test_non_numeric_cash_defaults_to_zero_with_warning(self, tmp_path, caplog):
        (tmp_path / "glenn_ira.yaml").write_text(
            "weights:\n  SPY: 0.5\ncash: not_a_number\n", encoding="utf-8"
        )
        with caplog.at_level(logging.WARNING):
            result = load_account_weights("glenn_ira", accounts_dir=tmp_path)

        assert result["weights"] == {"SPY": 0.5}
        assert result["cash"] == 0.0
        assert any("non-numeric" in r.message for r in caplog.records)


# ── security convention (V5): yaml.safe_load only, no incumbent reuse ────────


class TestSecurityConvention:
    def test_uses_yaml_safe_load_not_yaml_load(self):
        import inspect

        from trading_crab_lib.platform.report import holdings

        source = inspect.getsource(holdings)
        assert "yaml.safe_load" in source
        assert "yaml.load(" not in source

    def test_does_not_import_incumbent_load_portfolio(self):
        import inspect

        from trading_crab_lib.platform.report import holdings

        source = inspect.getsource(holdings)
        assert "import load_portfolio" not in source
        assert "from trading_crab_lib.config import" not in source
