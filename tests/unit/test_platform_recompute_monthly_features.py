"""Tests for scripts/recompute_monthly_features.py — the D-02-A offline recompute.

None of these tests touch the real ``data/checkpoints/platform/`` tree: they
operate on synthetic frames passed directly to the script's pure frame-builder
helper (``rebuild_monthly_features``), or statically parse the script's own
source with ``ast`` to prove it never imports a network-capable module.

The recompute exists to rebuild the dev ``monthly_features`` checkpoint from
the CACHED ``monthly_raw`` checkpoint without ever re-ingesting — see
``07-CONTEXT.md`` D-02-A and ``07-02-PLAN.md`` Task 2. The single most
important assertion in this file is Test 1: writing
``compute_lean_features()``'s 13-column output directly (instead of
reproducing ``build_monthly_spine``'s ``pd.concat([monthly_raw, lean], axis=1)``
+ dedupe tail) would silently destroy 40 raw columns of the real checkpoint.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from recompute_monthly_features import rebuild_monthly_features

from trading_crab_lib.platform.honesty.holdout import split_by_holdout_boundary

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "recompute_monthly_features.py"

TAXONOMY_CFG: dict = {
    "fast": [
        "curve_10y3m", "curve_10y2y", "credit_spread_baa_aaa", "fred_vix", "gold", "oil",
        "trailing_return_1m", "trailing_return_3m", "realized_vol_1m", "realized_vol_3m",
    ],
    "slow": ["cape_shiller", "div_yield", "real_rate_level"],
    "agency": ["fred_gdp", "fred_cpi", "fred_unrate", "fred_indpro", "fred_payems"],
}


def _make_cfg() -> dict:
    return {"taxonomy": TAXONOMY_CFG}


def _make_monthly_raw(periods: int = 24) -> pd.DataFrame:
    """A synthetic monthly_raw carrying both lean-source columns AND
    non-lean columns (fred_gdp, equities_tr, cash), matching the real
    checkpoint's shape: some columns feed compute_lean_features, most do
    not and must simply survive the rebuild untouched."""
    idx = pd.date_range("2010-01-31", periods=periods, freq="ME")
    n = len(idx)
    return pd.DataFrame(
        {
            # Lean-source / passthrough columns.
            "fred_gs10": np.linspace(0.02, 0.035, n),
            "fred_tb3ms": np.linspace(0.01, 0.018, n),
            "fred_baa": np.linspace(0.05, 0.06, n),
            "fred_aaa": np.linspace(0.04, 0.045, n),
            "fred_vix": np.linspace(15.0, 22.0, n),
            "fred_cpi": 100.0 * (1.002 ** np.arange(n)),
            "gold": 1000.0 + np.arange(n, dtype=float),
            "oil": 50.0 + np.arange(n, dtype=float) * 0.5,
            "cape_shiller": [25.0] * n,
            "div_yield": [0.02] * n,
            "equities_tr": 100 * (1.01 ** np.arange(n)),
            # Non-lean columns that compute_lean_features never reads —
            # these are the 40-column "raw" columns Test 1 protects.
            "fred_gdp": [20000.0] * n,
            "cash": [1.0] * n,
        },
        index=idx,
    )


class TestRebuildMonthlyFeatures:
    def test_rebuild_frame_keeps_every_raw_column(self):
        raw = _make_monthly_raw()
        cfg = _make_cfg()

        rebuilt = rebuild_monthly_features(raw, cfg)

        missing = set(raw.columns) - set(rebuilt.columns)
        assert not missing, (
            f"rebuild dropped raw column(s) {sorted(missing)} — this is the exact "
            "13-column bug: writing compute_lean_features()'s output directly "
            "instead of reproducing build_monthly_spine's concat+dedupe tail."
        )

    def test_passthrough_columns_are_deduped_keeping_the_lean_copy(self):
        raw = _make_monthly_raw()
        cfg = _make_cfg()

        rebuilt = rebuild_monthly_features(raw, cfg)

        oil_cols = [c for c in rebuilt.columns if c == "oil"]
        assert len(oil_cols) == 1, f"expected exactly one 'oil' column after dedupe, got {len(oil_cols)}"
        pd.testing.assert_series_equal(rebuilt["oil"], raw["oil"], check_names=False)

    def test_dev_side_stops_at_the_cutoff(self):
        raw = _make_monthly_raw(periods=180)  # spans well past a 2015-06-30 cutoff
        cfg = _make_cfg()

        rebuilt = rebuild_monthly_features(raw, cfg)
        cutoff = "2015-06-30"
        dev_df, holdout_df = split_by_holdout_boundary(rebuilt, cutoff=cutoff)

        cutoff_ts = pd.Timestamp(cutoff)
        assert dev_df.index.max() <= cutoff_ts
        assert holdout_df.empty or holdout_df.index.min() > cutoff_ts
        assert len(dev_df) + len(holdout_df) == len(rebuilt)

    def test_index_name_is_date(self):
        raw = _make_monthly_raw()
        raw.index.name = None  # simulate an unnamed source index
        cfg = _make_cfg()

        rebuilt = rebuild_monthly_features(raw, cfg)

        assert rebuilt.index.name == "date"


class TestScriptStaticImports:
    """Static AST checks — the script must never be capable of reaching the
    network. macrotrends/stooq egress is blocked in this container, so a
    script that imports a network-capable module cannot merely run slowly
    here; it cannot run at all."""

    @pytest.fixture
    def script_ast(self) -> ast.Module:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(SCRIPT_PATH))

    def _imported_module_names(self, tree: ast.Module) -> list[str]:
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        return names

    def test_no_network_module_is_imported(self, script_ast):
        forbidden_modules = ("fredapi", "yfinance", "requests", "curl_cffi")
        imported = self._imported_module_names(script_ast)

        for forbidden in forbidden_modules:
            offenders = [name for name in imported if name == forbidden or name.startswith(forbidden + ".")]
            assert not offenders, f"script imports network-capable module {offenders!r}"

        ingestion_offenders = [
            name for name in imported
            if name.startswith("trading_crab_lib.platform.ingestion")
        ]
        assert not ingestion_offenders, f"script imports ingestion submodule(s) {ingestion_offenders!r}"

        referenced_names = {
            node.id for node in ast.walk(script_ast) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(script_ast) if isinstance(node, ast.Attribute)
        }
        assert "build_monthly_spine" not in referenced_names, (
            "script references build_monthly_spine — that function always re-fetches "
            "from FRED/macrotrends/stooq/yfinance, which this offline recompute must not do."
        )

    def test_imports_only_platform_and_stdlib(self, script_ast):
        imported = self._imported_module_names(script_ast)
        trading_crab_imports = [name for name in imported if name.startswith("trading_crab_lib")]

        offenders = [
            name for name in trading_crab_imports
            if not (name == "trading_crab_lib" or name.startswith("trading_crab_lib.platform"))
        ]
        assert not offenders, (
            f"non-platform trading_crab_lib import(s) {offenders!r} — criterion 8 requires "
            "platform/ code (and its entry-point scripts) to only ever reach into "
            "trading_crab_lib.platform.*, never the legacy incumbent library surface."
        )
