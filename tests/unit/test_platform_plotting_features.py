"""Unit tests for ``platform/plotting/features.py`` (P2 — features & taxonomy).

Follows ``tests/unit/test_plotting.py``'s shape: Agg backend forced in the
module header, seeded synthetic frames, one ``TestXxx`` class per public
function with a does-not-crash case and an empty-input case. No network, no
real checkpoint reads — the plan's own ``<verify>`` scripts cover the live
data outside pytest, where the session-scoped checkpoint-isolation fixture in
``tests/conftest.py`` does not apply.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

# matplotlib.use("Agg") must precede pyplot import — import order is intentional.
# pylint: disable=wrong-import-position,wrong-import-order
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from trading_crab_lib.platform.plotting import features as pfeatures  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order


SYNTHETIC_CFG: dict = {
    "taxonomy": {
        "fast": ["f_alpha", "f_beta", "f_gamma"],
        "slow": ["s_alpha", "s_beta"],
        "agency": ["a_alpha"],
    }
}


@pytest.fixture
def monthly_df() -> pd.DataFrame:
    """40 monthly rows: 3 fast-tagged, 2 slow-tagged, 1 agency-tagged, 2 untagged columns."""
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    rng = np.random.default_rng(42)
    columns = ["f_alpha", "f_beta", "f_gamma", "s_alpha", "s_beta", "a_alpha", "u_alpha", "u_beta"]
    return pd.DataFrame(rng.standard_normal((40, len(columns))), index=idx, columns=columns)


def _real_shaped_range_table() -> pd.DataFrame:
    """A range table matching the live dev checkpoint's observed ranges for every named bound.

    Deliberately synthetic (not a live checkpoint read) so the test stays
    network-free and deterministic, while carrying the real numbers verified
    against ``monthly_features`` during planning.
    """
    rows = [
        ("fred_vix", "fast", 9.51, 59.89),
        ("realized_vol_1m", "fast", 0.0001, 0.2014),
        ("realized_vol_3m", "fast", 0.0007, 0.1132),
        ("credit_spread_baa_aaa", "fast", 0.32, 3.38),
        ("cape_shiller", "slow", 6.64, 44.19),
        ("div_yield", "slow", 0.0111, 0.0624),
        ("gold", "fast", 254.60, 1971.68),
        ("oil", "fast", 10.25, 139.96),
    ]
    return pd.DataFrame(
        [
            {
                "feature": name,
                "tier": tier,
                "n_obs": 708,
                "first_valid": pd.Timestamp("1962-01-31"),
                "last_valid": pd.Timestamp("2020-12-31"),
                "min": lo,
                "max": hi,
                "mean": (lo + hi) / 2,
                "std": (hi - lo) / 4,
            }
            for name, tier, lo, hi in rows
        ]
    )


def _drift_report_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"column": "gold", "standardized_mean_shift": 2.4, "flag": True},
            {"column": "fred_vix", "standardized_mean_shift": -1.3, "flag": True},
            {"column": "cape_shiller", "standardized_mean_shift": 0.4, "flag": False},
            {"column": "div_yield", "standardized_mean_shift": float("nan"), "flag": False},
        ]
    )


# ── tier_frames ──────────────────────────────────────────────────────────────


class TestTierFrames:
    def test_groups_columns_into_the_four_tiers(self, monthly_df):
        tiers = pfeatures.tier_frames(monthly_df, SYNTHETIC_CFG)
        assert set(tiers) == {"fast", "slow", "agency", "untagged"}
        assert list(tiers["fast"].columns) == ["f_alpha", "f_beta", "f_gamma"]
        assert list(tiers["slow"].columns) == ["s_alpha", "s_beta"]
        assert list(tiers["agency"].columns) == ["a_alpha"]
        assert list(tiers["untagged"].columns) == ["u_alpha", "u_beta"]

    def test_slices_preserve_the_index(self, monthly_df):
        tiers = pfeatures.tier_frames(monthly_df, SYNTHETIC_CFG)
        assert tiers["fast"].index.equals(monthly_df.index)

    def test_empty_input_returns_four_empty_frames(self):
        tiers = pfeatures.tier_frames(pd.DataFrame(), SYNTHETIC_CFG)
        assert set(tiers) == {"fast", "slow", "agency", "untagged"}
        for frame in tiers.values():
            assert isinstance(frame, pd.DataFrame)
            assert frame.shape[1] == 0

    def test_tier_membership_comes_from_taxonomy_not_a_local_list(self, monthly_df):
        """An empty taxonomy config puts every column in 'untagged' — no hard-coded fallback list."""
        tiers = pfeatures.tier_frames(monthly_df, {"taxonomy": {}})
        assert tiers["untagged"].shape[1] == monthly_df.shape[1]
        assert tiers["fast"].shape[1] == 0


# ── feature_range_table ──────────────────────────────────────────────────────


class TestFeatureRangeTable:
    def test_one_row_per_column_with_expected_fields(self, monthly_df):
        table = pfeatures.feature_range_table(monthly_df, SYNTHETIC_CFG)
        assert len(table) == monthly_df.shape[1]
        assert list(table.columns) == [
            "feature", "tier", "n_obs", "first_valid", "last_valid", "min", "max", "mean", "std",
        ]
        assert set(table["feature"]) == set(monthly_df.columns)

    def test_untagged_columns_report_the_literal_untagged_tier(self, monthly_df):
        table = pfeatures.feature_range_table(monthly_df, SYNTHETIC_CFG).set_index("feature")
        assert table.loc["u_alpha", "tier"] == "untagged"
        assert table.loc["u_beta", "tier"] == "untagged"
        assert table.loc["a_alpha", "tier"] == "agency"

    def test_sorted_by_tier_then_feature(self, monthly_df):
        table = pfeatures.feature_range_table(monthly_df, SYNTHETIC_CFG)
        pairs = list(zip(table["tier"], table["feature"]))
        assert pairs == sorted(pairs)

    def test_all_nan_column_reports_zero_obs_and_does_not_raise(self, monthly_df):
        df = monthly_df.copy()
        df["u_alpha"] = np.nan
        table = pfeatures.feature_range_table(df, SYNTHETIC_CFG).set_index("feature")
        assert table.loc["u_alpha", "n_obs"] == 0
        for field in ("min", "max", "mean", "std"):
            assert np.isnan(table.loc["u_alpha", field])

    def test_leading_nans_report_the_first_valid_index(self, monthly_df):
        df = monthly_df.copy()
        df.iloc[:10, df.columns.get_loc("f_alpha")] = np.nan
        table = pfeatures.feature_range_table(df, SYNTHETIC_CFG).set_index("feature")
        assert table.loc["f_alpha", "first_valid"] == df.index[10]
        assert table.loc["f_alpha", "n_obs"] == 30

    def test_empty_input(self):
        table = pfeatures.feature_range_table(pd.DataFrame(), SYNTHETIC_CFG)
        assert table.empty


# ── plot_feature_ranges ──────────────────────────────────────────────────────


class TestPlotFeatureRanges:
    def test_does_not_crash(self, monthly_df):
        table = pfeatures.feature_range_table(monthly_df, SYNTHETIC_CFG)
        assert isinstance(pfeatures.plot_feature_ranges(table), plt.Figure)

    def test_tier_filtered_subset(self, monthly_df):
        table = pfeatures.feature_range_table(monthly_df, SYNTHETIC_CFG)
        fig = pfeatures.plot_feature_ranges(table, tier="fast")
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes[0].get_yticklabels()) == 3

    def test_tier_with_no_members_does_not_raise(self, monthly_df):
        table = pfeatures.feature_range_table(monthly_df, SYNTHETIC_CFG)
        assert isinstance(pfeatures.plot_feature_ranges(table, tier="nonexistent"), plt.Figure)

    def test_all_nan_rows_do_not_raise(self, monthly_df):
        df = monthly_df.copy()
        df["u_alpha"] = np.nan
        table = pfeatures.feature_range_table(df, SYNTHETIC_CFG)
        assert isinstance(pfeatures.plot_feature_ranges(table), plt.Figure)

    def test_empty_input(self):
        assert isinstance(pfeatures.plot_feature_ranges(pd.DataFrame()), plt.Figure)

    def test_saves_to_explicit_path(self, monthly_df, tmp_path):
        table = pfeatures.feature_range_table(monthly_df, SYNTHETIC_CFG)
        out = tmp_path / "ranges.png"
        pfeatures.plot_feature_ranges(table, save_path=out)
        assert out.exists()


# ── assert_feature_ranges_plausible (D-11) ───────────────────────────────────


class TestAssertFeatureRangesPlausible:
    def test_real_shaped_table_passes_and_returns_a_verdict_frame(self):
        verdict = pfeatures.assert_feature_ranges_plausible(_real_shaped_range_table())
        assert list(verdict.columns) == ["feature", "bound", "observed", "verdict"]
        assert len(verdict) == len(pfeatures._RANGE_CHECKS)
        assert set(verdict["verdict"]) == {"pass"}

    @pytest.mark.parametrize(
        "feature,field,bad_value",
        [
            ("fred_vix", "min", -5.0),
            ("fred_vix", "min", 0.0),
            ("realized_vol_1m", "min", -0.01),
            ("realized_vol_3m", "min", -0.0001),
            ("credit_spread_baa_aaa", "min", -0.5),
            ("cape_shiller", "min", 0.0),
            ("gold", "min", -1.0),
            ("oil", "min", 0.0),
            ("div_yield", "max", 0.40),
            ("div_yield", "min", -0.01),
        ],
    )
    def test_out_of_band_value_raises_naming_the_feature(self, feature, field, bad_value):
        table = _real_shaped_range_table().set_index("feature")
        table.loc[feature, field] = bad_value
        with pytest.raises(ValueError) as exc_info:
            pfeatures.assert_feature_ranges_plausible(table.reset_index())
        assert feature in str(exc_info.value)

    def test_collects_every_violation_before_raising_once(self):
        table = _real_shaped_range_table().set_index("feature")
        table.loc["fred_vix", "min"] = -5.0
        table.loc["div_yield", "max"] = 0.40
        with pytest.raises(ValueError) as exc_info:
            pfeatures.assert_feature_ranges_plausible(table.reset_index())
        message = str(exc_info.value)
        assert "fred_vix" in message and "div_yield" in message
        assert "2 domain-impossible" in message

    def test_absent_feature_is_skipped_silently(self):
        table = _real_shaped_range_table()
        table = table[table["feature"] != "gold"]
        verdict = pfeatures.assert_feature_ranges_plausible(table)
        assert "gold" not in set(verdict["feature"])
        assert set(verdict["verdict"]) == {"pass"}

    def test_all_nan_feature_has_no_range_to_violate(self):
        table = _real_shaped_range_table().set_index("feature")
        table.loc["fred_vix", ["min", "max", "mean", "std"]] = np.nan
        verdict = pfeatures.assert_feature_ranges_plausible(table.reset_index()).set_index("feature")
        assert verdict.loc["fred_vix", "verdict"] == "pass"
        assert verdict.loc["fred_vix", "observed"] == "no observations"

    def test_empty_input(self):
        verdict = pfeatures.assert_feature_ranges_plausible(pd.DataFrame())
        assert verdict.empty
        assert list(verdict.columns) == ["feature", "bound", "observed", "verdict"]


# ── check_agency_level_discontinuities (A4 regression guard) ─────────────────


class TestCheckAgencyLevelDiscontinuities:
    @staticmethod
    def _smooth_cpi(n: int = 60) -> pd.DataFrame:
        idx = pd.date_range("1970-01-31", periods=n, freq="ME")
        return pd.DataFrame({"fred_cpi": np.linspace(30.0, 45.0, n)}, index=idx)

    def test_clean_series_returns_empty_dict(self):
        assert pfeatures.check_agency_level_discontinuities(self._smooth_cpi()) == {}

    def test_tripling_series_raises_naming_column_and_date(self):
        df = self._smooth_cpi()
        break_date = df.index[24]
        df.loc[break_date:, "fred_cpi"] = df.loc[break_date:, "fred_cpi"] * 3.0
        with pytest.raises(ValueError) as exc_info:
            pfeatures.check_agency_level_discontinuities(df)
        message = str(exc_info.value)
        assert "fred_cpi" in message
        assert str(break_date.date()) in message
        assert "A4" in message

    def test_absent_column_warns_and_does_not_raise(self, caplog):
        with caplog.at_level("WARNING"):
            assert pfeatures.check_agency_level_discontinuities(pd.DataFrame()) == {}
        assert "fred_cpi" in caplog.text

    def test_collects_every_offending_column(self):
        df = self._smooth_cpi()
        df["fred_indpro"] = df["fred_cpi"].to_numpy()
        for column in ("fred_cpi", "fred_indpro"):
            df.loc[df.index[24]:, column] = df.loc[df.index[24]:, column] * 3.0
        with pytest.raises(ValueError) as exc_info:
            pfeatures.check_agency_level_discontinuities(df, columns=("fred_cpi", "fred_indpro"))
        message = str(exc_info.value)
        assert "fred_cpi" in message and "fred_indpro" in message
        assert "2 column(s)" in message

    def test_does_not_reimplement_the_discontinuity_math(self):
        """The guard must delegate to drift.assert_no_level_discontinuity, not duplicate it."""
        source = Path(pfeatures.__file__).read_text()
        assert "drift.assert_no_level_discontinuity" in source
        assert "ratio_threshold" not in source.split("def check_agency_level_discontinuities")[1]


# ── plot_drift_summary ───────────────────────────────────────────────────────


class TestPlotDriftSummary:
    def test_does_not_crash(self):
        assert isinstance(pfeatures.plot_drift_summary(_drift_report_frame()), plt.Figure)

    def test_flagged_bars_use_a_distinct_color(self):
        fig = pfeatures.plot_drift_summary(_drift_report_frame())
        colors = {patch.get_facecolor()[:3] for patch in fig.axes[0].patches}
        assert len(colors) >= 2

    def test_empty_input(self):
        assert isinstance(pfeatures.plot_drift_summary(pd.DataFrame()), plt.Figure)

    def test_saves_to_explicit_path(self, tmp_path):
        out = tmp_path / "drift.png"
        pfeatures.plot_drift_summary(_drift_report_frame(), save_path=out)
        assert out.exists()


# ── D-01 fresh-package boundary + module hygiene ─────────────────────────────

_FORBIDDEN_LEGACY_MODULE = "trading_crab_lib.plotting"


def _local_imports(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return {n for n in names if n.startswith("trading_crab_lib")}


def _transitive_closure(start_module: str) -> set[str]:
    seen: set[str] = set()
    stack = [start_module]
    while stack:
        mod_name = stack.pop()
        if mod_name in seen:
            continue
        seen.add(mod_name)
        spec = importlib.util.find_spec(mod_name)
        if spec is None or spec.origin is None:
            continue
        path = Path(spec.origin)
        if path.suffix != ".py":
            continue
        for dep in _local_imports(path):
            if dep not in seen:
                stack.append(dep)
    return seen


class TestFreshPackageBoundary:
    def test_no_reachable_module_is_legacy_plotting(self):
        reachable = _transitive_closure("trading_crab_lib.platform.plotting.features")
        offending = {
            m for m in reachable
            if m == _FORBIDDEN_LEGACY_MODULE or m.startswith(_FORBIDDEN_LEGACY_MODULE + ".")
        }
        assert not offending, (
            f"platform.plotting.features transitively imports legacy plotting module(s), "
            f"which D-01 forbids: {offending}"
        )

    def test_matplotlib_is_reached_only_through_core(self):
        """features.py owns no plotting-library import of its own (core owns the Agg guard)."""
        source = Path(pfeatures.__file__).read_text()
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        offending = {name for name in imported if name.split(".")[0] in {"matplotlib", "seaborn", "pylab"}}
        assert not offending, f"features.py imports a plotting library directly: {offending}"

    def test_features_is_not_re_exported_from_the_package_barrel(self):
        """Wave-2 submodules are imported by path so the five plans never contend for __init__.py."""
        from trading_crab_lib.platform import plotting as pplot

        assert not hasattr(pplot, "tier_frames")


# ── P2 notebook source discipline (guarded until the notebook lands) ─────────

_P2_NOTEBOOK = Path("notebooks/platform/P2_features_taxonomy.ipynb")


def _p2_code_source() -> str:
    import nbformat

    nb = nbformat.read(_P2_NOTEBOOK, as_version=4)
    return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")


class TestP2NotebookSource:
    def test_no_code_cell_builds_a_centered_variant(self):
        """Amendment 3 item I: P2 builds/seeks/overlays no centered feature, ever."""
        if not _P2_NOTEBOOK.exists():
            pytest.skip("P2_features_taxonomy.ipynb not yet built (task 3 produces it)")
        code = _p2_code_source()
        assert "center=True" not in code, "a code cell constructs a two-sided (centered) window"
        for suffix in ("_centered", "_c5", "_zerophase"):
            assert f'"{suffix}' not in code and f"'{suffix}" not in code, (
                f"a code cell references a {suffix}-suffixed column, which "
                f"honesty.gating.FORBIDDEN_CENTERED_SUFFIXES rejects outright"
            )

    def test_carries_no_sign_off_cell(self):
        """D-15: the sign-off is P3's exclusive responsibility."""
        if not _P2_NOTEBOOK.exists():
            pytest.skip("P2_features_taxonomy.ipynb not yet built (task 3 produces it)")
        import nbformat

        nb = nbformat.read(_P2_NOTEBOOK, as_version=4)
        combined = "\n".join(cell.source for cell in nb.cells)
        assert "Sign-Off" not in combined
        assert "sign_off" not in combined
