"""Tests for src/trading_crab_lib/platform/plotting/{core,loaders}.py.

Follows the tests/unit/test_plotting.py pattern: force the Agg backend
before importing pyplot, seed synthetic frames, and assert does-not-crash +
edge-case shapes rather than pixel content.
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
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from trading_crab_lib.platform import plotting as pplot  # noqa: E402
from trading_crab_lib.platform.checkpoints import PLATFORM_CHECKPOINT_DIR  # noqa: E402
from trading_crab_lib.platform.plotting import core as pcore  # noqa: E402
from trading_crab_lib.platform.plotting import loaders  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


# ── _save_or_show ────────────────────────────────────────────────────────────

class TestSaveOrShow:
    def test_saves_file_at_explicit_path(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        save_path = tmp_path / "sub" / "test_plot.png"
        result = pcore._save_or_show(fig, save_path=save_path, show=False)
        assert save_path.exists()
        assert isinstance(result, plt.Figure)
        assert result is fig

    def test_no_save_without_path(self, tmp_path):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        result = pcore._save_or_show(fig, save_path=None, show=False)
        assert isinstance(result, plt.Figure)
        assert list(tmp_path.iterdir()) == []


class TestRegimeColor:
    def test_wraps_modulo_palette_length(self):
        n = len(pcore.CUSTOM_COLORS)
        assert pcore._regime_color(0) == pcore.CUSTOM_COLORS[0]
        assert pcore._regime_color(n) == pcore.CUSTOM_COLORS[0]
        assert pcore._regime_color(n + 2) == pcore.CUSTOM_COLORS[2]


class TestPalette:
    def test_custom_colors_has_five_entries(self):
        # Matches the platform's configured labeling.K of 5.
        assert len(pplot.CUSTOM_COLORS) == 5

    def test_a13_caveat_names_audit_item(self):
        assert "A13" in pplot.A13_CAVEAT
        assert "not interpretable" in pplot.A13_CAVEAT.lower()


# ── D-01 fresh-package boundary ──────────────────────────────────────────────

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
    """Statically compute the set of trading_crab_lib modules reachable from *start_module*."""
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
        reachable = _transitive_closure("trading_crab_lib.platform.plotting")
        offending = {
            m for m in reachable
            if m == _FORBIDDEN_LEGACY_MODULE or m.startswith(_FORBIDDEN_LEGACY_MODULE + ".")
        }
        assert not offending, (
            f"trading_crab_lib.platform.plotting transitively imports legacy plotting "
            f"module(s), which D-01 forbids: {offending}"
        )


# ── D-10 loaders: load-or-raise-actionably ──────────────────────────────────

class TestLoadPlatformCheckpoint:
    def test_missing_checkpoint_raises_actionable(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            loaders.load_platform_checkpoint("does_not_exist", rebuild_hint="X")
        message = str(exc_info.value)
        assert "does_not_exist" in message
        assert "X" in message

    def test_real_monthly_raw_shape(self):
        df = loaders.load_platform_checkpoint(
            "monthly_raw", rebuild_hint="python scripts/build_platform_data.py"
        )
        assert df.shape[0] >= 700
        assert df.shape[1] >= 40
        assert df.index.max() >= _ts("2020-12-31")


class TestLoadFullSpanCheckpoint:
    def test_missing_dev_checkpoint_raises_actionable(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            loaders.load_full_span_checkpoint("does_not_exist", rebuild_hint="Y")
        message = str(exc_info.value)
        assert "does_not_exist" in message
        assert "Y" in message

    def test_full_span_has_more_rows_than_dev(self, tmp_path, monkeypatch):
        # tests/conftest.py's session-scoped checkpoint isolation redirects
        # PLATFORM_CHECKPOINT_DIR to a session tmp dir but deliberately does
        # NOT seed it with real data/holdout/ content (that data is the one
        # dataset the honesty framework exists to protect and is not
        # reproducible from a dev-fenced rebuild) — so this test builds its
        # own dev+holdout pair rather than depending on real holdout data
        # being present under pytest.
        import trading_crab_lib.platform.checkpoints as platform_ckpt_mod
        import trading_crab_lib.platform.honesty.holdout as holdout_mod
        from trading_crab_lib.checkpoints import CheckpointManager

        dev_dir = tmp_path / "dev"
        holdout_dir = tmp_path / "holdout"
        monkeypatch.setattr(platform_ckpt_mod, "PLATFORM_CHECKPOINT_DIR", dev_dir)
        monkeypatch.setattr(holdout_mod, "HOLDOUT_CHECKPOINT_DIR", holdout_dir)

        dev_idx = pd.date_range("2019-01-31", periods=12, freq="ME")
        holdout_idx = pd.date_range("2021-01-31", periods=4, freq="ME")
        CheckpointManager(checkpoint_dir=dev_dir).save(
            pd.DataFrame({"x": range(12)}, index=dev_idx), "synthetic_features"
        )
        CheckpointManager(checkpoint_dir=holdout_dir).save(
            pd.DataFrame({"x": range(4)}, index=holdout_idx), "synthetic_features"
        )

        dev = loaders.load_platform_checkpoint("synthetic_features", rebuild_hint="x")
        full = loaders.load_full_span_checkpoint("synthetic_features", rebuild_hint="x")
        assert len(full) > len(dev)
        assert len(full) == 16
        assert full.index.max() >= _ts("2021-01-31")


class TestLoadReportArtifact:
    def test_missing_artifact_raises_actionable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loaders, "OUTPUT_DIR", tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            loaders.load_report_artifact("missing_file.parquet", rebuild_hint="Z command")
        message = str(exc_info.value)
        assert "missing_file.parquet" in message
        assert "Z command" in message


class TestLoadFullSampleStatesAndFilteredProbs:
    """These artifacts are written by plan 06-02, absent in this plan's wave.

    Monkeypatching OUTPUT_DIR to an empty tmp_path keeps this deterministic
    even after 06-02 lands and the real artifacts exist elsewhere on disk.
    """

    def test_load_full_sample_states_missing_raises_actionable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loaders, "OUTPUT_DIR", tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            loaders.load_full_sample_states()
        assert "trading_crab_lib.platform.evaluation.report" in str(exc_info.value)

    def test_load_filtered_state_probs_missing_raises_actionable(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loaders, "OUTPUT_DIR", tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            loaders.load_filtered_state_probs()
        assert "trading_crab_lib.platform.evaluation.report" in str(exc_info.value)


class TestRedactedConfig:
    def test_redacts_nested_api_key_without_mutating_input(self):
        cfg = {
            "fred_monthly": {"api_key": "SECRET123", "other": 1},
            "nested": {"deep": {"fred_vintage_api_key": "SECRET456"}},
            "series": ["a", "b"],
        }
        redacted = loaders.redacted_config(cfg)
        assert redacted["fred_monthly"]["api_key"] == "<redacted>"
        assert redacted["nested"]["deep"]["fred_vintage_api_key"] == "<redacted>"
        assert redacted["fred_monthly"]["other"] == 1
        assert redacted["series"] == ["a", "b"]
        # original untouched
        assert cfg["fred_monthly"]["api_key"] == "SECRET123"
        assert cfg["nested"]["deep"]["fred_vintage_api_key"] == "SECRET456"


class TestNotebookScratchDir:
    def test_differs_from_production_platform_checkpoint_dir(self):
        assert loaders.NOTEBOOK_SCRATCH_DIR != PLATFORM_CHECKPOINT_DIR
