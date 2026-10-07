"""The no_regime weekly page needs no regime model (phase 08.4, K-10 / D-T7).

From a fresh build alone (monthly_raw + monthly_features, no nowcaster, no regime labels,
no serving artifacts) the page renders and persists the executed book. regime_tilt without a
model still raises; with a model the page is unchanged (test_platform_weekly_no_regime.py).
"""

from __future__ import annotations

import pandas as pd
import pytest
from test_platform_report_serving import _serving_world

import trading_crab_lib.platform.checkpoints as platform_ckpt
from trading_crab_lib.platform.report import weekly
from trading_crab_lib.platform.report.serving import SERVING_BUILD_COMMAND


def _fresh_build_world(tmp_path, monkeypatch, *, mode: str | None) -> dict:
    """What a fresh build leaves: no serving artifacts, no regime labels, no backtest outputs."""
    world = _serving_world(tmp_path, monkeypatch)
    if mode is not None:
        world["cfg"]["report"]["allocation_mode"] = mode
    (world["platform_dir"] / "regime_labels.parquet").unlink()
    (world["platform_dir"] / "regime_labels.meta.json").unlink(missing_ok=True)
    empty_reports = tmp_path / "no_reports"
    empty_reports.mkdir()
    monkeypatch.setattr(platform_ckpt, "PLATFORM_REPORT_DIR", empty_reports)
    return world


def test_the_no_regime_page_builds_with_no_regime_model(tmp_path, monkeypatch):
    world = _fresh_build_world(tmp_path, monkeypatch, mode="no_regime")
    cm = platform_ckpt.get_platform_checkpoint_manager()

    markdown, path = weekly.build_weekly_page(world["cfg"], cm, output_dir=tmp_path / "page")

    assert path.read_text(encoding="utf-8") == markdown
    assert weekly._NO_MODEL_SENTENCE.startswith("Regime view: suspended")
    assert weekly._NO_MODEL_SENTENCE in markdown
    assert weekly._SUSPENDED_SENTENCE not in markdown
    assert "Scoreboard: not yet measured" in markdown
    assert "Crash Tripwire" in markdown
    assert "**Allocation mode:** no_regime" in markdown
    assert "### Target allocation" in markdown

    # The executed book and the mode are persisted; no model-derived state is written.
    written = {p.name for p in world["platform_dir"].iterdir()}
    assert {"executed_weights.parquet", "allocation_mode.parquet"} <= written
    for absent in ("regime_belief", "hysteresis_state", "nowcaster", "asset_returns", "returns_by_regime"):
        assert not [n for n in written if n.startswith(absent)], (absent, sorted(written))


def test_the_executed_book_of_the_model_free_page_is_the_no_regime_target(tmp_path, monkeypatch):
    world = _fresh_build_world(tmp_path, monkeypatch, mode="no_regime")
    cm = platform_ckpt.get_platform_checkpoint_manager()

    inputs = weekly._build_report_inputs(world["cfg"], cm)

    assert inputs["suspended_sentence"] == weekly._NO_MODEL_SENTENCE
    assert inputs["allocation_mode"] == "no_regime"
    assert inputs["regime_belief"] is None and inputs["active_regime"] is None
    weights = inputs["target_weights"]
    assert abs(float(weights.sum()) + float(inputs["cash"]) - 1.0) < 1e-9
    assert weekly.load_last_executed_weights(cm) is not None


def test_regime_tilt_without_a_model_still_raises_naming_the_serving_command(tmp_path, monkeypatch):
    world = _fresh_build_world(tmp_path, monkeypatch, mode="regime_tilt")
    cm = platform_ckpt.get_platform_checkpoint_manager()

    with pytest.raises(FileNotFoundError, match=SERVING_BUILD_COMMAND):
        weekly.build_weekly_page(world["cfg"], cm, output_dir=tmp_path / "page")
    assert not [p for p in world["platform_dir"].iterdir() if p.name.startswith("executed_weights")]


def test_a_bad_mode_value_still_raises_before_anything_is_loaded(tmp_path, monkeypatch):
    world = _fresh_build_world(tmp_path, monkeypatch, mode="no_regime")
    world["cfg"]["report"]["allocation_mode"] = "sideways"
    with pytest.raises(ValueError, match="allocation_mode"):
        weekly.build_weekly_page(world["cfg"], platform_ckpt.get_platform_checkpoint_manager())


def test_assemble_weekly_report_keeps_the_old_sentence_when_none_is_given():
    kwargs = dict(
        regime_probs=pd.Series(dtype=float), transition_matrix=pd.DataFrame(), returns_by_regime=pd.DataFrame(),
        target_weights=pd.Series({"SPY": 0.6}), accounts=[], active_regime=None, regime_view_suspended=True,
    )
    assert weekly._SUSPENDED_SENTENCE in weekly.assemble_weekly_report(**kwargs)
    assert weekly._NO_MODEL_SENTENCE in weekly.assemble_weekly_report(
        **kwargs, suspended_sentence=weekly._NO_MODEL_SENTENCE
    )
