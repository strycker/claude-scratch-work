"""
Tests for the serving builder (plan 08-13, gap G-08-2).

``python -m trading_crab_lib.platform.report.serving`` builds the three artifacts the weekly
report reads — ``nowcaster.pkl``, ``returns_by_regime`` and ``asset_returns`` — through the
evaluated recipe's own function (``backtest/driver.py::fit_l2_nowcaster``), at zero registry
cost (Glenn's 2026-09-28 ruling: the serving fit is NOT a registry trial).

Every test here runs on synthetic checkpoints in per-test tmp dirs. The session fixture in
``tests/conftest.py`` already redirects the checkpoint namespaces to a session dir holding
COPIES of the real data; these tests redirect again, per test, so no serving artifact ever
lands in the shared session dir (or anywhere real).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import logging
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform.backtest import driver
from trading_crab_lib.platform.prediction.nowcaster import build_nowcaster_training_set, fit_nowcaster

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_REGISTRY = _REPO_ROOT / "registry" / "trials.jsonl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── the shared L2 fit (characterization pin + extraction) ────────────────────


def _pin_frame():
    """200 month-ends: ``long`` over every row, ``late`` from row 150 (50 observations,
    admitted at min_history 30), three classes, and a class change inside the last six
    months so the 6-month label embargo changes what is trained on."""
    n = 200
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(7)
    states = np.array([(i // 7) % 3 for i in range(n)])
    states[-4:] = (states[-5] + 1) % 3  # a class change inside the embargoed tail
    mu = np.array([-1.0, 0.0, 1.0])
    X = pd.DataFrame(
        {
            "long": mu[states] + rng.normal(scale=0.8, size=n),
            "late": -mu[states] + rng.normal(scale=0.8, size=n),
        },
        index=idx,
    )
    X.iloc[:150, X.columns.get_loc("late")] = np.nan
    cfg = {
        "labeling": {"embargo_months": 6},
        "backtest": {"feature_min_history": 30, "nowcaster_cv_splits": 3},
    }
    return X, pd.Series(states, index=idx, name="state"), cfg


class TestFitL2IsShared:
    def test_refit_l2_equals_the_inline_backtest_recipe(self):
        """CHARACTERIZATION pin, written and run GREEN against the unmodified driver before
        the extraction. Both sides are computed in-process, so exact equality is right."""
        features, labels, cfg = _pin_frame()
        row = features.iloc[[-1]]

        X, y = build_nowcaster_training_set(features, labels, embargo_months=6)
        assert y.index.max() < labels.index.max(), "the embargo must actually drop rows"
        assert labels.iloc[-4:].nunique() == 1 and labels.iloc[-4] != labels.iloc[-5]
        active = driver._cv_safe_active_features(X, y, list(X.columns), min_history=30, n_splits=3)
        assert active == ["long", "late"], "fixture must admit the late column"
        model = fit_nowcaster(X[active], y, n_splits=3)
        expected = pd.Series(model.predict_proba(row[active])[0], index=model.classes_)

        got = driver._refit_l2(features, labels, row, cfg)

        pd.testing.assert_series_equal(got, expected, check_exact=True)

    def test_fit_l2_nowcaster_returns_the_model_and_its_columns(self):
        features, labels, cfg = _pin_frame()
        model, columns = driver.fit_l2_nowcaster(features, labels, cfg)
        assert list(model.feature_names_in_) == columns
        assert columns == ["long", "late"]

    def test_refit_l2_delegates_to_the_shared_fit(self, monkeypatch):
        """``_refit_l2`` calls ``fit_l2_nowcaster`` by its module-level name — one recipe,
        not two copies that could drift."""
        features, labels, cfg = _pin_frame()
        calls = []
        real = driver.fit_l2_nowcaster
        monkeypatch.setattr(driver, "fit_l2_nowcaster", lambda *a: calls.append(a) or real(*a))
        driver._refit_l2(features, labels, features.iloc[[-1]], cfg)
        assert len(calls) == 1


# ── the one research-to-tradable mapping ─────────────────────────────────────


class TestTradableAssetReturns:
    _SPLICE = {
        "equities": {"research_name": "eq_r", "tradable": "SPY"},
        "long_duration": {"research_name": "bond_r", "tradable": "TLT"},
        "gold": {"research_name": "gold", "tradable": "IAU", "optional": True},
        "cash": {"research_name": "cash_r", "tradable": "CASH"},
    }

    def _returns(self, cols):
        idx = pd.date_range("2000-01-31", periods=4, freq="ME")
        return pd.DataFrame({c: np.arange(4, dtype=float) + i for i, c in enumerate(cols)}, index=idx)

    def test_maps_research_name_to_tradable(self):
        from trading_crab_lib.platform.assets.returns import tradable_asset_returns

        returns = self._returns(["eq_r", "bond_r", "gold", "cash_r"])
        out = tradable_asset_returns(returns, self._SPLICE)
        pd.testing.assert_series_equal(out["SPY"], returns["eq_r"], check_names=False)
        pd.testing.assert_series_equal(out["TLT"], returns["bond_r"], check_names=False)

    def test_excludes_the_cash_class(self):
        from trading_crab_lib.platform.assets.returns import tradable_asset_returns

        out = tradable_asset_returns(self._returns(["eq_r", "bond_r", "gold", "cash_r"]), self._SPLICE)
        assert "CASH" not in out.columns and "cash_r" not in out.columns

    def test_skips_a_class_whose_research_column_is_absent(self):
        from trading_crab_lib.platform.assets.returns import tradable_asset_returns

        out = tradable_asset_returns(self._returns(["eq_r", "bond_r", "cash_r"]), self._SPLICE)
        assert list(out.columns) == ["SPY", "TLT"]

    def test_preserves_splice_order(self):
        from trading_crab_lib.platform.assets.returns import tradable_asset_returns

        reordered = {k: self._SPLICE[k] for k in ("gold", "long_duration", "cash", "equities")}
        out = tradable_asset_returns(self._returns(["eq_r", "bond_r", "gold", "cash_r"]), reordered)
        assert list(out.columns) == ["IAU", "TLT", "SPY"]


# ── the synthetic serving world ──────────────────────────────────────────────

_K = 3
_EMBARGO = 12
_MIN_HISTORY = 60
_N_SPLITS = 3
_DEV_END = pd.Timestamp("2020-12-31")


def _world_cfg() -> dict:
    return {
        "labeling": {"K": _K, "embargo_months": _EMBARGO},
        "backtest": {"feature_min_history": _MIN_HISTORY, "nowcaster_cv_splits": _N_SPLITS},
        "allocation": {
            "no_trade_band": 0.05,
            "target_vol_annual": 0.10,
            "ewma_halflife_months": 6,
            "portfolio_vol_min_obs": 12,
        },
        "report": {"accounts": []},
        "splice": {
            "equities": {"research_name": "eq_r", "method": "single_source", "source_col": "px_eq", "tradable": "SPY"},
            "long_duration": {
                "research_name": "bond_r", "method": "single_source", "source_col": "px_bond", "tradable": "TLT",
            },
        },
    }


def _world_states(idx: pd.DatetimeIndex) -> np.ndarray:
    """Runs of 8 months cycling 0/1/2, except that class 2 is rare after 2015: from
    2015-01 to the embargo cutoff it appears exactly twice. A column starting in 2015 is
    therefore old enough by min_history yet its induced block starves class 2 below
    n_splits — the shape ``_cv_safe_active_features`` exists for."""
    states = np.array([(i // 8) % 3 for i in range(len(idx))])
    late = idx >= pd.Timestamp("2015-01-31")
    states[late] = np.array([(i // 6) % 2 for i in range(int(late.sum()))])
    rare = np.flatnonzero(late)[[10, 30]]
    states[rare] = 2
    return states


def _serving_world(
    tmp_path: Path,
    monkeypatch,
    *,
    short_hist: bool = False,
    starver: bool = False,
    nan_tail: int = 0,
) -> dict:
    """Write synthetic dev + holdout checkpoints into per-test tmp dirs and redirect every
    path the builder and the weekly report touch (checkpoints, holdout, outputs, registry)."""
    import trading_crab_lib.platform.assets.returns as returns_mod
    import trading_crab_lib.platform.checkpoints as platform_ckpt
    import trading_crab_lib.platform.honesty.holdout as holdout_mod
    from trading_crab_lib.platform.honesty import registry
    from trading_crab_lib.platform.report import serving, weekly

    platform_dir = tmp_path / "platform"
    holdout_dir = tmp_path / "holdout"
    out_dir = tmp_path / "out"
    registry_copy = tmp_path / "registry" / "trials.jsonl"
    registry_copy.parent.mkdir(parents=True)
    shutil.copy2(_REAL_REGISTRY, registry_copy)

    monkeypatch.setattr(platform_ckpt, "PLATFORM_CHECKPOINT_DIR", platform_dir)
    monkeypatch.setattr(holdout_mod, "HOLDOUT_CHECKPOINT_DIR", holdout_dir)
    monkeypatch.setattr(weekly, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(returns_mod, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(registry, "DEFAULT_REGISTRY_PATH", registry_copy)
    cfg = _world_cfg()
    monkeypatch.setattr(serving, "load_platform_config", lambda: cfg)
    monkeypatch.setattr(weekly, "load_platform_config", lambda: cfg)

    dev_idx = pd.date_range("1995-01-31", _DEV_END, freq="ME")
    hold_idx = pd.date_range("2021-01-31", "2021-06-30", freq="ME")
    full_idx = dev_idx.append(hold_idx)
    rng = np.random.default_rng(11)
    dev_states = _world_states(dev_idx)
    hold_states = np.array([0, 1, 2, 1, 0, 2])
    states = np.concatenate([dev_states, hold_states])
    mu = np.array([-1.5, 0.0, 1.5])
    features = pd.DataFrame(
        {
            "f_level": mu[states] + rng.normal(scale=0.6, size=len(full_idx)),
            "f_slope": -0.5 * mu[states] + rng.normal(scale=0.6, size=len(full_idx)),
        },
        index=full_idx,
    )
    if short_hist:
        # < min_history non-NaN dev months (48 of them), NaN in the latest holdout row.
        col = pd.Series(np.nan, index=full_idx)
        col.iloc[len(dev_idx) - 48: len(full_idx) - 1] = rng.normal(size=48 + len(hold_idx) - 1)
        features["short_hist"] = col
    if starver:
        # Old enough (2015-01 onward: > 60 dev months) but its block holds class 2 twice.
        col = pd.Series(np.nan, index=full_idx)
        start = int(np.flatnonzero(full_idx >= pd.Timestamp("2015-01-31"))[0])
        col.iloc[start:] = mu[states[start:]] + rng.normal(scale=0.6, size=len(full_idx) - start)
        features["starver"] = col
    if nan_tail:
        # A publication lag: the last ``nan_tail`` holdout rows lack the model column f_slope.
        # Holdout rows only, so the dev fit (and its columns) is unchanged.
        assert nan_tail <= len(hold_idx)
        features.iloc[-nan_tail:, features.columns.get_loc("f_slope")] = np.nan

    labels = pd.DataFrame({"state": pd.Series(dev_states, index=dev_idx, dtype=int)})

    prices_rng = np.random.default_rng(3)
    raw = pd.DataFrame(
        {
            "px_eq": 100 * np.cumprod(1 + prices_rng.normal(0.006, 0.04, len(full_idx))),
            "px_bond": 100 * np.cumprod(1 + prices_rng.normal(0.003, 0.02, len(full_idx))),
        },
        index=full_idx,
    )

    dev_cm = platform_ckpt.get_platform_checkpoint_manager()
    dev_cm.save(features.loc[dev_idx], "monthly_features")
    dev_cm.save(labels, "regime_labels")
    dev_cm.save(raw, "monthly_raw")
    holdout_mod.get_holdout_checkpoint_manager().save(features.loc[hold_idx], "monthly_features")

    return {
        "cfg": cfg,
        "platform_dir": platform_dir,
        "out_dir": out_dir,
        "registry_copy": registry_copy,
        "dev": features.loc[dev_idx],
        "full": features,
        "labels": labels["state"],
    }


# ── the tracer: serving.main then the real weekly.main ───────────────────────


class TestServingEndToEnd:
    def test_build_then_report_completes_and_leaves_the_registry_untouched(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.honesty import registry
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch)
        real_sha_before = _sha256(_REAL_REGISTRY)
        copy_bytes_before = world["registry_copy"].read_bytes()
        count_before = registry.total_trial_count(world["registry_copy"])
        assert count_before == registry.total_trial_count(_REAL_REGISTRY)

        calls = []
        real_append = registry.append_trial

        def spy(**kwargs):
            calls.append(kwargs)
            return real_append(**kwargs)

        monkeypatch.setattr(registry, "append_trial", spy)

        assert serving.main([]) == 0
        assert weekly.main([]) == 0

        for name in ("nowcaster.pkl", "returns_by_regime.parquet", "asset_returns.parquet"):
            assert (world["platform_dir"] / name).exists(), name
        report = (world["out_dir"] / "reports" / "platform" / "weekly_report.md").read_text()
        assert "## Active Regime" in report
        assert "gates no weight" in report
        assert "EXECUTED book after the 5.0% no-trade band" in report

        assert world["registry_copy"].read_bytes() == copy_bytes_before
        assert registry.total_trial_count(world["registry_copy"]) == count_before
        assert _sha256(_REAL_REGISTRY) == real_sha_before
        assert len(calls) == 1 and calls[0]["path"] is registry.NO_REGISTRY

        # The model is not degenerate on this fixture: its posterior depends on the input,
        # so a scoring bug cannot hide behind a constant output.
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

        model = get_platform_checkpoint_manager().load_model("nowcaster")
        cols = list(model.feature_names_in_)
        complete = world["full"][cols].dropna()
        assert np.unique(model.predict_proba(complete), axis=0).shape[0] > 1

    def test_the_serving_fit_reads_no_holdout_path(self, tmp_path, monkeypatch):
        import trading_crab_lib.platform.honesty.holdout as holdout_mod
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)

        def poisoned(*_a, **_k):
            raise AssertionError("the serving fit touched a holdout path")

        monkeypatch.setattr(holdout_mod, "get_holdout_checkpoint_manager", poisoned)
        monkeypatch.setattr(holdout_mod, "load_full_span", poisoned)

        facts = serving.build_serving_artifacts(world["cfg"])

        assert facts["train_last"] <= _DEV_END - pd.DateOffset(months=_EMBARGO)

        tree = ast.parse(Path(serving.__file__).read_text(encoding="utf-8"))
        forbidden = {"load_full_span", "get_holdout_checkpoint_manager", "HOLDOUT_CHECKPOINT_DIR"}
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.alias):
                used.add(node.name.split(".")[-1])
                if node.asname:
                    used.add(node.asname)
            elif isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
        assert not (used & forbidden), used & forbidden

    def test_no_accuracy_is_computed_logged_or_returned(self, tmp_path_factory, monkeypatch, caplog):
        from trading_crab_lib.platform.honesty import registry
        from trading_crab_lib.platform.report import serving

        # Not tmp_path: its directory is named after this test, and the builder logs the
        # artifact paths it writes, so the scan would match the test's own name.
        world = _serving_world(tmp_path_factory.mktemp("world"), monkeypatch)
        calls = []
        real_append = registry.append_trial
        monkeypatch.setattr(registry, "append_trial", lambda **kw: calls.append(kw) or real_append(**kw))

        with caplog.at_level(logging.INFO):
            facts = serving.build_serving_artifacts(world["cfg"])

        assert caplog.records, "the builder must log what it built"
        assert not [r.getMessage() for r in caplog.records if re.search("accura", r.getMessage(), re.I)]
        bad = re.compile("accura|score", re.I)
        assert not [k for k in facts if bad.search(k)]
        assert len(calls) == 1
        assert not [k for k in calls[0]["metrics"] if bad.search(k)]
        assert facts["registry_row_written"] is False


# ── weekly scores the model's own columns (Task 2) ───────────────────────────


def _trades_section(report: str) -> str:
    start = report.index("## Target vs. Current — Trades Implied")
    rest = report[start + 1:]
    nxt = rest.find("\n## ")
    return report[start:] if nxt < 0 else report[start: start + 1 + nxt]


class TestWeeklyScoresTheModelsColumns:
    def test_report_scores_the_models_columns_not_the_whole_row(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.honesty.holdout import load_full_span
        from trading_crab_lib.platform.report import serving, weekly

        _serving_world(tmp_path, monkeypatch, short_hist=True, starver=True)
        assert serving.main([]) == 0
        model = get_platform_checkpoint_manager().load_model("nowcaster")
        cols = list(model.feature_names_in_)
        full_span = load_full_span("monthly_features")

        # Preconditions that make this fixture discriminating: the latest row carries
        # columns the model was not fit on, so the pre-fix whole-row scorer raises here.
        assert "short_hist" not in cols and "starver" not in cols
        assert {"short_hist", "starver"} <= set(full_span.columns)
        with pytest.raises(ValueError):
            model.predict_proba(full_span.iloc[[-1]])

        assert weekly.main([]) == 0

    def test_selected_columns_equal_cv_safe_active_features(self, tmp_path, monkeypatch):
        """Parity: the builder saves exactly ``_cv_safe_active_features``'s columns, on a
        fixture where BOTH exclusion mechanisms fire."""
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch, short_hist=True, starver=True)
        serving.build_serving_artifacts(world["cfg"])
        model = get_platform_checkpoint_manager().load_model("nowcaster")

        X, y = build_nowcaster_training_set(world["dev"], world["labels"], embargo_months=_EMBARGO)
        window = driver._window_active_features(X, list(X.columns), min_history=_MIN_HISTORY)
        assert "short_hist" not in window, "the window (min_history) rule must fire"
        assert "starver" in window, "starver must be old enough by min_history"
        expected = driver._cv_safe_active_features(
            X, y, list(X.columns), min_history=_MIN_HISTORY, n_splits=_N_SPLITS
        )
        assert "starver" not in expected, "the CV narrowing must fire"
        assert expected != list(X.columns)

        assert list(model.feature_names_in_) == expected

    def test_served_posterior_equals_refit_l2_at_full_dev_history(self, tmp_path, monkeypatch):
        """No skew: the saved model's posterior equals ``_refit_l2``'s, bit for bit."""
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch, short_hist=True, starver=True)
        serving.build_serving_artifacts(world["cfg"])
        model = get_platform_checkpoint_manager().load_model("nowcaster")
        cols = list(model.feature_names_in_)
        dev, labels = world["dev"], world["labels"]

        served = pd.Series(model.predict_proba(dev.iloc[[-1]][cols])[0], index=model.classes_)
        backtest = driver._refit_l2(dev, labels, dev.iloc[[-1]], world["cfg"])

        pd.testing.assert_series_equal(served, backtest, check_exact=True)

    @pytest.mark.parametrize(
        "artifact, filename",
        [
            ("nowcaster", "nowcaster.pkl"),
            ("returns_by_regime", "returns_by_regime.parquet"),
            ("asset_returns", "asset_returns.parquet"),
        ],
    )
    def test_a_missing_serving_artifact_names_the_command_that_builds_it(
        self, tmp_path, monkeypatch, artifact, filename
    ):
        from trading_crab_lib.platform.report import serving, weekly
        from trading_crab_lib.platform.report.serving import SERVING_BUILD_COMMAND

        module_path = SERVING_BUILD_COMMAND.split()[-1]
        assert importlib.util.find_spec(module_path) is not None
        assert serving.__name__ == module_path

        world = _serving_world(tmp_path, monkeypatch)
        assert serving.main([]) == 0
        (world["platform_dir"] / filename).unlink()

        with pytest.raises(FileNotFoundError) as excinfo:
            weekly.main([])

        msg = str(excinfo.value)
        assert SERVING_BUILD_COMMAND in msg
        assert artifact in msg

    def test_a_same_month_rerun_serves_the_same_targets(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch)
        # One account with no holdings file (a neutral, all-cash account — never a crash),
        # so the Trades Implied section carries per-asset rows the comparison can bite on.
        world["cfg"]["report"]["accounts"] = ["serving_rerun_no_holdings_file"]
        assert serving.main([]) == 0
        report_path = world["out_dir"] / "reports" / "platform" / "weekly_report.md"
        cm = get_platform_checkpoint_manager()

        assert weekly.main([]) == 0
        first_book, first_report = cm.load("executed_weights"), report_path.read_text()
        assert weekly.main([]) == 0
        second_book, second_report = cm.load("executed_weights"), report_path.read_text()

        pd.testing.assert_frame_equal(first_book, second_book)
        assert _trades_section(first_report) == _trades_section(second_report)
        assert "- SPY:" in _trades_section(first_report) and "- TLT:" in _trades_section(first_report)


# ── Glenn's 08-12 rulings at serve (plan 08-14): q1-c and q2-ii ──────────────
#
# 08-SERVING.md §2.1 (q1-c): score the newest month observed in EVERY model column, say so on
# the page, step the belief and the band on that month, and refuse (before any save) when it
# is more than MAX_SCORING_LAG_MONTHS = 3 month-ends behind the newest monthly_features row.
# §2.2 (q2-ii): print the exact count of distinct posteriors across every full-span month
# complete in the model columns, directly under the distribution; never withhold on it.

_SCORED_STATE = ("regime_belief", "hysteresis_state", "executed_weights")


def _report_path(world: dict) -> Path:
    return world["out_dir"] / "reports" / "platform" / "weekly_report.md"


def _assert_nothing_written(world: dict) -> None:
    assert not _report_path(world).exists()
    for name in _SCORED_STATE:
        assert not (world["platform_dir"] / f"{name}.parquet").exists(), name


class _ScoringProxy:
    """Delegates to the world's REAL fitted nowcaster and records every frame handed to
    ``predict_proba``. With ``constant`` set it returns that one vector for every row: a
    deliberately input-independent model, built by wrapping the real one (the q2 degenerate
    arm; the real world's posterior varies, which the 08-13 e2e asserts)."""

    def __init__(self, model, constant=None):
        self._model = model
        self._constant = constant
        self.calls: list[pd.DataFrame] = []

    def __getattr__(self, name):
        return getattr(self._model, name)

    def predict_proba(self, X):
        self.calls.append(X.copy())
        proba = self._model.predict_proba(X)
        if self._constant is None:
            return proba
        return np.tile(np.asarray(self._constant, dtype=float), (len(X), 1))


def _install_proxy(monkeypatch, *, constant: bool = False) -> list:
    """Route weekly's nowcaster load through ``_ScoringProxy``; the returned list receives it."""
    from trading_crab_lib.platform.report import weekly

    real_load = weekly._load_serving_artifact
    holder: list[_ScoringProxy] = []

    def load(cm, name, *, model=False):
        obj = real_load(cm, name, model=model)
        if not model:
            return obj
        cols = list(obj.feature_names_in_)
        fixed = None
        if constant:
            first = load_full_span_frame()[cols].dropna(how="any").iloc[[0]]
            fixed = obj.predict_proba(first)[0]
        holder.append(_ScoringProxy(obj, constant=fixed))
        return holder[-1]

    def load_full_span_frame():
        from trading_crab_lib.platform.honesty.holdout import load_full_span

        return load_full_span("monthly_features")

    monkeypatch.setattr(weekly, "_load_serving_artifact", load)
    return holder


class TestQ1cLatestCompleteMonth:
    def test_q1c_scores_the_latest_complete_month_and_names_the_lag(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch, nan_tail=2)
        assert serving.main([]) == 0
        holder = _install_proxy(monkeypatch)
        assert weekly.main([]) == 0

        report = _report_path(world).read_text()
        assert "Scored as of 2021-04-30" in report
        assert "2021-05-31 lacks f_slope" in report and "2021-06-30 lacks f_slope" in report
        assert "Nothing is imputed." in report
        heading = report.index("## Current Regime Distribution")
        assert heading < report.index("Scored as of") < report.index("- regime ")

        # The spy: the one scored row IS the observed row, value for value. An imputation, a
        # forward-fill or the wrong row all fail here.
        proxy = holder[-1]
        cols = list(proxy.feature_names_in_)
        assert pd.isna(world["full"].iloc[-1]["f_slope"]), "precondition: the newest row is ragged"
        scored = [c for c in proxy.calls if len(c) == 1]
        assert len(scored) == 1
        pd.testing.assert_frame_equal(
            scored[0], world["full"].loc[[pd.Timestamp("2021-04-30")], cols], check_exact=True
        )

    @pytest.mark.parametrize("nan_tail, as_of", [(2, "2021-04-30"), (0, "2021-06-30")])
    def test_q1c_belief_and_band_step_on_the_scored_month(self, tmp_path, monkeypatch, nan_tail, as_of):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving, weekly

        _serving_world(tmp_path, monkeypatch, nan_tail=nan_tail)
        assert serving.main([]) == 0
        assert weekly.main([]) == 0

        cm = get_platform_checkpoint_manager()
        assert weekly.load_regime_belief(cm).name == pd.Timestamp(as_of)
        executed = cm.load("executed_weights")
        assert not executed.empty
        assert set(pd.to_datetime(executed["as_of"])) == {pd.Timestamp(as_of)}

    def test_q1c_a_complete_latest_row_is_scored_as_itself_with_no_lag(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch)
        assert serving.main([]) == 0
        assert weekly.main([]) == 0

        report = _report_path(world).read_text()
        assert "Scored as of 2021-06-30" in report
        assert " lacks " not in report
        assert "Nothing is imputed." in report

    def test_q1c_no_complete_month_raises(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch)
        assert serving.main([]) == 0
        # After the build: every full-span month lacks f_slope, so no month is scorable.
        ragged = world["full"].copy()
        ragged["f_slope"] = np.nan
        monkeypatch.setattr(weekly, "load_full_span", lambda name: ragged)

        with pytest.raises(ValueError) as excinfo:
            weekly.main([])

        msg = str(excinfo.value)
        assert "f_slope" in msg and "f_level" in msg, "the message names the model columns"
        assert "impute" in msg
        _assert_nothing_written(world)

    def test_q1c_staleness_cap_refuses_a_month_more_than_3_behind(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        assert weekly.MAX_SCORING_LAG_MONTHS == 3
        world = _serving_world(tmp_path, monkeypatch, nan_tail=4)
        assert serving.main([]) == 0

        with pytest.raises(ValueError) as excinfo:
            weekly.main([])

        msg = str(excinfo.value)
        assert "2021-02-28" in msg, "the latest complete month"
        assert "2021-06-30" in msg, "the newest row"
        assert "4 month-ends" in msg and "MAX_SCORING_LAG_MONTHS = 3" in msg
        _assert_nothing_written(world)

    def test_q1c_staleness_cap_boundary_exactly_3_behind_serves(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch, nan_tail=3)
        assert serving.main([]) == 0
        assert weekly.main([]) == 0
        assert "Scored as of 2021-03-31" in _report_path(world).read_text()

