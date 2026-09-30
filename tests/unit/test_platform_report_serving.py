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

        got, _class_prior = driver._refit_l2(features, labels, row, cfg)

        pd.testing.assert_series_equal(got, expected, check_exact=True)

    def test_fit_l2_nowcaster_returns_the_model_and_its_columns(self):
        features, labels, cfg = _pin_frame()
        model, columns, _class_prior = driver.fit_l2_nowcaster(features, labels, cfg)
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


# ── CR-01: the likelihood's class prior is the prior of the rows the model was fit on ──


def _restricted_label_prior(labels: pd.Series, classes) -> pd.Series:
    """The whole-label prior (the pre-CR-01 rule) restricted to ``classes`` and renormalized."""
    from trading_crab_lib.platform.prediction.regime_filter import unconditional_belief

    states = sorted({int(v) for v in labels.dropna().unique()})
    full = unconditional_belief(labels, state_index=states)
    sub = full.reindex([int(c) for c in classes])
    return sub / sub.sum()


class TestTrainingClassPrior:
    def test_fit_l2_nowcaster_returns_the_prior_of_the_rows_it_fit(self, monkeypatch):
        features, labels, cfg = _pin_frame()
        received = []
        real_fit = driver.fit_nowcaster

        def spy(X, y, **kwargs):
            received.append((X.copy(), y.copy()))
            return real_fit(X, y, **kwargs)

        monkeypatch.setattr(driver, "fit_nowcaster", spy)

        model, active, class_prior = driver.fit_l2_nowcaster(features, labels, cfg)

        assert len(received) == 1
        X_fit, y_fit = received[0]
        # Every row handed to fit_nowcaster is finite, so its own drop keeps them all.
        assert np.isfinite(X_fit.to_numpy(dtype=float)).all()
        expected = y_fit.value_counts(normalize=True).reindex(list(model.classes_))
        np.testing.assert_array_equal(class_prior.to_numpy(dtype=float), expected.to_numpy(dtype=float))
        assert [int(s) for s in class_prior.index] == [int(c) for c in model.classes_]
        assert abs(float(class_prior.sum()) - 1.0) < 1e-12

        # Precondition: the fixture discriminates. The embargo and the late column truncate
        # the block, so the fit's prior is not the whole-label prior on the same classes.
        label_prior = _restricted_label_prior(labels, model.classes_)
        gap = float(np.abs(class_prior.to_numpy(dtype=float) - label_prior.to_numpy(dtype=float)).max())
        assert gap > 1e-3, f"training prior equals the label prior (max abs gap {gap}); fixture cannot discriminate"

    def test_training_class_prior_refuses_labels_that_are_not_the_models_classes(self):
        y = pd.Series([0, 0, 1, 2, 2, 2])
        with pytest.raises(ValueError):
            driver.training_class_prior(y, [0, 1])  # y carries a class the model does not
        with pytest.raises(ValueError):
            driver.training_class_prior(y, [0, 1, 2, 3])  # the model has a class y lacks
        with pytest.raises(ValueError):
            driver.training_class_prior(pd.Series([], dtype=int), [0, 1, 2])
        got = driver.training_class_prior(y, [0, 1, 2])
        assert list(got.index) == [0, 1, 2]
        np.testing.assert_array_equal(got.to_numpy(), np.array([2, 1, 3]) / 6)


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
_RUN_DATE = "2021-07-08"  # the world's "today": a week into July, after the 06-30 row


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
        # 08.1: both world series are real-time (lag 0), so the weekly staleness check judges
        # them against the run date alone.
        "publication_lags": {"f_level": 0, "f_slope": 0},
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
    # The world ends 2021-06-30; the wall clock would call every world series years stale.
    monkeypatch.setattr(weekly, "_run_date", lambda: pd.Timestamp(_RUN_DATE), raising=False)

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
        """No skew: the saved model's posterior equals ``_refit_l2``'s, bit for bit, and the
        persisted ``nowcaster_class_prior`` equals the prior ``_refit_l2`` returns, bit for
        bit (plan 08-17: train/serve parity on the likelihood's prior, not only the posterior)."""
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch, short_hist=True, starver=True)
        serving.build_serving_artifacts(world["cfg"])
        model = get_platform_checkpoint_manager().load_model("nowcaster")
        cols = list(model.feature_names_in_)
        dev, labels = world["dev"], world["labels"]

        served = pd.Series(model.predict_proba(dev.iloc[[-1]][cols])[0], index=model.classes_)
        backtest, backtest_prior = driver._refit_l2(dev, labels, dev.iloc[[-1]], world["cfg"])

        # rtol 1e-9, abs 0 — not bit-for-bit: the two fits see identical values in different
        # memory layouts, and Apple Accelerate's BLAS sums in a different order than OpenBLAS
        # (1-ULP disagreement seen on macOS, 2026-09-29; the G-08-1 class). A real skew (other
        # columns, other embargo, other rows) moves these probabilities by ~1e-2 or more.
        pd.testing.assert_series_equal(served, backtest, check_exact=False, rtol=1e-9, atol=0.0)
        frame = get_platform_checkpoint_manager().load("nowcaster_class_prior")
        persisted = pd.Series(frame["prior"].to_numpy(dtype=float), index=[int(v) for v in frame["state"]])
        assert list(persisted.index) == [int(c) for c in backtest_prior.index]
        np.testing.assert_array_equal(persisted.to_numpy(), backtest_prior.to_numpy(dtype=float))

    @pytest.mark.parametrize(
        "artifact, filename",
        [
            ("nowcaster", "nowcaster.pkl"),
            ("returns_by_regime", "returns_by_regime.parquet"),
            ("asset_returns", "asset_returns.parquet"),
            ("nowcaster_class_prior", "nowcaster_class_prior.parquet"),
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
        # The allocation table's last-week column legitimately differs (n/a, then the book the
        # first run executed); everything else in the section, and the table's targets, do not.
        def without_table(section: str) -> str:
            return "\n".join(ln for ln in section.splitlines() if not ln.startswith("|"))

        assert without_table(_trades_section(first_report)) == without_table(_trades_section(second_report))
        assert [c[2] for c in _allocation_rows(first_report).values()] == [
            c[2] for c in _allocation_rows(second_report).values()
        ]
        assert "- SPY:" in _trades_section(first_report) and "- TLT:" in _trades_section(first_report)


# ── CR-01 at serve: the fit's training prior travels with the model ──────────


class TestTheServedClassPrior:
    def test_the_builder_persists_the_fits_training_prior_beside_the_model(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving

        world = _serving_world(tmp_path, monkeypatch)
        facts = serving.build_serving_artifacts(world["cfg"])
        cm = get_platform_checkpoint_manager()
        model = cm.load_model("nowcaster")
        frame = cm.load("nowcaster_class_prior")
        persisted = pd.Series(frame["prior"].to_numpy(dtype=float), index=[int(s) for s in frame["state"]])

        assert list(persisted.index) == [int(c) for c in model.classes_]
        _, _, fit_prior = driver.fit_l2_nowcaster(world["dev"], world["labels"], world["cfg"])
        np.testing.assert_array_equal(persisted.to_numpy(), fit_prior.to_numpy(dtype=float))
        # Cross-check the record's re-derivation (IN-04 stays out of scope; watched here).
        block = serving._training_block(world["dev"], world["labels"], list(model.feature_names_in_), world["cfg"])
        np.testing.assert_array_equal(
            persisted.to_numpy(), block.value_counts(normalize=True).reindex(persisted.index).to_numpy(dtype=float)
        )
        assert facts["class_prior"] == {int(k): float(v) for k, v in persisted.items()}

    def test_weekly_divides_by_the_served_training_prior_not_the_label_prior(self, tmp_path, monkeypatch):
        """The discriminating end-to-end: serving.main, then weekly.main, cold start. The
        filter's 4th argument is the persisted training prior; its start is the label
        distribution over all K states (two roles, two rules)."""
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.prediction.regime_filter import (
            filter_step,
            transition_matrix_for,
            unconditional_belief,
        )
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch)
        assert serving.main([]) == 0
        cm = get_platform_checkpoint_manager()
        model = cm.load_model("nowcaster")
        frame = cm.load("nowcaster_class_prior")
        served_prior = pd.Series(frame["prior"].to_numpy(dtype=float), index=[int(s) for s in frame["state"]])
        labels = world["labels"]
        label_prior = _restricted_label_prior(labels, model.classes_)
        gap = float((served_prior - label_prior.reindex(served_prior.index)).abs().max())
        print(f"served training prior vs restricted label prior: max abs diff {gap!r}")
        assert gap > 1e-6, f"precondition: the two priors differ by only {gap}; the arm cannot discriminate"

        calls = []
        real_filter = weekly.filter_step
        monkeypatch.setattr(weekly, "filter_step", lambda *a: calls.append(a) or real_filter(*a))
        assert weekly.main([]) == 0

        assert len(calls) == 1
        start, transition, posterior, class_prior = calls[0]
        states = list(range(_K))
        pd.testing.assert_series_equal(
            pd.Series(class_prior, dtype=float), served_prior, check_names=False, check_exact=True
        )
        pd.testing.assert_series_equal(start, unconditional_belief(labels, state_index=states), check_exact=True)

        cols = list(model.feature_names_in_)
        row = world["full"][cols].dropna(how="any").iloc[[-1]]
        own_posterior = pd.Series(model.predict_proba(row)[0], index=model.classes_)
        pd.testing.assert_series_equal(posterior, own_posterior, check_exact=True)

        expected = filter_step(
            unconditional_belief(labels, state_index=states),
            transition_matrix_for(labels, state_index=states),
            own_posterior,
            served_prior,
        )
        pd.testing.assert_series_equal(weekly.load_regime_belief(cm), expected, check_names=False, check_exact=True)

    def test_a_class_prior_that_is_not_the_models_classes_is_refused_before_any_save(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving, weekly
        from trading_crab_lib.platform.report.serving import SERVING_BUILD_COMMAND

        world = _serving_world(tmp_path, monkeypatch)
        assert serving.main([]) == 0
        cm = get_platform_checkpoint_manager()
        classes = {int(c) for c in cm.load_model("nowcaster").classes_}
        # A prior from another build: a different state set (a superset, so the filter itself
        # would silently accept it — only the served-prior check can refuse it).
        other = sorted(classes | {max(classes) + 1})
        cm.save(pd.DataFrame({"state": other, "prior": [1.0 / len(other)] * len(other)}), "nowcaster_class_prior")

        with pytest.raises(ValueError) as excinfo:
            weekly.main([])

        msg = str(excinfo.value)
        assert SERVING_BUILD_COMMAND in msg
        assert "nowcaster_class_prior" in msg
        _assert_nothing_written(world)


# ── CR-01 on the REAL served model (tracked dev data, session copy, read-only) ──
#
# 08-SERVING.md §1 items 3, 5, 6 and 08-VERIFICATION.md gap 1 recorded these numbers. The
# session fixture in tests/conftest.py serves COPIES of data/checkpoints/platform, so this reads
# the tracked data without touching it. No holdout path is opened; nothing is saved.

#: 08-SERVING.md §1 item 6: the served posterior on the last complete dev month, exact floats.
_RECORDED_POSTERIOR = [0.41800356506238856, 0.5639928698752229, 0.018003565062388593]
#: 08-SERVING.md §1 item 3: dev label counts over 695 months, 1963-02-28 -> 2020-12-31.
_RECORDED_LABEL_COUNTS = {0: 40, 1: 228, 2: 71, 3: 200, 4: 84, 5: 72}
#: 08-SERVING.md §1 item 5: the fit's training block, 153 rows, 2007-04-30 -> 2019-12-31.
_RECORDED_TRAIN_COUNTS = {0: 11, 3: 137, 4: 5}


class TestTheServedModelOnTheTrackedData:
    def test_the_training_prior_flips_state_3s_evidence_and_the_served_belief(self, monkeypatch):
        import trading_crab_lib.platform.honesty.holdout as holdout_mod
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.config import load_platform_config
        from trading_crab_lib.platform.honesty.holdout import DEFAULT_HOLDOUT_CUTOFF, split_by_holdout_boundary
        from trading_crab_lib.platform.prediction.regime_filter import transition_matrix_for, unconditional_belief
        from trading_crab_lib.platform.report import serving, weekly

        def poisoned(*_a, **_k):
            raise AssertionError("the real-data CR-01 test touched a holdout path")

        monkeypatch.setattr(holdout_mod, "get_holdout_checkpoint_manager", poisoned)
        monkeypatch.setattr(holdout_mod, "load_full_span", poisoned)
        monkeypatch.setattr(weekly, "load_full_span", poisoned)

        cutoff = pd.Timestamp(DEFAULT_HOLDOUT_CUTOFF)
        cm = get_platform_checkpoint_manager()
        dev, _ = split_by_holdout_boundary(cm.load("monthly_features"), cutoff=DEFAULT_HOLDOUT_CUTOFF)
        label_frame, _ = split_by_holdout_boundary(cm.load("regime_labels"), cutoff=DEFAULT_HOLDOUT_CUTOFF)
        labels = label_frame["state"]
        assert dev.index.max() <= cutoff and labels.index.max() <= cutoff
        cfg = load_platform_config()
        states = list(range(int(cfg["labeling"]["K"])))
        assert states == [0, 1, 2, 3, 4, 5]

        # The fit's own training prior: the rows it was fit on, over classes_ [0, 3, 4].
        model, active, prior = driver.fit_l2_nowcaster(dev, labels, cfg)
        assert [int(c) for c in model.classes_] == [0, 3, 4]
        assert list(prior.index) == [0, 3, 4]
        assert prior.to_dict() == pytest.approx({k: v / 153 for k, v in _RECORDED_TRAIN_COUNTS.items()}, rel=1e-12)
        block = serving._training_block(dev, labels, active, cfg)
        assert len(block) == 153
        assert block.index.min() == pd.Timestamp("2007-04-30") and block.index.max() == pd.Timestamp("2019-12-31")

        # The whole-label prior (the pre-CR-01 rule).
        assert len(labels) == 695
        label_prior = unconditional_belief(labels, state_index=states)
        assert label_prior.to_dict() == pytest.approx({k: v / 695 for k, v in _RECORDED_LABEL_COUNTS.items()}, rel=1e-12)

        # The served posterior on the last dev row complete in the model's columns. A fitted
        # float: compared by 08-11's portable rule (rel 1e-9, abs 0), never == (G-08-1).
        row = dev[active].dropna(how="any").iloc[[-1]]
        post = pd.Series(model.predict_proba(row)[0], index=[int(c) for c in model.classes_])
        assert list(post) == pytest.approx(_RECORDED_POSTERIOR, rel=1e-9, abs=0)

        # State 3's likelihood ratio: evidence AGAINST under the training prior, FOR under
        # the label prior.
        ratio_train = post[3] / prior[3]
        ratio_label = post[3] / label_prior[3]
        assert ratio_train < 1.0 and ratio_train == pytest.approx(0.63, abs=5e-3)
        assert ratio_label > 1.0 and ratio_label == pytest.approx(1.96, abs=5e-3)

        # The new rule, through the served path's own function: cold start, one step.
        belief = weekly.advance_regime_belief(
            None, labels, post, class_prior=prior, state_index=states, as_of=labels.index[-1]
        )
        assert int(belief.idxmax()) == 0
        assert belief[0] == pytest.approx(0.300, abs=5e-3)
        assert belief[1] == pytest.approx(0.293, abs=5e-3)
        assert belief[3] == pytest.approx(0.163, abs=5e-3)

        # The old rule by explicit arithmetic (not through filter_step): normalize((pi_0 A) x r_old),
        # r_old = post / label_prior on classes_ and 1.0 elsewhere.
        pi0 = label_prior.to_numpy(dtype=float)
        a = transition_matrix_for(labels, state_index=states).to_numpy(dtype=float)
        r_old = np.ones(len(states))
        for state in post.index:
            r_old[state] = post[state] / label_prior[state]
        old = (pi0 @ a) * r_old
        old = old / old.sum()
        assert int(np.argmax(old)) == 3
        assert old[3] == pytest.approx(0.369, abs=5e-3)


# ── Glenn's 08-12 rulings at serve (plan 08-14): q1-c and q2-ii ──────────────
#
# 08-SERVING.md §2.1 (q1-c): score the newest month observed in EVERY model column, say so on
# the page, and step the belief and the band on that month. Its 3-month-end refusal
# (MAX_SCORING_LAG_MONTHS) was superseded by DECISIONS A-12 (plan 08.1-02): a series later than
# the run date allows is named in a STALE DATA banner and the report still writes.
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


def _spy_on_every_fit(monkeypatch) -> list:
    """Record any fit reached after the serving build: the disclosure only LOOKS."""
    from sklearn.calibration import CalibratedClassifierCV

    import trading_crab_lib.platform.prediction.nowcaster as nowcaster_mod

    calls: list[str] = []
    real_cal_fit = CalibratedClassifierCV.fit

    def cal_fit(self, *a, **k):
        calls.append("CalibratedClassifierCV.fit")
        return real_cal_fit(self, *a, **k)

    monkeypatch.setattr(CalibratedClassifierCV, "fit", cal_fit)
    for module, name in ((nowcaster_mod, "fit_nowcaster"), (driver, "fit_nowcaster"), (driver, "fit_l2_nowcaster")):
        real = getattr(module, name)
        monkeypatch.setattr(module, name, lambda *a, _n=name, _r=real, **k: calls.append(_n) or _r(*a, **k))
    return calls


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
        assert "STALE DATA" not in report
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

    def test_a12_a_stale_series_is_named_in_a_banner_and_the_report_still_writes(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch, nan_tail=4)
        assert serving.main([]) == 0
        assert weekly.main([]) == 0  # HEAD raises: 4 month-ends behind

        report = _report_path(world).read_text()
        assert "## STALE DATA" in report
        h1, banner = report.index("# Trading-Crab Platform Weekly Report"), report.index("## STALE DATA")
        dist = report.index("## Current Regime Distribution")
        assert h1 < banner < dist
        section = report[banner:dist]
        assert "f_slope: 4 months late" in section
        assert "last value 2021-02-28" in section and "expected through 2021-06-30" in section
        assert "f_level" not in section, "a fresh series is never named"
        assert "imputes nothing" in section
        assert "Scored as of 2021-02-28" in report
        assert "Nothing is imputed." in report
        assert not hasattr(weekly, "MAX_SCORING_LAG_MONTHS"), "A-12 removed the 3-month-end cap"

    def test_a12_grace_window_a_series_inside_it_is_not_stale(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch, nan_tail=1)  # f_slope ends 2021-05-31
        assert serving.main([]) == 0

        # Inside the 7-day grace: (07-05 minus 7d) is 06-28, so only May must exist.
        monkeypatch.setattr(weekly, "_run_date", lambda: pd.Timestamp("2021-07-05"))
        assert weekly.main([]) == 0
        assert "STALE DATA" not in _report_path(world).read_text()

        # Past it: (07-08 minus 7d) is 07-01, so June must exist and f_slope lacks it.
        monkeypatch.setattr(weekly, "_run_date", lambda: pd.Timestamp("2021-07-08"))
        assert weekly.main([]) == 0
        report = _report_path(world).read_text()
        section = report[report.index("## STALE DATA"): report.index("## Current Regime Distribution")]
        assert "f_slope: 1 month late" in section and "f_level" not in section

    def test_q1c_staleness_cap_boundary_exactly_3_behind_serves(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch, nan_tail=3)
        assert serving.main([]) == 0
        assert weekly.main([]) == 0
        assert "Scored as of 2021-03-31" in _report_path(world).read_text()

class TestQ2iiDistinctPosteriorDisclosure:
    _SENTENCE = "The distribution above does not depend on the features: it is the same every week."

    def test_q2ii_the_page_states_the_distinct_count_and_window(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
        from trading_crab_lib.platform.report import serving, weekly

        # nan_tail=2: the newest two rows are NOT complete, so "only complete rows" bites.
        world = _serving_world(tmp_path, monkeypatch, nan_tail=2)
        assert serving.main([]) == 0
        model = get_platform_checkpoint_manager().load_model("nowcaster")
        cols = list(model.feature_names_in_)
        complete = world["full"][cols].dropna(how="any")
        n = np.unique(model.predict_proba(complete), axis=0).shape[0]
        assert n > 1, "precondition: the world's posterior varies"

        fits = _spy_on_every_fit(monkeypatch)
        holder = _install_proxy(monkeypatch)
        assert weekly.main([]) == 0

        report = _report_path(world).read_text()
        first, last = complete.index[0].date().isoformat(), complete.index[-1].date().isoformat()
        line = f"{n} distinct posterior vectors across {len(complete)} complete months ({first} → {last})"
        assert line in report
        assert self._SENTENCE not in report
        assert "does not depend" not in report
        dist = report.index("## Current Regime Distribution")
        assert report.index("- regime ", dist) < report.index(line) < report.index("## Filtered Regime Belief")

        counted = [c for c in holder[-1].calls if len(c) > 1]
        assert len(counted) == 1
        # check_freq=False: index freq is metadata the parquet round trip sets; values are exact.
        pd.testing.assert_frame_equal(counted[0], complete, check_exact=True, check_freq=False)
        assert fits == [], f"the disclosure fitted something: {fits}"

    def test_q2ii_a_constant_posterior_is_disclosed_as_such(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch, nan_tail=2)
        assert serving.main([]) == 0
        complete = world["full"][["f_level", "f_slope"]].dropna(how="any")

        fits = _spy_on_every_fit(monkeypatch)
        holder = _install_proxy(monkeypatch, constant=True)
        assert weekly.main([]) == 0  # disclosed, NOT withheld (q2-iii was not chosen)

        report = _report_path(world).read_text()
        first, last = complete.index[0].date().isoformat(), complete.index[-1].date().isoformat()
        assert f"1 distinct posterior vector across {len(complete)} complete months ({first} → {last})" in report
        assert self._SENTENCE in report
        counted = [c for c in holder[-1].calls if len(c) > 1]
        assert len(counted) == 1
        assert list(counted[0].index) == list(complete.index)
        assert fits == [], f"the disclosure fitted something: {fits}"


# ── the always-printed target allocation table, end to end (08.1, D-09) ──────


def _allocation_rows(report: str) -> dict[str, list[str]]:
    """ticker -> [class, ticker, target, last week, change] from the rendered table."""
    start = report.index("### Target allocation")
    lines = [ln for ln in report[start:].splitlines() if ln.startswith("|")][2:]
    cells = [[c.strip() for c in ln.strip("|").split("|")] for ln in lines]
    return {row[1]: row for row in cells}


class TestAllocationTableEndToEnd:
    def test_no_account_page_prints_the_table_and_a_same_month_rerun_shows_no_change(self, tmp_path, monkeypatch):
        from trading_crab_lib.platform.report import serving, weekly

        world = _serving_world(tmp_path, monkeypatch)
        assert world["cfg"]["report"]["accounts"] == [], "precondition: no account is configured"
        assert serving.main([]) == 0

        assert weekly.main([]) == 0
        first = _report_path(world).read_text()
        assert "### Target allocation" in first and "### Account:" not in first
        assert "| Class | Ticker | Target % | Last week % | Change |" in first
        rows = _allocation_rows(first)
        assert {"SPY", "TLT"} <= set(rows)
        for ticker in ("SPY", "TLT"):
            assert rows[ticker][3:] == ["n/a", "n/a"], rows[ticker]  # no executed book before this run
        assert rows["SPY"][0] == "equities" and rows["TLT"][0] == "long_duration"

        assert weekly.main([]) == 0
        second = _report_path(world).read_text()
        rows = _allocation_rows(second)
        for cells in rows.values():
            assert cells[4] == "+0.0 pp", cells
            assert cells[2] == cells[3], "last week's column is the book this month already executed"
        # Targets are unchanged by the re-run (same held book, same belief).
        assert [c[2] for c in _allocation_rows(first).values()] == [c[2] for c in rows.values()]
        # The executed book (the table's target column) is the saved checkpoint's book.
        from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager

        saved = get_platform_checkpoint_manager().load("executed_weights")
        saved = saved[(saved["basis"] == "executed") & saved["asset"].notna()].set_index("asset")["weight"]
        assert rows["SPY"][2] == f"{float(saved['SPY']):.1%}"

