"""Point-in-time guarantee for the monthly spine (phase 08.1, D-01..D-03).

A feature at month t may depend only on raw values published by the end of t.
The value of series s for reference month m is published at m + FLOOR[s]
(measured release schedules, 08.1-RESEARCH.md §1). ``FLOOR`` lives HERE, not in
config: if the config were the truth, lowering a config lag would lower the
truth with it and this file could only ever agree.

The proof is truncation-by-poisoning through the REAL ``build_monthly_spine``
(only the three network fetchers are mocked): multiply every raw value not yet
published at t, rebuild, and require every ``monthly_features`` cell at rows
<= t to be unchanged. The detector self-test runs the same harness with
``div_yield`` lagged 0 and requires it to go red — the permanent mutation proof
that the harness can fail.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_crab_lib.platform import splice, transforms_monthly
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.ingestion import publication_lags as pl

START, END = "1962-01-01", "1976-12-31"
IDX = pd.date_range(START, END, freq="ME")

# Independent publication floor, in month-ends after the reference month.
# Everything not listed is observed by its own month's close (0).
FLOOR: dict[str, int] = {"fred_m2sl": 1, "fred_totalsl": 2, "div_yield": 3, "sentinel": 2}
GDP_FALLBACK_FLOOR = 3

TRUNCATION_POINTS = [pd.Timestamp("1972-01-31"), pd.Timestamp("1974-06-30"), pd.Timestamp("1976-12-31")]

# Not the plan's 1e6: splice.assert_yield_units_plausible raises on any yield
# above 100% annualized, so a 1e6 poison on fred_gs10 would crash the build
# rather than test it. Any factor != 1 is detected by the exact (rel 1e-9)
# comparison; 1.5 keeps every poisoned yield plausible.
POISON = 1.5

AGENCY = ("fred_gdp", "fred_cpi", "fred_unrate", "fred_indpro", "fred_payems")


# ── Synthetic world ─────────────────────────────────────────────────────────


def _price_ingest_tickers(cfg: dict) -> list[str]:
    uni = cfg["universe"]
    skip = set(uni.get("no_price_ingest", []))
    names = [*uni.get("satellites", []), *uni.get("holdings", []), *uni.get("watchlist", [])]
    return [t for t in dict.fromkeys(names) if t not in skip]


def _macro_names(cfg: dict) -> list[str]:
    return [
        *(meta["name"] for meta in cfg["fred_monthly"]["series"].values()),
        *(row[0] for row in cfg["multpl_monthly"]["datasets"]),
        *(entry["name"] for entry in cfg["macrotrends_monthly"]["series"]),
        *(entry["name"] for entry in (cfg.get("worldbank_monthly") or {}).get("series", [])),
        *(meta["name"] for meta in cfg["index_monthly"].values()),
    ]


def _walk(rng: np.random.Generator, level: float, n: int) -> np.ndarray:
    return level * np.exp(np.cumsum(rng.normal(0.0, 0.03, n)))


def _synthetic_macro(cfg: dict, rng: np.random.Generator, idx: pd.DatetimeIndex = IDX) -> pd.DataFrame:
    cols = {}
    for name in [*_macro_names(cfg), "sentinel"]:
        # FRED yields are percent (~5); multpl div_yield is a decimal (~0.03).
        level = 0.03 if name == "div_yield" else 5.0 if name.startswith("fred_") else 100.0
        cols[name] = _walk(rng, level, len(idx))
    return pd.DataFrame(cols, index=idx)


def _synthetic_prices(cfg: dict, rng: np.random.Generator, idx: pd.DatetimeIndex = IDX) -> pd.DataFrame:
    return pd.DataFrame({t: _walk(rng, 50.0, len(idx)) for t in _price_ingest_tickers(cfg)}, index=idx)


def _synthetic_vintages(rng: np.random.Generator, end: str = END) -> dict[str, pd.DataFrame]:
    """Monthly agency releases, each reference month published 45 days after it
    starts. Identical in every build — the agency path is not under test here."""
    refs = pd.date_range("1960-01-01", end, freq="MS")
    return {
        name: pd.DataFrame(
            {
                "realtime_start": refs + pd.Timedelta(days=45),
                "date": refs,
                "value": _walk(rng, 100.0, len(refs)),
            }
        )
        for name in AGENCY
    }


def _pit_cfg() -> dict:
    cfg = copy.deepcopy(load_platform_config())
    cfg["data"]["start_date"], cfg["data"]["end_date"] = START, END
    cfg["publication_lags"]["sentinel"] = FLOOR["sentinel"]
    # 08.4 (2026-10-05): synthetic worlds omit sources by design; the gate is tested in test_platform_build_fail_loud.py
    cfg["build"] = {"fail_loud": False}
    return cfg


def _poison(frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Multiply every value whose publication month (m + FLOOR) is after t."""
    out = frame.copy()
    for col in out.columns:
        floor = FLOOR.get(col, 0)
        published = out.index + pd.offsets.MonthEnd(floor) if floor else out.index
        out.loc[published > t, col] *= POISON
    return out


def _build(cfg: dict, macro: pd.DataFrame, prices: pd.DataFrame, vintages: dict, root: Path) -> pd.DataFrame:
    """The real build_monthly_spine with only the network fetchers mocked, writing
    into its own checkpoint and holdout directories."""
    with (
        patch("trading_crab_lib.platform.ingestion.macro_monthly.fetch_macro_monthly", return_value=macro),
        patch(
            "trading_crab_lib.platform.ingestion.prices_daily.fetch_universe_prices",
            return_value=(pd.DataFrame(), prices),
        ),
        patch("trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages", return_value=vintages),
        patch("trading_crab_lib.platform.checkpoints.PLATFORM_CHECKPOINT_DIR", root / "platform"),
        patch("trading_crab_lib.platform.honesty.holdout.HOLDOUT_CHECKPOINT_DIR", root / "holdout"),
    ):
        return transforms_monthly.build_monthly_spine(cfg)


def _differing_cells(clean: pd.DataFrame, other: pd.DataFrame, t: pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
    """Cells at rows <= t that differ (H-10: rel 1e-9, abs 0, NaN == NaN)."""
    a, b = clean.loc[:t], other.loc[:t]
    assert list(a.columns) == list(b.columns)
    assert a.index.equals(b.index)
    av, bv = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
    same = (np.isnan(av) & np.isnan(bv)) | np.isclose(av, bv, rtol=1e-9, atol=0.0)
    rows, cols = np.nonzero(~same)
    return [(a.columns[j], a.index[i]) for i, j in zip(rows, cols)]


@pytest.fixture(scope="module")
def world():
    cfg = _pit_cfg()
    rng = np.random.default_rng(81)
    return cfg, _synthetic_macro(cfg, rng), _synthetic_prices(cfg, rng), _synthetic_vintages(rng)


@pytest.fixture(scope="module")
def clean_features(world, tmp_path_factory):
    cfg, macro, prices, vintages = world
    return _build(cfg, macro, prices, vintages, tmp_path_factory.mktemp("clean"))


# ── 1. Truncation proof ─────────────────────────────────────────────────────


class TestTruncation:
    @pytest.mark.parametrize("t", TRUNCATION_POINTS, ids=lambda t: str(t.date()))
    def test_no_feature_at_or_before_t_moves_when_unpublished_values_are_poisoned(
        self, t, world, clean_features, tmp_path
    ):
        cfg, macro, prices, vintages = world
        poisoned = _build(cfg, _poison(macro, t), _poison(prices, t), vintages, tmp_path)

        diffs = _differing_cells(clean_features, poisoned, t)
        assert diffs == [], f"look-ahead: {len(diffs)} cell(s) at rows <= {t.date()} moved, first {diffs[:10]}"
        # The poison did land after t — otherwise the equality above is vacuous.
        assert _differing_cells(clean_features, poisoned, IDX[-1]) or t == IDX[-1]


# ── 2. Detector self-test (the permanent mutation proof) ────────────────────


class TestDetectorCanFail:
    T = pd.Timestamp("1974-06-30")

    def test_harness_is_not_vacuous(self, clean_features):
        # The columns the proof expects to go red must exist, or red is impossible.
        for col in ("div_yield", "equities_tr", "trailing_return_1m", "sentinel"):
            assert col in clean_features.columns, col
        # A centred window reads future rows, which would turn the proof red on
        # CORRECT lags — the harness is only meaningful over causal builders.
        for module in (transforms_monthly, splice):
            assert "center=True" not in inspect.getsource(module), module.__name__

    def test_div_yield_lag_zero_is_detected(self, world, tmp_path):
        cfg, macro, prices, vintages = world
        mutated = copy.deepcopy(cfg)
        mutated["publication_lags"]["div_yield"] = 0

        clean = _build(mutated, macro, prices, vintages, tmp_path / "clean")
        poisoned = _build(mutated, _poison(macro, self.T), _poison(prices, self.T), vintages, tmp_path / "poisoned")
        diffs = set(_differing_cells(clean, poisoned, self.T))

        assert ("div_yield", self.T) in diffs
        assert ("equities_tr", self.T) in diffs
        assert ("trailing_return_1m", self.T) in diffs

    def test_bypassing_the_lag_is_detected(self, world, tmp_path):
        cfg, macro, prices, vintages = world
        identity = lambda frame, cfg: frame.copy()  # noqa: E731
        with patch.object(transforms_monthly, "apply_publication_lags", identity):
            clean = _build(cfg, macro, prices, vintages, tmp_path / "clean")
            poisoned = _build(cfg, _poison(macro, self.T), _poison(prices, self.T), vintages, tmp_path / "poisoned")
        cols = {c for c, _ in _differing_cells(clean, poisoned, self.T)}
        assert {"div_yield", "fred_m2sl", "fred_totalsl", "sentinel"} <= cols


# ── 3. Floor ────────────────────────────────────────────────────────────────


class TestFloor:
    def test_config_lags_are_at_least_the_measured_floor(self):
        table = pl.lag_table(load_platform_config())
        for name, floor in FLOOR.items():
            if name == "sentinel":
                continue
            assert isinstance(table[name], int) and table[name] >= floor, (name, table[name], floor)

    def test_gdp_pre_vintage_fallback_is_at_least_three(self):
        table = pl.lag_table(load_platform_config())
        assert table["fred_gdp"]["vintage"] is True
        assert table["fred_gdp"]["fallback_months"] >= GDP_FALLBACK_FLOOR


# ── 4. Completeness ─────────────────────────────────────────────────────────


class TestCompleteness:
    def test_every_emittable_column_has_exactly_the_right_kind_of_entry(self):
        cfg = load_platform_config()
        table = pl.lag_table(cfg)
        ingest = set(_macro_names(cfg)) | set(_price_ingest_tickers(cfg))
        research = {params["research_name"] for params in cfg["splice"].values()}
        agency = {meta["name"] for meta in cfg["fred_vintage"]["series"].values()}

        assert set(table) == ingest | research | agency, (
            f"missing {sorted((ingest | research | agency) - set(table))}, "
            f"stale {sorted(set(table) - (ingest | research | agency))}"
        )
        assert all(isinstance(table[n], int) for n in ingest), sorted(n for n in ingest if not isinstance(table[n], int))
        assert all(table[n] == pl.DERIVED for n in research)
        assert all(isinstance(table[n], dict) and table[n]["vintage"] for n in agency)


# ── apply_publication_lags / lag_table contracts ────────────────────────────


def _mini_cfg(**lags) -> dict:
    return {"data": {"monthly_freq": "ME"}, "publication_lags": lags}


class TestApplyPublicationLags:
    def test_shifts_timestamps_not_positions_and_never_mutates(self):
        idx = pd.DatetimeIndex(["2020-01-31", "2020-02-29", "2020-04-30"])  # March missing
        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}, index=idx)
        before = frame.copy()

        out = pl.apply_publication_lags(frame, _mini_cfg(a=2, b=0))

        pd.testing.assert_frame_equal(frame, before)
        assert out.loc["2020-03-31", "a"] == 1.0  # Jan -> Mar
        assert out.loc["2020-04-30", "a"] == 2.0  # Feb -> Apr
        assert out.loc["2020-06-30", "a"] == 3.0  # Apr -> Jun, past the input's end
        assert out["b"].dropna().to_dict() == before["b"].to_dict()

    @pytest.mark.parametrize(
        "entry, match",
        [(None, "no publication_lags entry"), ("derived", "derived or vintage"),
         ({"vintage": True, "fallback_months": 1}, "derived or vintage")],
    )
    def test_refuses_unlisted_derived_and_vintage_columns(self, entry, match):
        frame = pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex(["2020-01-31"]))
        cfg = _mini_cfg() if entry is None else _mini_cfg(x=entry)
        with pytest.raises(ValueError, match=match):
            pl.apply_publication_lags(frame, cfg)

    @pytest.mark.parametrize(
        "entry",
        [-1, 1.5, True, "3", "vintage", {"vintage": True}, {"vintage": True, "fallback_months": 0},
         {"vintage": False, "fallback_months": 1}, {"vintage": True, "fallback_months": 1, "extra": 1}],
    )
    def test_lag_table_rejects_malformed_entries_naming_the_column(self, entry):
        with pytest.raises(ValueError, match=r"publication_lags\.bad_col"):
            pl.lag_table(_mini_cfg(bad_col=entry))


# ── 5. ALFRED: vintage era and pre-vintage fallback are both point-in-time ──


def _quarterly_gdp_releases(first_vintage_year: int = 1975) -> tuple[pd.DataFrame, pd.Series]:
    """GDP-shaped synthetic ALFRED history: quarterly reference periods (dated at
    the quarter start, as GDPC1 is), each released 30 days after its quarter ends.
    The archive only begins in *first_vintage_year*: its first vintage carries
    every earlier reference period at once, as GDPC1's 1991-12-04 vintage does.

    Returns (all_releases, release date per reference period)."""
    refs = pd.date_range("1955-01-01", "1980-10-01", freq="QS")
    released = pd.Series(refs + pd.offsets.QuarterEnd(0) + pd.Timedelta(days=30), index=refs)
    first_vintage = released[released >= pd.Timestamp(f"{first_vintage_year}-01-01")].iloc[0]
    realtime_start = released.where(released >= first_vintage, first_vintage)
    values = 100.0 * 1.01 ** np.arange(len(refs))
    frame = pd.DataFrame({"realtime_start": realtime_start.to_numpy(), "date": refs, "value": values})
    return frame, released


class TestAlfredPointInTime:
    def test_value_at_every_month_end_is_the_latest_release_known_by_then_in_both_eras(self):
        releases, released = _quarterly_gdp_releases()
        spine = pd.date_range("1962-01-01", "1980-12-31", freq="ME")
        cfg = load_platform_config()

        with patch(
            "trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages",
            return_value={"fred_gdp": releases},
        ):
            aligned = transforms_monthly.align_agency_monthly(spine, cfg)["fred_gdp"]

        value_of = releases.set_index("date")["value"]
        wrong = []
        for t in spine[spine >= "1970-01-01"]:
            latest_ref = released[released <= t].index.max()
            expected = value_of[latest_ref]
            if not np.isclose(aligned[t], expected, rtol=1e-9, atol=0.0):
                wrong.append((str(t.date()), aligned[t], expected))
        assert wrong == [], f"{len(wrong)} month-ends show a value not yet released (or a stale one): {wrong[:6]}"

    def test_fallback_lag_is_a_parameter_applied_on_the_reference_month_grid(self):
        releases, _ = _quarterly_gdp_releases()
        spine = pd.date_range("1970-01-01", "1970-12-31", freq="ME")
        q1 = releases.set_index("date").loc["1970-01-01", "value"]

        lag3 = transforms_monthly._shift_fallback_series(releases, spine, "ME", lag=3)
        lag1 = transforms_monthly._shift_fallback_series(releases, spine, "ME")

        assert lag3["1970-04-30"] == q1 and lag3["1970-03-31"] != q1  # Q1 visible from Apr 30
        assert lag1["1970-02-28"] == q1  # the default keeps the old behaviour for callers

    def test_agency_series_without_a_vintage_entry_raises(self):
        releases, _ = _quarterly_gdp_releases()
        cfg = copy.deepcopy(load_platform_config())
        del cfg["publication_lags"]["fred_gdp"]
        with (
            patch(
                "trading_crab_lib.platform.ingestion.alfred.fetch_all_vintages",
                return_value={"fred_gdp": releases},
            ),
            pytest.raises(ValueError, match="fred_gdp"),
        ):
            transforms_monthly.align_agency_monthly(pd.date_range("1970-01-01", periods=3, freq="ME"), cfg)


# ── 6. Index end: the running month is never a row ─────────────────────────


class TestLastCompleteMonthEnd:
    @pytest.mark.parametrize(
        "today, expected",
        [("2026-09-30", "2026-08-31"), ("2026-10-01", "2026-09-30"), ("2026-10-31", "2026-09-30"),
         ("2026-09-30 17:45", "2026-08-31")],
    )
    def test_returns_the_last_complete_month_end(self, today, expected):
        assert transforms_monthly.last_complete_month_end(pd.Timestamp(today)) == pd.Timestamp(expected)

    def test_a_last_day_of_month_build_writes_no_row_for_that_month(self, tmp_path):
        from datetime import date as real_date

        class _LastDayOfSeptember(real_date):
            @classmethod
            def today(cls):
                return cls(2026, 9, 30)

        cfg = _pit_cfg()
        cfg["data"]["start_date"], cfg["data"]["end_date"] = "2024-01-01", None
        idx = pd.date_range("2024-01-31", "2026-09-30", freq="ME")  # includes the running month
        rng = np.random.default_rng(3)
        with patch.object(transforms_monthly, "date", _LastDayOfSeptember):
            out = _build(
                cfg, _synthetic_macro(cfg, rng, idx), _synthetic_prices(cfg, rng, idx),
                _synthetic_vintages(rng, "2026-09-30"), tmp_path,
            )
        assert out.index[-1] == pd.Timestamp("2026-08-31")
        assert pd.Timestamp("2026-09-30") not in out.index


# ── 7. Merge guard: never merge lagged data onto an unlagged monthly_raw ────


MARKER_IDX = pd.date_range("1970-01-31", "1972-12-31", freq="ME")


@pytest.fixture
def small_world():
    cfg = _pit_cfg()
    cfg["data"]["start_date"], cfg["data"]["end_date"] = "1970-01-01", "1972-12-31"
    rng = np.random.default_rng(7)
    return (
        cfg, _synthetic_macro(cfg, rng, MARKER_IDX), _synthetic_prices(cfg, rng, MARKER_IDX),
        _synthetic_vintages(rng, "1972-12-31"),
    )


def _seed_monthly_raw(root: Path) -> Path:
    from trading_crab_lib.checkpoints import CheckpointManager

    cm = CheckpointManager(checkpoint_dir=root / "platform")
    return cm.save(pd.DataFrame({"old_only": np.arange(len(MARKER_IDX), dtype=float)}, index=MARKER_IDX), "monthly_raw")


class TestLagMarkerMergeGuard:
    def test_empty_dir_saves_and_writes_a_matching_marker(self, small_world, tmp_path):
        cfg, macro, prices, vintages = small_world
        _build(cfg, macro, prices, vintages, tmp_path)
        marker = tmp_path / "platform" / pl.LAG_MARKER_FILENAME
        assert (tmp_path / "platform" / "monthly_raw.parquet").is_file()
        assert pl.lag_marker_matches(marker, cfg)

    @pytest.mark.parametrize("marker_state", ["missing", "different"])
    def test_unmarked_or_differently_marked_monthly_raw_raises_before_any_write(
        self, marker_state, small_world, tmp_path
    ):
        cfg, macro, prices, vintages = small_world
        raw_path = _seed_monthly_raw(tmp_path)
        before = raw_path.read_bytes()
        if marker_state == "different":
            other = copy.deepcopy(cfg)
            other["publication_lags"]["div_yield"] = 4
            pl.write_lag_marker(tmp_path / "platform" / pl.LAG_MARKER_FILENAME, other)

        with pytest.raises(RuntimeError, match="migrate_publication_lags"):
            _build(cfg, macro, prices, vintages, tmp_path)

        assert raw_path.read_bytes() == before
        assert not (tmp_path / "platform" / "daily_raw.parquet").exists()
        assert not (tmp_path / "platform" / "monthly_features.parquet").exists()

    def test_matching_marker_merges(self, small_world, tmp_path):
        cfg, macro, prices, vintages = small_world
        _seed_monthly_raw(tmp_path)
        pl.write_lag_marker(tmp_path / "platform" / pl.LAG_MARKER_FILENAME, cfg)

        _build(cfg, macro, prices, vintages, tmp_path)

        saved = pd.read_parquet(tmp_path / "platform" / "monthly_raw.parquet")
        assert "old_only" in saved.columns and "div_yield" in saved.columns

    def test_marker_round_trip_detects_a_changed_table(self, tmp_path):
        cfg = load_platform_config()
        marker = pl.write_lag_marker(tmp_path / pl.LAG_MARKER_FILENAME, cfg)
        assert pl.lag_marker_matches(marker, cfg)
        changed = copy.deepcopy(cfg)
        changed["publication_lags"]["fred_totalsl"] = 3
        assert not pl.lag_marker_matches(marker, changed)
        assert not pl.lag_marker_matches(tmp_path / "absent.json", cfg)


# ── 8. No lag is applied twice ──────────────────────────────────────────────


class TestShiftGuard:
    def test_fred_monthly_shift_true_makes_lag_table_raise(self):
        cfg = copy.deepcopy(load_platform_config())
        cfg["fred_monthly"]["series"]["M2SL"]["shift"] = True
        with pytest.raises(ValueError, match="fred_m2sl"):
            pl.lag_table(cfg)
