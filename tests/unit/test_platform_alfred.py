"""Mocked unit tests for platform/ingestion/alfred.py — ALFRED point-in-time
vintage ingestion (DATA-03).

All network access is mocked — no real HTTP/FRED calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_three_column_all_releases_df() -> pd.DataFrame:
    """Synthetic get_series_all_releases() response in the REAL fredapi shape
    — realtime_start / date / value only, no realtime_end column.

    Same two-revision structure as :func:`_make_all_releases_df`:

    - 2020-01-01: two revisions — an original vintage (value=100, known from
      2020-02-01) and a later revision (value=150, known from 2020-05-01).
    - 2020-06-01: a single vintage (value=200, known from 2020-06-15).
    """
    return pd.DataFrame(
        {
            "realtime_start": pd.to_datetime(
                ["2020-02-01", "2020-05-01", "2020-06-15"]
            ),
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-06-01"]),
            "value": [100.0, 150.0, 200.0],
        }
    )


def _make_all_releases_df() -> pd.DataFrame:
    """Synthetic get_series_all_releases() response with two reference periods:

    - 2020-01-01: two revisions — an original vintage (value=100, known from
      2020-02-01) and a later revision (value=150, known from 2020-05-01).
    - 2020-06-01: a single vintage (value=200, known from 2020-06-15) — used
      to exercise the "at/after the earliest vintage" branch of
      align_with_fallback with a matching reference period.
    """
    return pd.DataFrame(
        {
            "realtime_start": pd.to_datetime(
                ["2020-02-01", "2020-05-01", "2020-06-15"]
            ),
            "realtime_end": pd.to_datetime(
                ["2020-05-01", "2099-12-31", "2099-12-31"]
            ),
            "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-06-01"]),
            "value": [100.0, 150.0, 200.0],
        }
    )


# ── _detect_vintage_columns ─────────────────────────────────────────────────


def test_detect_vintage_columns_case_insensitive():
    from trading_crab_lib.platform.ingestion.alfred import _detect_vintage_columns

    df = pd.DataFrame(
        columns=["Realtime_Start", "REALTIME_END", "Date", "VALUE"]
    )
    mapping = _detect_vintage_columns(df)
    assert mapping == {
        "realtime_start": "Realtime_Start",
        "realtime_end": "REALTIME_END",
        "date": "Date",
        "value": "VALUE",
    }


def test_detect_vintage_columns_raises_on_missing_role():
    from trading_crab_lib.platform.ingestion.alfred import _detect_vintage_columns

    df = pd.DataFrame(columns=["realtime_start", "date"])  # no value — genuinely required
    with pytest.raises(ValueError, match="value"):
        _detect_vintage_columns(df)


def test_detect_vintage_columns_accepts_real_fredapi_three_column_shape():
    from trading_crab_lib.platform.ingestion.alfred import _detect_vintage_columns

    df = _make_three_column_all_releases_df()
    mapping = _detect_vintage_columns(df)
    assert set(mapping) == {"realtime_start", "date", "value"}
    assert "realtime_end" not in mapping


def test_detect_vintage_columns_maps_realtime_end_when_present():
    from trading_crab_lib.platform.ingestion.alfred import _detect_vintage_columns

    df = _make_all_releases_df()  # four-column frame, carries realtime_end
    mapping = _detect_vintage_columns(df)
    assert mapping["realtime_end"] == "realtime_end"


# ── value_as_of ──────────────────────────────────────────────────────────────


def test_value_as_of_respects_vintage():
    from trading_crab_lib.platform.ingestion.alfred import value_as_of

    all_releases = _make_all_releases_df()

    # Between the two revisions: only the first vintage (100) was known.
    between = value_as_of(all_releases, pd.Timestamp("2020-03-01"))
    assert between.loc[pd.Timestamp("2020-01-01")] == pytest.approx(100.0)

    # After the second revision: the later value (150) is now known.
    after = value_as_of(all_releases, pd.Timestamp("2020-06-01"))
    assert after.loc[pd.Timestamp("2020-01-01")] == pytest.approx(150.0)


def test_value_as_of_ignores_future_revisions():
    from trading_crab_lib.platform.ingestion.alfred import value_as_of

    all_releases = _make_all_releases_df()

    # Before the first vintage was even published: nothing is known yet.
    before = value_as_of(all_releases, pd.Timestamp("2020-01-15"))
    assert before.empty


def test_value_as_of_on_three_column_frame():
    """Proves reconstruction genuinely works on the real (three-column)
    fredapi shape, not only on the padded four-column fixture."""
    from trading_crab_lib.platform.ingestion.alfred import value_as_of

    all_releases = _make_three_column_all_releases_df()

    # Between the two revisions: only the first vintage (100) was known.
    between = value_as_of(all_releases, pd.Timestamp("2020-03-01"))
    assert between.loc[pd.Timestamp("2020-01-01")] == pytest.approx(100.0)

    # After the second revision: the later value (150) is now known.
    after = value_as_of(all_releases, pd.Timestamp("2020-06-01"))
    assert after.loc[pd.Timestamp("2020-01-01")] == pytest.approx(150.0)


# ── align_with_fallback ──────────────────────────────────────────────────────


def test_pre_vintage_fallback():
    from trading_crab_lib.platform.ingestion.alfred import align_with_fallback

    all_releases = _make_all_releases_df()  # earliest vintage: 2020-02-01
    # Second as-of date matches the 2020-06-01 reference period, which was
    # first published (realtime_start) on 2020-06-15 — already known by
    # 2020-06-20.
    as_of_dates = pd.DatetimeIndex(["2019-01-01", "2020-06-20"])
    shift_series = pd.Series(
        {pd.Timestamp("2019-01-01"): 42.0, pd.Timestamp("2020-06-20"): 60.0}
    )

    result = align_with_fallback(all_releases, as_of_dates, shift_series)

    # Before the earliest recorded vintage: falls back to the shift value —
    # never NaN, never a raised error (D-06).
    assert result.loc[pd.Timestamp("2019-01-01")] == pytest.approx(42.0)
    # At the vintage-era join the series is ratio-spliced onto shift_series'
    # base, so the two agree AT the join by construction and diverge after it
    # along the vintage growth path. (Previously this returned the raw
    # published level, 200.0, which splices two different index bases together
    # — see test_vintage_rebasing_does_not_create_a_level_discontinuity.)
    assert result.loc[pd.Timestamp("2020-06-20")] == pytest.approx(60.0)


# ── fetch_vintage_series ──────────────────────────────────────────────────────


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_vintage_series_calls_bulk_endpoint_once(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.alfred import fetch_vintage_series

    mock_fred = MagicMock()
    mock_fred.get_series_all_releases.return_value = _make_all_releases_df()

    result = fetch_vintage_series(mock_fred, "PAYEMS")

    mock_fred.get_series_all_releases.assert_called_once_with("PAYEMS")
    assert isinstance(result, pd.DataFrame)


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_vintage_series_accepts_three_column_response(mock_fred_cls):
    """The real fredapi response (three columns, no realtime_end) must not
    raise — this is exactly the shape returned by every live call."""
    from trading_crab_lib.platform.ingestion.alfred import fetch_vintage_series

    mock_fred = MagicMock()
    mock_fred.get_series_all_releases.return_value = _make_three_column_all_releases_df()

    result = fetch_vintage_series(mock_fred, "PAYEMS")

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["realtime_start", "date", "value"]


# ── fetch_all_vintages ────────────────────────────────────────────────────────


def _make_vintage_cfg():
    return {
        "fred_vintage": {
            "api_key": "fake_key_for_testing",
            "series": {
                "GDPC1": {"name": "fred_gdp", "tier": "agency"},
                "CPIAUCSL": {"name": "fred_cpi", "tier": "agency"},
            },
        }
    }


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_all_vintages_basic(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.alfred import fetch_all_vintages

    mock_fred = MagicMock()
    mock_fred.get_series_all_releases.return_value = _make_all_releases_df()
    mock_fred_cls.return_value = mock_fred

    result = fetch_all_vintages(_make_vintage_cfg())
    assert "fred_gdp" in result
    assert "fred_cpi" in result


@patch("trading_crab_lib.platform.ingestion.alfred.Fred")
def test_fetch_all_vintages_handles_single_series_failure(mock_fred_cls):
    from trading_crab_lib.platform.ingestion.alfred import fetch_all_vintages

    mock_fred = MagicMock()

    def _side_effect(series_id):
        if series_id == "GDPC1":
            raise OSError("API rate limit")
        return _make_all_releases_df()

    mock_fred.get_series_all_releases.side_effect = _side_effect
    mock_fred_cls.return_value = mock_fred

    result = fetch_all_vintages(_make_vintage_cfg())
    assert "fred_cpi" in result
    assert "fred_gdp" not in result


def test_fetch_all_vintages_missing_api_key_raises():
    from trading_crab_lib.platform.ingestion.alfred import fetch_all_vintages

    cfg = {"fred_vintage": {"api_key": None, "series": {}}}
    with pytest.raises(OSError, match="FRED_API_KEY"):
        fetch_all_vintages(cfg)


# ── Index-base discontinuities across vintages ──────────────────────────────


def _rebasing_releases() -> pd.DataFrame:
    """The real CPIAUCSL situation, in miniature.

    BLS rebased CPI from 1967=100 to 1982-84=100 in January 1988. Vintages
    published before the rebasing carry the old base; vintages published after
    carry the new one, ~2.99x smaller. Both describe the same price level.

    Four reference periods, each published a month later. The vintage released
    on 1988-02-15 restates the whole history on the new base.
    """
    old_base = {"1987-10-01": 340.2, "1987-11-01": 344.5, "1987-12-01": 345.5}
    new_base_factor = 1 / 2.9845

    rows = []
    # Pre-rebasing vintages: each release adds one period, on the OLD base.
    for i, (ref, val) in enumerate(old_base.items()):
        rows.append(
            {
                "realtime_start": pd.Timestamp(ref) + pd.DateOffset(months=1, days=14),
                "date": pd.Timestamp(ref),
                "value": val,
            }
        )
        _ = i
    # The 1988-02-15 vintage: adds 1988-01 AND restates every earlier period on
    # the NEW base.
    restated = {**old_base, "1988-01-01": 345.9}
    for ref, val in restated.items():
        rows.append(
            {
                "realtime_start": pd.Timestamp("1988-02-15"),
                "date": pd.Timestamp(ref),
                "value": val * new_base_factor,
            }
        )
    return pd.DataFrame(rows)


def test_vintage_rebasing_does_not_create_a_level_discontinuity():
    """The A4 regression. Splicing published LEVELS across a rebasing put a
    ~2.99x cliff into fred_cpi (1988-01: 345.9 -> 1988-02: 115.9), which drove
    real_rate_level to a range of -209..+74 — in a feature that defines 64% of
    labeler occupancy. Growth measured inside a vintage is base-invariant."""
    from trading_crab_lib.platform.ingestion.alfred import align_with_fallback

    releases = _rebasing_releases()
    as_of_dates = pd.DatetimeIndex(
        ["1987-11-30", "1987-12-31", "1988-01-31", "1988-02-29"]
    )
    shift_series = pd.Series(340.0, index=as_of_dates)

    result = align_with_fallback(releases, as_of_dates, shift_series)

    step_ratios = (result / result.shift(1)).dropna()
    assert step_ratios.max() < 1.10, f"level discontinuity survived: {result.to_dict()}"
    assert step_ratios.min() > 0.90, f"level discontinuity survived: {result.to_dict()}"


def test_growth_is_preserved_across_the_rebasing():
    """Removing the cliff must not flatten the series — the real month-over-
    month growth published in the rebased vintage has to survive."""
    from trading_crab_lib.platform.ingestion.alfred import align_with_fallback

    releases = _rebasing_releases()
    as_of_dates = pd.DatetimeIndex(["1988-01-31", "1988-02-29"])
    shift_series = pd.Series(340.0, index=as_of_dates)

    result = align_with_fallback(releases, as_of_dates, shift_series)

    # In the rebased vintage 1987-12 -> 1988-01 grows 345.9/345.5.
    expected_growth = 345.9 / 345.5
    actual_growth = result.iloc[-1] / result.iloc[-2]
    assert actual_growth == pytest.approx(expected_growth, rel=1e-6)


def test_chaining_is_a_no_op_when_no_rebasing_occurs():
    """The property that bounds the blast radius: with a single consistent
    base the ratios telescope, so chained output equals the raw published
    level exactly. Rates (fred_unrate) and headcounts (fred_payems) are
    therefore untouched by this change."""
    from trading_crab_lib.platform.ingestion.alfred import align_with_fallback

    refs = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    levels = [3.5, 3.8, 4.4]
    releases = pd.DataFrame(
        {
            "realtime_start": refs + pd.DateOffset(months=1, days=14),
            "date": refs,
            "value": levels,
        }
    )
    as_of_dates = pd.DatetimeIndex(["2020-02-29", "2020-03-31", "2020-04-30"])
    # shift_series agrees with the first vintage value, so the join scale is 1.0
    # and the whole segment must reproduce the published levels untouched.
    shift_series = pd.Series(3.5, index=as_of_dates)

    result = align_with_fallback(releases, as_of_dates, shift_series)

    assert list(result.to_numpy()) == pytest.approx(levels)


def test_point_in_time_guarantee_is_unaffected_by_chaining():
    """Chaining must not let a later revision leak backward: every value still
    derives only from rows with realtime_start <= as_of."""
    from trading_crab_lib.platform.ingestion.alfred import align_with_fallback

    all_releases = _make_all_releases_df()  # 2020-01 revised 100 -> 150 on 2020-05-01
    as_of_dates = pd.DatetimeIndex(["2020-03-01", "2020-06-20"])
    shift_series = pd.Series(100.0, index=as_of_dates)

    result = align_with_fallback(all_releases, as_of_dates, shift_series)

    # At 2020-03-01 only the original 100.0 was published; the 150.0 revision
    # lands 2020-05-01 and must be invisible.
    assert result.loc[pd.Timestamp("2020-03-01")] == pytest.approx(100.0)
