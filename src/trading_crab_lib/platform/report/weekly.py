"""
Weekly report assembly + trades-implied + opt-in email delivery (L4-02, design
§7, 04-CONTEXT.md D-02).

``assemble_weekly_report()`` builds the markdown Glenn reads before trading in
Fidelity: current regime distribution + trajectory (nowcaster probabilities +
the empirical transition matrix), per-asset signals (the returns-by-regime
table, flagging D11 short-history cells as low-confidence), and
target-vs-current with explicit per-account trades implied.

``write_weekly_report()`` ALWAYS writes the markdown to
``outputs/reports/platform/weekly_report.md`` — a distinct path from the
incumbent's ``outputs/reports/weekly_report.md`` (D-02, avoids collision).
Email delivery is opt-in behind ``--send-email`` and reuses the incumbent
``email.py`` machinery READ-ONLY (``build_weekly_email_body`` /
``load_email_config`` / ``send_weekly_email``) — never forked, never
modified, and never invoked on the default path.

``main()`` orchestrates the full allocation cycle before assembly: load the
previous hysteresis state, update it with the current nowcaster
probabilities, compute target weights via ``vol_targeted_tilt``, pass them
through the no-trade band, and persist the new state (load-before-save order,
Pitfall 3) — all isolated in
``_build_report_inputs()`` so it can be swapped out in tests without real
Phase 1/3 checkpoint data on disk.

**The Bayes filter at serve (plan 08-08).** The nowcaster's posterior is filtered
into a belief (``prediction/regime_filter.py``) before the hysteresis and the tilt
see it — the same recursion ``backtest/driver.py`` and ``joint_driver.py`` run in
their loops, so the allocator consumes the same kind of object at train and serve.
The belief persists across runs in the ``regime_belief`` checkpoint, loaded BEFORE
it is saved (the Pitfall 3 ordering the hysteresis state uses), in the same block
as the hysteresis and immediately before it. The cold start — no checkpoint, or a
persisted null — is ``unconditional_belief`` on the ``regime_labels`` the nowcaster
was trained on: the SAME function object the drivers use, because a cold start that
differed between train and serve would itself be train/serve skew. ``A`` is
``transition_matrix_for`` on those labels. One filter step is one MONTH: the belief
carries the as-of date of the feature row it absorbed, a weekly re-run inside the
same month reuses it unchanged (re-filtering would count the same month's evidence
again), and a gap of several months advances by ``predict_only_step`` for each
unobserved month.

**Two roles, two rules (plan 08-16, CR-01).** The likelihood ``L_t(j) = posterior(j) /
prior(j)`` divides by the served model's TRAINING class prior: the class distribution of
the rows ``fit_l2_nowcaster`` fit it on, over its ``classes_`` only, built beside the model
by ``serving.py`` as ``nowcaster_class_prior`` and refused unless its states are the model's
``classes_``. Bayes: p(x | j) ∝ p(j | x) / p_train(j); only that inversion needs the training
prior. The cold start π_0 and ``A`` stay on ``regime_labels`` over all K states: they
describe the L1 regime process, and π_0 is a belief over every state, whereas the training
prior lives on ``classes_`` alone (3 of 6 at serve, because the model's feature block starts
2007-04) and would put zero initial mass on the others for a reason that is about the fit's
rows, not the world. π_0 enters once and decays under ``A``; the likelihood prior enters
every step.

**The no-trade band and the active regime at serve (plan 08-09, ``08-A7.md``).** The
tilt's target passes through ``allocation/hysteresis.py::execute_rebalance`` — the 5pp
no-trade band, NOT SWEPT, the same function both backtest drivers call — so the weights
the report shows are the EXECUTED book, not the raw target. ``held`` is the last
executed book, persisted in the ``executed_weights`` checkpoint (load before save). One
band step is one MONTH, like the belief: a same-month re-run re-bands this month's
target against the SAME held book the first run used (the previous month's execution),
so repeated weekly runs neither compound the band nor drift. No checkpoint yet trades in
full. ``assemble_weekly_report`` now receives the hysteresis output as
``active_regime`` instead of recomputing ``probs.idxmax()`` — it no longer narrates a
state machine whose output it does not show. ``active_regime`` gates no weight: A7
closed by rewording, and the report says so beside the value.

**What it scores (plans 08-13, 08-14, 08.1-02).** The nowcaster's own columns
(``feature_names_in_``), in its own order; never the whole row. Under Glenn's 08-12 ruling q1-c
(08-SERVING.md §2.1) it scores the latest month observed in every model column, prints "Scored as
of <month>" naming the model columns each newer row lacks, and steps the belief and the band on
that month. It never imputes. Staleness is judged per series (DECISIONS A-12, which supersedes
A-07's 3-month-end cap): ``stale_series`` compares each ``publication_lags`` series' last value
with the month the run date allows (``report.staleness_grace_days`` of slack), and the page
carries a STALE DATA banner naming each late series and its months late; it warns, never
refuses. Under ruling q2-ii (§2.2) it prints, directly under the distribution, the exact
count of distinct posteriors the served model gives across every full-span month complete in
its columns, and says so in plain words when that count is 1; it never withholds on it.

**The allocation mode (plan 08.2-01, DECISIONS A-13, A-15).** ``report.allocation_mode`` picks
what the weekly page trades. ``no_regime`` (the live config) is the measured no-regime ablation
leg: ``no_regime_target`` hands a constant one-state belief and ``returns_by_regime_stats`` over
the asset returns up to the scored month to the same ``vol_targeted_tilt``, exactly as the
driver's tilt-off path does, then the same no-trade band. ``regime_tilt`` (the code default when
the key is absent, so synthetic configs keep the Phase 8 path) tilts on the filtered belief. The
belief and the hysteresis still load, advance and save in both modes, so flipping back does not
cold-start the filter; in ``no_regime`` the page simply shows them as suspended (D-03). The mode
persists in the ``allocation_mode`` checkpoint (gated like the executed book): a run whose mode
differs from the last one executes the new target in full, not banded against the old mode's
book, and the page says so (A-15).

Run order::

    # 1. the data (FRED key plus network)
    python scripts/build_platform_data.py
    # 2. the serving artifacts (nowcaster, nowcaster_class_prior, returns_by_regime,
    #    asset_returns) from the evaluated recipe; NOT a registry trial
    python -m trading_crab_lib.platform.report.serving
    # 3. the report
    python -m trading_crab_lib.platform.report.weekly [--send-email]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from trading_crab_lib import OUTPUT_DIR
from trading_crab_lib.email import build_weekly_email_body, load_email_config, send_weekly_email
from trading_crab_lib.platform.allocation.hysteresis import (
    execute_rebalance,
    hysteresis_thresholds,
    load_active_regime,
    no_trade_band_from_config,
    save_active_regime,
    update_active_regime,
)
from trading_crab_lib.platform.allocation.tilt import vol_targeted_tilt
from trading_crab_lib.platform.assets.returns import returns_by_regime_stats
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.honesty.holdout import load_full_span
from trading_crab_lib.platform.ingestion.publication_lags import DERIVED
from trading_crab_lib.platform.prediction.regime_filter import (
    filter_step,
    predict_only_step,
    transition_matrix_for,
    unconditional_belief,
)
from trading_crab_lib.platform.prediction.transition_matrix import empirical_transition_matrix
from trading_crab_lib.platform.report.holdings import load_account_weights
from trading_crab_lib.platform.report.serving import SERVING_BUILD_COMMAND, SERVING_CLASS_PRIOR
from trading_crab_lib.platform.tripwire.monitor import evaluate_tripwire

log = logging.getLogger(__name__)

# report.trade_threshold_pct / report.min_obs_flag defaults (config/platform_settings.yaml
# `report:` section) — used only when cfg omits the key (defensive .get() pattern).
_DEFAULT_TRADE_THRESHOLD_PCT = 0.03
_DEFAULT_MIN_OBS_FLAG = 6

# Plan 08-15: what §3 prints in neutral posture instead of rows. The active regime selects the
# per-asset rows (the A7 sentence); with none active there is no regime to select, so no rows.
_NEUTRAL_PER_ASSET_SENTENCE = (
    "Neutral posture: no regime is active, so no per-asset regime rows are shown (the active "
    "regime selects them, as stated above). The executed book below is built from the filtered "
    "belief across all regimes."
)

# report.staleness_grace_days default (08.1 ruling 4): FRED posts a month's average 1-2 business
# days into the next month, so a series is late only once the run date is this many days past
# the month-end it should cover.
_DEFAULT_STALENESS_GRACE_DAYS = 7

_BELIEF_CHECKPOINT = "regime_belief"
_EXECUTED_CHECKPOINT = "executed_weights"
_MODE_CHECKPOINT = "allocation_mode"

# report.allocation_mode (D-01, A-13): what the page trades. regime_tilt stays available but off (G-06).
ALLOCATION_MODES = ("regime_tilt", "no_regime")

# D-03: what the page shows in place of the regime sections in no_regime mode.
_SUSPENDED_SENTENCE = (
    "Regime view: suspended — the served nowcaster is input-independent (fixed in the regime "
    "rebuild); the allocation does not use it."
)


def allocation_mode_from_config(cfg: dict) -> str:
    """The weekly allocation mode from ``cfg["report"]["allocation_mode"]``.

    Absent or null means ``regime_tilt`` (ruling A2: synthetic test configs keep the Phase 8
    path); the live config carries ``no_regime``, pinned by a test, so deleting the key goes
    red rather than silently trading the tilt again. Any value outside ``ALLOCATION_MODES``
    raises ValueError naming it.
    """
    value = cfg.get("report", {}).get("allocation_mode")
    if value is None:
        return "regime_tilt"
    if not isinstance(value, str) or value not in ALLOCATION_MODES:
        raise ValueError(f"report.allocation_mode must be one of {ALLOCATION_MODES} or null, got {value!r}")
    return value


def no_regime_target(asset_returns: pd.DataFrame, as_of: pd.Timestamp, cfg: dict) -> dict:
    """The no-regime core mix at ``as_of``: the driver's tilt-off path, by its own functions.

    ``backtest/driver.py`` with ``use_regime_tilt=False`` labels every training month state 0,
    feeds the belief ``{0: 1.0}`` and conditions ``returns_by_regime_stats`` on those labels over
    the returns up to the decision month. This does the same over ``asset_returns`` up to and
    including ``as_of`` (served returns run past the scored month; those rows never enter), then
    the same ``vol_targeted_tilt`` with the same ``allocation`` keys and defaults. No new
    allocation math; the returns themselves are untouched (D-02 defers E-07).

    Returns ``vol_targeted_tilt``'s dict: ``weights``, ``cash``, ``scale``, ``portfolio_vol``.
    """
    window = asset_returns.loc[asset_returns.index <= pd.Timestamp(as_of)]
    stats = returns_by_regime_stats(window, pd.Series(0, index=window.index))
    allocation_cfg = cfg.get("allocation", {})
    return vol_targeted_tilt(
        pd.Series({0: 1.0}),
        stats,
        window,
        target_vol_annual=allocation_cfg.get("target_vol_annual", 0.10),
        halflife=allocation_cfg.get("ewma_halflife_months", 6),
        min_obs=allocation_cfg.get("portfolio_vol_min_obs", 12),
    )


def load_regime_belief(cm=None) -> pd.Series | None:
    """Load the previous run's filtered regime belief. Cold start (no checkpoint yet,
    or a persisted null) returns None — the caller then uses ``unconditional_belief``.

    The returned Series is indexed by integer state and its ``name`` is the as-of
    month (``pd.Timestamp``) of the feature row it absorbed, or None if unrecorded.
    """
    cm = cm or get_platform_checkpoint_manager()
    try:
        frame = cm.load(_BELIEF_CHECKPOINT)
    except FileNotFoundError:
        return None
    if frame.empty or frame["belief"].isna().all() or frame["state"].isna().all():
        return None
    belief = pd.Series(frame["belief"].to_numpy(dtype=float), index=[int(v) for v in frame["state"]])
    as_of = frame["as_of"].iloc[0] if "as_of" in frame.columns else None
    belief.name = None if as_of is None or pd.isna(as_of) else pd.Timestamp(as_of)
    return belief


def save_regime_belief(belief: pd.Series | None, cm=None, *, as_of: pd.Timestamp | None = None) -> None:
    """Persist the filtered belief (or None) with the as-of month it absorbed."""
    cm = cm or get_platform_checkpoint_manager()
    if belief is None:
        frame = pd.DataFrame([{"state": None, "belief": None, "as_of": as_of}])
    else:
        frame = pd.DataFrame({
            "state": [int(v) for v in belief.index],
            "belief": belief.to_numpy(dtype=float),
            "as_of": [as_of] * len(belief),
        })
    cm.save(frame, _BELIEF_CHECKPOINT)


def _months_between(earlier: pd.Timestamp, later: pd.Timestamp) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _run_date() -> pd.Timestamp:
    """Today, midnight. A function so tests can pin the run date (the wall clock is the point)."""
    return pd.Timestamp.today().normalize()


def expected_month_end(run_date: pd.Timestamp, grace_days: int) -> pd.Timestamp:
    """The newest month-end a current series must already cover on ``run_date``.

    ``(run_date - grace_days)`` is the date the data is judged as of; the month that has fully
    closed before it is the expected one. On a month-end itself that month is still open, so
    the expected month is the previous one (``MonthEnd(1)`` steps back from an on-offset date).
    """
    judged = (pd.Timestamp(run_date) - pd.Timedelta(days=int(grace_days))).normalize()
    return judged - pd.offsets.MonthEnd(1)


def stale_series(
    frame: pd.DataFrame, columns: list[str], *, run_date: pd.Timestamp, grace_days: int
) -> dict[str, int | None]:
    """The series in ``columns`` whose last value is older than ``run_date`` allows (A-12).

    Returns ``{column: months_late}`` for each stale series, ``None`` for one never observed;
    fresh series are absent. A publication lag is already absorbed into the data's timestamps
    (08.1-01), so "within its lag" is simply "has a value at the expected month-end"; an
    interior gap is not staleness, only the last valid observation counts. Never imputes,
    never mutates ``frame``.
    """
    expected = expected_month_end(run_date, grace_days)
    late: dict[str, int | None] = {}
    for column in columns:
        valid = frame[column].dropna()
        if valid.empty:
            late[column] = None
            continue
        last = pd.Timestamp(valid.index[-1])
        if last < expected:
            late[column] = _months_between(last, expected)
    return late


def _stale_banner(late: dict[str, int | None], expected: pd.Timestamp | None) -> list[str]:
    """The ``## STALE DATA`` section: one line per late series, worst first, then the A-12 sentence."""
    lines = ["## STALE DATA", ""]
    for name, months in sorted(late.items(), key=lambda kv: (-(kv[1] if kv[1] is not None else 10**6), kv[0])):
        parts: list[str] = []
        if months is not None and expected is not None:
            parts.append(f"last value {(expected - pd.offsets.MonthEnd(months)).date().isoformat()}")
        if expected is not None:
            parts.append(f"expected through {expected.date().isoformat()}")
        what = "no value ever observed" if months is None else f"{months} month{'s' if months != 1 else ''} late"
        lines.append(f"- {name}: {what}" + (f" ({', '.join(parts)})" if parts else ""))
    lines.append("")
    lines.append(
        "The report below scores the latest month complete in the model's columns and imputes "
        "nothing; guidance that leans on the series above is older than the run date allows "
        "(DECISIONS A-12)."
    )
    lines.append("")
    return lines


def _weights_rows(weights: pd.Series | None, basis: str, as_of: pd.Timestamp) -> list[dict]:
    """Rows for one book. An empty book (all cash) is one null-asset marker row, so it
    stays distinguishable from "no book" (no rows at all)."""
    if weights is None:
        return []
    if len(weights) == 0:
        return [{"asset": None, "weight": None, "basis": basis, "as_of": as_of}]
    return [
        {"asset": str(a), "weight": float(w), "basis": basis, "as_of": as_of} for a, w in weights.items()
    ]


def load_held_weights(cm=None, *, as_of: pd.Timestamp) -> pd.Series | None:
    """The ``held`` book the no-trade band compares this month's target against.

    - no checkpoint: None — nothing executed yet, the first execution trades in full;
    - checkpoint from an EARLIER month: that month's executed book;
    - checkpoint from THIS month (a weekly re-run): the held book that run used, so the
      band steps once per month and a re-run reproduces rather than compounds;
    - checkpoint dated AFTER ``as_of``: WARNING and None, mirroring the belief's rule.
    """
    cm = cm or get_platform_checkpoint_manager()
    try:
        frame = cm.load(_EXECUTED_CHECKPOINT)
    except FileNotFoundError:
        return None
    if frame.empty:
        return None
    saved_as_of = pd.Timestamp(frame["as_of"].iloc[0])
    gap = _months_between(saved_as_of, as_of)
    if gap < 0:
        log.warning("executed_weights checkpoint is dated %s, after this run's %s — no held book", saved_as_of, as_of)
        return None
    rows = frame[frame["basis"] == ("executed" if gap > 0 else "held_in")]
    if rows.empty:
        return None
    rows = rows[rows["asset"].notna()]
    return pd.Series(rows["weight"].to_numpy(dtype=float), index=[str(a) for a in rows["asset"]], dtype=float)


def load_last_executed_weights(cm=None) -> pd.Series | None:
    """The book the PREVIOUS run executed: the ``executed_weights`` checkpoint's
    ``basis == "executed"`` rows, for the report's "last week" column (D-09).

    Deliberately not ``load_held_weights``: that returns the ``held_in`` book on a same-month
    re-run, which is what the band compared against, not what was executed. Call this BEFORE
    ``save_executed_weights`` overwrites the checkpoint (load before save, Pitfall 3).
    No checkpoint: None. An all-cash book (the null-asset marker row) is an empty Series.
    """
    cm = cm or get_platform_checkpoint_manager()
    try:
        frame = cm.load(_EXECUTED_CHECKPOINT)
    except FileNotFoundError:
        return None
    rows = frame[frame["basis"] == "executed"] if not frame.empty else frame
    if rows.empty:
        return None
    rows = rows[rows["asset"].notna()]
    return pd.Series(rows["weight"].to_numpy(dtype=float), index=[str(a) for a in rows["asset"]], dtype=float)


def load_allocation_mode(cm=None) -> dict | None:
    """The mode the previous run traded (A-15): ``{"mode", "as_of", "changed_from"}`` or None.

    ``changed_from`` is the mode a switch left this month (None when there was no switch), so a
    same-month re-run can repeat the switch note. No checkpoint, or an empty one: None.
    """
    cm = cm or get_platform_checkpoint_manager()
    try:
        frame = cm.load(_MODE_CHECKPOINT)
    except FileNotFoundError:
        return None
    if frame.empty or pd.isna(frame["mode"].iloc[0]):
        return None
    as_of, changed_from = frame["as_of"].iloc[0], frame["changed_from"].iloc[0]
    return {
        "mode": str(frame["mode"].iloc[0]),
        "as_of": None if pd.isna(as_of) else pd.Timestamp(as_of),
        "changed_from": None if pd.isna(changed_from) else str(changed_from),
    }


def save_allocation_mode(mode: str, cm=None, *, as_of: pd.Timestamp, changed_from: str | None) -> None:
    """Persist this run's mode, the month it scored and the mode a switch left (or None)."""
    cm = cm or get_platform_checkpoint_manager()
    frame = pd.DataFrame(
        [{"mode": mode, "as_of": as_of, "changed_from": changed_from}], columns=["mode", "as_of", "changed_from"]
    )
    cm.save(frame, _MODE_CHECKPOINT)


def _has_executed_book(cm) -> bool:
    try:
        cm.load(_EXECUTED_CHECKPOINT)
    except FileNotFoundError:
        return False
    return True


def mode_switch(
    record: dict | None, *, mode: str, as_of: pd.Timestamp, executed_exists: bool
) -> tuple[bool, str | None]:
    """The A-15 rule: ``(switched, changed_from)`` for this run.

    - previous mode: the loaded record's; else ``regime_tilt`` when an ``executed_weights``
      checkpoint exists (every pre-8.2 book was the tilt's); else None (nothing traded yet);
    - ``switched``: a previous mode exists and differs from ``mode`` — the caller then
      executes the new target in full (held None), never a band against the old mode's book;
    - ``changed_from``: the previous mode on a switch; on a same-month re-run of the same mode,
      the record's own ``changed_from`` (the note survives the re-run); otherwise None.
    """
    if record is not None:
        previous = record["mode"]
    elif executed_exists:
        previous = "regime_tilt"
    else:
        previous = None
    switched = previous is not None and previous != mode
    if switched:
        return True, previous
    if record is not None and record["mode"] == mode and record["as_of"] == pd.Timestamp(as_of):
        return False, record["changed_from"]
    return False, None


def _mode_note(mode: str, changed_from: str) -> str:
    return (
        f"**Note:** allocation mode changed to {mode} (was {changed_from}) — this run executes the new "
        "target in full, so a large change against last week's book is expected (DECISIONS A-15)."
    )


def _class_by_ticker(cfg: dict) -> dict[str, str]:
    """``{tradable ticker: asset class}``: the reverse of ``cfg["splice"]``, in splice order
    (SPY equities, TLT long_duration, IAU gold, USO oil, FZFXX cash)."""
    return {
        str(spec["tradable"]): str(name) for name, spec in cfg.get("splice", {}).items() if spec.get("tradable")
    }


def _pct(value: float) -> str:
    return f"{0.0 if abs(value) < 1e-12 else value:.1%}"


def _change_pp(delta: float) -> str:
    return f"{round(delta * 100.0, 1) + 0.0:+.1f} pp"  # + 0.0 turns -0.0 into 0.0


def _allocation_table(
    target_weights: pd.Series,
    cash: float | None,
    last_week_weights: pd.Series | None,
    asset_classes: dict[str, str] | None,
) -> list[str]:
    """The "### Target allocation" block (D-09): one row per asset class, in ``asset_classes``
    order, with its tradable ticker, target %, last week's %, and the change in percentage
    points. Target is the EXECUTED book (``target_weights``, absent = 0.0%); the cash row's
    target is ``cash`` and its last-week value the residual ``1 - sum(last week's risky
    weights)``, which is what ``execute_rebalance`` leaves in cash. ``last_week_weights`` None
    reads n/a. A ticker outside the map gets its own "unmapped" row; the cash row falls back to
    ticker CASH when the map carries no cash class (a config without one, e.g. a test world).
    """
    classes = dict(asset_classes or {})
    cash_ticker = next((t for t, c in classes.items() if c == "cash"), "CASH")
    rows = list(classes.items())
    if cash is not None and cash_ticker not in classes:
        rows.append((cash_ticker, "cash"))
    seen = {t for t, _ in rows}
    known = set(target_weights.index) | (set(last_week_weights.index) if last_week_weights is not None else set())
    rows.extend((t, "unmapped") for t in sorted(known - seen))

    risky_target = float(target_weights.drop(labels=[cash_ticker], errors="ignore").sum())
    lines = ["### Target allocation", ""]
    if not rows:
        lines.append("(no target allocation available)")
        lines.append("")
        return lines
    lines.append("| Class | Ticker | Target % | Last week % | Change |")
    lines.append("|---|---|---|---|---|")
    for ticker, asset_class in rows:
        if ticker == cash_ticker:
            target = float(cash) if cash is not None else max(0.0, 1.0 - risky_target)
            last = (
                None if last_week_weights is None
                else 1.0 - float(last_week_weights.drop(labels=[cash_ticker], errors="ignore").sum())
            )
        else:
            target = float(target_weights.get(ticker, 0.0))
            last = None if last_week_weights is None else float(last_week_weights.get(ticker, 0.0))
        last_cell, change_cell = ("n/a", "n/a") if last is None else (_pct(last), _change_pp(target - last))
        lines.append(f"| {asset_class} | {ticker} | {_pct(target)} | {last_cell} | {change_cell} |")
    lines.append("")
    lines.append(
        "Target is the executed book after the no-trade band; last week is the book this report "
        "executed before this run (n/a when there was none)."
    )
    lines.append("")
    return lines


def save_executed_weights(
    executed: pd.Series, held_in: pd.Series | None, cm=None, *, as_of: pd.Timestamp
) -> None:
    """Persist this month's executed book and the held book it was banded against."""
    cm = cm or get_platform_checkpoint_manager()
    rows = _weights_rows(executed, "executed", as_of) + _weights_rows(held_in, "held_in", as_of)
    cm.save(pd.DataFrame(rows, columns=["asset", "weight", "basis", "as_of"]), _EXECUTED_CHECKPOINT)


def advance_regime_belief(
    prev_belief: pd.Series | None,
    regime_labels: pd.Series,
    regime_probs: pd.Series,
    *,
    class_prior: pd.Series,
    state_index: list[int],
    as_of: pd.Timestamp,
) -> pd.Series:
    """This run's belief from the loaded one: cold start, same-month reuse, or filter.

    - the likelihood divides the posterior by ``class_prior``, the served model's training
      prior (CR-01); the cold start π_0 is ``unconditional_belief(regime_labels)`` over all
      of ``state_index`` (the drivers' rule), and ``A`` is ``transition_matrix_for`` on the
      same labels — two roles, two rules (module docstring). ``class_prior`` is REQUIRED,
      with no default: a default would silently re-open CR-01;
    - no previous belief (or its states differ from ``state_index``): cold start from π_0,
      then one filter step;
    - previous belief already absorbed ``as_of``: returned unchanged (no double count);
    - otherwise ``predict_only_step`` once per unobserved month in between, then one
      ``filter_step`` with this month's posterior.
    """
    transition = transition_matrix_for(regime_labels, state_index=state_index)
    start = unconditional_belief(regime_labels, state_index=state_index)
    if prev_belief is not None and sorted(int(v) for v in prev_belief.index) == sorted(state_index):
        prev_as_of = prev_belief.name
        if prev_as_of is not None and _months_between(pd.Timestamp(prev_as_of), as_of) == 0:
            return prev_belief.rename(None)
        if prev_as_of is not None and _months_between(pd.Timestamp(prev_as_of), as_of) < 0:
            log.warning(
                "regime_belief checkpoint is dated %s, after this run's %s — cold-starting the filter",
                prev_as_of, as_of,
            )
        else:
            start = prev_belief.rename(None)
            gap = 1 if prev_as_of is None else _months_between(pd.Timestamp(prev_as_of), as_of)
            for _ in range(gap - 1):
                start = predict_only_step(start, transition)
    elif prev_belief is not None:
        log.warning(
            "regime_belief checkpoint covers states %s, not %s — cold-starting the filter",
            list(prev_belief.index), state_index,
        )
    return filter_step(start, transition, regime_probs, class_prior)


def trades_implied(
    target_weights: pd.Series,
    current_weights: pd.Series,
    *,
    threshold: float = _DEFAULT_TRADE_THRESHOLD_PCT,
) -> pd.DataFrame:
    """One row per asset (union of target and current index): current_pct,
    target_pct, delta_pct, signal in {BUY, SELL, HOLD}.

    Flat no-trade band (design §21's full band-width math is deferred to v2,
    L4-V2-01): ``|delta| < threshold`` -> HOLD, ``delta >= +threshold`` ->
    BUY, ``delta <= -threshold`` -> SELL. New implementation — inspiration
    from incumbent ``reporting.generate_recommendation`` but NOT imported
    (D-01).
    """
    all_assets = sorted(set(target_weights.index) | set(current_weights.index))
    rows = []
    for asset in all_assets:
        current = float(current_weights.get(asset, 0.0))
        target = float(target_weights.get(asset, 0.0))
        delta = target - current
        if delta >= threshold:
            signal = "BUY"
        elif delta <= -threshold:
            signal = "SELL"
        else:
            signal = "HOLD"
        rows.append(
            {"asset": asset, "current_pct": current, "target_pct": target, "delta_pct": delta, "signal": signal}
        )
    return pd.DataFrame(rows, columns=["asset", "current_pct", "target_pct", "delta_pct", "signal"])


def assemble_weekly_report(
    *,
    regime_probs: pd.Series | dict,
    transition_matrix: pd.DataFrame,
    returns_by_regime: pd.DataFrame,
    target_weights: pd.Series,
    accounts: list[str],
    active_regime: int | None,
    regime_belief: pd.Series | dict | None = None,
    cash: float | None = None,
    no_trade_band: float | None = None,
    accounts_dir: Path | None = None,
    min_obs_flag: int = _DEFAULT_MIN_OBS_FLAG,
    trade_threshold_pct: float = _DEFAULT_TRADE_THRESHOLD_PCT,
    scored_as_of_note: str | None = None,
    input_sensitivity_note: str | None = None,
    stale_series: dict[str, int | None] | None = None,
    stale_expected_through: pd.Timestamp | None = None,
    last_week_weights: pd.Series | None = None,
    asset_classes: dict[str, str] | None = None,
    regime_view_suspended: bool = False,
    allocation_mode: str | None = None,
    mode_note: str | None = None,
    tripwire: dict | None = None,
) -> str:
    """Assemble the weekly report markdown (design §7 output list) from
    pre-computed inputs — a pure function, no I/O beyond the per-account
    holdings YAML reads (``load_account_weights``).

    Sections: (1) current regime distribution, then the hysteresis state
    machine's OUTPUT (``active_regime``) with the cold-start rule that produced it,
    (2) trajectory from the empirical transition matrix out of that regime, (3)
    per-asset signals: the ``returns_by_regime`` rows of ``active_regime`` only, each
    naming its regime, flagging cells with ``n_obs < min_obs_flag`` as low-confidence
    (D11); in neutral posture (``active_regime`` None) no rows, one sentence saying why
    (plan 08-15), (4) target-vs-current + trades implied
    PER account, each asset annotated with a one-line regime rationale.

    ``active_regime`` is ``update_active_regime``'s output, passed in — plan 08-09
    removed the internal ``probs.idxmax()`` recomputation. ``None`` is the neutral
    posture. ``target_weights`` should be the EXECUTED book (after the no-trade band);
    ``no_trade_band`` names the band so the report says so. ``scored_as_of_note`` (ruling
    q1-c) is rendered directly under the distribution heading: the month scored and the
    model columns each newer row lacks. ``input_sensitivity_note`` (ruling q2-ii) is rendered
    directly under the distribution: the exact count of distinct posteriors across history.
    ``stale_series`` (``stale_series()``'s output) puts a STALE DATA banner right after the
    title when non-empty; ``stale_expected_through`` names the month-end those series should
    have covered. The banner warns only: the report is still assembled in full (A-12).
    Directly under the trades heading, with or without ``accounts``, a "Target allocation" table
    lists every class in ``asset_classes`` (``{ticker: class}``, in row order) with target %,
    ``last_week_weights`` (last run's executed book, None = n/a) and the change (D-09, A-11).

    ``regime_view_suspended`` (D-03, the no_regime mode) replaces sections 1-3 with one
    ``## Regime View (suspended)`` block: the suspended sentence, ``scored_as_of_note`` and
    ``input_sensitivity_note`` (A-08: the distinct-posterior count is the evidence); per-account
    lines read "no-regime core mix". ``allocation_mode``, when given, is printed directly under
    the trades heading, before the table, followed by ``mode_note`` (A-15, a mode switch). With
    all three at their defaults the page is byte-identical to the pre-08.2 page.

    ``tripwire`` (``evaluate_tripwire``'s output, D-04) renders the ``## Crash Tripwire`` section
    directly before the trades heading: one bullet per signal with its state, value, threshold
    and as-of date, then the escalation over current signals (``_tripwire_section``). Advisory:
    nothing else on the page reads it. None (the default) renders nothing.
    """
    probs = pd.Series(regime_probs, dtype=float)

    lines: list[str] = ["# Trading-Crab Platform Weekly Report", ""]
    if stale_series:
        lines.extend(_stale_banner(stale_series, stale_expected_through))

    if regime_view_suspended:
        lines.extend(_suspended_regime_view(scored_as_of_note, input_sensitivity_note))
    else:
        lines.extend(_regime_sections(
            probs,
            transition_matrix=transition_matrix,
            returns_by_regime=returns_by_regime,
            active_regime=active_regime,
            regime_belief=regime_belief,
            min_obs_flag=min_obs_flag,
            scored_as_of_note=scored_as_of_note,
            input_sensitivity_note=input_sensitivity_note,
        ))

    if tripwire is not None:
        lines.extend(_tripwire_section(tripwire))

    # ── 4. Target-vs-current + trades implied, per account ────────────────
    lines.append("## Target vs. Current — Trades Implied")
    lines.append("")
    if allocation_mode is not None:
        lines.append(f"**Allocation mode:** {allocation_mode}")
        lines.append("")
    if mode_note:
        lines.append(mode_note)
        lines.append("")
    lines.extend(_allocation_table(target_weights, cash, last_week_weights, asset_classes))
    if no_trade_band is not None:
        lines.append(
            f"Targets below are the EXECUTED book after the {no_trade_band:.1%} no-trade band "
            "(design §5.3 bounded turnover, 08-A7.md): an asset whose target moved by no more "
            "than the band from its last executed weight keeps that weight."
        )
        lines.append("")
    if regime_view_suspended:
        rationale = "no-regime core mix"
    else:
        rationale = f"regime {active_regime}" if active_regime is not None else "neutral posture"
    for account in accounts:
        holdings = load_account_weights(account, accounts_dir=accounts_dir)
        current_weights = pd.Series(holdings["weights"], dtype=float)
        implied = trades_implied(target_weights, current_weights, threshold=trade_threshold_pct)
        lines.append(f"### Account: {account} (cash on file: {holdings['cash']:.1%})")
        lines.append("")
        if implied.empty:
            lines.append("(no target or current holdings)")
        for row in implied.itertuples():
            lines.append(
                f"- {row.asset}: {row.signal} current={row.current_pct:.1%} "
                f"target={row.target_pct:.1%} delta={row.delta_pct:+.1%} ({rationale})"
            )
        lines.append("")

    if cash is not None:
        lines.append(f"_Target allocation cash residual: {cash:.1%}_")
        lines.append("")

    return "\n".join(lines)


_DATA_BUILD_COMMAND = "python scripts/build_platform_data.py"
_TRIPWIRE_HEADING = "## Crash Tripwire (advisory — changes no weight)"


def _tripwire_value(name: str, sig: dict) -> str:
    """A signal's value and trip level, in its own units."""
    value, threshold = sig["value"], sig["threshold"]
    if name == "vol_spike":
        return f"{value:.2f}x short/baseline EWMA vol (trips above {threshold:.2f}x)"
    if name == "credit_velocity":
        return f"{value:+.0f} bps widening (trips at {threshold:+.0f} bps)"
    return f"{value:.2%} (trips at {threshold:.2%})"


def _tripwire_section(tripwire: dict) -> list[str]:
    """The ``## Crash Tripwire`` block (D-04): advisory, per-signal as-of, nothing imputed.

    - current signals read RED or GREEN;
    - a STALE signal names its date and the rebuild command, and reads "STALE (would be RED)"
      when its last value trips;
    - an UNAVAILABLE signal names what is missing and the rebuild command;
    - the escalation counts current signals only, and reads "none" only when all three are
      current and green: with no red and fewer than three current it reads UNKNOWN, never an
      all-clear (T-08.2-08).
    """
    signals = tripwire["signals"]
    run_date = pd.Timestamp(tripwire["run_date"]).date().isoformat()
    n_days = int(tripwire["stale_business_days"])
    n_total = len(signals)
    lines = [_TRIPWIRE_HEADING, ""]
    for name, sig in signals.items():
        state = sig["state"]
        if state == "unavailable":
            lines.append(
                f"- {sig['label']}: UNAVAILABLE — {sig['reason']}; nothing is imputed. "
                f"Build it with `{_DATA_BUILD_COMMAND}`."
            )
            continue
        as_of = pd.Timestamp(sig["as_of"]).date().isoformat()
        if state == "stale":
            label = "STALE (would be RED)" if sig["triggered"] else "STALE"
            lines.append(
                f"- {sig['label']}: {label} — {_tripwire_value(name, sig)}, as of {as_of}, more than "
                f"{n_days} business days before the run date {run_date}; refresh with `{_DATA_BUILD_COMMAND}`."
            )
            continue
        lines.append(f"- {sig['label']}: {state.upper()} — {_tripwire_value(name, sig)}, as of {as_of}")
    lines.append("")
    n_current = int(tripwire["n_current"])
    any_red = any(sig["state"] == "red" for sig in signals.values())
    if not any_red and n_current < n_total:
        lines.append(f"Escalation: UNKNOWN ({n_current} of {n_total} signals current — not green)")
    else:
        lines.append(
            f"Escalation: {tripwire['escalation'].value} ({n_current} of {n_total} signals current; "
            "nothing is imputed)"
        )
    lines.append("")
    lines.append(
        f"Run date {run_date}. A signal older than {n_days} business days is STALE and a missing input "
        "UNAVAILABLE; neither counts as green. The tripwire changes no weight on this page."
    )
    lines.append("")
    return lines


def _suspended_regime_view(scored_as_of_note: str | None, input_sensitivity_note: str | None) -> list[str]:
    """Sections 1-3 in no_regime mode (D-03): one block, no probabilities, belief or posture.

    No distribution is printed here, so the q2-ii constant-posterior sentence, which points at
    "the distribution above", names the served posterior instead.
    """
    lines = ["## Regime View (suspended)", "", _SUSPENDED_SENTENCE, ""]
    if input_sensitivity_note:
        input_sensitivity_note = input_sensitivity_note.replace(
            _CONSTANT_POSTERIOR_SENTENCE, _SUSPENDED_CONSTANT_POSTERIOR_SENTENCE
        )
    for note in (scored_as_of_note, input_sensitivity_note):
        if note:
            lines.append(note)
            lines.append("")
    return lines


def _regime_sections(
    probs: pd.Series,
    *,
    transition_matrix: pd.DataFrame,
    returns_by_regime: pd.DataFrame,
    active_regime: int | None,
    regime_belief: pd.Series | dict | None,
    min_obs_flag: int,
    scored_as_of_note: str | None,
    input_sensitivity_note: str | None,
) -> list[str]:
    """Sections 1-3 in regime_tilt mode: distribution, belief, active regime, trajectory and
    per-asset signals, exactly as the pre-08.2 page rendered them."""
    lines: list[str] = []
    # ── 1. Current regime distribution ────────────────────────────────────
    lines.append("## Current Regime Distribution")
    lines.append("")
    if scored_as_of_note:
        lines.append(scored_as_of_note)
        lines.append("")
    if probs.empty:
        lines.append("(no regime probabilities available)")
    else:
        for regime_id, p in probs.sort_values(ascending=False).items():
            lines.append(f"- regime {regime_id}: {p:.1%}")
    lines.append("")
    if input_sensitivity_note:
        lines.append(input_sensitivity_note)
        lines.append("")
    if regime_belief is not None:
        belief = pd.Series(regime_belief, dtype=float)
        lines.append("## Filtered Regime Belief (what the allocation consumed)")
        lines.append("")
        for regime_id, p in belief.sort_values(ascending=False).items():
            lines.append(f"- regime {regime_id}: {p:.1%}")
        lines.append("")

    # The hysteresis OUTPUT, and — adjacent to it — the rule that produced it.
    lines.append("## Active Regime (hysteresis state machine output)")
    lines.append("")
    if active_regime is None:
        lines.append("- active regime: none (neutral posture)")
    else:
        lines.append(f"- active regime: regime {active_regime}")
    lines.append("")
    lines.append(
        "Hysteresis cold-start rule (A1): with no prior active regime, the "
        "platform acts immediately on the highest-probability regime if it "
        "already clears the act threshold; otherwise it stays neutral "
        "until some regime first crosses the act threshold. Once active, a regime "
        "is held until its own probability falls below the unwind threshold."
    )
    lines.append("")
    lines.append(
        "The active regime is a reported label: it selects the trajectory and "
        "per-asset rows below and gates no weight (audit item A7, 08-A7.md). "
        "The weights come from the filtered belief through the no-trade band."
    )
    lines.append("")

    # ── 2. Trajectory (empirical transition matrix) ───────────────────────
    lines.append("## Trajectory (Empirical Transition Matrix)")
    lines.append("")
    if active_regime is not None and active_regime in transition_matrix.index:
        row = transition_matrix.loc[active_regime].sort_values(ascending=False)
        lines.append(f"From regime {active_regime}, next-regime probabilities:")
        for to_regime, p in row.items():
            lines.append(f"- -> regime {to_regime}: {p:.1%}")
    else:
        lines.append("(no trajectory available for the current regime)")
    lines.append("")

    # ── 3. Per-asset signals (returns-by-regime), flagging D11 cells ──────
    lines.append("## Per-Asset Signals (Returns by Regime)")
    lines.append("")
    if returns_by_regime.empty:
        lines.append("(no returns-by-regime data available)")
    elif active_regime is None:
        lines.append(_NEUTRAL_PER_ASSET_SENTENCE)
    else:
        sub = returns_by_regime[returns_by_regime["regime"] == active_regime]
        for row in sub.itertuples():
            flag = " [LOW-CONFIDENCE — short history, D11]" if row.n_obs < min_obs_flag else ""
            lines.append(
                f"- {row.asset} (regime {row.regime}): mean={row.mean_monthly_return:.2%} "
                f"sharpe={row.sharpe_annualized:.2f} n_obs={row.n_obs}{flag}"
            )
    lines.append("")
    return lines


def write_weekly_report(markdown: str, *, output_dir: Path | None = None) -> Path:
    """ALWAYS write the markdown to
    ``(output_dir or OUTPUT_DIR / "reports" / "platform") / "weekly_report.md"``
    (mkdir parents) — D-02: this happens unconditionally, regardless of
    ``--send-email``."""
    target_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR / "reports" / "platform"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "weekly_report.md"
    path.write_text(markdown, encoding="utf-8")
    log.info("Weekly report written: %s", path)
    return path


_RUN_ORDER = (
    f"python scripts/build_platform_data.py, then {SERVING_BUILD_COMMAND}, "
    "then python -m trading_crab_lib.platform.report.weekly"
)


def _load_serving_artifact(cm, name: str, *, model: bool = False):
    """Load one serving artifact; a missing one names the command that builds it."""
    try:
        return cm.load_model(name) if model else cm.load(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Serving artifact '{name}' is missing ({exc}). It is built by: {SERVING_BUILD_COMMAND}. "
            f"Run order: {_RUN_ORDER}."
        ) from exc


def _served_class_prior(cm, nowcaster) -> pd.Series:
    """The served model's training class prior (CR-01), validated against the model.

    Loaded from the ``nowcaster_class_prior`` artifact ``serving.py`` writes beside the
    model; never recomputed here from ``regime_labels`` or any other frame. Raises
    ValueError, naming the artifact and the build command, when its states are not
    ``set(model.classes_)``, when any prior is not > 0, or when it does not sum to 1 within
    1e-9 — each means the prior and the model are not from the same build.
    """
    frame = _load_serving_artifact(cm, SERVING_CLASS_PRIOR)
    prior = pd.Series(frame["prior"].to_numpy(dtype=float), index=[int(v) for v in frame["state"]])
    classes = sorted(int(c) for c in nowcaster.classes_)
    problem = None
    if sorted(prior.index) != classes:
        problem = f"covers states {sorted(prior.index)}, but the nowcaster's classes_ are {classes}"
    elif not (prior > 0.0).all():
        problem = f"has a non-positive prior: {prior.to_dict()}"
    elif abs(float(prior.sum()) - 1.0) > 1e-9:
        problem = f"sums to {float(prior.sum())!r}, not 1"
    if problem is not None:
        raise ValueError(
            f"Serving artifact '{SERVING_CLASS_PRIOR}' {problem}. The model and its training prior "
            f"must come from the same build; rebuild them together with: {SERVING_BUILD_COMMAND}"
        )
    return prior


def _model_columns(nowcaster, monthly_features: pd.DataFrame) -> list[str]:
    """The model's own columns (``feature_names_in_``), in its own order.

    Never the whole row: the served model is fit on ``fit_l2_nowcaster``'s active columns,
    and a row carrying any other column is not what it was trained on. Raises ValueError,
    before any state is loaded or saved, when the model declares no columns or when a model
    column is absent from ``monthly_features``.
    """
    names = getattr(nowcaster, "feature_names_in_", None)
    if names is None:
        raise ValueError(
            "The loaded nowcaster declares no feature_names_in_, so the report cannot tell "
            f"which columns it was trained on. Rebuild it with: {SERVING_BUILD_COMMAND}"
        )
    cols = [str(c) for c in names]
    absent = [c for c in cols if c not in monthly_features.columns]
    if absent:
        raise ValueError(
            f"monthly_features lacks {len(absent)} of the nowcaster's {len(cols)} columns: {absent}. "
            f"Rebuild the data and the serving artifacts ({_RUN_ORDER})."
        )
    return cols


def _scored_row(monthly_features: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, str]:
    """The row the report scores, and the "Scored as of" note that says which (ruling q1-c).

    Glenn's 08-12 ruling q1-c (08-SERVING.md §2.1): score the NEWEST month observed in
    every model column, exactly as observed — ``monthly_features[cols].dropna(how="any")``,
    last row. Nothing is imputed: no forward-fill, no interpolation, no fill from another
    series. Raises ValueError, before any state is loaded or saved, when no month is complete
    in the model columns. A latest complete month far behind the newest row does NOT refuse:
    staleness is judged per series against the run date and shown as a banner (A-12).
    """
    complete = monthly_features[cols].dropna(how="any")
    if complete.empty:
        raise ValueError(
            f"No monthly_features row is observed in all {len(cols)} of the nowcaster's model columns "
            f"{cols}, so there is no month the report can score. The report does not impute. "
            f"Rebuild the data ({_RUN_ORDER})."
        )
    as_of = pd.Timestamp(complete.index[-1])
    newest = pd.Timestamp(monthly_features.index[-1])
    lag = _months_between(as_of, newest)
    newer = monthly_features.loc[monthly_features.index > as_of, cols]
    lacking = [
        (pd.Timestamp(date).date().isoformat(), [c for c in cols if pd.isna(row[c])])
        for date, row in newer.iterrows()
    ]
    note = (
        f"Scored as of {as_of.date().isoformat()}, the latest month observed in all {len(cols)} of the "
        "nowcaster's model columns"
    )
    if lacking:
        note += (
            f" ({lag} month-end{'s' if lag != 1 else ''} behind the newest row). Newer rows lack model "
            "columns: " + "; ".join(f"{d} lacks {', '.join(missing)}" for d, missing in lacking) + "."
        )
    else:
        note += " (the newest row)."
    note += " Nothing is imputed."
    return complete.iloc[[-1]], note


_CONSTANT_POSTERIOR_SENTENCE = (
    "The distribution above does not depend on the features: it is the same every week."
)
_SUSPENDED_CONSTANT_POSTERIOR_SENTENCE = (
    "The served posterior does not depend on the features: it is the same every week."
)


def _input_sensitivity_note(nowcaster, monthly_features: pd.DataFrame, cols: list[str]) -> str:
    """How many DISTINCT posteriors the served model gives across history (ruling q2-ii).

    Glenn's 08-12 ruling q2-ii (08-SERVING.md §2.2): score every full-span month observed in
    all model columns with the already-fitted model and count the distinct posterior vectors
    with ``np.unique(..., axis=0)`` — exact float comparison, no rounding, no threshold. This
    only LOOKS: nothing is fitted. When the count is 1 the page says in plain words that the
    distribution does not depend on the features. The report is never withheld on the count
    (that was q2-iii, not chosen).
    """
    frame = monthly_features[cols].dropna(how="any")
    n_distinct = int(np.unique(np.asarray(nowcaster.predict_proba(frame), dtype=float), axis=0).shape[0])
    n_months = len(frame)
    note = (
        f"{n_distinct} distinct posterior vector{'s' if n_distinct != 1 else ''} across {n_months} complete "
        f"month{'s' if n_months != 1 else ''} ({pd.Timestamp(frame.index[0]).date().isoformat()} → "
        f"{pd.Timestamp(frame.index[-1]).date().isoformat()}): the served model scored on every month "
        f"observed in all {len(cols)} model columns, compared exactly (no rounding, no threshold)."
    )
    if n_distinct == 1:
        note += " " + _CONSTANT_POSTERIOR_SENTENCE
    return note


def _build_report_inputs(cfg: dict, cm=None) -> dict:
    """The full allocation-cycle orchestration (load -> update -> tilt ->
    save, load-before-save order per Pitfall 3): load the previous
    hysteresis state, update it with the current nowcaster probabilities,
    compute target weights via ``vol_targeted_tilt``, and persist the new
    state.

    Isolated as its own function — not inlined in ``main()`` — so tests can
    monkeypatch it directly without needing real Phase 1/3 checkpoint data
    (nowcaster model, monthly_features, regime_labels, returns_by_regime,
    asset_returns) on disk.

    Returns:
        dict with keys ``regime_probs`` (raw posterior), ``regime_belief``
        (the filtered belief the allocation consumed), ``active_regime`` (the
        hysteresis output), ``transition_matrix``, ``returns_by_regime``,
        ``target_weights`` / ``cash`` (the EXECUTED book, after the no-trade band),
        ``pre_band_target_weights`` / ``pre_band_cash`` (the tilt's target), ``no_trade_band``, and
        ``stale_series`` / ``stale_expected_through`` (A-12: series later than the run date
        allows, for the banner; empty when every watched series is current) and
        ``last_week_weights`` (the previous run's executed book, read before this run saves
        its own; None with no band configured or no earlier run), ``allocation_mode`` (D-01:
        ``no_regime`` targets ``no_regime_target``; ``regime_tilt`` tilts on the belief) and
        ``mode_note`` (A-15: set on the run that switched mode and on its same-month re-runs).
    """
    cm = cm or get_platform_checkpoint_manager()
    # D-01: validated before any artifact load or state I/O, so a bad value writes nothing.
    mode = allocation_mode_from_config(cfg)

    # The four serving artifacts are built by SERVING_BUILD_COMMAND (report/serving.py);
    # a missing one says so.
    nowcaster = _load_serving_artifact(cm, "nowcaster", model=True)
    # CR-01: the prior the filter's likelihood divides by, validated against this model
    # before anything is scored and before any state is loaded or saved.
    class_prior = _served_class_prior(cm, nowcaster)
    # Live scoring is "looking", not "fitting", so it takes the explicit
    # full-span opt-in. The dev checkpoint stops at the 2020-12 holdout
    # boundary; loading it here would silently score December 2020 as "today",
    # every week, forever.
    monthly_features = load_full_span("monthly_features")
    regime_labels = cm.load("regime_labels")["state"]
    returns_by_regime = _load_serving_artifact(cm, "returns_by_regime")
    asset_returns = _load_serving_artifact(cm, "asset_returns")

    # Score the model's own columns, never the whole row, on the latest month complete in
    # them (ruling q1-c); validated before any state load or save, so a refusal writes nothing.
    cols = _model_columns(nowcaster, monthly_features)
    row, scored_as_of_note = _scored_row(monthly_features, cols)
    # A-12: per-series staleness against the run date, over the series the lag table lists
    # (derived research series inherit their inputs' lags) that the full-span frame carries.
    report_cfg = cfg.get("report", {})
    grace_days = int(report_cfg.get("staleness_grace_days", _DEFAULT_STALENESS_GRACE_DAYS))
    watched = [
        str(name) for name, entry in (cfg.get("publication_lags") or {}).items()
        if entry != DERIVED and name in monthly_features.columns
    ]
    run_date = _run_date()
    late = stale_series(monthly_features, watched, run_date=run_date, grace_days=grace_days)
    stale_expected_through = expected_month_end(run_date, grace_days)
    proba = nowcaster.predict_proba(row)[0]
    regime_probs = pd.Series(proba, index=nowcaster.classes_)
    # Ruling q2-ii: disclose how input-dependent that posterior is (looking, not fitting).
    input_sensitivity_note = _input_sensitivity_note(nowcaster, monthly_features, cols)

    allocation_cfg = cfg.get("allocation", {})
    act_threshold, unwind_threshold = hysteresis_thresholds(cfg)
    no_trade_band = no_trade_band_from_config(cfg)

    # Bayes filter (plan 08-08): load BEFORE save, immediately before the hysteresis,
    # so the belief the hysteresis sees is this run's.
    state_index = list(range(int(cfg.get("labeling", {}).get("K", 5))))
    # The SCORED month, not the newest row: the belief and the band step once per scored
    # month, so a re-run while the newest row stays ragged neither double-counts nor compounds.
    as_of = pd.Timestamp(row.index[0])
    prev_belief = load_regime_belief(cm)  # load BEFORE save (Pitfall 3)
    regime_belief = advance_regime_belief(
        prev_belief, regime_labels, regime_probs, class_prior=class_prior, state_index=state_index, as_of=as_of
    )
    save_regime_belief(regime_belief, cm, as_of=as_of)  # save AFTER load

    prev_active = load_active_regime(cm)  # load BEFORE save (Pitfall 3)
    active_regime = update_active_regime(
        regime_belief,
        prev_active,
        act_threshold=act_threshold,
        unwind_threshold=unwind_threshold,
    )
    save_active_regime(active_regime, cm)  # save AFTER load

    if mode == "no_regime":
        # D-01: the measured ablation leg's target; the belief above is kept, not consumed.
        tilt = no_regime_target(asset_returns, as_of, cfg)
    else:
        tilt = vol_targeted_tilt(
            regime_belief,
            returns_by_regime,
            asset_returns,
            target_vol_annual=allocation_cfg.get("target_vol_annual", 0.10),
            halflife=allocation_cfg.get("ewma_halflife_months", 6),
            min_obs=allocation_cfg.get("portfolio_vol_min_obs", 12),
        )

    # The no-trade band (plan 08-09): the SAME function the drivers call. Held-book I/O
    # happens only when a band is configured — with none, the executed book is the target.
    # The mode record (A-15) is gated the same way, so band-free configs write nothing new.
    last_week, held, changed_from = None, None, None
    if no_trade_band is not None:
        last_week = load_last_executed_weights(cm)  # load BEFORE save
        switched, changed_from = mode_switch(
            load_allocation_mode(cm), mode=mode, as_of=as_of, executed_exists=_has_executed_book(cm)
        )  # load BEFORE save
        # A-15: a switch trades the new target in full; a band against the old mode's book
        # would hold a hybrid no measured leg ever held.
        held = None if switched else load_held_weights(cm, as_of=as_of)  # load BEFORE save
    executed = execute_rebalance(tilt["weights"], tilt["cash"], held, band=no_trade_band)
    if no_trade_band is not None:
        save_executed_weights(executed["weights"], held, cm, as_of=as_of)  # save AFTER load
        save_allocation_mode(mode, cm, as_of=as_of, changed_from=changed_from)  # save AFTER load

    return {
        "regime_probs": regime_probs,
        "regime_belief": regime_belief,
        "active_regime": active_regime,
        "transition_matrix": empirical_transition_matrix(regime_labels),
        "returns_by_regime": returns_by_regime,
        "target_weights": executed["weights"],
        "cash": executed["cash"],
        "pre_band_target_weights": tilt["weights"],
        "pre_band_cash": tilt["cash"],
        "no_trade_band": no_trade_band,
        "scored_as_of_note": scored_as_of_note,
        "input_sensitivity_note": input_sensitivity_note,
        "stale_series": late,
        "stale_expected_through": stale_expected_through,
        "last_week_weights": last_week,
        "allocation_mode": mode,
        "mode_note": _mode_note(mode, changed_from) if changed_from else None,
    }


def build_weekly_page(cfg: dict, cm=None, *, output_dir: Path | None = None) -> tuple[str, Path]:
    """Run the allocation cycle, evaluate the tripwire, assemble the page and write it.

    The page ``main`` writes, with the checkpoint manager and the output directory explicit, so
    a notebook or a smoke can build it in a scratch copy (N4) without touching the tracked
    namespace. Order: ``_build_report_inputs`` (load -> advance -> save of the belief, the
    hysteresis and the executed book), then ``evaluate_tripwire`` on the same ``cm`` at
    ``_run_date()``, then ``assemble_weekly_report`` and ``write_weekly_report(output_dir=...)``.
    The tripwire is read after the book is executed and feeds only its own section (D-04).

    Returns ``(markdown, path)``.
    """
    cm = cm or get_platform_checkpoint_manager()
    report_cfg = cfg.get("report", {})
    inputs = _build_report_inputs(cfg, cm)
    tripwire = evaluate_tripwire(cfg, cm, run_date=_run_date())
    markdown = assemble_weekly_report(
        regime_probs=inputs["regime_probs"],
        regime_belief=inputs.get("regime_belief"),
        active_regime=inputs["active_regime"],
        transition_matrix=inputs["transition_matrix"],
        returns_by_regime=inputs["returns_by_regime"],
        target_weights=inputs["target_weights"],
        accounts=report_cfg.get("accounts", []),
        cash=inputs["cash"],
        no_trade_band=inputs.get("no_trade_band"),
        min_obs_flag=report_cfg.get("min_obs_flag", _DEFAULT_MIN_OBS_FLAG),
        trade_threshold_pct=report_cfg.get("trade_threshold_pct", _DEFAULT_TRADE_THRESHOLD_PCT),
        scored_as_of_note=inputs.get("scored_as_of_note"),
        input_sensitivity_note=inputs.get("input_sensitivity_note"),
        stale_series=inputs.get("stale_series"),
        stale_expected_through=inputs.get("stale_expected_through"),
        last_week_weights=inputs.get("last_week_weights"),
        asset_classes=_class_by_ticker(cfg),
        regime_view_suspended=inputs.get("allocation_mode") == "no_regime",
        allocation_mode=inputs.get("allocation_mode"),
        mode_note=inputs.get("mode_note"),
        tripwire=tripwire,
    )
    return markdown, write_weekly_report(markdown, output_dir=output_dir)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: assemble + always write the markdown; email opt-in.

    ``main([])`` writes the markdown and does NOT call
    ``send_weekly_email``. ``main(["--send-email"])`` calls
    ``build_weekly_email_body`` / ``load_email_config`` /
    ``send_weekly_email`` exactly once each (D-02) — never on the default
    path.
    """
    parser = argparse.ArgumentParser(
        description="Assemble (and optionally email) the platform weekly report"
    )
    parser.add_argument(
        "--send-email", action="store_true", help="Send the report via SMTP (opt-in, D-02)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    cfg = load_platform_config()
    _, report_path = build_weekly_page(cfg)

    if args.send_email:
        subject, body = build_weekly_email_body(
            report_path.parent, subject_prefix="Trading-Crab Platform Weekly Report"
        )
        email_cfg = load_email_config()
        if email_cfg:
            send_weekly_email(email_cfg, subject, body)  # plot_paths=None in v1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
