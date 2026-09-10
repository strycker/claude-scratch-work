# Phase 6: Platform Notebook Suite - Research

**Researched:** 2026-09-09
**Domain:** Visualization library authoring (matplotlib) + Jupyter notebook authoring, over a
walk-forward financial backtesting/regime-labeling codebase
**Confidence:** HIGH (all claims below are read from the actual source tree and actual
artifacts on disk in this session, not from training-data assumptions about a generic
plotting/notebook task)

## Summary

Phase 6 has two genuinely separate halves. **Half A** is a brand-new visualization package,
`src/trading_crab_lib/platform/plotting/`, which does not exist today — confirmed by direct
read of all 48 `platform/` modules: zero `matplotlib`/`seaborn` imports anywhere. **Half B** is
six thin notebooks under `notebooks/platform/` that call Half A's functions against real
checkpoints. The legacy `src/trading_crab_lib/plotting/` package (9 submodules, `_save_or_show`
/ `_regime_color` / `CUSTOM_COLORS` / `REGIME_CMAP` idioms) is confirmed to be a clean pattern
source with no `RunConfig` equivalent needed on the platform side — `platform/config.py` has no
`RunConfig`-like object, so D-02's `plot_x(data, *, save_path=None, show=False) -> Figure`
signature is the correct, verified-necessary shape.

The environment has changed materially since discussion. `data/checkpoints/platform/` now
holds real dev data (`monthly_raw` 776×45, `monthly_features` 708×53 fenced at 2020-12-31,
`daily_raw` 14,290×22), and `data/holdout/monthly_features` holds the 68 post-cutoff rows —
`honesty.holdout.load_full_span("monthly_features")` was run in this session and returns
776×53, 1962-01→2026-08, confirming the D-06 opt-in works exactly as documented. **No
`regime_labels`/`regime_confidences`/`regime_profiles` checkpoint exists on disk** — confirmed
by `CheckpointManager.list()` — so P3 must call `label_regimes()` itself; this is a fast,
in-process call (a single `fit_jump_model`, not the 588-step backtest) and is not a blocker.

The single highest-value finding of this research answers the phase's most consequential open
question: **the A13 side-by-side panel does not require re-running the expensive 588-step
walk-forward.** The per-window active-feature-count timeline (`driver._window_active_features`,
private) was reproduced in this session in **1.2 seconds** for all 588 steps and its 7 change
points matched the audit's reported dates exactly (1972-01→4, 1972-02→6, 1972-04→8, 1973-02→9,
1986-06→10, 1995-02→12, 2000-01→13). The smoothed reference labeling
(`report._reference_label_columns`, fixed 9 columns) was reproduced with a single
`fit_jump_model` call in **0.6 seconds**. Only the walk-forward's own *filtered* state path
(what the L2 nowcaster believed at each of the 588 decision dates) requires the full,
multi-minute `run_full_backtest_evaluation()` — and P6 already needs that computation for its
own success criterion 5. This reframes A13 from "new expensive work" to "one new cheap helper
plus reuse of a computation P6 must do anyway" — but it also surfaces a real wave-sequencing
decision the planner must make explicitly (see Wave/Dependency Shape below).

Also load-bearing: **the Faber/60-40 anomalies named in the original CONTEXT.md are gone from
the artifacts on disk today.** `outputs/reports/platform/backtest_kpi_table.parquet`, read
directly in this session, shows Faber at 6.372592/-0.189421 and 60/40 at 5.047073/-0.269625 —
matching `BASELINE-v1-tracer-bullet.md`'s "current reference run" exactly, not the void
-99.7%/-2.3% numbers from the original discussion. P6 should still narrate this history (an
operator opening the notebook needs to know these were once wrong and are now fixed — that is
part of "periodic V&V"), but must not present the old numbers as current.

**Primary recommendation:** Build `platform/plotting/` as a fresh, dependency-light package
with `plot_x(data, *, save_path: Path | None = None, show: bool = False) -> Figure` signatures,
decomposed by layer (core/data/features/regime/nowcaster/allocation/backtest); author the six
notebooks via `nbformat`'s programmatic API (already installed, zero new dependency) rather than
hand-written JSON or jupytext; keep D-09 drift-against-baseline AND add D-11's reinstated
plausibility bands, using the codebase's own existing plausibility-assertion patterns
(`assert_yield_units_plausible`, `TestKpisAreOnAPlausibleScale`) as the template rather than
inventing new ones; and resolve the A13 computation-source decision (embed vs. persist vs.
defer-to-P6) as an explicit planning decision, not an implicit one.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Plot rendering (matplotlib figures) | Library (`trading_crab_lib.platform.plotting`) | — | Pure functions, no I/O side effects beyond optional `save_path`; ADR #11 mandates library placement |
| Checkpoint/artifact loading | App (notebook cell) | Library (shared load helper, D-10) | Notebooks are app-side (D-03); a shared "load or raise with actionable instructions" helper is thin enough to live in either `platform/plotting/core.py` or a small `notebooks/platform/_utils.py` — Claude's Discretion per CONTEXT.md |
| Drift/plausibility computation | Library | — | Testable pure functions; notebooks call, never define inline (same ADR #11 doctrine extended to non-plot diagnostics) |
| Regime label generation (`label_regimes`) | Library (`platform.labeling.diagnostics`) | — | Already exists; P3 calls it, does not reimplement it |
| Full backtest evaluation (`run_full_backtest_evaluation`) | Library (`platform.evaluation.report`) | — | Already exists (Phase 5); P6 calls it, does not reimplement it |
| Economic-history overlay data (USREC + event list) | App (notebook-time fetch) or Library helper | — | D-12 requires no new ingestion pipeline column; a direct `fredapi` call at notebook-render time, independent of `monthly_features`, avoids touching the Phase-1 ingestion schema and avoids a rebuild — recommended placement below |
| Cold-start sign-off record | App (P3 notebook markdown cell) | — | D-15: plain markdown cell, no library code |
| Notebook static checks (CI) | Test suite (`tests/unit/`) | — | D-17: parses `.ipynb` JSON, asserts import resolvability and "no inline plotting" — a test-suite concern, not library or app runtime |

## User Constraints

<user_constraints>
### Locked Decisions (from 06-CONTEXT.md, verbatim, as amended)

**D-01 (still locked):** Fresh, self-contained `platform/plotting/` importing nothing from the
legacy `trading_crab_lib.plotting` package. Verified this session: `platform/` genuinely
imports zero matplotlib/seaborn today.

**D-02 (still locked):** Plain kwargs, return the `Figure`.
`plot_x(data, *, save_path: Path | None = None, show: bool = False) -> Figure`. No `RunConfig`
equivalent is invented for the platform. Verified this session: `platform/config.py` has no
`RunConfig`-shaped object — `load_platform_config()` returns a plain dict, so there is nothing
to port even if D-02 were reconsidered.

**D-03 (still locked):** Notebooks live in `notebooks/platform/`.

**D-04 (still locked):** `platform/plotting/` lives in the library
(`src/trading_crab_lib/platform/plotting/`).

**D-05 (still locked):** Notebooks only — no report wiring this phase. `backtest_report.md`
and `weekly_report.md` stay text-only.

**D-06 (still locked):** The fence is on fitting, not on looking. Notebooks read the full span
via `honesty.holdout.load_full_span(name)`. **Implementation confirmed to exist and work** —
verified this session by calling it directly: returns 776×53 rows, 1962-01→2026-08, for
`monthly_features`.

**D-07 (still locked):** Post-2020 observations that change a decision get recorded with their
date.

**D-08 (still locked):** No conditional "coverage-metadata exemption" — deleted.

**D-09 (still locked, now paired with reinstated D-11):** Drift baseline is the pre-2021 fitted
window.

**D-10 (still locked):** Missing artifacts stop with actionable instructions. A shared load
helper raises naming the missing checkpoint and the command that produces it. Notebooks never
write production checkpoints.

**D-11 — REVERSED by Amendment 1 item A.** The original rejection of a plausibility banner is
disproven by the percent-vs-decimal yield defect (a uniformly-wrong series shows zero drift).
**Amended decision: keep D-09 drift AND add a plausibility check.** Audit item A3. Existing
patterns to extend: `assert_yield_units_plausible` (`platform/splice.py:124`),
`TestKpisAreOnAPlausibleScale` (`tests/integration/test_mini_backtest.py:218`),
`test_percent_yields_produce_a_plausible_treasury_index`
(`tests/unit/test_platform_splice.py:194`).

**D-12 (still locked):** Economic history overlay = FRED `USREC` + a plain dated event list
(1973 oil, 1980 Volcker, 1987, 1998 LTCM, 2000 dot-com, 2008 GFC) as a module constant or a few
YAML lines. No dedicated events file, no per-entry provenance schema.

**D-13 (still locked):** Overlay + regime×era contingency table. No significance test.

**D-14 (still locked):** No (K, λ) sweep. P3 renders diagnostics for the shipped config
(K=5, λ=52.0, n_restarts=10) — verified this session directly from
`config/platform_settings.yaml`'s `labeling:` section.

**D-15 (still locked):** P3 carries the only sign-off cell, plain markdown, no helper, no YAML
ledger.

**D-16 (still locked, expectation revised by Amendment 2 item E):** A negative P3 verdict is
recorded and does not block. **The prior "expect negative" framing is now WRONG** — the
corrected labeling (post A4/A12 fix) maps onto real economic history (1973-09 oil shock,
1981-10 Volcker, 1988-09 disinflation, 1996-08 late-90s expansion, 2008-07→GFC crisis state,
2009-06 recovery). Build P3 expecting a genuine question, not a foregone answer. D-16 remains
the safety valve either way.

**D-17 (still locked):** CI does library unit tests plus static notebook checks — no cell
execution (`nbmake`/`papermill` rejected).

**D-18/D-19 — PARTLY STALE, corrected this session.** `daily_raw` premise ("empty 0,0") is
stale: verified this session as **14,290×22** (equity tickers only — see the discrepancy note
in Section B below). The `regime_labels` premise ("does not exist") **still holds** — verified
this session via `CheckpointManager.list()`: only `daily_raw`, `monthly_raw`, `monthly_features`
exist in `data/checkpoints/platform/`. P3/P4/P5 still have no persisted labeler output; P3 must
call `label_regimes()` itself (cheap — see Section B).

**D-20 — SATISFIED, verified this session.** `data/holdout/monthly_features.parquet` exists
(68 rows). `monthly_features` dev checkpoint verified at 708 rows, 1962-01→**2020-12-31**
exactly (no post-cutoff rows). `honesty.holdout.load_full_span()` verified callable and correct.

**D-21 (still locked):** Package split is a Phase 7 concern.

**NEW — Amendment 2 item F (P3 scope addition, confirmed not a D-14 breach):** P3 must render
**both labelings side by side** — the fixed 9-column reference
(`report._reference_label_columns`) and the walk-forward's per-window labeling
(`driver._window_active_features`, changing 4→6→8→9→10→12→13 across the backtest) — with
disagreement quantified and feature-set change dates marked. The §5.4 ratio (currently 0.591,
164-month median lag) must carry an explicit "not interpretable — see A13" caveat wherever
shown, until this is resolved.

**NEW — Amendment 2 item G:** A14 (canonicalization fallback) is closed as negligible (1 of 588
steps, 0.2%). P3's churn panel needs no special handling for it.

### Claude's Discretion (from CONTEXT.md, verbatim)

- Submodule split inside `platform/plotting/` (per-layer vs per-notebook) and figure
  sizing/DPI conventions.
- Where the shared checkpoint-load helper lives (plotting package vs a notebook utils module)
  and its exact error text.
- The specific drift statistic(s) comparing current window to the pre-2021 baseline
  (e.g. standardized mean shift, KS distance, rolling z vs baseline σ) — must be simple,
  transparent, and explained in the notebook.
- Exact panel composition per notebook, beyond what ROADMAP criteria 4 and 5 mandate.
- Whether the post-2020 decision record (D-07) is a markdown log, a YAML file, or a
  registry row — pick the lightest option that stays greppable.

### Deferred Ideas (OUT OF SCOPE — verbatim)

- Wire `platform/plotting/` figures into the report writers (`backtest_report.md`, weekly
  email) — mirrors legacy D39's `attach_plots`. Out of scope here.
- Tuning surfaces: (K, λ) sweep, feature/metric noise screening, asset-class selection.
- `sign_off()` helper + machine-readable ledger — dropped for a markdown cell (D-15).
- Relocating pipeline-shaped platform modules (`backtest/driver.py`, `report/weekly.py`,
  `tripwire/monitor.py`) into the app package — Phase 7 (D-21).
- Notebook execution in CI via `nbmake`/`papermill` with synthetic fixtures — D-17.
- Holdout carve repair — SATISFIED (D-20), not phase work.
- Fixing the Faber / 60-40 KPI anomalies — **already fixed as of 2026-09-08/09**, confirmed by
  reading the current artifacts (see Section E below). P6 narrates the history; does not
  re-diagnose it.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| NB-01 | Six platform notebooks (P1–P6) covering L0–L4 + evaluation, serving as a periodic V&V surface; each runs top-to-bottom against real checkpoints, reads the full span via the explicit holdout opt-in while fitting stays fenced at 2020-12, compares current behavior against the pre-2021 fitted baseline, and calls plotting logic from `platform/plotting/` rather than defining it inline. P3 additionally carries a cold-start sign-off cell. | Sections A (plotting library design), B (data availability), D (plausibility bands), F (wave shape) below directly scope the implementation. The A13 side-by-side requirement (Amendment 2 item F) is folded into P3's NB-01 scope and is addressed in detail in Section B and F. |
</phase_requirements>

## Standard Stack

### Core

| Library | Version (verified) | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | 3.8+ constraint in `pyproject.toml` extras; import succeeds in `.venv` | All figure rendering | Already the incumbent's plotting engine (legacy `plotting/core.py`); zero new dependency — `[plotting]` extra (`matplotlib>=3.8`, `seaborn>=0.13`) already exists in `src/trading_crab_lib/pyproject.toml:58-61` and is already installed (env description: `trading-crab-lib[all,dev]`) |
| nbformat | **5.11.1, verified installed in `.venv` this session** | Programmatic notebook (.ipynb) construction and static parsing for CI checks | `[VERIFIED: .venv import check]` Already present — no new dependency. Confirmed by `import nbformat; nbformat.__version__` in this session. |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| seaborn | 0.13+ (same `[plotting]` extra) | Optional statistical plots (e.g. correlation heatmaps in P2) | Only if a heatmap/violin plot genuinely benefits; matplotlib alone covers most of P1-P6's needs (line plots, bar charts, scatter, stacked area) |
| fredapi | already a core dependency (used throughout `platform/ingestion/`) | Direct USREC fetch for D-12's recession-bar overlay | Notebook-time (or plotting-helper-time) call, independent of the `monthly_features` pipeline — see Section B |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Programmatic `nbformat` generation | jupytext paired `.py` files | jupytext is **NOT installed** `[VERIFIED: .venv import check — ImportError]` — would be a new dependency, and the existing 12 legacy notebooks are stored as plain committed `.ipynb` with outputs, not jupytext pairs (verified via `git ls-files notebooks/` and reading `01_ingestion.ipynb`'s JSON — `nbformat 4.5`, stored cell outputs present). Introducing jupytext for only the 6 new platform notebooks would create two different notebook-authoring conventions in one repo. |
| Programmatic `nbformat` generation | Hand-written raw `.ipynb` JSON | Error-prone (malformed JSON, wrong `nbformat_minor`, missing `execution_count: null`) with no payoff — nbformat's Python API produces valid JSON by construction and is trivially scriptable, so there is no reason to hand-edit JSON |
| `nbmake`/`papermill` CI execution | Static notebook checks only | **NOT installed** `[VERIFIED: .venv import check — ImportError for both]`. D-17 already rejected this route for the same reasons (new deps, fixture generator per artifact, "green means ran on fake data"). Confirmed unchanged. |

**Installation:** No new packages required. `pip install -e "src/trading_crab_lib/[plotting,all,dev]"` (already the documented dev-install command) covers everything Phase 6 needs.

**Version verification:** `matplotlib`, `seaborn`, `nbformat` all confirmed importable in `.venv` this session; `jupytext`, `nbmake`, `papermill` all confirmed **not installed** (`ImportError`) this session. No registry lookups were needed — this phase adds zero new third-party packages.

## Package Legitimacy Audit

**Not applicable — this phase installs no new external packages.** `platform/plotting/` reuses
the already-present `matplotlib`/`seaborn` extra; notebook authoring uses the already-installed
`nbformat`. No `Package Legitimacy Gate` run was required.

## Architecture Patterns

### System Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │  data/checkpoints/platform/  (dev, fenced ≤2020-12-31)   │
                    │  monthly_raw · monthly_features · daily_raw              │
                    └───────────────┬───────────────────────┬─────────────────┘
                                    │                        │
                    ┌───────────────▼─────────┐   ┌──────────▼──────────────┐
                    │ data/holdout/            │   │ outputs/reports/        │
                    │  monthly_features (68 rows│   │  platform/              │
                    │  >2020-12-31)             │   │  backtest_*, model_*    │
                    └───────────────┬───────────┘   └──────────┬──────────────┘
                                    │                           │
              honesty.holdout.load_full_span()      loaded directly (already
              [the ONLY sanctioned "looking" path]    on disk, no opt-in needed
                                    │                           │
                                    ▼                           ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │                    notebooks/platform/*.ipynb  (app tier)         │
        │  P1 P2 P3 P4 P5 P6 — each: load checkpoint/artifact → call        │
        │  plotting.<layer>.plot_x(...) → render inline → prose commentary  │
        └───────────────────────────┬─────────────────────────────────────┘
                                    │  calls (never inline logic — ADR #11)
                                    ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │      src/trading_crab_lib/platform/plotting/  (library tier)      │
        │  core.py (save/show, palette)  ·  data.py  ·  features.py         │
        │  regime.py (incl. A13 side-by-side)  ·  nowcaster.py              │
        │  allocation.py  ·  backtest.py  ·  drift.py (D-09/D-11 checks)    │
        └───────────────────────────┬─────────────────────────────────────┘
                                    │  reads (never writes — D-10)
                                    ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │  platform.labeling.diagnostics.label_regimes()   (P3, on demand)  │
        │  platform.evaluation.report.run_full_backtest_evaluation()  (P6)  │
        │  — both PRE-EXISTING, Phase-3/5 code, called but never modified   │
        └─────────────────────────────────────────────────────────────────┘
```

A reader traces P3's flow left-to-right: `monthly_features` (dev, fenced) →
`label_regimes()` (in-process, ~seconds) → `regime_labels`/`regime_confidences` →
`plotting.regime.plot_regime_timeline_with_history(..., usrec, events)` → rendered inline →
sign-off markdown cell. P6's flow: `monthly_raw`/`monthly_features` (dev, fenced, via
`run_full_backtest_evaluation()`, which internally applies `split_by_holdout_boundary`) →
equity curves / KPI table / `full_sample_states` / `per_step_metrics` (all in-memory, only
some persisted) → `plotting.backtest.*` functions → rendered inline.

### Recommended Project Structure

```
src/trading_crab_lib/platform/plotting/
├── __init__.py          # re-exports, mirrors legacy plotting/__init__.py pattern
├── core.py              # _save_or_show(fig, save_path, show), CUSTOM_COLORS/REGIME_CMAP
│                         #   (reuse legacy palette or pick a fresh one — Claude's Discretion),
│                         #   _in_jupyter(), a shared load-or-raise-actionably helper (D-10)
├── data.py               # P1: coverage timelines, splice join points, ALFRED vintage vs
│                         #   revised overlay, NaN maps
├── features.py            # P2: lean-13 time series, fast/slow/agency grouping, causal-vs-
│                         #   centered overlay (never mixed — C2), correlation matrix
├── regime.py              # P3: regime timeline + USREC/event overlay, sojourn distribution,
│                         #   transition matrix, soft confidences, churn, per-regime profiles,
│                         #   PLUS the A13 side-by-side (reference vs walk-forward) panel and
│                         #   feature-set-change-date markers
├── nowcaster.py            # P4: calibration curve, transition-window/overall/steady-state
│                         #   accuracy, detection lag, filtered-probability paths
├── allocation.py           # P5: returns-by-regime, EWMA vol, tilt weights over time,
│                         #   hysteresis state path, turnover
├── backtest.py             # P6: equity curves (strategy/ablation/SPY/60-40/Faber), drawdown,
│                         #   KPI table, Brier, confusion, sojourn-lag headline w/ resolved count
└── drift.py                # D-09 drift-against-baseline + D-11 plausibility bands, shared
                            #   across all six notebooks; the one genuinely new "logic" module

notebooks/platform/
├── P1_data_spine.ipynb
├── P2_features_taxonomy.ipynb
├── P3_regime_labeling.ipynb        # carries the sign-off cell
├── P4_nowcaster.ipynb
├── P5_assets_allocation.ipynb
└── P6_backtest_evaluation.ipynb
```

This decomposition maps 1:1 onto P1..P6 (plus a cross-cutting `drift.py`), mirroring the legacy
convention (`plotting/{ingestion,features,clustering,regime,prediction,assets,diagnostics}.py`
map onto legacy pipeline steps 01-09). `core.py` is the only module every notebook imports.

### Pattern 1: Save/show without a config object

**What:** Every legacy plot function takes `run_cfg: RunConfig` and calls
`_save_or_show(fig, filename, run_cfg)`, which checks `run_cfg.save_plots` / `run_cfg.show_plots`
and calls `plt.close(fig)`. The platform has no `RunConfig` — `load_platform_config()` returns a
plain dict (`platform/config.py:56-94`), confirmed by direct read this session.

**When to use:** Every `platform/plotting/` function.

**Example:**
```python
# Source: pattern adapted from src/trading_crab_lib/plotting/core.py:69-83
# (legacy _save_or_show — NOT imported, D-01), re-derived for D-02's plain-kwargs signature.
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt


def _save_or_show(fig: plt.Figure, *, save_path: Path | None, show: bool) -> plt.Figure:
    """Finalize a figure per D-02: caller decides save/show explicitly, no RunConfig."""
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show or _in_jupyter():
        plt.show()
    return fig  # notebooks render inline naturally on return in a Jupyter cell


def plot_regime_timeline(
    states: "pd.Series",
    *,
    usrec: "pd.Series | None" = None,
    events: list[tuple[str, str, str]] | None = None,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 4))
    # ... plotting logic ...
    return _save_or_show(fig, save_path=save_path, show=show)
```

### Pattern 2: The shared "load-or-raise-actionably" helper (D-10)

**What:** A single function every notebook's first cell calls, so a missing checkpoint fails
with the exact remediation command rather than a bare `FileNotFoundError`.

**When to use:** Every notebook's data-loading cell.

**Example:**
```python
# Source: derived from CheckpointManager.load()'s existing FileNotFoundError contract
# (src/trading_crab_lib/checkpoints.py) — this wraps it, does not reimplement it.
from __future__ import annotations
import pandas as pd
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager


def load_or_explain(name: str, *, rebuild_hint: str) -> pd.DataFrame:
    """Load *name* from the platform checkpoint namespace or raise with the fix command."""
    try:
        return get_platform_checkpoint_manager().load(name)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Checkpoint '{name}' not found in data/checkpoints/platform/. "
            f"Run: {rebuild_hint}"
        ) from exc
```

### Pattern 3: The A13 side-by-side (verified cheap; see Section B/F for the full analysis)

```python
# Source: composed from real function signatures read this session —
# report.py:442 (_reference_label_columns), driver.py:98 (_window_active_features),
# honesty/walkforward.py:38 (expanding_steps). All three confirmed callable and their
# combined cost measured empirically at ~1.8s total in this session (see Section B).
from trading_crab_lib.platform.honesty.walkforward import expanding_steps
from trading_crab_lib.platform.backtest.driver import _window_active_features  # currently private

def active_feature_count_timeline(dev_features, lean_cols, *, min_history, min_train):
    steps = list(expanding_steps(dev_features.index, min_train=min_train))
    counts = []
    for t, train_index, _test_index in steps:
        active = _window_active_features(dev_features.loc[train_index], lean_cols, min_history=min_history)
        counts.append((t, len(active)))
    return counts  # -> [(1972-01-31, 4), (1972-02-29, 6), ..., (2000-01-31, 13)]
```

`_window_active_features` is currently `_`-prefixed (private) in `backtest/driver.py:98`. The
plan must decide: promote it to a public function (e.g. re-export from a new
`backtest/__init__.py` name, or rename without the underscore), or duplicate its four-line body
inside `platform/plotting/regime.py`. Promoting is cleaner (single source of truth) and is a
one-line change to `driver.py`; duplication risks drift if `min_history` semantics change later.
**Recommend promoting** — this is a trivial, low-risk edit to Phase-5 code that does not touch
its behavior (renaming/re-exporting a pure function, not modifying logic), which is a much
smaller "touch" than the CONTEXT.md fence anticipated when it said "no report writers modified."

### Anti-Patterns to Avoid

- **Reimplementing `_save_or_show` with `RunConfig`:** the platform genuinely has no such
  object; do not invent one just because the legacy pattern has one (D-02 already settled this,
  verified correct this session).
- **Calling `run_backtest()` or `run_full_backtest_evaluation()` more than once per notebook
  execution:** each full walk-forward run is "minutes of ... sklearn work" per the driver's own
  docstring (`backtest/driver.py:354`) — P6 (and any notebook needing the walk-forward's
  filtered path for A13) should call it exactly once per notebook run and reuse the returned
  dict for every panel.
- **Persisting a new production checkpoint from inside a notebook:** violates D-10 explicitly
  ("Notebooks never write production checkpoints"). If A13 needs `full_sample_states`/
  `per_step_metrics` persisted for reuse across notebook re-opens, that persistence must be
  added to `platform/evaluation/report.py` (library code, Phase-5-owned), not to a notebook.
- **Mixing centered and causal features in one P2 plot (C2, ADR #1):** the platform's
  `monthly_features` checkpoint is causal-only (no `features.parquet`/`features_supervised.parquet`
  split exists on the platform side the way the legacy quarterly pipeline has one) — verify
  this assumption against `transforms_monthly.py` before assuming there is nothing to
  distinguish; if the platform genuinely only produces one (causal) feature file, P2's
  "causal-vs-centered overlay" panel from pre-planning's table may not apply as literally as
  the legacy pattern suggests, and the plan should confirm this rather than copy the legacy
  panel description verbatim.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Regime state fitting | A new labeler in the notebook | `platform.labeling.diagnostics.label_regimes()` | Already implements the exact DP-decode jump model (design §4.1), canonicalization, confidences, churn, and diagnostics persistence — verified as the strongest-tested module in the project (Phase 3's DP-decode oracle test) |
| Backtest KPIs / equity curves | Recomputing terminal log wealth, drawdown, CVaR by hand in a notebook cell | `platform.evaluation.kpis.{terminal_log_wealth, max_drawdown_and_duration, cvar, crisis_capture_ratio}` | These are the exact functions that produced the numbers in `backtest_kpi_table.parquet`; a notebook reimplementation risks silently drifting from the report's own math |
| Sojourn/detection-lag headline | A new comparison of two label series | `platform.evaluation.sojourn_lag.{build_filtered_probs_matrix, compute_sojourn_lag_headline}` | Already implements the per-target-state grouping fix (review F1) that a naive class-agnostic max would get wrong |
| Multiclass calibration/confusion | `sklearn.metrics.brier_score_loss` or a hand-rolled Brier | `platform.evaluation.model_metrics.compute_brier_multiclass` | **This codebase's Brier formula is NOT the textbook one** — see Section D. Reimplementing with `sklearn`'s or the textbook sum-per-sample formula would produce a different-scaled number than what's already in `model_metrics_brier.parquet`, silently breaking comparability |
| Economic recession bars | Manually typing NBER dates | FRED `USREC` series via `fredapi` (D-12) | Free, authoritative, already the credential path the platform uses everywhere else |

**Key insight:** every non-trivial computation P1-P6 needs already exists in Phase 1-5 library
code. The only genuinely new logic this phase must write is (1) the plotting functions
themselves, (2) the D-09/D-11 drift+plausibility checks, and (3) the A13 active-feature-count
reconstruction helper (four lines, shown above). Everything else is load-and-call.

## Common Pitfalls

### Pitfall 1: Assuming `regime_labels` exists

**What goes wrong:** A notebook or plan step assumes `get_platform_checkpoint_manager().load("regime_labels")` succeeds because pre-planning's table listed it as an existing checkpoint.
**Why it happens:** Pre-planning's checkpoint table was aspirational/schema documentation, not
a snapshot of what's on disk. Verified this session: `CheckpointManager.list()` on the platform
namespace returns exactly `daily_raw`, `monthly_raw`, `monthly_features` — no `regime_labels`.
**How to avoid:** P3 must call `label_regimes(monthly_features, cfg)` itself before it can plot
anything regime-related. This is fast (a single `fit_jump_model` call, not a walk-forward) but
it is a *fitting* action — confirm the plan explicitly treats it as "P3 computes its own inputs
on first run" rather than "P3 loads a pre-existing checkpoint."
**Warning signs:** A `FileNotFoundError` on `regime_labels` the first time P3 is opened for real.

### Pitfall 2: Conflating the checkpoint inventory pre-planning described with reality

**What goes wrong:** Pre-planning's table lists `monthly_raw_daily` and `fred_daily_raw` as
separate checkpoints feeding P1. **Verified this session: neither exists.** The dev namespace
holds only `daily_raw` (14,290×22 — all-equity tickers: AGG, COST, DBA, EEM, EFA, GDX, HYG,
IAU, IEF, IWM, LQD, MCD, O, QQQ, SHY, SLV, SPY, TSM, UEC, UNG, VNQ, VYM), and there is no
`monthly_raw_daily` or `fred_daily_raw` checkpoint on disk. FRED daily series (T10Y3M, T10Y2Y,
VIXCLS per `config/platform_settings.yaml`'s `fred_monthly.series` `daily: true` entries) are
folded directly into `monthly_raw`'s columns (`fred_t10y3m`, `fred_t10y2y`, `fred_vix` are
present in the 45-column `monthly_raw`), not kept as a separate daily-frequency checkpoint.
**Why it happens:** the ingestion pipeline evolved past whatever pre-planning read in August.
**How to avoid:** treat "what checkpoints exist" as something the plan verifies fresh at
execution time (`CheckpointManager.list()`), not something carried forward from pre-planning.
**Warning signs:** a P1 plot function signature that expects a `fred_daily_raw` argument that
no loader can actually produce.

### Pitfall 3: Treating the multiclass Brier score as the textbook [0,2] metric

**What goes wrong:** A plausibility band written for the textbook multiclass Brier score
(sum over K classes per sample, range [0,2]) will be silently wrong for this codebase's actual
metric.
**Why it happens:** `compute_brier_multiclass` (`platform/evaluation/model_metrics.py:58-81`)
computes `np.mean(diff * diff)` over the **full (n_samples, K) array**, not
`np.mean(per_sample_sum)`. Quoted verbatim: `diff = proba - onehot` then
`return float(np.mean(diff * diff))` — the mean is taken over every cell of the (n, K) matrix,
not summed per row first. This makes the metric range **[0, 1]**, not [0, 2], and gives a
"no-skill" (uniform 1/K probability) reference point of **(K-1)/K²** — for K=5, that is
**0.16** (verified by direct arithmetic this session, cross-checked against the observed
value: `model_metrics_brier.parquet` currently holds **0.208746**, matching
`BASELINE-v1-tracer-bullet.md`'s reported 0.2087 exactly).
**How to avoid:** any plausibility band for the calibration Brier score in P4/P6 must be
`[0, 1]` with a K-dependent no-skill reference `(K-1)/K²`, not `[0, 2]`. See Section D for the
full reasoning and its implication (the current 0.2087 is *above* the 0.16 no-skill floor).
**Warning signs:** a notebook cell asserting `brier < 2.0` — that assertion is true for almost
any conceivable input under this formula and catches nothing.

### Pitfall 4: Assuming the A13 panel requires a fresh 588-step walk-forward every notebook open

**What goes wrong:** Planning P3 (or P6) to call `run_backtest()`/`run_full_backtest_evaluation()`
fresh, unconditionally, every time the notebook is opened for "periodic V&V" makes P3 a
multi-minute operation for a notebook whose stated purpose (D-12/D-13/D-15) is a human glancing
at regime timelines — a mismatch with the "opened when something needs checking, not on every
run" framing in CONTEXT.md's `<domain>` section.
**Why it happens:** A13's fixed-vs-changing feature set framing makes it *sound* like it needs
the full walk-forward output, and in the strictest reading (the actual filtered probability
path) it does. But two of the three A13 ingredients (the reference labeling, the feature-count
timeline) are cheap and standalone.
**How to avoid:** Decouple the three ingredients: (1) reference labeling — cheap, ~0.6s,
computable independently; (2) feature-count-change timeline — cheap, ~1.2s, computable
independently; (3) the walk-forward's own filtered path — expensive, needs the full backtest.
See Section F for the three concrete architectural options the plan must choose among.
**Warning signs:** a plan task that says "P3 calls `run_full_backtest_evaluation()`" without
a stated reason connecting P3 (a regime-labeling notebook) to the backtest evaluation module.

### Pitfall 5: Reporting Faber/60-40 anomalies as current

**What goes wrong:** Prose describing "known anomalies" in P6 cites the original CONTEXT.md
numbers (Faber -99.7% DD, 60/40 -2.3% DD) as still-live facts to warn the operator about.
**Why it happens:** those numbers were accurate when CONTEXT.md was written (2026-08-04) but
were fixed by the 2026-09-08/09 units-defect corrections.
**How to avoid:** read `outputs/reports/platform/backtest_kpi_table.parquet` and
`backtest_report.md` fresh — verified this session to already show the corrected values
(Faber 6.372592/-18.94%, 60/40 5.047073/-26.96%, matching BASELINE's current-reference-run
table exactly). P6 should narrate the history ("these were once implausible, here is what
changed and why — see UAT-AUDIT-2026-09-09.md") rather than restate the old numbers as live
findings.
**Warning signs:** a plan task whose acceptance criterion literally reproduces "-99.7%" as an
expected value to display.

## Code Examples

### Loading the full span vs. the fenced dev span

```python
# Source: src/trading_crab_lib/platform/honesty/holdout.py:77-105, verified callable this
# session — returns 776 rows, 1962-01-31 to 2026-08-31, when dev alone is 708 rows to 2020-12-31.
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.honesty.holdout import load_full_span

# Fitting paths (never in a notebook that trains a model): dev only, fenced at 2020-12-31.
dev_features = get_platform_checkpoint_manager().load("monthly_features")  # 708 rows

# Notebook "looking" paths (D-06 opt-in, explicit): full span including post-2020.
full_features = load_full_span("monthly_features")  # 776 rows, through 2026-08
```

### Generating a labeling on demand (P3's first real cell)

```python
# Source: src/trading_crab_lib/platform/labeling/diagnostics.py:257-316, read in full this
# session. Verified this checkpoint namespace currently has no regime_labels — this call
# creates it.
from trading_crab_lib.platform.config import load_platform_config
from trading_crab_lib.platform.checkpoints import get_platform_checkpoint_manager
from trading_crab_lib.platform.labeling.diagnostics import label_regimes

cfg = load_platform_config()
monthly_features = get_platform_checkpoint_manager().load("monthly_features")  # dev, fenced
result = label_regimes(monthly_features, cfg)
# result: {"states": ..., "confidences": ..., "churn": ..., "diagnostics_path": ...}
# Also persists (verbatim from source): cm.save(labels_df, "regime_labels");
# cm.save(confidences_df, "regime_confidences"); cm.save(profiles_df, "regime_profiles").
# labels_df columns: ["state"]. confidences_df columns: [f"state_{k}" for k in range(K)].
# profiles_df columns: ["state", "profile"].
```

### Reconstructing the A13 feature-count timeline (verified in this session, 1.2s for 588 steps)

```python
# Source: composed and TIMED in this session against real data:
# lean_cols (13): cape_shiller, credit_spread_baa_aaa, curve_10y2y, curve_10y3m, div_yield,
#   fred_vix, gold, oil, real_rate_level, realized_vol_1m, realized_vol_3m, trailing_return_1m,
#   trailing_return_3m  (verified via lean_feature_set(cfg) & set(dev_features.columns))
# ref_cols (9, from _reference_label_columns): cape_shiller, credit_spread_baa_aaa,
#   curve_10y3m, div_yield, real_rate_level, realized_vol_1m, realized_vol_3m,
#   trailing_return_1m, trailing_return_3m  (drops fred_vix, curve_10y2y, gold, oil)
# Change points reproduced exactly (verified this session, matching UAT-AUDIT-2026-09-09.md
# Part V's reported sequence):
#   1972-01-31 -> 4, 1972-02-29 -> 6, 1972-04-30 -> 8, 1973-02-28 -> 9,
#   1986-06-30 -> 10, 1995-02-28 -> 12, 2000-01-31 -> 13
```

### Multiclass Brier's real bound (see Pitfall 3)

```python
# Source: src/trading_crab_lib/platform/evaluation/model_metrics.py:58-81, quoted verbatim:
#   diff = proba - onehot
#   return float(np.mean(diff * diff))
# This means the metric is mean-over-(n, K), bounding it to [0, 1], with a no-skill
# (uniform 1/K) reference of (K-1)/K^2. For K=5: (5-1)/25 = 0.16.
# Observed value in outputs/reports/platform/model_metrics_brier.parquet: 0.208746
# (matches BASELINE-v1-tracer-bullet.md's reported 0.2087).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| D-11 rejected (drift-only) | D-09 drift + D-11 reinstated plausibility | Amendment 1, 2026-09-09 | Notebooks (esp. P6) need TWO check classes, not one — see Validation Architecture |
| "Expect P3 to fail" (Amendment 1) | "Expect a genuine question" (Amendment 2) | Same day, 2026-09-09 (A4/A12 fix) | Changes the tone the plan should set for P3's prose — a real diagnostic question, not a foregone negative |
| `backtest_kpi_table` showing Faber -99.7%/60-40 -2.3% | Corrected: Faber -18.94%/60-40 -26.96% | 2026-09-08/09 (`75dedc7`, `976f7c8`, `18f68af`) | P6 must narrate the *history* of this fix, not the old numbers |
| A13 unresolved with the old degenerate labeling (49.6%/1.7% occupancy) | A13 confirmed real (7 feature-set changes) even after the labeling fix (1.6/14.0/31.9/40.6/11.9 occupancy) | 2026-09-09, Part V of the audit | A13 is not an artifact of the CPI defect — it is an independent, still-open issue P3 must surface regardless |

**Deprecated/outdated:**
- The original CONTEXT.md's D-11 rejection reasoning ("guessed thresholds would get tuned to
  whatever the current data happens to look like") — Amendment 1 shows the actual proposed
  bounds are economics-derived, not data-derived, and the rejection reasoning does not survive
  the incident it would have prevented.
- Any plan reference to `regime_labels`/`regime_confidences`/`regime_profiles` as "already
  persisted" — they are not, as of this session.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `platform/plotting/` submodule split (per-layer, 7 files: core + 6 layer modules) is the right decomposition | Architecture Patterns | Low — this is explicitly Claude's Discretion per CONTEXT.md; an alternative split (e.g. per-notebook instead of per-layer) is equally valid and low-cost to change |
| A2 | Promoting `driver._window_active_features` to a public name is lower-risk than duplicating its logic in the plotting library | Pattern 3 / Pitfall 4 | Medium — touches Phase-5 code (`backtest/driver.py`), which CONTEXT.md's boundary section says should stay untouched ("no report writers modified... no platform modules relocated"). This is a promotion/rename, not a relocation or behavior change, but the planner should confirm this reading is acceptable before assigning the task, or choose the duplication alternative instead. |
| A3 | The "walk-forward's per-window labeling" in Amendment 2 item F means the L2 nowcaster's per-step filtered probability path (`per_step_metrics["proba"]`), not the L1 hard-state labels discarded inside each step of `_refit_l1` | Section B/F | High if wrong — this determines whether A13's third ingredient needs the expensive full backtest (if it means the L2 path) or could be satisfied more cheaply (if it means something else). Reasoning: `_refit_l1`'s states cover `train_index` (strictly before t), never `t` itself, so there is no L1-only "label at t" to compare against the reference at date t without the L2 nowcaster step. This inference is sound given the code read this session but was not explicitly confirmed by re-reading Amendment 2's author intent. |
| A4 | `platform_settings.yaml` has no existing USREC/economic-events config section, so D-12's overlay data is best fetched independently at notebook/plotting-time rather than added to the `fred_monthly` ingestion schema | Standard Stack / Architecture | Low — verified no USREC entry exists in `fred_monthly.series` this session; the "notebook-time fetch, no schema change" recommendation is a reasonable inference from D-12's wording ("free, already an authenticated FRED path") but the plan could also legitimately add USREC to the ingestion schema (would require a rebuild, touching Phase-1 code) |
| A5 | The platform's `monthly_features` checkpoint has no centered/causal split analogous to the legacy `features.parquet`/`features_supervised.parquet` pair, making P2's "causal-vs-centered overlay" panel (carried over from pre-planning's table) potentially inapplicable as literally stated | Anti-Patterns | Medium — this was flagged as a caution, not fully verified; the planner should grep `transforms_monthly.py` for a second, centered variant before finalizing P2's panel list |

**If this table is empty:** N/A — see entries above; none of these are compliance/retention/
security-policy items, all are implementation-detail confirmations the planner should make
explicit decisions about rather than inherit silently from pre-planning or the original
CONTEXT.md.

## Open Questions

1. **How should the A13 panel's expensive ingredient (the walk-forward's actual filtered path)
   be sourced?**
   - What we know: two of three A13 ingredients are cheap (~2s combined, verified this
     session); the third (the L2 nowcaster's per-step filtered probability path) requires the
     full `run_full_backtest_evaluation()`, which is "minutes" per the driver's own docstring
     and is currently only available in-memory, not persisted to any checkpoint or artifact.
   - What's unclear: whether the plan should (a) have P3 call the full evaluation itself
     (self-contained but slow and conceptually odd for a "regime labeling" notebook to run a
     full backtest), (b) extend `platform/evaluation/report.py` to additionally persist
     `full_sample_states` and a per-date filtered-state artifact so P3 can load rather than
     recompute (small, targeted addition to Phase-5 code — a data-artifact addition, not
     "report wiring" in the forbidden sense, but it does touch verified Phase-5 code), or
     (c) put the full side-by-side only in P6 (which already computes everything) and have P3
     show a lighter version (reference labeling + feature-count timeline only, with a pointer
     to P6 for the full comparison).
   - Recommendation: **(b)** is architecturally cleanest and matches the codebase's existing
     pattern of persisting evaluation artifacts (`backtest_equity_curve_*.parquet` etc. are
     already produced by this same function) — but the planner or a `/gsd-discuss-phase`
     follow-up should make this an explicit decision rather than have an executor improvise it,
     since it is the one place this phase's stated boundary ("touches nothing verified in
     Phases 1-5") is genuinely in tension with Amendment 2's explicit new requirement.

2. **Does the platform have a centered-feature variant at all (for P2's causal-vs-centered
   panel)?**
   - What we know: `monthly_features` (the only feature checkpoint on the platform side) is
     described everywhere in the code and docs as causal. The legacy quarterly pipeline's ADR
     #1 (centered for clustering, causal for supervised) does not obviously have a platform
     analog based on what was read this session.
   - What's unclear: whether `transforms_monthly.py` computes and discards a centered variant
     internally, or whether the platform's L1 labeler's "intentionally non-causal at the batch
     level" framing (per `jump_model.py`'s own docstring — the DP-decode is non-causal by
     design even though the *features* are causal) is meant to substitute for a
     centered/causal feature distinction entirely.
   - Recommendation: the planner should grep `transforms_monthly.py` for a second feature
     variant before locking P2's panel list; if none exists, P2's "causal-vs-centered overlay"
     item should be replaced with "features are causal-only; the labeler's own non-causal
     batch-fit is the source of any hindsight, not the features" as the prose framing.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| matplotlib | `platform/plotting/` (all notebooks) | ✓ | 3.8+ (extra pinned, installed in `.venv`) | — |
| seaborn | Optional heatmaps/violin plots | ✓ | 0.13+ (extra pinned, installed in `.venv`) | Skip; matplotlib alone covers line/bar/scatter/stacked-area needs |
| nbformat | Notebook authoring + static CI checks | ✓ | 5.11.1, verified this session | — |
| jupytext | (not used — see Standard Stack alternatives) | ✗ | — | Not needed; `nbformat` covers the chosen authoring approach |
| nbmake / papermill | (not used — D-17 rejects execution-based CI) | ✗ | — | Not needed; static checks only |
| `FRED_API_KEY` | D-12's USREC fetch; any rebuild-dependent work | ✓ (per STATE.md — confirmed functional 2026-07-23 and again during the 2026-09-09 rebuild) | — | — |
| Network egress (FRED) | USREC fetch at notebook-render time; `label_regimes()`/`run_full_backtest_evaluation()` need NO network (checkpoint-only) | Unconfirmed in this specific session/sandbox (prior sessions note network is proxy-reset for some sources, but FRED itself was confirmed reachable during the 2026-09-09 CPI rebuild) | — | If USREC fetch fails in a given environment, D-12's overlay degrades to the event-list only (no recession shading) — this should be an explicit, loud degradation, not a silent one, consistent with D-10's "fail loudly, actionably" doctrine |

**Missing dependencies with no fallback:** None — all required tooling is already present.

**Missing dependencies with fallback:** USREC network fetch (fallback: event-list-only overlay,
loud warning).

## Validation Architecture

### The standing lesson this section is built around

Every criterion across Phases 1-5 was satisfiable by evidence of one of four shapes —
existence, shape, pure-function correctness, or placement — and **none of them could fail on a
value that is physically impossible.** Two concrete historical failures, both confirmed by
direct reading of the phase artifacts and audit in this session:

- **60/40 baseline recorded at −2.27% max drawdown** through a window spanning 1973-74, 1980-82,
  2000-02, and 2008-09 — a span containing multiple bear markets a diversified 60/40 mix
  historically drew down 15-30%+ in. A pure `max_drawdown ∈ [-1, 0]` bound check does **NOT**
  catch this (−2.27% is inside that interval); catching it requires a **domain-informed** band
  specific to that leg's expected behavior, not a universal mathematical bound.
- **Terminal log wealth of 111.06** (e¹¹¹ ≈ 10⁴⁸). This one IS out of reach of any reasonable
  universal band. `abs(wealth) < 10.0` (the bound already coded in
  `tests/integration/test_mini_backtest.py:258`, `TestKpisAreOnAPlausibleScale`, verified read
  this session) catches it immediately.

The distinction that must survive into every Phase 6 criterion: **a check that would have
caught the 60/40 case needs domain knowledge about that specific leg's expected drawdown floor;
a check that would have caught the log-wealth case needs only a wide universal band.** Both are
required; neither substitutes for the other.

### Drift vs. plausibility — two disjoint failure modes (D-09 + D-11)

| Check type | Question it answers | Catches | Does NOT catch | Existing pattern to extend |
|---|---|---|---|---|
| **Drift (D-09)** | "Is the current window still like the pre-2021 fitted window?" | A feature that *decays* — was predictive, quietly stopped (the seasonality motivating example) | A feature that was **always** wrong — the percent-vs-decimal defect corrupted the ENTIRE span uniformly, 1962→2026, so drift-against-baseline reports it as perfectly stable. **Verified in the audit's own words**: "A uniformly wrong series shows zero drift." | None existing — this is the new logic to write in `platform/plotting/drift.py` |
| **Plausibility (D-11, reinstated)** | "Is this value physically/economically possible at all?" | A value outside a domain-derived band regardless of trend (a 60/40 with too-shallow a drawdown; a Brier score at or below the no-skill floor; a terminal log wealth requiring >20%/yr sustained for 49 years) | A feature that has *drifted* to a new-but-still-plausible level (e.g. VIX regime shift to a persistently higher baseline — plausible on its own, but different from history) | `assert_yield_units_plausible` (`platform/splice.py:124-152`), `TestKpisAreOnAPlausibleScale` (`tests/integration/test_mini_backtest.py:218-258`) |

Both must be present in **every** notebook that displays a numeric KPI, not just P6 — P1's
coverage counts, P2's feature ranges, P4's calibration numbers, and P5's turnover/weights all
qualify.

### Plausibility bands by number (the deliverable this section exists to produce)

| Quantity | Band | Reasoning | Would have caught |
|---|---|---|---|
| **Terminal log wealth** (49-year equity-leg span, ~588 months) | `abs(x) < 10.0` (matches the existing coded test bound, `tests/integration/test_mini_backtest.py:258`); tighter operational band `[-3, 12]` for a single diversified leg | `e^10 ≈ 22,000×` over 49 years implies >21%/yr CAGR sustained without interruption — beyond any real long-only asset-class history over that span. SPY's real value in this dataset is 5.68 (verified, `backtest_kpi_table.parquet`), consistent with documented ~10-12%/yr equity CAGR. | The void 111.06 (e¹¹¹≈10⁴⁸) instantly; also the historical 2.3e128 `long_duration_tr` compounding defect at the series level |
| **Max drawdown** (any leg) | Universal bound: `x ∈ [-1, 0]`. Per-leg domain bound: SPY/buy-and-hold legs should show `x < -0.30` somewhere in a 1972-2020 span (contains 1973-74, 2000-02, 2008-09); a 60/40 blended leg should show `x < -0.10`; a defensive/trend-following leg (Faber) should show `x > -0.50` (its entire design purpose is limiting drawdown) | The universal bound alone does not catch either historical anomaly (−2.27% and −99.7% both satisfy `∈[-1,0]`) — **this is the sharpest illustration of why domain bands are required, not just interval bounds** | The void 60/40 at −2.27% (too shallow for its known crisis exposure) AND the void Faber at −99.7% (too deep for a trend-following strategy with a cash refuge) — verified current values are −26.96% and −18.94% respectively, both inside their domain bands |
| **Multiclass Brier score, K=5** (this codebase's specific formula) | `x ∈ [0, 1]` (NOT [0,2] — see Pitfall 3); no-skill reference `(K-1)/K² = 0.16` for K=5 | `compute_brier_multiclass` (`model_metrics.py:58-81`) computes `mean(diff²)` over the full `(n, K)` array, verified by direct source read and arithmetic cross-check against the observed 0.208746 | A Brier ≥ 1.0 (impossible under one-hot proba, or a units/shape bug); **currently flags an amber condition, not a red one** — 0.2087 sits ABOVE the 0.16 no-skill floor, meaning this metric alone cannot currently claim the nowcaster beats random guessing (echoes audit item A8's finding: "uses a calibrator" ≠ "is calibrated") |
| **Monthly turnover** | `x ∈ [0, 2]` per period (100% long-only reallocation both directions); operational band for a hysteresis-gated strategy: mean `< 0.30` | Full liquidation-and-reallocation of a long-only book is `Σ|Δw| = 2`; a hysteresis mechanism (act 0.70 / unwind 0.40 thresholds, verified in `config/platform_settings.yaml`) exists specifically to keep this low | Current value 0.0734 (verified, matches `BASELINE`) — well inside band, no flag |
| **CVaR(5%)** (monthly) | `x ∈ [-0.5, 0]`; operational band `x ∈ [-0.15, -0.01]` for a vol-targeted book | A single-month CVaR beyond −50% for a 10%-vol-targeted diversified book would require an extreme, near-total loss in the worst 5% of months — inconsistent with the vol-targeting mechanism itself | Current value −0.0463 (verified) — inside band |
| **Regime occupancy fractions** | Each `∈ [0, 1]`, `Σ = 1.0` exactly (verified: `occupancy_and_sojourns` computes `int((arr==state).sum())/n_total` summed over all states by construction); soft warning below `0.05` (already coded, `_MIN_OCCUPANCY_THRESHOLD`, `labeling/diagnostics.py:67`) | K=5 states must partition the sample | The pre-fix degenerate labeling (49.6% / 1.7% split) — the 1.7% state would have tripped the existing `_MIN_OCCUPANCY_THRESHOLD` warning (already coded, not new) |
| **Detection lag / sojourn (months)** | `∈ [0, 588]` (bounded by the walk-forward's own step count, verified: 588 steps from 1972-01 to 2020-12); **currently NOT interpretable per A13** regardless of band membership | A lag or sojourn outside [0, 588] would be a units/index error; but band membership alone says nothing about *meaning* while A13 is open (the 164-month lag/0.591 ratio is inside the band and still not trustworthy) | N/A directly, but the finding must be stated: **being "in-band" is necessary, not sufficient — this is a third failure mode (a valid-looking number computed against a mismatched target) that neither drift nor plausibility bands can catch, and the plan should not claim the A13 caveat is resolved by adding a numeric band around the ratio** |
| **Portfolio/tilt weights** | Each `∈ [0, 1]`, long-only `Σw_assets + w_cash = 1.0` exactly | `vol_targeted_tilt` is long-only by design (no options/shorts per PROJECT.md constraint) | A weight `< 0` or `Σ ≠ 1` would indicate a normalization bug in `allocation/tilt.py` |

### Would-have-caught summary (the two named historical failures)

| Historical failure | Caught by drift (D-09)? | Caught by plausibility (D-11)? |
|---|---|---|
| 60/40 at −2.27% max DD | **No** — the corruption was uniform across the whole span (percent-vs-decimal), so nothing "drifted" relative to itself | **Yes** — a domain-informed per-leg drawdown-floor band, NOT the universal [-1,0] bound |
| Terminal log wealth 111.06 | **No**, for the same reason | **Yes** — the universal `abs(x)<10` band already coded in the test suite |

This table is the concrete answer the phase's `<validation_architecture_requirement>` demands:
neither historical failure was a drift failure (both were present from the start of the
corrupted window, uniformly), so **only plausibility bands would have caught them** — which is
exactly why D-11's reversal is correct and D-09 alone (the original decision) was insufficient.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ (verified: repo-wide convention, `pyproject.toml` `[tool.pytest.ini_options]`) |
| Config file | root `pyproject.toml` |
| Quick run command | `pytest tests/unit/test_platform_plotting.py -x` (new file, per D-17's static-check pattern) |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| NB-01 (plotting fns) | Every `platform/plotting/` function runs without crashing on synthetic frames, empty-input edge cases | unit | `pytest tests/unit/test_platform_plotting.py -x` | ❌ Wave 0 — new file, follow `tests/unit/test_plotting.py`'s `does_not_crash` pattern |
| NB-01 (notebook structure) | Each `.ipynb` parses as valid JSON, its code cells' imports resolve, no forbidden plotting-library imports (`matplotlib`, `seaborn`) appear inline outside `platform/plotting/`'s own module | unit (static) | `pytest tests/unit/test_platform_notebooks.py -x` | ❌ Wave 0 — new file |
| NB-01 (drift/plausibility logic) | `platform/plotting/drift.py`'s drift-statistic and plausibility-band functions are pure and unit-testable | unit | `pytest tests/unit/test_platform_plotting_drift.py -x` | ❌ Wave 0 — new file |
| NB-01 (human: notebooks run top-to-bottom) | All six notebooks execute against real checkpoints end-to-end | manual-only | N/A — human verification item, mirroring Phase 1/5 precedent | N/A |
| NB-01 (P3 sign-off) | The sign-off markdown cell exists and is filled with date/verdict/reasoning after a real run | manual-only | N/A | N/A |

### Sampling Rate

- **Per task commit:** `pytest tests/unit/test_platform_plotting*.py tests/unit/test_platform_notebooks.py -x`
- **Per wave merge:** `pytest tests/ -v` (full suite)
- **Phase gate:** Full suite green before `/gsd-verify-work`; human-verification item
  (notebooks run top-to-bottom against real checkpoints, P3 sign-off recorded) tracked open
  per D-18, exactly like Phase 1's `FRED_API_KEY` item — **except this environment CAN now
  partially complete it**: `monthly_features`/`monthly_raw`/`daily_raw` are real, so P1/P2/P6
  are executable here; P3/P4/P5 depend on `label_regimes()` being called first (cheap, doable
  here) but the human sign-off itself remains inherently human.

### Wave 0 Gaps

- [ ] `tests/unit/test_platform_plotting.py` — covers every `platform/plotting/` function
      (does-not-crash + empty-input pattern, per `tests/unit/test_plotting.py`)
- [ ] `tests/unit/test_platform_plotting_drift.py` — covers D-09 drift statistic + D-11
      plausibility bands as pure functions
- [ ] `tests/unit/test_platform_notebooks.py` — static notebook checks (D-17): valid JSON,
      resolvable imports, no inline plotting-library imports
- [ ] Framework install: none — pytest, matplotlib, seaborn, nbformat all already present

## Security Domain

`security_enforcement` is enabled in `.planning/config.json` (ASVS level 1, block on `high`).
This phase's attack surface is narrow: it is a visualization library plus local Jupyter
notebooks reading local parquet checkpoints, with one new outbound network call (USREC via
FRED). No user-facing input, no authentication surface, no new secrets.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | No auth surface introduced |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Marginal | Checkpoint loading is via `CheckpointManager.load()` (parquet, not pickle — the legacy P27 pickle-arbitrary-code-execution pitfall does not apply to the platform's parquet-only checkpoints, verified: `get_platform_checkpoint_manager()` wraps the same `CheckpointManager` used elsewhere, and platform checkpoints observed this session are all `.parquet`). The one new external input is the USREC fetch response — treat it as data, never `eval`/`exec` it (obviously already the convention; no risk identified) |
| V6 Cryptography | No | `FRED_API_KEY` reuse only (already-established credential path per `platform/config.py:88-92`), no new secret material |

### Known Threat Patterns for this stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| A notebook accidentally committing a real `FRED_API_KEY` value into cell output (e.g. printing `cfg` which contains the injected key) | Information Disclosure | Never `print(cfg)` wholesale in a notebook cell; if config needs to be displayed, redact `cfg["fred_monthly"]["api_key"]`/`cfg["fred_vintage"]["api_key"]` first — this is a real risk specific to notebooks (which persist cell outputs in the committed `.ipynb`, verified: legacy notebooks DO store outputs) that library code does not have |
| A notebook writing to `data/checkpoints/platform/` by accident (e.g. calling `.save()` instead of `.load()`) | Tampering | D-10 already forbids this explicitly; the static notebook check (D-17) should additionally grep for `.save(` calls in notebook cells and flag them |

## Sources

### Primary (HIGH confidence — read directly, this session)

- `.planning/phases/06-platform-notebook-suite/06-CONTEXT.md` (full file, including both
  amendments) — locked decisions, scope fences
- `.planning/phases/06-platform-notebook-suite/06-PRE-PLANNING.md` — codebase inventory context
  (used as a cross-check baseline, several claims found stale and corrected — see Pitfall 2)
- `.planning/BASELINE-v1-tracer-bullet.md` ("Current reference run" section only)
- `.planning/UAT-AUDIT-2026-09-09.md` (Parts I-V, full)
- `.planning/STATE.md`, `.planning/REQUIREMENTS.md` (NB-01), `.planning/ROADMAP.md` (Phase 6 +
  Progress table)
- `src/trading_crab_lib/platform/` — grep for matplotlib/seaborn imports (zero found); full
  file listing of 48 modules
- `src/trading_crab_lib/plotting/core.py` — legacy pattern source, read in full
- `src/trading_crab_lib/platform/config.py` — read in full, confirmed no `RunConfig` analog
- `src/trading_crab_lib/platform/checkpoints.py` — read in full
- `src/trading_crab_lib/platform/honesty/holdout.py` — read in full, `load_full_span()` called
  and its output verified (776×53, 1962-01→2026-08)
- `src/trading_crab_lib/platform/labeling/diagnostics.py` — read in full
- `src/trading_crab_lib/platform/backtest/driver.py` — read substantially (docstrings,
  `_window_active_features`, `_refit_l1`, `_refit_l2`, `run_backtest` signature and body)
- `src/trading_crab_lib/platform/evaluation/report.py` — read substantially
  (`_reference_label_columns`, `run_full_backtest_evaluation` body and return contract)
- `src/trading_crab_lib/platform/evaluation/model_metrics.py` — `compute_brier_multiclass` read
  in full and its bound derived by direct arithmetic
- `src/trading_crab_lib/platform/splice.py` — `assert_yield_units_plausible` read in full
- `tests/integration/test_mini_backtest.py` — `TestKpisAreOnAPlausibleScale` read in full
- `data/checkpoints/platform/`, `data/holdout/`, `outputs/reports/platform/` — directory
  listings and file contents read directly this session (`CheckpointManager.list()`,
  `backtest_kpi_table.parquet`, `backtest_report.md`, `model_metrics_*.parquet`)
- `config/platform_settings.yaml` — `labeling`, `backtest`, `report`, `fred_monthly` sections
  read directly
- `notebooks/*.ipynb` — one file's JSON structure read directly to confirm storage convention
  (nbformat 4.5, stored outputs, no jupytext pairing)
- `.venv` package availability — `matplotlib`, `seaborn`(implied by extras), `nbformat` (5.11.1)
  confirmed importable; `jupytext`, `nbmake`, `papermill` confirmed NOT installed, this session
- `CLAUDE.md` (root) — ADR #11, conventions
- Live timing experiment run in this session: single `fit_jump_model` reference-labeling fit
  (0.621s) and full 588-step active-feature-count reconstruction (1.199s), both against the
  real `monthly_features` checkpoint, both cross-checked against `UAT-AUDIT-2026-09-09.md`
  Part V's reported figures (exact match on lean-column set, ref-column set, and all 7 change
  points/dates)

### Secondary (MEDIUM confidence)

- None — this research relied exclusively on direct codebase/artifact reads and live
  computation rather than external web search, since the domain (this specific project's
  internal architecture) has no external authoritative source beyond the codebase itself.

### Tertiary (LOW confidence)

- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — zero new dependencies; every package claim verified by direct
  `import`/`ImportError` check in `.venv` this session
- Architecture: HIGH — every function signature, checkpoint shape, and artifact content cited
  was read directly this session, not inferred from documentation
- Pitfalls: HIGH — all five pitfalls are drawn from direct discrepancies found between
  pre-planning/CONTEXT.md claims and the actual codebase/artifacts, verified this session
- Validation Architecture / plausibility bands: HIGH for the two named historical-failure
  reconstructions (both independently re-derived from source and cross-checked against
  recorded values); MEDIUM for the specific numeric operational bands proposed for
  turnover/CVaR/drawdown-per-leg, which are reasoned from domain knowledge and the observed
  current values rather than pulled from an authoritative external financial-industry standard
  — the planner/discuss-phase should treat these specific numbers as a starting proposal, not
  a locked spec

**Research date:** 2026-09-09
**Valid until:** This research is tied to a specific, fast-moving state of the repo (multiple
same-day fixes on 2026-09-08/09). Treat as valid only until the next rebuild or the next
`run_full_backtest_evaluation()` run changes the artifacts on disk — re-verify checkpoint
inventory and `backtest_kpi_table.parquet` contents at planning time if more than a few days
have elapsed.
