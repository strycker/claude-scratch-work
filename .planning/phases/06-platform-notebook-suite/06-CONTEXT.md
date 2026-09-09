# Phase 6: Platform Notebook Suite - Context

**Gathered:** 2026-08-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Build `platform/plotting/` — a visualization library that does not exist today (zero
matplotlib/seaborn imports across 48 platform modules) — and six notebooks that load
real artifacts, call those functions, and let a human see five phases of verified but
un-inspected work. Requirement NB-01.

**What these notebooks are for** (reframed during this discussion — supersedes the
earlier "human-in-the-loop gate for migration" framing):

1. **Periodic verification & validation.** Are the regimes still well-behaved and
   clearly defined? Is the data still behaving as it did historically? This is the
   recurring job.
2. **One cold-start selection gate at P3.** Regime detection and labeling is the thing
   that genuinely needs a human choice before it can be trusted for live use. P3 carries
   a sign-off; the other five do not.

**Weekly operation is scoring, not re-fitting** — so the notebooks are not in the weekly
path. They are opened when something needs checking, not on every run.

**In scope:**
- `platform/plotting/` package: `core.py` (palette, save/show, regime coloring) plus
  per-layer submodules, with tests.
- Six notebooks under `notebooks/platform/`: `P1_data_spine`, `P2_features_taxonomy`,
  `P3_regime_labeling`, `P4_nowcaster`, `P5_assets_allocation`, `P6_backtest_evaluation`.
- Drift-against-baseline checks (current behavior vs the pre-2021 fitted window).
- A cold-start sign-off cell in P3 only.

**Out of scope (hard boundaries):**
- **No tuning.** Phase 6 builds the viewing surface. (K, λ) sweeps, feature screening,
  and asset-class selection arrive as their own work — consistent with Phase 3 D-02,
  which already deferred (K, λ) tuning to v2 (L1-V2-01).
- **No report wiring.** `backtest_report.md` and `weekly_report.md` stay text-only.
- **No package restructuring.** Relocating pipeline-shaped platform modules
  (`backtest/driver.py`, `report/weekly.py`, `tripwire/monitor.py`) out of the library
  is a Phase 7 migration concern.
- **No holdout-carve repair.** Applying the physical split is separate pre-Phase-6 work
  (see D-17 — it is urgent, but it is not notebook work).

</domain>

<decisions>
## Implementation Decisions

### Plotting library (D1)

- **D-01: Fresh, self-contained `platform/plotting/`** importing nothing from the legacy
  `trading_crab_lib.plotting` package. The legacy package is a *pattern source only* —
  copy the `_save_or_show` / palette idioms, not the imports.
  **Rationale beyond cleanliness:** `plotting/core.py` imports `runtime.RunConfig`,
  `OUTPUT_DIR`, and `checkpoints`. `MIGRATION-PLAN.md` §2 states the coupling surface is
  *exactly four seams* and verifies that `platform/` currently imports nothing from the
  legacy lib (grep confirmed empty). Reusing legacy plotting would add seams to a
  migration plan built on there being four.
  — **Reversibility:** costly — undoing means either re-coupling to the legacy lib
  (invalidating MIGRATION-PLAN §2's seam count and growing P0) or rewriting every plot
  call site across six notebooks.

- **D-02: Plain kwargs, return the `Figure`.**
  `plot_x(data, *, save_path: Path | None = None, show: bool = False) -> Figure`.
  No `RunConfig` equivalent is invented for the platform. Notebooks render inline
  naturally; tests assert on axes without mocking. This *removes* an abstraction rather
  than porting one.

- **D-03: Notebooks live in `notebooks/platform/`.** Phase 7 moves one directory with no
  per-file filtering, and the split from the 12 legacy notebooks is self-evident.
  Notebooks belong to the **app** side of the two-package split (`trading-crab`), not the
  function library.

- **D-04: `platform/plotting/` lives in the library** (`src/trading_crab_lib/platform/`).
  It is pure functions with no `main()` and no pipeline, which is exactly what
  `trading-crab-lib` is for. Notebooks (app-side) call into it.

- **D-05: Notebooks only — no report wiring this phase.** The report writers stay
  text-only. Adding figures to them is a deferred idea, not Phase 6 scope.

### Holdout, drift, and what notebooks may see (D2 — MAJOR REVISION)

This section supersedes an earlier decision in this same discussion that notebooks would
hard-stop at 2020-12. That was over-restrictive and contradicted the platform's purpose.

- **D-06: The fence is on fitting, not on looking.** Notebooks read the **full span,
  including post-2020**, via the explicit `get_holdout_checkpoint_manager()` opt-in.
  Model-fitting paths keep the default dev manager, which structurally cannot reach the
  holdout tree (`holdout.py`: "There is no fallback code path from the default (dev)
  manager to the holdout tree ... This absence of a fallback IS the guarantee").
  **The architecture already anticipated this** — the holdout module's own docstring says
  "Live-scoring mode opts in explicitly via `get_holdout_checkpoint_manager()`."
  **Why:** the platform exists to score current data for real trades, and a human must be
  able to see a previously predictive feature stop working (e.g. seasonality decaying).
  Refusing to look would let a dead feature stay weighted.
  — **Reversibility:** one-way in the epistemic sense — once an operator has seen
  post-2020 behavior it cannot be un-seen, and any later claim of an untouched holdout is
  void. The trade is accepted knowingly (see `PROJECT.md` amendment).

- **D-07: Post-2020 observations that change a decision get recorded** with their date.
  This is the residual honesty guarantee: the influence stays traceable even though it is
  no longer firewalled. The record is what the deflated-Sharpe denominator would need.

- **D-08: The earlier "P1 coverage-metadata exemption" is DELETED.** It created a
  conditional rule (metadata may span full range, values may not) that someone had to
  remember, apply per cell, and test. With D-06 the whole conditional dissolves.

- **D-09: Drift baseline is the pre-2021 fitted window.** "Behaving as historic" means
  compared against the distribution over the same span the model was fit on (≤2020-12).
  This directly answers "is the world still the one this model learned?"
  A rolling trailing baseline was rejected: it decays alongside the feature, so gradual
  decay never trips it — precisely the failure mode this is meant to catch.

- **D-10: Missing artifacts stop with actionable instructions.** A shared load helper
  raises naming the missing checkpoint and the command that produces it. **Notebooks
  never write production checkpoints** (constraint C6; legacy pitfall P20/D5). Recompute-
  if-missing and degrade-and-continue were both rejected — the latter makes "runs
  top-to-bottom" meaningless as a criterion, since a nearly empty notebook still passes.

- **D-11: The earlier "plausibility banner" with fixed thresholds is REPLACED by D-09's
  drift check.** Guessed thresholds (|max DD| < 95% and similar) would get tuned to
  whatever the current data happens to look like. Drift-against-baseline is both simpler
  and more useful. Known anomalies are still named explicitly in prose — see
  `<specifics>`.

### P3 — the cold-start notebook (D3)

- **D-12: Economic history overlay = FRED `USREC` + a plain dated event list.**
  Recession bars come from `USREC` (free, already an authenticated FRED path, monthly,
  1854+). Inflation eras and credit events are a short list of `(start, end, label)`
  tuples — 1973 oil, 1980 Volcker, 1987, 1998 LTCM, 2000 dot-com, 2008 GFC — as a module
  constant or a few lines in `platform_settings.yaml`. **No dedicated events file, no
  per-entry provenance schema, no migration obligation** for six well-known date ranges.
  USREC alone was rejected: recessions cannot separate a stagflation regime from a
  credit-crisis regime, which is much of what a 5-state labeler should distinguish.

- **D-13: Overlay + regime×era contingency table.** The timeline answers it visually;
  the table (share of recession months landing in each regime, share of each regime
  falling in recessions) stops the eye from seeing patterns that are not there. Both are
  descriptive. A significance test was rejected — a p-value here invites treating "the
  regimes are real" as a passed test, and any number that informs a choice belongs in the
  registry.

- **D-14: No (K, λ) sweep.** P3 renders diagnostics for the shipped configuration
  (K=5, λ=52.0, n_restarts=10): sojourn distribution, occupancy, churn, transition
  matrix, §4.4 acceptance criteria as report-only. Phase 6 is the viewing surface;
  tuning arrives later and stays consistent with Phase 3 D-02.

- **D-15: P3 carries the only sign-off cell, and it is a plain markdown cell.** The
  operator types date, verdict, and reasoning, then saves the notebook. **No `sign_off()`
  helper, no YAML ledger.** The helper was justified by machine-checkable gating that is
  explicitly not wanted; for one notebook a markdown cell is proportionate. Easy to
  upgrade later if it ever needs to be greppable.
  After cold start the cell remains but is **not** required per run — subsequent use of
  P3 is behavior-checking, not re-signing.

- **D-16: A negative P3 verdict is recorded and does not block the phase.** Verification
  passes on "renders honestly and the human recorded a verdict," mirroring Phase 5 D-01
  (diagnostic, not gate) and design §14's "beats nothing yet — that's fine." A finding
  that the 5 regimes do not map onto recognizable history is an *input* to Phases 7 and
  8, which is exactly what `STATUS-REVIEW-2026-08.md` §7 wants learned before migrating.

### Testing, CI, and verification (D4)

- **D-17: CI does library unit tests plus static notebook checks — no cell execution.**
  Every `platform/plotting/` function is tested on synthetic frames (the legacy
  `test_plotting.py` `does_not_crash` + empty-input pattern); each notebook is checked to
  parse with its imports resolving. Catches the realistic breakage (renamed function, bad
  import) at near-zero cost. `nbmake`/`papermill` execution was rejected: new
  dependencies, a fixture generator per artifact, and a green CI that would mean "ran on
  fake data" — not the V&V the notebooks exist for.

- **D-18: Verification splits automated from human.** Automated coverage: the plotting
  library, notebook static checks, drift-check math. The "all six run against real
  checkpoints" criterion becomes an explicit **human-verification item**, exactly like
  Phase 1's `FRED_API_KEY` item and Phase 5's. Phase closes on the automated half with
  the human item tracked open. **This environment cannot complete the human half**:
  `daily_raw` is empty (0,0) and there is no `regime_labels` checkpoint, so P3/P4/P5 have
  no inputs here.

- **D-19: Build order — plotting core → P3 → the rest.** Wave 1: `core.py` + package
  scaffold + tests. Wave 2: P3 first (it carries the cold-start question), then the rest
  parallelizable by layer (P1/P2, P4, P5/P6).
  **Caveat the planner must handle:** P3 needs `regime_labels`, which does not exist in
  this environment. Its real-data verdict lands in the human item regardless, so "P3
  first" orders the *code*, not the answer.

### Scope boundaries confirmed elsewhere

- **D-20: The holdout carve repair is separate pre-Phase-6 work.** Verified during this
  discussion: `data/holdout/` does not exist, the dev-tree `monthly_features` holds
  **66 post-cutoff rows** running to 2026-06, and nothing outside
  `tests/unit/test_platform_holdout.py` calls `write_monthly_features_split()` or
  `assert_dev_checkpoint_within_boundary()`. The carve is a tested mechanism that was
  never applied to built data — so **fitting is currently unfenced**, and
  `assert_dev_checkpoint_within_boundary("monthly_features")` would raise today.
  Fix = call the split in `scripts/build_platform_data.py` and assert at fitting entry
  points. **Urgent, and independent of Phase 6.**

- **D-21: Package split is a Phase 7 concern.** `platform/` sits inside the library today
  and holds pipeline-shaped modules (`backtest/driver.py`, `report/weekly.py`,
  `tripwire/monitor.py`) that arguably belong in the `trading-crab` app package under the
  functions-only-library doctrine. Relocating them is migration work, not notebook work.
  `platform/plotting/` is pure functions and belongs in the library either way (D-04).

### Claude's Discretion

- Submodule split inside `platform/plotting/` (per-layer vs per-notebook) and figure
  sizing/DPI conventions.
- Where the shared checkpoint-load helper lives (plotting package vs a notebook utils
  module) and its exact error text.
- The specific drift statistic(s) comparing current window to the pre-2021 baseline
  (e.g. standardized mean shift, KS distance, rolling z vs baseline σ) — must be simple,
  transparent, and explained in the notebook.
- Exact panel composition per notebook, beyond what ROADMAP criteria 4 and 5 mandate.
- Whether the post-2020 decision record (D-07) is a markdown log, a YAML file, or a
  registry row — pick the lightest option that stays greppable.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design (authoritative)
- `platform_design/platform_design.md` §14 — tracer-bullet phase plan; "beats nothing
  yet — that's fine" underwrites D-16.
- `platform_design/platform_design.md` §4.4 — jump-model acceptance criteria that P3
  renders report-only.
- `platform_design/platform_design.md` §5.1, §5.4 — nowcaster spec, transition-window
  accuracy, label churn, smoothed-vs-filtered gap, detection lag (P4 and P6 content).
- `platform_design/platform_design.md` §8.7–8.9 — baseline gauntlet, no-regime ablation,
  strategy KPIs (P6 content).

### Requirements & roadmap (AMENDED 2026-08-04 — read the current text)
- `.planning/REQUIREMENTS.md` — **NB-01, reworded this session**: V&V framing, full-span
  reads with fitting fenced, pre-2021 drift baseline, P3-only sign-off.
- `.planning/ROADMAP.md` — **Phase 6 goal and criterion 2 rewritten this session**;
  new criterion 2b on the holdout opt-in.
- `.planning/PROJECT.md` — **honesty-discipline constraint amended this session**: the
  fence is on fitting, not looking; post-2020 decisions are recorded rather than
  firewalled. The amendment carries its own rationale note.
- `.planning/phases/06-platform-notebook-suite/06-PRE-PLANNING.md` — codebase inventory
  (§2 artifact surface, §4 constraints C1–C7). **Note: its §5 open decisions D1–D6 are
  now answered here, and its D-numbering does not match this document's.**

### Analysis
- `.planning/STATUS-REVIEW-2026-08.md` §5 — the notebook gap and why it precedes
  migration; §7 — why P3 is load-bearing.
- `MIGRATION-PLAN.md` §1 (what migrates), §2 (**the four-seam coupling surface** — D-01
  exists to keep this count honest), §3 PRE-1 (Phase 6 blocks migration), §4 (per-step
  notebook validation gates).

### Codebase
- `src/trading_crab_lib/platform/honesty/holdout.py` — `get_holdout_checkpoint_manager()`
  (the D-06 opt-in), `split_by_holdout_boundary()`, `write_monthly_features_split()`,
  `assert_dev_checkpoint_within_boundary()`, `DEFAULT_HOLDOUT_CUTOFF`. **Read the module
  docstring — it already describes the live-scoring opt-in D-06 relies on.**
- `src/trading_crab_lib/platform/checkpoints.py` — `get_platform_checkpoint_manager()`,
  the fenced dev namespace.
- `src/trading_crab_lib/plotting/core.py` — **pattern source only, do not import.**
  `_save_or_show`, `_regime_color`, `CUSTOM_COLORS`, `REGIME_CMAP`, the Jupyter-backend
  guard, `load_or_generate`.
- `src/trading_crab_lib/plotting/{regime,prediction,assets,clustering}.py` — analogous
  functions for quarterly shapes; adapt the ideas, not the code.
- `tests/unit/test_plotting.py` — the `does_not_crash` + empty-input test pattern D-17
  follows.
- `tests/unit/test_platform_holdout.py` — the only current caller of the carve functions.
- `src/trading_crab_lib/platform/labeling/diagnostics.py`,
  `platform/evaluation/{sojourn_lag,model_metrics,kpis}.py`,
  `platform/allocation/{tilt,hysteresis}.py` — producers of what P3–P6 display.
- `config/platform_settings.yaml` — 18 sections; lean 13-feature set; K=5, λ=52.0.
- `CLAUDE.md` (root) — conventions; **ADR #11** (plotting logic in the library, never
  inline) is the constraint behind D-01/D-04.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `holdout.get_holdout_checkpoint_manager()` — already built and tested as the explicit
  live-scoring opt-in. D-06 needs no new mechanism, only its first non-test caller.
- `src/trading_crab_lib/plotting/` (9 submodules) — direct analogs for regime timelines,
  transition matrices, calibration curves, soft probabilities, CV fold accuracy. Pattern
  source; the data shapes are quarterly, so these are not drop-ins.
- `platform/evaluation/` and `platform/labeling/diagnostics.py` already persist parquet
  artifacts — the notebooks load and render; **no new computation is required** beyond
  the drift statistic.
- The legacy `monitoring/` package's `validate_step_output` idiom is the closest existing
  shape for D-09's drift reporting.

### Established Patterns
- Functions-only library, `from __future__ import annotations`, type hints on public
  functions, `log = logging.getLogger(__name__)`, no `print()` in library code.
- New config sections read via `.get()`, never added to `_REQUIRED_PLATFORM_SECTIONS`
  (Phase 2/4 pattern).
- Synthetic-frame, no-network tests; matplotlib forced to `Agg` outside Jupyter.
- Optional deps raise `ImportError` with an install hint (`pip install
  'trading-crab-lib[plotting]'`).

### Integration Points
- **Consumes:** `monthly_features`, `monthly_raw`, `daily_raw` (platform checkpoints);
  `regime_labels`, `regime_confidences`, `regime_profiles`, `nowcaster`,
  `returns_by_regime`; and `outputs/reports/platform/*.parquet`.
- **Produces:** `platform/plotting/` package, six notebooks, saved PNGs, P3's sign-off
  cell content, and the D-07 post-2020 decision record.
- **Touches nothing verified in Phases 1–5** — no report writers modified, no platform
  modules relocated, no incumbent quarterly code touched.

### Current data state (matters for planning)
- `monthly_features`: 774 rows, 1962-01 → 2026-06, **66 rows past the cutoff**.
- `daily_raw`: empty (0, 0). No `regime_labels` checkpoint exists.
- `outputs/reports/platform/` holds the Phase 5 backtest artifacts.
- ⇒ P1/P2/P6 have real inputs here; **P3/P4/P5 do not**.

</code_context>

<specifics>
## Specific Ideas

- **The two-package doctrine drives placement.** `trading-crab-lib` is a pure function
  library — no `main()`, no pipeline — so that other people can build their own models on
  the core ideas. `trading-crab` holds pipelines, templates, installation, and the
  notebooks. Plot functions are functions; notebooks are app.
- **Known anomalies to surface, not hide.** `backtest_kpi_table` currently shows Faber
  SMA at terminal log wealth 1.14 with **−99.7% max drawdown**, and 60/40 at **−2.3%**
  max drawdown. Both are implausible on their face. P6 should name these directly in its
  prose so the operator confirms or flags them — no threshold machinery, just prose.
- **The sojourn/lag headline resolves on only 2 of 6 transitions.** P6 must show the
  resolved-transition count alongside the ratio (ROADMAP criterion 5), because a ratio
  computed on two events is a very different object from one computed on sixty.
- **The seasonality case is the motivating example for D-09.** A feature that was
  predictive and quietly stopped being predictive is the failure the drift check exists
  to catch — and the reason looking at post-2020 data is worth its bias cost.

</specifics>

<deferred>
## Deferred Ideas

- **Wire `platform/plotting/` figures into the report writers** — `backtest_report.md`
  and the weekly report/email, mirroring legacy D39's `attach_plots`. Out of scope here.
- **Tuning surfaces:** (K, λ) sweep, feature/metric noise screening, asset-class
  selection. You raised these as things notebooks might eventually do; Phase 6 is the
  viewing surface only. Consistent with Phase 3 D-02 → v2 (L1-V2-01). Feature screening
  in particular wants walk-forward evidence, not a notebook glance.
- **`sign_off()` helper + machine-readable ledger** — dropped for a markdown cell (D-15).
  Revisit only if sign-off ever needs to be checked by a program.
- **Relocating pipeline-shaped platform modules** (`backtest/driver.py`,
  `report/weekly.py`, `tripwire/monitor.py`) into the app package — Phase 7 (D-21).
- **Notebook execution in CI** via `nbmake`/`papermill` with synthetic fixtures — D-17;
  revisit if static checks prove insufficient.

### Adjacent work surfaced, not part of this phase
- **Holdout carve repair (D-20)** — urgent and separate. Fitting is unfenced today.
- **Fixing the Faber / 60-40 KPI anomalies** — Phase 6 surfaces them; diagnosing them is
  Phase 5 follow-up work.

</deferred>

---

*Phase: 6-Platform Notebook Suite*
*Context gathered: 2026-08-04*

---

# AMENDMENT — 2026-09-09 (read before `/gsd-plan-phase 6`)

Four decisions above rest on facts that have since changed, and one is now
**empirically disproven**. Source: `.planning/UAT-AUDIT-2026-09-09.md` and
`.planning/BASELINE-v1-tracer-bullet.md`.

## A. D-11 is REVERSED — reinstate the plausibility check alongside drift

**D-11 rejected a "plausibility banner" in favour of D-09's drift-against-baseline.**
That decision is now disproven by the event it would have prevented.

The percent-vs-decimal yield defect compounded `long_duration_tr` to **2.3e128** across
the *entire* span, 1962→2026. D-09 compares the current window against the pre-2021
fitted window. **A uniformly wrong series shows zero drift.** Drift-against-baseline
was structurally incapable of catching the largest defect in the project's history; it
would have reported the corrupted series as perfectly stable.

D-11's stated rejection reason — "guessed thresholds would get tuned to whatever the
current data happens to look like" — does not survive contact with the actual case.
The bounds do not come from the data; they come from economics:

- a US Treasury yield has never exceeded 100% annualized
- a long-only book cannot average **+21.9% per month** (the corrupted strategy did)
- a 60/40 portfolio cannot cross 1972–2020 with a **−2.27%** max drawdown
- a total-return index cannot reach 10¹²⁸

None of those is tuned. They are facts about the domain, and each one alone would have
tripped instantly.

**Amended decision:** keep D-09 drift **and** add a plausibility check. They catch
different failure modes — drift catches a feature *decaying*, plausibility catches a
value that *cannot exist*. Neither substitutes for the other. Three such assertions
already exist in the suite (`assert_yield_units_plausible`,
`TestKpisAreOnAPlausibleScale`, `test_percent_yields_produce_a_plausible_treasury_index`);
Phase 6 should surface the same idea in the notebooks rather than leave it as an
artifact of one incident. This is audit item **A3**.

## B. D-20 is SATISFIED — the holdout carve is done

D-20 called the carve "urgent, and independent of Phase 6," and recorded that
`data/holdout/` did not exist and fitting was unfenced. **Completed 2026-09-08**
(commit `7f99548`): `build_monthly_spine()` writes through
`write_monthly_features_split()`, `scripts/build_platform_data.py` asserts the boundary
and fails the build on violation, and the dev checkpoint now runs 1962-01 → 2020-12
(708 rows) with 68 post-cutoff rows in `data/holdout/`.

**Planner note for D-06:** the full-span opt-in now has an implementation —
`honesty.holdout.load_full_span(name)` returns dev + holdout concatenated in index
order. Notebooks should call it rather than opening two managers by hand.

## C. D-18 and D-19 environment premises are PARTLY stale

*(Corrected 2026-09-09 — the first version of this section overstated the case; see
the note at the end.)*

Both state that the human half cannot be completed here because "`daily_raw` is empty
(0,0) and there is no `regime_labels` checkpoint." As of 2026-09-09:

| Checkpoint | D-18/D-19 premise | Actual |
|---|---|---|
| `daily_raw` | empty (0,0) | **14,290 × 22 — premise is stale** |
| `monthly_features` | — | 708 × 53 dev (776 via `load_full_span`) |
| `regime_labels` | does not exist | **still does not exist as a committed artifact — premise HOLDS** |

`daily_raw` is genuinely fixed (Tiingo). The **`regime_labels` half of D-18/D-19
still stands**: no `regime_labels` / `regime_confidences` / `regime_profiles`
checkpoint is produced by the build or committed to the repo. `label_regimes()` will
generate them on demand from `monthly_features`, but nothing in the pipeline persists
them, so D-19's caveat — *"P3 needs `regime_labels`, which does not exist in this
environment"* — remains true and the planner must still handle it.

**Correction note:** the first draft of this amendment claimed `regime_labels` existed
at 372 × 1. It did — but only because a diagnostic `label_regimes()` call made during
this audit had created it minutes earlier. Checking a file's existence right after your
own tooling created it is not evidence. The files were removed; the claim is withdrawn.
Worth recording precisely because it is the same error shape the audit is about:
confirming what you went looking for.

## D. NEW BLOCKER for P2/P3 — audit item A4

An unfixed defect sits directly upstream of the two notebooks Phase 6 cares most about.
`fred_cpi` carries two artificial ~3× discontinuities (1970-12 `39.60` → 1971-01
`119.03`; 1988-01 `345.9` → 1988-02 `115.9`) because ALFRED point-in-time vintages of a
**rebased index** are not level-comparable across rebasings. `real_rate_level` inherits
it and ranges **−209.49 … +74.51**, and it is a defining feature of labeler states 2
and 3 — **64.3% of total occupancy**.

Building P2 (features/taxonomy) and P3 (regime labeling) on top of that means the
cold-start sign-off in D-15 would be a human signing off on a corrupted feature.
**Recommend fixing A4 before Phase 6 planning**, on exactly the same reasoning that made
D-20 pre-Phase-6 work: it is upstream, it is not notebook work, and it changes what the
notebooks would show.

## E. Expectation-setting for D-14 / D-16 (no change, but the planner should know)

The first trustworthy backtest is now recorded in `BASELINE-v1-tracer-bullet.md`. The
labeler produces **6 transitions in 59 years**, median sojourn 95 months, median
detection lag **76 months**, headline ratio **1.25** against a design bar of ~5. The
strategy finishes **last of five legs** and the regime layer adds +0.0732 log wealth
while making drawdown 1.95pp worse.

D-14 (no (K, λ) sweep) still holds as scope. D-16 (a negative P3 verdict is recorded and
does not block) is very likely to be exercised — P3 should be built expecting a negative
verdict, and that is the phase working as designed, not failing.

---

# AMENDMENT 2 — 2026-09-09 (supersedes items B and D of Amendment 1)

## D. **CLEARED.** The A4 blocker is fixed and verified.

Amendment 1 recommended fixing audit item A4 before planning. Done and confirmed
against rebuilt data: zero discontinuity warnings, `real_rate_level` −4.91…9.27
(economically correct — +3.38 mean through 1981, −3.26 through 1974), occupancy
matching the repair experiment exactly.

**Phase 6 planning is unblocked.**

## E. **REVISED — the P3 expectation has flipped.**

Amendment 1 said "a negative P3 verdict is very likely; build P3 expecting one."
That was written when the labeling was degenerate (one state at 49.6%, another at
1.7%). It is no longer the right prior. The corrected labeling maps onto
recognizable economic history:

| transition | state | reading |
|---|---|---|
| 1973-09 | → 1 | oil shock / stagflation |
| 1981-10 | → 4 | Volcker |
| 1988-09 | → 2 | post-Volcker disinflation |
| 1996-08 | → 3 | late-90s expansion |
| **2008-07** | **→ 0** | **GFC** — state 0 is the 1.6% crisis state |
| 2009-06 | → 3 | recovery |

D-12's economic-history overlay and D-13's regime×era contingency table now have
a real chance of showing agreement rather than noise. **Build P3 expecting a
genuine question, not a foregone negative.** D-16 (a negative verdict is recorded
and does not block) still stands as the safety valve.

## F. **NEW SCOPE ITEM for P3 — surface audit item A13.**

Two labelings coexist and are compared as if they were the same thing:

- `report._reference_label_columns` → a **fixed 9 columns**, 695 months from 1963.
- `driver._window_active_features` → a set that **changes 7 times** across the
  backtest (4 → 6 → 8 → 9 → 10 → 12 → 13 features), admitting each column once it
  clears `feature_min_history` in-window.

§5.4's detection lag is defined as how long the *filtered* path takes to agree
with the *smoothed reference*. With different and time-varying inputs, their
disagreement is not purely detection delay. The current run reports a 164-month
lag and a 0.591 ratio, and **that number is not interpretable** until this is
settled.

**A13 is not a blocker for Phase 6 — it is Phase 6's most valuable subject.**
Deciding what the labeler *should* see is a judgment call that benefits from the
viewing surface, which is exactly what the notebooks are for (D-16 anticipates
"a finding that ... is an *input* to Phases 7 and 8"). So:

- **P3 must render both labelings side by side** — the fixed reference and the
  walk-forward's per-window labeling — with their disagreement quantified and the
  feature-set change dates marked.
- This does **not** breach D-14 (no (K, λ) sweep). It changes no hyperparameter;
  it displays two labelings the code already produces.
- Until resolved, the §5.4 ratio should be shown with an explicit "not
  interpretable — see A13" caveat wherever it appears.

## G. A14 was measured and closed — no Phase 6 impact.

The `trailing_return_1m` canonicalization fallback fires on **1 of 588 steps**
(0.2%), the first step, which degrades anyway. It is not distorting churn or
detection lag, and P3's churn panel needs no special handling for it.

---

# AMENDMENT 3 — 2026-09-09 (settled during `/gsd-plan-phase 6`)

Two open questions from `06-RESEARCH.md` were resolved before planning. Both are binding.

## H. A13's filtered path — **PERSIST A NEW ARTIFACT** (resolves RESEARCH OQ1)

A13's side-by-side needs three ingredients. Two are cheap, timed live during research:
the smoothed reference labeling fits in **0.62s**, and reconstructing the 588-step
active-feature-count timeline takes **1.2s** — and that reconstruction reproduced the
audit's exact change points (4→6→8→9→10→12→13 at 1972-01, 1972-02, 1972-04, 1973-02,
1986-06, 1995-02, 2000-01), which independently corroborates the A13 finding.

The third ingredient — the walk-forward's actual **filtered probability path** — exists
only in memory during `run_full_backtest_evaluation()` (minutes) and is persisted nowhere.

**Decision: extend `platform/evaluation/report.py` to additionally persist
`full_sample_states` and a per-date filtered-state artifact**, alongside the
`backtest_equity_curve_*.parquet` artifacts that same function already writes. P3 then
**loads** rather than recomputes.

Rationale and boundary:
- It matches the codebase's existing pattern — that function is already the place where
  evaluation artifacts get persisted.
- It is an **additive write only**. No logic, no metric, and no hyperparameter changes.
  Every existing Phase-5 output must be byte-identical afterward; a task that changes an
  existing artifact has exceeded this decision.
- It is a knowing, bounded exception to "touches nothing verified in Phases 1–5." The
  exception is granted because Amendment 2 item F makes the P3 side-by-side a stated
  requirement, and the alternatives were worse: having P3 run a full backtest is
  conceptually wrong for a labeling notebook and duplicates P6's work, while deferring the
  comparison to P6 would not satisfy Amendment 2 item F.
- P6 consumes the same artifact — it is written once and read twice.

**This does not resolve A13.** It builds the surface on which A13 can be judged. The §5.4
ratio stays labelled "not interpretable — see A13" wherever it appears, and no plan task
may claim otherwise.

## I. P2 has no causal-vs-centered panel — **the variant does not exist and is forbidden**
   (resolves RESEARCH OQ2)

Verified during planning: `transforms_monthly.py` contains **zero** occurrences of
`center` / `centered` / `causal`, and `honesty/gating.py` defines
`FORBIDDEN_CENTERED_SUFFIXES = ("_centered", "_c5", "_zerophase")` and refuses loudly on
sight. There is no centered feature variant to overlay, and creating one would trip the
platform's own gating rail.

P2 must **not** attempt a causal-vs-centered comparison. Its prose framing instead:
*platform features are causal-only by construction, and a centered variant is actively
rejected by the gating rail; the only hindsight in the system is the L1 labeler's
deliberate non-causal batch DP-decode, not the features.*

This is a genuine difference from the legacy quarterly pipeline's ADR #1 (centered for
clustering, causal for supervised) — the platform did not inherit that split, and P2
should say so rather than imply a missing feature.
