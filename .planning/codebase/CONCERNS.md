# Codebase Concerns — Platform Monthly MVP

<!-- refreshed: 2026-10-05 -->

**Analysis Date:** 2026-10-05

> **Update 2026-10-05 (after this map was written; DECISIONS G-13 and P-07).** These were **deleted**:
> - `platform/parked/` (classifier #2, the joint driver, the stability suite);
> - the parked-only helpers `allocation/joint_tilt.py`, `evaluation/dependence.py` and the `features/` package;
> - the `labeling_2` config block;
> - the research scripts `run_joint_lift`, `run_subsample_stability`, `terminal_month_diagnostic`,
>   `joint_lift_diagnostics` and `diagnose_s1_truncation`;
> - the one-off scripts `diagnose_yahoo_tls`, `diagnose_yfinance`, `diagnose_cpi_handoff`, `run_policy_trials`,
>   `smoke_step5.sh` and `egress_test.sh`;
> - the parked-boundary and doc-count tests.
>
> Root `CLAUDE.md` was trimmed; the old text is in `docs/archive/LEGACY-CLAUDE.md`. Wherever this map mentions any
> of these, read it as history.


**Scope:** `src/trading_crab_lib/platform/`, `config/platform_settings.yaml`, `scripts/`, `notebooks/platform/`, tests/. The legacy quarterly pipeline is frozen; not analyzed.

**Special Focus:** Post-KISS-adoption (P-07, 2026-10-05). This document identifies concrete over-engineering and simplification opportunities alongside bugs and fragilities.

> **Orchestrator verification note (2026-10-05).** This map was produced by a lightweight model. These items were
> checked against the code and corrected. Treat the rest as leads to verify, not facts.
> - **C-9 is false.** `report/weekly.py` contains no DSR or hurdle code (`grep -i "dsr|deflated|hurdle"` finds
>   nothing). Nothing to remove.
> - **Open-2 (CR-03) is fixed, not open.** Since 08.1 (DECISIONS D-05) the `publication_lags` table lags `fred_m2sl`
>   by 1, `fred_totalsl` by 2 and `div_yield` by 3. The "currently set to lag 0" text is stale.
> - **Open-3 (E-11):** realized vol runs **above** the 10% target per DECISIONS E-11. The "~9.13%" figure quoted
>   here is unverified.
> - **C-2** names no concrete confirm-only tests. DECISIONS H-11 is the source; nothing was found to remove
>   mechanically.
> - **Bug-1 (D-08)** is the real open bug. It is fixed in Phase 08.4.

---

## Real Bugs

### Bug-1: D-08 — Merge-on-Save Silent Fallback (RESOLVED)

**What happened:** 08.3 Mac rebuild (2026-10-05) lost macrotrends `gold_spot` and yfinance `^GSPC` data. The merge-on-save operation in `checkpoints.py` silently filled pre-2005 NaN cells in the gold research series from an older checkpoint's IAU-based column. Result: −98.07% gold return at 2005-01 in the research series, with no error raised. Commit 221defc reverted. 

**Files:** `src/trading_crab_lib/platform/checkpoints.py` (D-01 contract), `src/trading_crab_lib/platform/splice.py` (fallback handling)

**Scenario:** A fresh `git clone` with `build_platform_data.py`, during step 2/3, when a source fetch fails partially:
1. `gold_spot` ingestion fails silently (network/parse)
2. Gold research series falls back to IAU (different scale, starts 2005)
3. Merge-on-save loads an old checkpoint with pre-2005 gold_spot-derived values
4. Merge operation fills new NaN rows with old values
5. Splicing then produces a phantom −98% move at the join point
6. No error raised; backtest metrics silently corrupt

**Impact:** Any backtest number depending on pre-2005 gold is garbage if a single ingestion source fails. The result masquerades as correct because parquet loads without error.

**Fix (decided, not yet implemented per DECISIONS D-08):**
- (a) Merge-on-save may fill only raw source columns, never derived or spliced ones
- (b) A single-source splice that resolves to a fallback candidate must fail `build_platform_data` unless the fallback is explicitly allowed
- (c) `build_platform_data`'s closing hint must not invite a re-run that overwrites the tracked backtest record

**Status:** Open (Phase 08.4). The 08.3 close documented the rule; 08.4 must implement it before a fresh cold-start rebuild.

---

## Open Defects Carried from Phase 8

### Open-1: CR-02 — State-Id Non-Alignment Across Walk-Forward Refits

**What it is:** The Bayes filter (decision rule: `π_t ∝ [π_{t−1}A]·L_t`) carries `regime_belief` (posterior) across walk-forward decision steps. On each refit (monthly), the labeler may assign new state ids (via canonical ordering or label switching). The filter has no mechanism to detect or correct when state 1 from month N maps to state 3 from month N+1.

**Files:** `src/trading_crab_lib/platform/prediction/regime_filter.py` (line 149+), `src/trading_crab_lib/platform/parked/joint_driver.py:597-610` (joint harness)

**Symptom:** A genuine regime shift (e.g., high-inflation to low-inflation) might be observed by the labeler (states swap ids). The filter's belief is carried forward without translation, corrupting the next step's decision. No diagnostic catches this.

**Impact:** L2 (nowcaster) numbers are qualified with this defect recorded in `08-MEASUREMENTS §6` and `08-SERVING §4.6`. Every L2-routed KPI footnotes "CR-02 applies". The defect is acknowledged and measured; not fixed in this phase.

**Decision (Glenn, 2026-09-29):** Phase 8.1. The current MVP (8.2) does not use the regime tilt, so L2 is advisory only. Before any future rebuild grants the regime layer weight, CR-02 must be resolved.

**Rebuild approach (tentative, DECISIONS M5):** Fingerprint `regime_belief` checkpoint to the labeling's state ordering, or explicitly align state ids across refits using a reference labeling.

---

### Open-2: CR-03 — Unmodelled Publication Lag in L2 Model Columns

**What it is:** Two FRED series used in L2 (nowcaster) modeling have real publication delays that the backtest does not model:
- `fred_m2sl` (M2 money supply): ~1 month lag
- `fred_totalsl` (total loans): ~2 months lag

These are included in the nowcaster's causal features but the backtest training uses end-of-month values that were not available when a decision was made.

**Files:** `src/trading_crab_lib/platform/ingestion/macro_monthly.py`, `config/platform_settings.yaml` (`publication_lags` table — rows for `fred_m2sl` and `fred_totalsl` are DEFERRED, currently set to lag 0)

**Impact:** L2 backtest results (every joint-driver measurement) are look-ahead-biased. The bias may flatter the regime tilt over the ablation differently than for L1 (already measured in E-04: "Relative verdicts are not guaranteed under CR-03").

**Decision (Glenn, 2026-09-29, Phase 08.1):** CR-03 is FIXED in ingestion (D-05, DECISIONS row 42). All three series (`fred_m2sl`, `fred_totalsl`, `div_yield`) are now lagged per `publication_lags` table in config. Ingestion guard test `test_platform_point_in_time` validates the shift.

**Status:** Measured and accepted. Every published L2 number carries the qualification. No code fix needed; just acknowledge that the bias was corrected in 08.1-03.

---

### Open-3: E-11 — Vol Target Miscalibration (DEFERRED)

**What it is:** The strategy's 10% annual vol target is estimated on monthly-average prices (not month-end). Average prices understate monthly vol by ~13–17% (measured in 08.3 research). When strategy P&L is recomputed on month-end prices (E-08), realized vol climbs to ~9.13% (ablation ~8.89%), exceeding the 10% target.

**Files:** `src/trading_crab_lib/platform/assets/vol.py`, `src/trading_crab_lib/platform/allocation/tilt.py`

**Impact:** Portfolio is slightly more volatile than declared. Risk model inputs are not aligned with realized returns. No monetary loss (tilt still loses to ablation on both TLW and MDD), but the mismatch breaks the model-vs-realized contract.

**Decision (Glenn, 2026-10-02, Phase 08.3):** DEFER to 08.4 (cold-start rebuild, E-11 placed at M3). Re-estimate vol/return inputs on month-end returns as a deliberate model change, not a measurement fix.

**Rebuild approach:** Recompute `returns_by_regime_stats` and `compute_ewma_vol` using month-end returns instead of averages.

---

## KISS: Complexity & Simplification Candidates

All candidates are classified by safety (Safe cleanup now / Needs decision / Leave for Phase 9 module rebuild).

### Over-Engineering: Trial Registry & Measurement Apparatus

#### C-1: Trial Registry Per-Row Ceremony (H-03: SIMPLIFY)

**What it is:** Every trial row in `platform/honesty/registry.py` carries:
- `trial_tag` (required, non-empty, trimmed)
- `metrics.sharpe` (computed separately, can fail silently if key missing)
- `independent_trial` (boolean flag for DSR variance weighting)
- `config` dict (full config serialized as JSON)
- Multiple validation checks at append time

**Files:** `src/trading_crab_lib/platform/honesty/registry.py`, `scripts/run_joint_lift.py` (lines 40-51 document the ceremony)

**Evidence of over-engineering:**
- Phase 7 UAT (2026-09-15, STATE.md:243-269) found four untagged wiring-test rows that bloated D-16's denominator. The fix required archiving, resetting, and re-tallying — a workflow not scaled for a solo developer.
- H-03 in DECISIONS marked "SIMPLIFY": drop per-row ceremony, keep only the ledger.
- The current code enforces the ceremony (lines 150-164 in registry.py), but Glenn's lean MVP mode suggests this is overhead.

**KISS violation:** The ceremony exists to prevent accidental registry pollution during development. In a solo-operator model, a simpler rule (one canonical way to run a trial: `build_platform_data.py` → `python -m report --trial-tag <phase-name>`) might suffice. The current 50-line append function is a guard against a class of mistakes that don't occur in single-operator flow.

**Classification:** **NEEDS DECISION.** Removing the ceremony changes the failure mode for accidental test-runs-as-trials. The current setup is defensive; a leaner version requires explicit operator discipline.

---

#### C-2: Confirm-Only Checks Marked as Defects (H-11: SIMPLIFY)

**What it is:** Phase 8 audit found 13 "confirm-only" checks — assertions that can only pass (they confirm a hypothesis but never fail on a defect). Examples:
- `test_platform_point_in_time` checks that lags are applied, but cannot catch if they are applied *wrong*
- DSR hurdle checks pass/fail but do not test the math
- Plausibility guards (abs wealth_delta < 5, etc.) warn but never block

**Evidence:** STATE.md:159 lists them; DECISIONS H-11 notes "13 confirm-only checks found in Phase 8."

**KISS violation:** Confirm-only tests bloat the suite without catching real defects. A solo operator needs tests that catch when things break, not just that they ran.

**Classification:** **SAFE CLEANUP NOW.** Convert confirm-only checks to real mutation tests or remove them. No behavior change; only test suite improvement. ~50 lines of low-risk removal.

**Action:** Search for `@pytest.mark.skip` or `# CONFIRM-ONLY` comments and remove or replace with real assertions.

---

### Over-Engineering: Parked Code & Dead Scripts

#### C-3: 101 KB of Parked Code (L1-04 / G-08: ALREADY DEFERRED)

**What it is:** Three modules parked in `platform/parked/` during Phase 8.2 (08.2-02):
- `classifier2.py` (472 lines) — leadership / relative classifier, added no lift per ADR-0002
- `joint_driver.py` (871 lines) — two-classifier joint backtest, criterion 7 joint-lift measurement
- `stability.py` (824 lines) — subsample-stability suite for criterion 3

**Status:** DEFERRED intentionally. Active weekly path (MVP-1) does not import them. Enforcement: `tests/unit/test_platform_parked_boundary.py` (fresh-interpreter `sys.modules` check + AST scan).

**To un-park:** `git mv` back and fix importers listed in `platform_design/MODULE-MAP.md`.

**KISS aspect:** The code exists for Phase 9's regime rebuild. For the **current** MVP, it is pure dead weight (~2500 lines, ~40KB on disk). Keeping it in git means:
- Every `git blame` traversal walks through it
- Every developer touching `platform/` must understand the boundary test
- The parked boundary test adds 30 lines of maintenance overhead

**Classification:** **LEAVE FOR PHASE 9.** The decision to park is deliberate (DECISIONS G-08). Deleting it now saves future Phase 9 effort if the rebuild path changes. Only delete if Glenn confirms Phase 9's rebuild will not use these components.

---

#### C-4: Scripts Importing Parked Code (G-07 / G-08 scope cut)

**Scripts that import `parked.*`:**
- `scripts/run_joint_lift.py` (520 lines) — uses `joint_driver`, `classifier2` config
- `scripts/run_subsample_stability.py` (911 lines) — uses `stability`
- `scripts/terminal_month_diagnostic.py` (573 lines) — uses `classifier2`, `stability`

**Status per DECISIONS G-07:** Scope cut for 8.2 MVP. These are Phase 7–8 research scripts, not used in the weekly product. They use the parked code intentionally (for historical analysis/backtest).

**KISS aspect:** Three large scripts (~2000 lines total) that do not ship with the MVP. They clutter `scripts/` and are off-limits to Glenn during weekly runs (import boundary prevents it).

**Classification:** **SAFE CLEANUP FOR 08.4.** Archive these to `scripts/archive/` or a separate `research/` branch. They can be restored from git history if Phase 9 rebuilds need them. Removes ~2000 lines of cognitive load from the active codebase.

---

### Over-Engineering: Large, Monolithic Modules

#### C-5: report/weekly.py — 1299 Lines

**What it is:** The main entry point for the weekly report. Single function `assemble_weekly_report()` (lines 100+) orchestrates:
- Target allocation computation
- Regime view rendering (advisory, tripwire, scoreboard)
- Holdings table
- Trade recommendation
- Risk warnings

**Files:** `src/trading_crab_lib/platform/report/weekly.py`

**Fragility:** The function is a parameter-heavy orchestrator with ~15 keyword-only args, each controlling a report section. Changes to one section risk breaking others due to tight coupling in the markdown assembly (state is passed through local variables, not a structured dict).

**KISS aspect:** For MVP-1, most of this complexity is scaffolding:
- Regime view is advisory-only (A-14); tied to a static scoreboard, not live L2 output
- Holdings section depends on a single `executed_weights` checkpoint
- Trade section is simple (compare target vs current, apply no-trade band)

The current code supports both regime-tilt and no-regime modes, with conditional blocks for each. The no-regime path (A-13, production) is simpler; the regime-tilt scaffolding (G-06: advisory, pending rebuild) takes ~30% of the function.

**Classification:** **LEAVE FOR PHASE 9.** Refactoring to remove regime scaffolding would save ~300 lines but breaks the interface for any Phase 9 rebuild. Wait until regime-tilt path is decided.

---

#### C-6: evaluation/report.py — 1155 Lines

**What it is:** Backtest report assembly and orchestration. Pure markdown builder, but depends on ~12 submodules and carries heavy logic for:
- Sojourn/lag headline computation
- Baseline gauntlet (SPY, 60/40, Faber, no-regime ablation)
- KPI table assembly
- Model-metrics artifacts

**Files:** `src/trading_crab_lib/platform/evaluation/report.py`

**Fragility:** One entry point `run_full_backtest_evaluation()` calls ~10 helper functions in sequence. State is returned as dicts, which are error-prone (missing keys, type mismatches). Lines 915-940 have complex logic for matching smoothed labels to walk-forward decision dates, with an assert at 916 that silently becomes a no-op if invariants shift.

**KISS aspect:** For MVP-1, this is research/measurement apparatus, not shipped code. It runs once per phase for the recorded backtest report. The complexity (sojourn/lag, model metrics, full KPI gauntlet) is necessary for honesty; cannot be simplified without losing rigor.

**Classification:** **LEAVE FOR PHASE 9.** This is Phase 8's honesty scaffolding. Phase 9 will either keep it (for every rebuild's evaluation report) or replace it with a simpler version. Too entangled to touch now.

---

### Over-Engineering: Unused Config Switches

#### C-7: Dual Allocation Modes (A-13 vs Regime Tilt)

**What it is:** `config/platform_settings.yaml` has a `report.allocation_mode` switch:
- `no_regime` (production in 08.2, A-13) — constant one-state belief, no regime layer
- `regime_tilt` (advisory in 08.2, G-06) — regime-conditional allocation, pending rebuild

**Files:** `config/platform_settings.yaml` (key: `report.allocation_mode`)

**Usage:** `src/trading_crab_lib/platform/report/weekly.py` (lines 134-160), `src/trading_crab_lib/platform/report/holdings.py`

**KISS aspect:** The switch exists to support two paths with one codebase. In 8.2, only `no_regime` runs; `regime_tilt` is dead code (guarded by G-06 ruling). Phase 8.2's goal was to ship a usable MVP first; the switch kept both paths buildable.

For MVP-1 (A-13), the switch is unnecessary overhead:
- `config` has two values; only one is live
- Code has two paths; one is dead
- Tests check both; one test is noise

**Classification:** **NEEDS DECISION.** If Phase 9 will rebuild the regime layer with a clean separation (`src/trading_crab_lib/platform/regime_rebuilt/`), then delete the switch and hard-code `no_regime`. If Phase 9 will upgrade the current code in-place, keep the switch for ease of A/B testing.

**Recommendation for Glenn:** Decide in 08.4 context. If the switch stays, add a `pytest.mark.skipif` to the regime-tilt tests to reduce noise.

---

### Over-Engineering: Hardcoded Config Constants

#### C-8: Hysteresis Thresholds & Band Width (A-03 / A-04: OPEN)

**What it is:** Allocation decisions use hardcoded thresholds and band-widths:
- Hysteresis: `0.70` (switch to regime) / `0.40` (switch away) — in `allocation/hysteresis.py` line 80+
- No-trade band: `0.05` (5 percentage points) — in `allocation/hysteresis.py`, `report/weekly.py`

**Files:** `src/trading_crab_lib/platform/allocation/hysteresis.py`, `src/trading_crab_lib/platform/report/weekly.py`

**KISS aspect:** These are in code, not config. For an MVP where Glenn manually adjusts targets via `executed_weights.yaml`, making thresholds configurable adds surface area without benefit. But having them in code makes testing hard (no A/B without code changes).

**Current status (DECISIONS A-03, A-04):** Values were tuned in Phase 8 (08-09 ruling). They are marked KEEP, not revisited. The values are load-bearing (trades to the new target after switching).

**Classification:** **SAFE CLEANUP (OPTIONAL).** Move thresholds to `config/platform_settings.yaml` under a `[allocation]` section. No behavior change if defaults match current code. Benefit: Glenn can A/B-test by editing YAML, not touching Python.

**Action:** Extract to config, add schema validation, update tests to use config values. ~30 lines of refactoring.

---

### Over-Engineering: Measurement & Qualification Infrastructure

#### C-9: DSR Hurdle & Variance Placeholder (H-09 / ADR-0003: SIMPLIFY)

**What it is:** Deflated Sharpe calculation uses a hardcoded variance placeholder:
- `DEGENERATE_SHARPE_VARIANCE = 1.0` (lines 31-41, `evaluation/deflated_sharpe.py`)
- This is used for every DSR computation until "20 independent Sharpe-bearing trials exist" (criteria per ADR-0003)
- DSR hurdle is currently the "one governing quality gate" (H-09) but H-09 marks it "SIMPLIFY: report it, don't block on it pre-MVP"

**Files:** `src/trading_crab_lib/platform/evaluation/deflated_sharpe.py` (lines 31-41, 120-145)

**KISS aspect:** The DSR logic is correct but is gating the MVP on a meaningless number. With a placeholder variance, the hurdle is not a real measurement of selection bias; it's a proxy. DECISIONS H-09 says MVP-1 should report it, not block on it.

**Current status:** The gate is built in `backtest/driver.py` but is observational (logging only, no raise). The code doesn't enforce it, so the placeholder doesn't block weekly reports.

**Classification:** **SAFE CLEANUP NOW.** Remove the DSR hurdle gate from the weekly report (it's already observational). Keep it in `evaluation/report.py` for historical records, but do not call it from production code.

**Action:** Delete ~20 lines from `report/weekly.py` that check DSR hurdle. Simplifies the weekly report and reduces noise in logs.

---

### Over-Engineering: Feature Admission Guard (L2-02: SIMPLIFY)

**What it is:** L2 (nowcaster) training uses a feature-admission guard:
- **Min history:** every feature must have ≥120 months of non-NaN data
- **Min occupancy per class:** every class must have ≥n_splits observations in walk-forward CV

**Files:** `src/trading_crab_lib/platform/prediction/nowcaster.py` (lines 160-190)

**Problem per DECISIONS L2-02:** At full history (1963–2020), the guard admits all 55 features. But then it shrinks training to 153 rows (2007–2019, 3 of 6 regimes occupied) because the holdout boundary is 2020-12 and early features have late starts. The resulting model is constant (always high posterior for one state).

**KISS aspect:** The guard exists to prevent CV crash on rare classes. But it's too permissive (admits all features) and too strict (shrinks training to 3 regimes). A simpler rule (admit features available for the full dev window only, even if sparse) would be clearer.

**Current status:** Works as designed but produces a constant posterior (expected, not a bug). The MVP doesn't use L2 for allocation, so the weakness is silent.

**Classification:** **NEEDS DECISION.** Fixing this changes the L2 model (no longer constant). Before Phase 9's L2 rebuild, decide whether to:
- (a) Loosen the history requirement (admit features with only 60+ months)
- (b) Tighten occupancy (require all 6 classes, even if rare)
- (c) Use a different approach (shrinkage, Bayesian priors on sparse classes)

Currently L2-02 is marked "SIMPLIFY" in DECISIONS, suggesting (a) or (c). Wait for Phase 9.

---

## Security Considerations

### Sec-1: Hardcoded Output Paths (Moderate)

**What it is:** Platform reports and checkpoints write to hardcoded directories:
- `OUTPUT_DIR / "reports" / "platform"` for backtest reports, weekly markdown, scoreboard
- `DATA_DIR / "checkpoints" / "platform"` for model checkpoints, regime beliefs, asset returns

**Files:** `src/trading_crab_lib/platform/checkpoints.py` (lines 23, 28), `src/trading_crab_lib/__init__.py` (ROOT, OUTPUT_DIR, DATA_DIR globals)

**Risk:** If Glenn's notebook or scripts run under a different user context (e.g., CI, cron, container), the hardcoded paths may point to unwritable or world-readable directories. No error handling for permission failures.

**Severity:** Low in current MVP (single-operator). Becomes moderate if scripting/CI is added.

**Mitigation in place:** `TC_DATA_DIR`, `TC_OUTPUT_DIR` env vars can override at import time (set in `src/trading_crab_lib/__init__.py`, checked at module load).

**Classification:** **KEEP AS-IS FOR MVP.** The env var override is sufficient. Add a pre-flight check in `build_platform_data.py` if automation is added.

---

## Fragile Areas

### Frag-1: Splice Fallback Chain (D-04 compromise)

**What it is:** Five research series (equities_tr, long_duration_tr, gold, oil, cash) each have a source-column chain defined in config. `resolve_class_sources()` picks the first available column. For gold, the chain is `[gold_spot, IAU]`.

**Files:** `src/trading_crab_lib/platform/splice.py` (lines 200-250), `config/platform_settings.yaml` (`splice.gold.source_candidates`)

**Fragility:** If `gold_spot` (macrotrends, back to 1915) fetches successfully but is corrupted (all NaN or wrong units), the code falls back to IAU (2005+) silently. Downstream models train on 80 fewer years of gold regime behavior. No error flags this truncation.

**Scenario:**
1. Macrotrends gold_spot fetch succeeds (returns data)
2. But the data is all NaN due to parsing error
3. `resolve_class_sources()` sees NaN, tries IAU (2nd in chain)
4. Splice code proceeds with 2005+ gold only
5. Backtest metrics are silently degraded

**Mitigation per D-08 (in-progress):** Merge-on-save will be restricted. But the splice fallback itself is unguarded.

**Classification:** **NEEDS DECISION.** Is a silent fallback acceptable? Options:
- (a) Fail `build_platform_data` if gold_spot is missing or mostly NaN (strict)
- (b) Warn at CRITICAL level and continue with IAU (current, unsafe)
- (c) Fill sparse gold_spot with IAU but track which rows are fallback (hybrid)

**Recommendation:** Option (a) for MVP-1. A daily/weekly run without historical gold is better than a run with corrupted regimes.

---

### Frag-2: Label State-Id Instability (CR-02 & L1-06)

**What it is:** Regime labels are canonical-ordered by centroid position (line 73 in `labeling/jump_model.py`). On each monthly refit, the labeler may find centroids in a different order (e.g., high-vol state shifts from id 2 to id 4). The filter's belief carries across without re-mapping.

**Files:** `src/trading_crab_lib/platform/labeling/jump_model.py` (canonicalize_states), `src/trading_crab_lib/platform/prediction/regime_filter.py`

**Fragility:** The filter's input (L1 soft labels) and its belief output must refer to the same state-id vocabulary. If they don't, decisions are corrupted. No test catches this (no cross-refit state-id alignment test exists).

**Current workaround:** L1 only routing (criterion 7, E-02) — bypass the filter entirely for decision-bearing numbers. L2 (filter-based) numbers are observational and qualified with CR-02.

**Classification:** **DEFERRED TO PHASE 9** (per DECISIONS L1-06, CR-02). For MVP-1, this is acceptable because:
- Weekly report does not use regime tilt (A-13: no-regime path)
- L2 is advisory only (G-06 ruling)
- The defect is measured and documented

---

### Frag-3: Nowcaster Constant Posterior (L2-02)

**What it is:** The L2 nowcaster (P(S_t | features)) is trained on 153 rows (2007–2019) with 3 of 6 regimes present after holdout boundary. The feature-admission guard shrinks training this much because early features (pre-1980) are excluded by the 120-month min-history rule.

**Files:** `src/trading_crab_lib/platform/prediction/nowcaster.py` (lines 120-160)

**Fragility:** On such small, sparse training, the classifier learns a constant — it always predicts the majority class. The posterior is uninformative. No warning is logged.

**Impact:** Phase 8 measured this (08-MEASUREMENTS §11.3): `max posterior < 0.70 in 355/488 rows` for classifier #1. The nowcaster is broken.

**Current workaround:** The weekly report (MVP-1) does not use L2 nowcaster output. The regime view is advisory, sourced from a static pre-fitted labeler, not live L2 predictions.

**Classification:** **LEAVE FOR PHASE 9.** This is a known issue flagged in STATE.md:154. The fix is to rebuild L2 with a different feature-admission rule (option (a), (b), or (c) above). Cannot be patched without changing the model semantics.

---

## Testing Gaps

### Test-1: Cross-Refit State-Id Alignment (CR-02)

**Missing:** No test verifies that regime belief is correctly re-mapped when state ids change across refits.

**Files:** `src/trading_crab_lib/platform/prediction/regime_filter.py`, `tests/unit/test_platform_walkforward.py`

**Risk:** Silent state-id drift would go undetected. Current tests only check that the filter runs; they don't validate semantic correctness across refits.

**Recommendation:** Add a test that fits jump_model on two overlapping windows, permutes state ids, and verifies that the filter's belief correctly maps.

---

### Test-2: Splice Fallback Provenance (D-04)

**Missing:** No test for the fallback chain in splice. All tests use cached raw data with no missing sources.

**Files:** `src/trading_crab_lib/platform/splice.py` (resolve_class_sources), `tests/unit/test_platform_*splice*.py`

**Risk:** A partial ingestion failure (gold_spot present but all NaN) would not be caught by any test.

**Recommendation:** Add a test that intentionally zeros out `gold_spot` and verifies:
- Fallback to IAU is logged at WARNING level
- Provenance JSON records the fallback
- Downstream models are alerted to the truncation

---

### Test-3: Weekly Report Mode Switching (A-15)

**Missing:** No test for the mode-switch logic in `weekly.py` (allocation_mode changed from `regime_tilt` to `no_regime` or vice versa).

**Files:** `src/trading_crab_lib/platform/report/weekly.py` (lines 200-220), `tests/unit/test_platform_weekly_page.py`

**Risk:** A mode switch would execute the new target in full (no band against old book). If the test is missing, a bug here could cause unexpected trades.

**Recommendation:** Add a test that:
1. Sets `allocation_mode: regime_tilt` in executed_weights checkpoint
2. Runs weekly report with config `allocation_mode: no_regime`
3. Verifies (a) report says "mode changed", (b) target trades full, (c) no band applied

---

## Test Artifacts & Maintenance Debt

### Test-Debt-1: Untagged Registry Rows (Phase 7)

**What happened:** Phase 7 wave 1 appended 4 untagged rows to `platform/honesty/registry.jsonl` during wiring verification (07-04). These rows inflated D-16 (total trial count) from 38 to 42 without being policy evaluations.

**Resolution (2026-09-15):** Registry was archived, reset with `prior_genuine_trials=38` header, and `append_trial` now refuses rows without `trial_tag` or `NO_REGISTRY` sentinel.

**Current status:** Resolved. Future wiring runs must use `--smoke` flag (appends NO_REGISTRY) or provide `--trial-tag`.

**Monitoring:** `tests/unit/test_platform_registry_hygiene.py` checks that registry rows have tags. No test fails if this rule is violated (non-gating).

**Classification:** **SAFE TO LEAVE.** The guard is in place. Monitor registry appendage in weekly runs via the tag-check test.

---

## Performance Concerns

### Perf-1: Backtest Driver Runtime (No Blocking Issue, But Large)

**What it is:** `backtest/driver.py::run_backtest()` walks forward 588 months (1972–2020) in nested loops:
- Outer: walk-forward refits (monthly)
- Inner: per-month decision + allocation + P&L

**Runtime:** ~30–60 seconds on a 4-core laptop (measured, not gated).

**Risk:** If run_backtest is called multiple times per session (e.g., in a notebook for A/B analysis), user will wait. No progress bar or checkpoint recovery.

**Classification:** **LOW PRIORITY.** Not a blocker. MVP runs backtest ~once per phase. If interactivity is needed in notebooks, add a progress bar and checkpoint recovery (`load_partial_backtest`, save equity curve checkpoint per month).

---

## Data Quality & Validation

### Data-1: No Pre-flight Check for Required Columns

**What it is:** `build_platform_data.py` does not validate that ingested `monthly_raw` has all expected columns before proceeding to splice.

**Files:** `scripts/build_platform_data.py`, `src/trading_crab_lib/platform/ingestion/macro_monthly.py`

**Risk:** If FRED or multpl API changes and a column is dropped, the splice will fail with a cryptic "column not found" error downstream, not at the source.

**Mitigation in place:** `assert_yield_units_plausible()` catches yield-unit errors. But no general column-schema check.

**Recommendation:** Add a preflight check in `build_platform_data.py` that validates all required columns in `monthly_raw` are present and non-empty, with clear error messages.

---

## Summary by Classification

### SAFE CLEANUP NOW (No behavior change, low risk)
- C-2: Confirm-only checks (remove 50 lines, make tests real)
- C-4: Archive research scripts (move 2000 lines to archive/)
- C-9: DSR hurdle gate (remove 20 lines from weekly report)
- Test-1, Test-2, Test-3: Add missing tests (~200 lines total)

### NEEDS DECISION
- C-1: Trial registry ceremony (simplify or keep for rigor)
- C-7: Dual allocation modes (hard-code or keep switch)
- C-8: Hysteresis thresholds (move to config or keep in code)
- Frag-1: Splice fallback (fail or warn on gold_spot truncation)
- Frag-3: Nowcaster fix (Phase 9 scope decision)

### LEAVE FOR PHASE 9
- C-3: Parked code (100 KB, used by Phase 9 regime rebuild)
- C-5: weekly.py monolith (1299 lines, wait for regime refactor)
- C-6: evaluation/report.py (1155 lines, keep for honesty)
- Frag-2, Frag-3: CR-02, L2-02 (Phase 9 deferred, per DECISIONS)

### ALREADY RESOLVED / MONITORED
- Bug-1 (D-08): Merge-on-save — rule decided, implementation deferred to 08.4
- Test-Debt-1: Registry contamination — fixed with tag requirement, no-retest needed
- Open-1, Open-2, Open-3: CR-02, CR-03, E-11 — measured, qualified, deferred per DECISIONS

---

**Analysis Date:** 2026-10-05

*Codebase concerns analysis: 2026-10-05*
