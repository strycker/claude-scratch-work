---
phase: 05-honest-backtest-evaluation
verified: 2026-07-27T00:00:00Z
status: passed
score: 4/4 success criteria verified
behavior_unverified: 0
overrides_applied: 0
gaps: []
deferred:
  - "RESOLVED 2026-08-04 — Activate the regime layer pre-1990. Implemented as per-feature min_history (approach ii): backtest.feature_min_history (default 120) admits a late-starting feature once it has that many months in-window, so pre-1990 windows train on the features that exist then instead of collapsing to an any-NaN row-drop. Real-run effect: 512/588 steps predicted (was ~370), terminal log wealth 32.2 -> 111.06, max drawdown -82.5%/215mo -> -66.2%/12mo, no-regime-ablation delta -81.7 -> -2.8. Follow-on: driver._cv_safe_active_features additionally gates admission on the induced training block clearing n_splits examples per class, which is the actual cause of the activation-date degrade cluster (block SIZE is not - a 240-row block with a 3-example regime degrades exactly like a 120-row one)."
  - "OPEN — Re-include gold once a free long-history spot source is reachable (macrotrends 403-blocks datacenter/VPN IPs; FRED delisted LBMA gold). Currently excluded via the optional-splice toggle and flagged in the report's 'Excluded assets' note. Non-blocking for D-01; carried forward to the next milestone as a data-sourcing item, not a Phase 5 gap."
human_verification:
  - test: "Run python -m trading_crab_lib.platform.evaluation.report against the real Phase 1 monthly checkpoints with a live FRED_API_KEY and confirm the honest 1972–2020 walk-forward runs end-to-end, stops at the holdout boundary, and reports every required number/artifact."
    expected: "The run completes (588 monthly steps, 1972-01 → 2020-12) and writes the report + equity-curve/KPI/model-metrics parquets; the sojourn/detection-lag ratio is the first metrics section; the four-baseline gauntlet + no-regime-ablation delta + smoothed-vs-filtered gap + KPI table are present; the equity curve ends ≤ 2020-12; the trial registry gains exactly two rows (strategy + ablation); no DSR and no 2021+ read appear."
    why_human: "This sandbox has no network egress to Yahoo/macrotrends and cannot run the real data build; the automated suite (1144 tests) is synthetic/mocked at the ingestion boundary by design. The phase's exit criterion — a real 1972–2020 honest run — can only be demonstrated against live data."
    precondition_status: "SATISFIED 2026-07-27 — operator ran the real build + backtest locally (live FRED_API_KEY) and pasted the full report + boundary + registry + artifact evidence, which verifies all four success criteria. See 05-07-SUMMARY.md for the evidence."
---

# Phase 5: Honest Backtest & Evaluation — Verification Report

**Phase Goal:** The full tracer-bullet pipeline (L1–L4) can be evaluated honestly over
1972–2020 walk-forward, against real baselines, with first-class metrics that reveal
whether regime timing is actually worth anything.

**Verified:** 2026-07-27
**Status:** passed (4/4 success criteria) — human-verified real run

## Success Criteria

**1. Walk-forward 1972–2020 executes all layers (L1→L4) end-to-end and produces a
result, without touching the 2021+ holdout. — ✅ VERIFIED**
Real run completed 588 monthly steps, 1972-01 → 2020-12; strategy equity curve ends
2020-12-31 (no 2021+ row). `split_by_holdout_boundary` applied before `expanding_steps`
(Plan 02); grep gate confirms no DSR / no holdout eval (Plan 06, D-04).

**2. Baseline gauntlet — SPY, 60/40, Faber 10-month SMA — computed over the same window
for direct comparison (+ no-regime ablation, §8.7). — ✅ VERIFIED**
Report gauntlet table shows SPY (5.68 / −48.95%), 60/40 (131.03 / −2.27%), Faber (1.14 /
−99.69%), plus the no-regime ablation (113.89 / −68.08%) built via
`no_regime_ablation(use_regime_tilt=False)` — same code path, not a fork (D-02).

**3. Sojourn/detection-lag ratio reported prominently as the headline go/no-go number.
— ✅ VERIFIED**
"## Headline: Sojourn / Detection-Lag Ratio" is the FIRST metrics section: median
sojourn 46.5 mo / median detection lag 107.0 mo → ratio 0.435, per-target-state
construction (review F1).

**4. Model-metrics artifacts (multiclass Brier, calibration bins, confusion tables)
persisted per run. — ✅ VERIFIED**
`model_metrics_brier.parquet`, `model_metrics_calibration.parquet`,
`model_metrics_confusion.parquet` written under `outputs/reports/platform/` and
confirmed to round-trip.

## Diagnostic outcome (D-01)

The phase passes on running + reporting correctly, not on beating a benchmark. It
honestly reports that the naive regime layer does **not** pay rent yet (strategy 32.18
< ablation 113.89 < 60/40 131.03; ratio 0.43 ≪ ~5 bar). This is the intended §14
"beats nothing yet — that's fine" result; these are the standing targets for the
L4-upgrade milestone (D-01a).

## Code verification

All six implementing plans (05-01…05-06) are code-complete with the cross-AI review
findings (F1–F7) incorporated and covered by the automated suite (**1144 tests pass**,
no regressions). Data-source resilience added during the real-run bring-up (Stooq
price fallback, FRED oil fallback, gold-exclusion toggle, empty-CV-fold guard, nowcaster
scaling + non-finite-row drop) is on PR #112.

**Re-verification:** No — initial verification; human-verified real run.

---

## Phase Closure Record (2026-08-04)

Post-verification work landed on `main` after the 2026-07-27 sign-off. Recorded here so
the phase closes against what is actually in the tree, not what was verified mid-flight.

### Work completed after verification

| Item | Effect |
|---|---|
| **Pre-1990 regime activation** (`backtest.feature_min_history`, approach ii) | Deferred item 1 resolved. 512/588 steps predicted (was ~370) |
| **CV-safety guard** (`driver._cv_safe_active_features` + `backtest.nowcaster_cv_splits`) | Late features admitted only once the induced training block clears `n_splits` examples per class — the real cause of the activation-date degrade cluster |
| **Headline sample transparency** (`n_transitions` / `n_resolved` / `act_threshold`) | The sojourn/lag ratio now reports the sample it was computed over, plus a small-sample caveat below 8 resolved transitions |
| **Data-source resilience** (Stooq fallback, FRED oil fallback, gold-exclusion toggle, empty-CV-fold guard, nowcaster scaling + non-finite-row drop) | Real run reproducible from a fresh clone against degraded free sources |

### Updated real-run numbers (post-activation)

| Metric | At verification | After activation | Note |
|---|---|---|---|
| Terminal log wealth (strategy) | 32.18 | **111.06** | |
| Max drawdown | −82.5% (215mo underwater) | **−66.2% (12mo)** | |
| No-regime-ablation delta | −81.7 | **−2.83** | still negative — regime layer does not pay rent yet |
| Median sojourn / lag → ratio | 46.5 / 107.0 → 0.435 | 84.5 / 161.5 → **0.52** | resolves on only ~2 of ~6 transitions |
| Multiclass Brier | — | **0.20** | |
| Steps predicted | ~370 / 588 | **512 / 588** | |

### Diagnostic conclusion is unchanged

D-01 still passes on *running and reporting correctly*, and the honest finding still
stands: **the naive regime layer does not pay rent yet** (ablation 113.89 > strategy
111.06; ratio 0.52 ≪ the ~5 bar). The activation work narrowed the gap from −81.7 to
−2.83 without closing it. That remains the standing target for the L4-upgrade milestone,
exactly as design §14 anticipates.

The detection-lag figure is now understood: it is a **small-sample artifact**, not a
model defect. Detection requires P(target state) ≥ 0.70; unresolved transitions are
dropped from the median; and a long-history reference yields ~6 transitions of which ~2
resolve. The report now surfaces those counts so the number is never mistaken for a
large-sample estimate.

### Closure checklist

- [x] All 4 success criteria verified (4/4)
- [x] All 7 plans code-complete with SUMMARY.md
- [x] Cross-AI review findings F1–F7 incorporated
- [x] Human-verified real 1972–2020 run (operator, live FRED_API_KEY)
- [x] Deferred item 1 (pre-1990 activation) — **resolved**
- [x] Deferred item 2 (gold re-inclusion) — carried to next milestone as a data-sourcing item
- [x] Full suite green on `main` (**1157 passing**)
- [x] Holdout discipline intact — no DSR, no 2021+ read

**PHASE 5 CLOSED — 2026-08-04.**
