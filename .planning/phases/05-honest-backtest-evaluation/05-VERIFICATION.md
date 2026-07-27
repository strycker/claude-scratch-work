---
phase: 05-honest-backtest-evaluation
verified: 2026-07-27T00:00:00Z
status: passed
score: 4/4 success criteria verified
behavior_unverified: 0
overrides_applied: 0
gaps: []
deferred:
  - "Activate the regime layer pre-1990: fred_vix (VIXCLS) starts 1990-01 and L1's any-NaN row-drop collapses pre-1990 training, so the walk-forward holds the naive allocation 1972–~1990 (inflating the 107-month median detection lag). Fix: drop all-NaN-in-window feature columns so the model trains pre-1990 on available features. Non-blocking for the D-01 diagnostic."
  - "Re-include gold once a free long-history spot source is reachable (macrotrends 403-blocks datacenter/VPN IPs; FRED delisted LBMA gold). Currently excluded via the optional-splice toggle and flagged in the report. Non-blocking for D-01."
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
