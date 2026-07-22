---
phase: 03-regime-labeling-prediction
verified: 2026-07-22T00:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 3: Regime Labeling & Prediction Verification Report

**Phase Goal:** The system can label historical market regimes with a temporally-persistent
jump model and nowcast today's regime with calibrated probabilities, using only causal
information.
**Verified:** 2026-07-22
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running the labeler on the 1962+ monthly feature set produces regime labels (default λ, K=5) via a k-means-warm-started jump model with per-jump penalty, exact DP decode, and multiple restarts | ✓ VERIFIED | `src/trading_crab_lib/platform/labeling/jump_model.py::decode_states_dp` — exact O(TK) min/second-min DP (Bemporad & Boyd 2018 recurrence), proven identical to `itertools.product` brute-force enumeration across 7 `(T,K,λ)` cases + 1 large-λ constant-state case (`tests/unit/test_platform_labeling.py::TestDPDecodeExact`, ran live: 8/8 pass). `fit_jump_model` warm-starts via `sklearn.cluster.KMeans(n_init=1, init="k-means++")`, alternates DP-decode/recompute-centroids to convergence, keeps the lowest-`total_cost` of `n_restarts` independent restarts, and is proven bit-identical across two calls with the same seed (`TestFitJumpModel::test_determinism`, ran live: pass). `config/platform_settings.yaml` `labeling:` section: `K: 5`, `lambda: 52.0`, `n_restarts: 10` (defaults match design §4.1 + CONTEXT D-locked spec). |
| 2 | Labels come with soft confidences and are persisted; the trailing 6–12 months of labels are marked/embargoed so L2 training cannot see them | ✓ VERIFIED | `soft_confidences()` — temperature-free softmax over `-d`, row-stochastic (`TestSoftConfidences`, live pass). `label_regimes()` persists `regime_labels`, `regime_confidences` (rows sum to ~1.0, `TestLabelRegimesPersistence`), and `regime_profiles` via the platform `CheckpointManager`. Embargo is structural, not advisory: `build_nowcaster_training_set(..., embargo_months=12)` physically slices `labels.index <= labels.index.max() - DateOffset(months=12)` before ever building `y`, proven by `TestEmbargoInvariant::test_trailing_12_months_excluded_on_real_output` (asserts on real output, not a mocked boundary) plus a fail-fast negative-embargo guard — both ran live, pass. Config default `embargo_months: 12` (top of the 6–12 range per CONTEXT D-01). |
| 3 | A label-churn metric (fraction of trailing labels revised on each refresh) is computed and available for monitoring after each run | ✓ VERIFIED | `label_churn(prev, new, trailing_months=24)` operates on canonicalized labels (avoids false churn from arbitrary cluster-index relabeling). `label_regimes()` loads the *previous* `regime_labels` checkpoint BEFORE saving the new one (`_churn_against_previous`), so churn reflects a true before/after diff. `TestChurnTwoRun::test_first_run_nan_second_run_zero` proves the ordering end-to-end: run 1 (no prior checkpoint) → `nan`; run 2 (identical rerun) → `0.0` — ran live via `python3 -m trading_crab_lib.platform.labeling.diagnostics`, observed `self-check: 96 months labeled, churn=nan` on first pass and `0.0` on the pytest two-run case (confirmed passing). |
| 4 | Given causal features through today, the nowcaster returns a calibrated probability distribution over regimes (not a single argmax class) | ✓ VERIFIED | `fit_nowcaster()` gates inputs through `assert_causal_features(X.columns)` (Phase 2 honesty rail), fits a plain `LogisticRegression` (no `multi_class` kwarg — verified absent from the installed sklearn 1.9.0 signature by `test_no_multi_class_kwarg_on_logistic_regression`) inside `CalibratedClassifierCV(method="sigmoid", cv=PurgedEmbargoedKFold(...))`. `predict_proba(X)` returns `(n, n_classes)` rows summing to ~1.0 (`TestNowcasterCalibratedOutput`, live pass); a dedicated test confirms `predict_proba` (not `predict()`/argmax) is the documented contract. `transition_window_accuracy()` reports `overall_accuracy` + `transition_accuracy` + `steady_state_accuracy` together, never overall alone — proven against a hand-computed 10-month series with errors placed inside/outside the transition window (`test_hand_computed_transition_vs_steady_state`: overall=0.7, transition=0.6, steady_state=0.8, exact match) plus a no-transition nan-not-raise case. `evaluate_nowcaster()` logs exactly one registry trial per call (`TestRegistryLoggingOne::test_exactly_one_new_trial_per_evaluate_call`, live pass: 1 trial → 2 trials across two calls) and persists the model via `CheckpointManager.save_model()` (joblib, `pickle.dump` absent from module source — P27, verified by source-inspection test). |
| 5 | An empirical transition matrix is available as a diagnostic showing the forward regime distribution implied by history | ✓ VERIFIED | `empirical_transition_matrix(states)` — pure `pd.crosstab`-based row-normalized K×K table, `pairs["from"]`/`pairs["to"]` from consecutive label positions, `counts.div(counts.sum(axis=1), axis=0)`. Occupied rows sum to 1.0 (`test_occupied_rows_sum_to_one`), a hand-computed known-count case matches exactly (`test_hand_computed_known_counts`), a never-a-"from"-state degrades gracefully with no crash (`test_never_a_from_state_does_not_crash`), and it is confirmed a pure function with no I/O/model-state (`test_pure_function_no_io`) — all 4 ran live, pass. `__main__` self-check run live produces a correct 3×3 row-normalized table. |

**Score:** 5/5 truths verified

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| L1-01 | 03-01 | Jump-model labeler (k-means + per-jump λ, exact DP, multi-restart, k-means warm start), default λ=52.0/K=5 | ✓ SATISFIED | `jump_model.py`; `TestDPDecodeExact`, `TestFitJumpModel` (live pass) |
| L1-02 | 03-02, 03-03 | Labels + soft confidences persisted; trailing labels structurally embargoed from L2 | ✓ SATISFIED | `diagnostics.py::label_regimes` persistence; `nowcaster.py::build_nowcaster_training_set` structural embargo (live pass) |
| L1-03 | 03-02 | Label churn tracked as a monitoring metric | ✓ SATISFIED | `label_churn`, `TestChurnTwoRun` (live pass) |
| L2-01 | 03-03 | Calibrated logistic nowcaster returning probabilities, not argmax | ✓ SATISFIED | `nowcaster.py::fit_nowcaster`/`evaluate_nowcaster`; `TestNowcasterCalibratedOutput`, `TestTransitionWindowAccuracy`, `TestRegistryLoggingOne` (live pass) |
| L2-02 | 03-04 | Empirical transition matrix diagnostic | ✓ SATISFIED | `transition_matrix.py::empirical_transition_matrix`; `TestEmpiricalTransitionMatrix` (live pass) |

**Note (informational, non-blocking):** `.planning/REQUIREMENTS.md` still shows L1-01…L1-03/L2-01/L2-02 as unchecked `[ ]` / status `Pending` in its tracking table — a documentation-sync gap (the checkbox/tracking-table update step for this phase was not run), not a code gap. Every requirement is independently verified as satisfied by the codebase evidence above. Recommend a follow-up doc-only edit to check these boxes; not gating this verification.

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/trading_crab_lib/platform/labeling/jump_model.py` | DP decode, warm start, multi-restart, canonicalization | ✓ VERIFIED | Exists, substantive, wired (imported by `diagnostics.py`, exercised by 21 live-passing tests) |
| `src/trading_crab_lib/platform/labeling/diagnostics.py` | Persistence, churn, §4.4 report-only diagnostics, auto-profile | ✓ VERIFIED | Exists, substantive, wired; `__main__` self-check run live, produces correct WARNING-on-violation output |
| `src/trading_crab_lib/platform/prediction/nowcaster.py` | Structural embargo, calibrated fit, transition-window accuracy, registry logging | ✓ VERIFIED | Exists, substantive, wired; 12 live-passing tests |
| `src/trading_crab_lib/platform/prediction/transition_matrix.py` | Empirical row-normalized transition matrix | ✓ VERIFIED | Exists, substantive, wired; 4 live-passing tests + `__main__` self-check run live |
| `config/platform_settings.yaml` `labeling:` section | K=5, lambda=52.0, n_restarts=10, embargo_months=12 | ✓ VERIFIED | Confirmed present via direct file read; not added to `_REQUIRED_PLATFORM_SECTIONS` (per CONTEXT instruction), read defensively via `.get()` |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| `diagnostics.py::label_regimes` | `jump_model.py` functions | direct import + call chain (standardize → fit → canonicalize → confidences) | WIRED | Confirmed by reading source + live `__main__` run producing real labeled output |
| `diagnostics.py::label_regimes` | `platform/checkpoints.py::get_platform_checkpoint_manager` | persistence of `regime_labels`/`regime_confidences`/`regime_profiles` | WIRED | Live run logged 3 checkpoint saves |
| `nowcaster.py::fit_nowcaster` | `platform/honesty/{cv,gating}.py` | `PurgedEmbargoedKFold`, `assert_causal_features` | WIRED | Imported and called directly; `TestNowcasterCalibratedOutput` exercises the full call chain live |
| `nowcaster.py::evaluate_nowcaster` | `platform/honesty/registry.py` | `registry.append_trial` | WIRED | `TestRegistryLoggingOne` proves exactly one new row per call, live |
| `nowcaster.py::evaluate_nowcaster` | `checkpoints.py::CheckpointManager.save_model` (joblib) | model persistence | WIRED | Source-inspection test confirms no `pickle.dump`; `save_model` uses `joblib.dump` (verified in `checkpoints.py`) |

### Cross-Cutting Constraints (CONTEXT decisions)

| Constraint | Status | Evidence |
|---|---|---|
| D-02: §4.4 report-only, never raises, WARNING on violation | ✓ VERIFIED | `report_labeling_diagnostics` never raises on a low-occupancy state; live run showed `WARNING:__main__:State 0 occupancy 3.1%...report-only, not blocking (D-02)` followed by normal completion. `TestReportDiagnosticsReportOnly` (live pass). |
| D-03: temperature-free distance softmax, no `/T` knob | ✓ VERIFIED | `soft_confidences()` has no temperature parameter; docstring explicitly forbids adding one. |
| D-04: numeric labels + auto_profile, no human-pinned names | ✓ VERIFIED | `auto_profile()` returns `dict[int, str]` keyed by numeric state id; no YAML name file created in this phase (`grep` for `platform_regime_labels.yaml` — absent, as required). |
| Frozen incumbent untouched | ✓ VERIFIED | `git log` on phase-3 commits touches only `src/trading_crab_lib/platform/**`, `tests/unit/test_platform_*.py`, and `config/platform_settings.yaml`; no `legacy/` or non-platform `src/trading_crab_lib/*.py` diffs. |
| No incumbent-pipeline imports in platform code | ✓ VERIFIED | Only cross-boundary import is `from trading_crab_lib import OUTPUT_DIR` (package-level constant) and `from trading_crab_lib.checkpoints import CheckpointManager` (Phase-1-established shared infra reuse pattern, documented in `platform/checkpoints.py`'s own docstring — not the frozen quarterly *pipeline* logic in `clustering.py`/`prediction/`/etc., which is never imported). |
| Zero new pip dependencies | ✓ VERIFIED | All new imports are `numpy`, `pandas`, `sklearn.{cluster,preprocessing,calibration,linear_model}`, and stdlib (`pathlib`, `typing`, `logging`); no `pyproject.toml`/`requirements*.txt` diffs in the phase's commit range. |
| `from __future__ import annotations` in all new modules | ✓ VERIFIED | Present in all 4 new source files (`jump_model.py`, `diagnostics.py`, `nowcaster.py`, `transition_matrix.py`) and `platform/prediction/__init__.py`. |
| Models persisted via joblib, not pickle | ✓ VERIFIED | `evaluate_nowcaster` uses `CheckpointManager.save_model()` → `joblib.dump` (see `checkpoints.py:267`); no raw `pickle` import anywhere in `platform/prediction/`. |

### Behavioral Spot-Checks (empirical, run live during this verification)

| Behavior | Command | Result | Status |
|---|---|---|---|
| Targeted labeling tests | `pytest tests/unit/test_platform_labeling.py -q` | 34 passed | ✓ PASS |
| Targeted nowcaster tests | `pytest tests/unit/test_platform_nowcaster.py -q` | 12 passed | ✓ PASS |
| Targeted transition-matrix tests | `pytest tests/unit/test_platform_transition_matrix.py -q` | 4 passed | ✓ PASS |
| DP-decode brute-force equivalence | `pytest -k "TestDPDecodeExact or TestChurnTwoRun or TestLabelingConfig"` | 11 passed | ✓ PASS |
| Embargo/calibration/transition-window/registry invariants | `pytest -k "TestEmbargoInvariant or TestNowcasterCalibratedOutput or TestRegistryLoggingOne or TestTransitionWindowAccuracy"` | 12 passed | ✓ PASS |
| Labeler `__main__` self-check | `python3 -m trading_crab_lib.platform.labeling.diagnostics` | Real WARNING emitted on low-occupancy state, artifact written, `churn=nan` on first run | ✓ PASS |
| Transition-matrix `__main__` self-check | `python3 -m trading_crab_lib.platform.prediction.transition_matrix` | Correct row-normalized 3×3 table printed | ✓ PASS |
| Full repo test suite (run once) | `pytest tests/ -q` | 913 passed, 49 skipped | ✓ PASS (matches SUMMARY/context claim) |

### Anti-Patterns Found

None. No `TODO`/`FIXME`/`XXX`/`HACK`/`PLACEHOLDER` markers in the 4 new modules. No stub returns (`return null`/`return {}`/empty-array stubs). No hardcoded-empty data paths. No `console.log`/debug-only bodies. `print()` calls in `diagnostics.py`/`transition_matrix.py` are documented first-class CLI-run output (matching the existing `gap_lag.py` pattern), not debug noise, and marked `# noqa: T201` deliberately.

### Human Verification Required

None. All 5 success criteria and all cross-cutting constraints were verified via direct code reading, live pytest execution, and live `__main__` self-check execution — no visual, real-time, or external-service behavior is in scope for this phase.

### Gaps Summary

No gaps. All 5 ROADMAP success criteria, all 5 requirements (L1-01, L1-02, L1-03, L2-01, L2-02), and all CONTEXT cross-cutting decisions (D-02, D-03, D-04, frozen-incumbent, no-incumbent-imports, zero-new-deps, `from __future__ import annotations`, joblib-not-pickle) are verified against the actual codebase — not merely claimed by SUMMARY.md. The one documentation-sync issue (`.planning/REQUIREMENTS.md` checkbox table not updated) is informational only and does not affect the phase goal, which is independently proven true in code and tests.

---

_Verified: 2026-07-22_
_Verifier: Claude (gsd-verifier)_
