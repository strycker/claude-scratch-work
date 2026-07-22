---
phase: 03-regime-labeling-prediction
plan: 01
subsystem: ml-regime-labeling
tags: [jump-model, dynamic-programming, kmeans, sklearn, numpy, regime-detection]

# Dependency graph
requires:
  - phase: 01-monthly-data-layer
    provides: monthly_features platform checkpoint, taxonomy.lean_feature_set()
  - phase: 02-honesty-infrastructure
    provides: platform/config.py load_platform_config(), platform code conventions
provides:
  - "decode_states_dp: exact O(TK) DP decode of jump-model state sequence given fixed centroids"
  - "soft_confidences: temperature-free softmax-over-negative-squared-distance confidence matrix"
  - "standardize_features: winsorize [1%,99%] + StandardScaler for lean feature DataFrames"
  - "fit_jump_model: multi-restart k-means-warm-started alternation, deterministic, empty-state-safe"
  - "canonicalize_states: economic-sort state relabeling (stable numbering across restarts/refreshes)"
  - "config/platform_settings.yaml labeling: section (K=5, lambda=52.0, n_restarts=10, embargo_months=12)"
affects: [03-02-labeling-diagnostics, 03-03-nowcaster, phase-4-asset-prediction]

# Tech tracking
tech-stack:
  added: []  # zero new dependencies — numpy/pandas/sklearn already installed
  patterns:
    - "Exact DP decode via min/second-min O(TK) trick (Bemporad & Boyd 2018 recurrence)"
    - "Freeze-on-empty centroid guard prevents NaN poisoning in coordinate-descent alternation"
    - "Economic-sort canonicalization (not Hungarian matching — that's the deferred v2 subsample-stability tool)"

key-files:
  created:
    - src/trading_crab_lib/platform/labeling/__init__.py
    - src/trading_crab_lib/platform/labeling/jump_model.py
    - tests/unit/test_platform_labeling.py
  modified:
    - config/platform_settings.yaml

key-decisions:
  - "decode_states_dp implemented exactly per RESEARCH.md Pattern 1 (verbatim recurrence) — proven exact against itertools.product brute-force enumeration across 7 parametrized (T,K,lambda) cases plus a large-lambda constant-state case"
  - "Extracted _recompute_centroids as a private helper inside fit_jump_model's alternation loop (CLAUDE.md Function Design: break helpers out past ~40 lines) — also made the freeze-on-empty guard independently reasoned-about, though the acceptance test exercises it end-to-end through fit_jump_model per the plan's literal criteria"
  - "labeling: config block added to platform_settings.yaml but NOT added to _REQUIRED_PLATFORM_SECTIONS, per CONTEXT's explicit instruction — read defensively via cfg.get('labeling', {}) everywhere"

requirements-completed: [L1-01]

coverage:
  - id: D1
    description: "decode_states_dp returns the exact global minimizer of the jump-model objective, matching brute-force enumeration"
    requirement: "L1-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestDPDecodeExact::test_matches_brute_force_on_small_TK"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestDPDecodeExact::test_large_lambda_gives_constant_state"
        status: pass
    human_judgment: false
  - id: D2
    description: "soft_confidences returns a row-stochastic, temperature-free confidence matrix"
    requirement: "L1-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestSoftConfidences"
        status: pass
    human_judgment: false
  - id: D3
    description: "fit_jump_model produces deterministic K=5 states on separable synthetic data, warm-started via k-means, empty-state-safe"
    requirement: "L1-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestFitJumpModel"
        status: pass
    human_judgment: false
  - id: D4
    description: "canonicalize_states produces stable, idempotent, permutation-invariant state numbering sorted by trailing_return_1m"
    requirement: "L1-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestCanonicalize"
        status: pass
    human_judgment: false
  - id: D5
    description: "labeling: config section present with documented lambda derivation, read defensively, not a required section"
    requirement: "L1-01"
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling.py::TestLabelingConfig"
        status: pass
    human_judgment: false

# Metrics
duration: 25min
completed: 2026-07-22
status: complete
---

# Phase 3 Plan 01: Jump-Model Labeler Core Summary

**Exact DP-decode jump-model core (decode_states_dp, fit_jump_model, canonicalize_states, soft_confidences) proven against brute-force enumeration, with a `labeling:` config section defaulting K=5/lambda=52.0/n_restarts=10.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-22T19:11:30Z (first RED commit)
- **Completed:** 2026-07-22T19:15:53Z (final GREEN commit)
- **Tasks:** 2 (both TDD, RED then GREEN)
- **Files modified:** 4 (2 created new modules, 1 new test file, 1 config edit)

## Accomplishments
- `decode_states_dp(d, lam)` — exact O(TK) dynamic program (min/second-min trick), proven identical to `itertools.product` brute-force enumeration across 7 parametrized `(T, K, λ)` cases plus a large-λ constant-state case
- `soft_confidences(d)` — temperature-free softmax over `-d`, row-stochastic, numerically stable under per-row shifts
- `standardize_features(X)` — winsorize [1%, 99%] per column then `StandardScaler`
- `fit_jump_model(X, K, lam, *, n_restarts, max_iter, random_state)` — multi-restart k-means-warm-started coordinate-descent alternation; deterministic given `random_state`; freeze-on-empty centroid guard verified with no NaN even when 3 of 6 requested states have zero occupancy every iteration
- `canonicalize_states(states, centroids, feature_names)` — economic sort on `trailing_return_1m`, idempotent, stable under arbitrary input-label permutation
- `config/platform_settings.yaml` `labeling:` section: `K: 5`, `lambda: 52.0` (documented as `4 × len(lean_feature_set)`), `n_restarts: 10`, `embargo_months: 12` — deliberately excluded from `_REQUIRED_PLATFORM_SECTIONS`

## Task Commits

Each task was committed atomically (RED test commit before GREEN implementation commit, TDD):

1. **Task 1: Exact DP state decode + soft confidences**
   - `56e6607` (test) — RED: failing `TestDPDecodeExact` + `TestSoftConfidences`, confirmed `ModuleNotFoundError` before implementation existed
   - `9caa07b` (feat) — GREEN: `decode_states_dp` + `soft_confidences` implemented, all 11 tests pass
2. **Task 2: Multi-restart alternation, warm start, canonicalization + labeling config**
   - `2a689ae` (test) — RED: failing `TestStandardizeFeatures`/`TestFitJumpModel`/`TestCanonicalize`/`TestLabelingConfig`, confirmed `ImportError` before implementation existed (implementation was temporarily reverted via `git checkout --` to prove the RED gate honestly, then restored)
   - `4d71b16` (feat) — GREEN: `standardize_features`, `fit_jump_model`, `_recompute_centroids`, `canonicalize_states` implemented + `labeling:` config block added, all 20 tests pass

**Plan metadata:** this commit (docs: complete plan) — pending

_Note: both tasks are TDD; each has exactly a RED test commit followed by a GREEN implementation commit, no REFACTOR commit was needed (implementation matched RESEARCH.md's verified patterns on the first pass)._

## Files Created/Modified
- `src/trading_crab_lib/platform/labeling/__init__.py` - package init, module docstring only (mirrors `honesty/__init__.py`)
- `src/trading_crab_lib/platform/labeling/jump_model.py` - `decode_states_dp`, `soft_confidences`, `standardize_features`, `fit_jump_model`, `_recompute_centroids`, `canonicalize_states`
- `tests/unit/test_platform_labeling.py` - `TestDPDecodeExact`, `TestSoftConfidences`, `TestStandardizeFeatures`, `TestFitJumpModel`, `TestCanonicalize`, `TestLabelingConfig` (20 tests total)
- `config/platform_settings.yaml` - new `labeling:` section

## Decisions Made
- Extracted `_recompute_centroids(X, states, K, prev_centroids)` as a private helper (not a separately-planned symbol, but keeps `fit_jump_model` under CLAUDE.md's ~40-line-before-helper-extraction guidance and makes the freeze-on-empty behavior independently readable). This is additive scope only — does not change any public API surface listed in the plan's artifacts.
- Empty-state test scenario: `K=6` requested against synthetic data with only 3 real distinct clusters reliably starves exactly 3 states (verified empirically across 30 seeds before locking the test) — chosen over trying to force the freeze path via a large-λ mechanism alone, which was less deterministic to reproduce.
- Confirmed the RED gate honestly for Task 2 by temporarily reverting the already-drafted `jump_model.py` implementation via `git checkout --` (uncommitted working-tree file, not a destructive git op on history) before writing the test-commit, since the implementation had been drafted ahead of the test file during design/experimentation. Test file was proven to fail with `ImportError` before the implementation was restored and committed.

## Deviations from Plan

None - plan executed exactly as written. No Rule 1/2/3 auto-fixes were needed; RESEARCH.md's patterns were directly implementable without deviation.

## Issues Encountered
- Ad-hoc `python3 -c` shell probes outside pytest resolved `trading_crab_lib` from an editable install pointing at the main repo (`/home/user/claude-scratch-work/src`) rather than this worktree's copy, because no `PYTHONPATH` was set for those direct invocations. `pytest` itself was unaffected (its `pythonpath = ["src", "scripts"]` config in `pyproject.toml` takes precedence). Fixed by prefixing ad-hoc probes with `PYTHONPATH=src` per this worktree's CLAUDE.md instruction ("Do NOT run pip install -e anywhere ... use PYTHONPATH=src python3 -m pytest").

## User Setup Required

None - no external service configuration required. Zero new dependencies (numpy/pandas/scikit-learn already installed).

## Next Phase Readiness
- `jump_model.py`'s 5 functions are ready for Plan 03-02 (labeling diagnostics: occupancy/sojourn, label-churn against a previous checkpoint, auto-profile) to consume `fit_jump_model`'s output and `canonicalize_states`'s stable numbering.
- `labeling.embargo_months` (12) is present in config for Plan 03-03 (nowcaster) to read, distinct from `cv.default_embargo_months` (1) — the two "embargo" concepts documented in 03-RESEARCH.md Pitfall 4 remain separate config keys as required.
- Manual/deferred: once `FRED_API_KEY` is configured and the real 1962+ `monthly_features` checkpoint exists, running `fit_jump_model` on the real lean feature set and eyeballing report-only diagnostics is deferred to a later checkpoint per 03-RESEARCH.md Open Question 1/2 — does not block this plan (developed/tested purely on synthetic frames per the established Phase 1/2 pattern).

---
*Phase: 03-regime-labeling-prediction*
*Completed: 2026-07-22*
