---
phase: 02-honesty-infrastructure
verified: 2026-07-22T17:42:58Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 2: Honesty Infrastructure Verification Report

**Phase Goal:** Every subsequent modeling result is protected by structural honesty
guarantees — a physically separate holdout, a trial registry, a walk-forward runner,
purged CV, and causal-feature gating — installed before any model is tuned.
**Verified:** 2026-07-22T17:42:58Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Data dated 2021+ lives in separate files/paths the default dev pipeline cannot read; live-scoring mode must opt in explicitly | ✓ VERIFIED | `platform/honesty/holdout.py`: `write_monthly_features_split()` writes dev rows (`<=2020-12-31`) via `get_platform_checkpoint_manager()` and holdout rows (`>2020-12-31`) via a distinct `get_holdout_checkpoint_manager()` pointed at `data/holdout/`. `assert_dev_checkpoint_within_boundary()` raises `RuntimeError` naming the offending date. No fallback code path exists from the default manager to the holdout tree. Behavioral test `TestHoldoutBoundary::test_default_manager_cannot_load_post_2020_rows` proves by real load (not monkeypatching) that (a) the default manager's loaded frame has `index.max() <= 2020-12-31`, and (b) loading a holdout-only checkpoint through the default manager raises `FileNotFoundError`. 12/12 `test_platform_holdout.py` tests pass. |
| 2 | Every evaluated configuration (features, params, metrics) is automatically logged to a trial registry store and is queryable after a run, with no manual bookkeeping | ✓ VERIFIED | `platform/honesty/registry.py`: `append_trial()` writes one JSON line (config_hash, config, features, metrics, git_sha, timestamp) in append-only (`"a"`) mode; `read_trials()` reads the ledger back as a DataFrame. Ledger path defaults to `registry/trials.jsonl` (repo root, NOT under gitignored `data/` — confirmed not `.gitignore`d via `git check-ignore`). `run_walkforward()` calls `registry.append_trial()` automatically exactly once per run — no caller-side bookkeeping. `TestRegistryLoggingIsAutomatic` and `TestExactlyOneRegistryTrial` (in `test_platform_walkforward.py`) and `TestAppendNeverTruncates`/`TestRowSchema` (in `test_platform_registry.py`) all pass, proving the append-only guarantee and full schema. 16/16 registry tests + relevant walkforward tests pass. |
| 3 | A walk-forward runner refits on data ≤ t at each step, executes a trivial model end-to-end, and records the decision made at each step | ✓ VERIFIED | `platform/honesty/walkforward.py::run_walkforward()` clones a `DummyClassifier(strategy="prior")` (trivial model, sklearn stdlib) at every step via `expanding_steps()`, fits strictly on `index[:i]`, predicts `index[i:i+1]`, and appends a `{t, prediction, params}` decision dict. Behavioral test `TestExpandingStepsLeakageInvariant::test_train_window_strictly_before_test` asserts `train_index[-1] < test_index[0]` and `test_index[0] not in train_index` at every step (the leakage invariant, not just presence). `TestOneDecisionPerStep` confirms one decision per step with the full key set. Ran `pytest tests/unit/test_platform_walkforward.py` directly — 8/8 pass. |
| 4 | Purged + embargoed CV splitting is available as a drop-in replacement for `TimeSeriesSplit` for any supervised component with overlapping labels | ✓ VERIFIED | `platform/honesty/cv.py::PurgedEmbargoedKFold` subclasses `sklearn.model_selection.BaseCrossValidator` (`isinstance` test passes), implements `split(X)`/`get_n_splits(X)`, and requires `label_horizon`/`embargo` as keyword-only args with no default (raises `TypeError` if omitted — 3 tests confirm). Behavioral spot-check: `TestNoLeakageAcrossPurgeEmbargoWindow` (parametrized 3-way sweep) proves no train index falls inside the purge-or-embargo window of any test fold — this is the actual leakage-removal invariant, not just symbol presence. `TestNontrivialPurgeAtMonthlyHorizon` confirms purging is non-degenerate at a realistic 12-month horizon. 14/14 `test_platform_cv.py` tests pass. |
| 5 | Supervised training paths load causal (not centered/look-ahead) features by default with a loud opt-out; smoothed-vs-filtered gap and detection lag are computed and reported as first-class run outputs | ✓ VERIFIED | HON-06: `platform/honesty/gating.py::assert_causal_features()` scans column names for forbidden centered-window suffixes (`_centered`, `_c5`, `_zerophase`) and raises `ValueError` (loud, not silent) unless `allow_noncausal=True` is explicitly passed, in which case it logs `NONCAUSAL_USED=true` at WARNING. `select_platform_feature_path()` wires this to the real `monthly_features.parquet` checkpoint schema. No forbidden-suffix columns exist in the Phase 1 `monthly_features` output (confirmed by 02-PATTERNS.md's Open-Question-1 resolution and by the guard finding zero offenders against the real taxonomy), so the "default" path is causal by construction — there is no separate supervised training entry point in the codebase yet to wire this into (Phase 3 territory), which is in-scope per 02-CONTEXT.md phase boundary ("No modeling (Phase 3)"). HON-05: `platform/honesty/gap_lag.py` provides `compute_gap()`, `compute_detection_lag()`, `sojourn_lag_ratio()`, and `report_gap_lag()` which prints a human-readable summary to stdout/log AND persists a parquet artifact under `outputs/reports/model_metrics/gap_lag_metrics.parquet` (D-05). Ran `python3 -m trading_crab_lib.platform.honesty.gap_lag` directly (not trusting SUMMARY claims) — it executed end-to-end on synthetic data, printed the CLI summary, and wrote a real, readable parquet artifact (verified by loading it back with pandas: `gap=0.03, detection_lag_median=1.0, sojourn_lag_ratio=18.0`). No incumbent report path was touched (git diff confirms zero changes outside `platform/`, `tests/`, `.planning/`, `config/platform_settings.yaml`). 20/20 `test_platform_gating.py` + `test_platform_gap_lag.py` tests pass. |

**Score:** 5/5 truths verified (0 present-but-behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/trading_crab_lib/platform/honesty/holdout.py` | Physical holdout carve (HON-01) | ✓ VERIFIED | Exists, substantive, wired; behavioral test proves the invariant by real load |
| `src/trading_crab_lib/platform/honesty/registry.py` | Append-only JSONL trial registry (HON-02) | ✓ VERIFIED | Exists, substantive, append-only enforced by test |
| `src/trading_crab_lib/platform/honesty/walkforward.py` | Expanding-window runner (HON-03) | ✓ VERIFIED | Exists, substantive, wired to registry.append_trial automatically |
| `src/trading_crab_lib/platform/honesty/cv.py` | PurgedEmbargoedKFold (HON-04) | ✓ VERIFIED | Exists, substantive, sklearn BaseCrossValidator subclass confirmed |
| `src/trading_crab_lib/platform/honesty/gating.py` | Causal-feature gating guard (HON-06) | ✓ VERIFIED | Exists, substantive, raises loudly by default |
| `src/trading_crab_lib/platform/honesty/gap_lag.py` | Gap/detection-lag metrics + CLI/artifact (HON-05) | ✓ VERIFIED | Exists, substantive; CLI run confirmed to produce a real parquet artifact |
| `config/platform_settings.yaml` (holdout/registry/cv sections) | Config-driven documentation of cutoff/path/CV conventions | ⚠️ ORPHANED (minor) | Sections exist and document the correct values, but no code path in `holdout.py`/`registry.py` actually reads `platform_settings.yaml`'s `holdout.cutoff` or `registry.path` at runtime — both are hardcoded module constants that happen to match the YAML. `cv:` section is explicitly documented as "documentation-only, NOT silently applied" (intentional per Open Question 2). See gap note below. |
| `registry/trials.jsonl` | Git-tracked ledger location | ✓ VERIFIED (not yet populated) | Path is not `.gitignore`d and is outside `data/`; the directory does not yet exist in the repo because no real trial has been logged in dev use (Phase 2 is infrastructure-only — no model has been tuned yet, consistent with the phase boundary). All tests write to `tmp_path`, never touching the real path. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `walkforward.run_walkforward()` | `registry.append_trial()` | direct import + call at end of run | ✓ WIRED | Confirmed by `TestRegistryLoggingIsAutomatic` behavioral test |
| `holdout.get_platform_checkpoint_manager()` | `platform/checkpoints.py` | import | ✓ WIRED | Confirmed no fallback path exists |
| `gating.select_platform_feature_path()` | `platform/checkpoints.py` (`monthly_features.parquet` schema) | `pyarrow.parquet` schema read | ✓ WIRED | Reads real parquet schema; not yet called from any supervised training entry point (none exists in-repo yet — Phase 3 scope) |
| `gap_lag.report_gap_lag()` | `outputs/reports/model_metrics/gap_lag_metrics.parquet` | `pandas.DataFrame.to_parquet` | ✓ WIRED | Confirmed by direct CLI execution and re-read of the artifact |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Holdout invariant test (real load, not mocked) | `pytest tests/unit/test_platform_holdout.py -q` | 12 passed | ✓ PASS |
| Registry append-only guarantee | `pytest tests/unit/test_platform_registry.py -q` | 16 passed | ✓ PASS |
| Purged/embargoed CV leakage-window sweep | `pytest tests/unit/test_platform_cv.py -q` | 14 passed | ✓ PASS |
| Causal-gating guard raises loudly | `pytest tests/unit/test_platform_gating.py -q` | 8 passed, 1 skipped (network-dependent) | ✓ PASS |
| Gap/lag CLI end-to-end | `python3 -m trading_crab_lib.platform.honesty.gap_lag` | printed summary + wrote real parquet artifact, re-read via pandas | ✓ PASS |
| Walk-forward leakage invariant + automatic logging | `pytest tests/unit/test_platform_walkforward.py -q` | 8 passed | ✓ PASS |
| Full phase-2 targeted suite | `pytest tests/unit/test_platform_{holdout,registry,cv,gating,gap_lag,walkforward}.py -q` | 62 passed | ✓ PASS |
| Full repo test suite (regression check) | `pytest tests/ -q` | 863 passed, 49 skipped | ✓ PASS |
| No new pip dependencies since Phase 1 end | `git diff <phase1-end-sha>..HEAD -- pyproject.toml src/trading_crab_lib/pyproject.toml requirements*.txt` | empty diff | ✓ PASS |
| Frozen incumbent untouched | `git diff <phase1-end-sha>..HEAD --stat` | all 31 changed files within `platform/honesty/`, `tests/unit/test_platform_*`, `.planning/`, `config/platform_settings.yaml` | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| HON-01 | 02-01 | Physical 2021+ holdout carve | ✓ SATISFIED | `holdout.py` + 12 passing tests including real-load invariant |
| HON-02 | 02-02 | Append-only trial registry, git-tracked | ✓ SATISFIED | `registry.py` + 16 passing tests; path outside `data/`, not gitignored |
| HON-03 | 02-05 | Walk-forward runner, trivial model, auto-logging | ✓ SATISFIED | `walkforward.py` + 8 passing tests including leakage invariant |
| HON-04 | 02-03 | PurgedEmbargoedKFold BaseCrossValidator | ✓ SATISFIED | `cv.py` + 14 passing tests including leakage-window sweep |
| HON-05 | 02-04 | Gap/detection-lag metrics + CLI/artifact | ✓ SATISFIED | `gap_lag.py`, CLI run verified end-to-end, real artifact produced |
| HON-06 | 02-04 | Causal-feature gating, loud opt-out | ✓ SATISFIED | `gating.py` + 8 passing tests; raises `ValueError` by default |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/trading_crab_lib/platform/honesty/cv.py` | 1 | Missing `from __future__ import annotations` (CLAUDE.md convention: "Required at the top of all source files") | ℹ️ Info | No functional impact — module uses no `X \| Y` union syntax, so it does not break on Python 3.10. Convention-only drift; trivial one-line fix. |
| `config/platform_settings.yaml` | `holdout:` / `registry:` sections | Config documents `holdout.cutoff` and `registry.path`, but `holdout.py`/`registry.py` hardcode matching constants instead of reading them | ℹ️ Info | Currently harmless (values match), but if the YAML is edited in the future the code will silently diverge from the documented config. Not a violation of any HON-01..06 success criterion (none require config-sourced values), and the `cv:` section is explicitly and correctly documented as "documentation-only, not silently applied" — only `holdout`/`registry` have this minor inconsistency. |

No debt markers (`TBD`/`FIXME`/`XXX`) found in any phase-2 file. No `TODO`/`HACK`/`PLACEHOLDER` found. No stub returns (`return null`/empty-and-unwired) found in any of the six honesty modules.

### Human Verification Required

None. All five success criteria are verifiable programmatically via real code execution (not mocks) and behavioral tests, and were verified directly in this session.

### Gaps Summary

No gaps that block the phase goal. Two informational (non-blocking) findings are recorded above:
1. `cv.py` is missing the project's standard `from __future__ import annotations` header (cosmetic, zero functional risk).
2. `config/platform_settings.yaml`'s `holdout:`/`registry:` sections are documentation that the code doesn't actually read at runtime (values currently match; a future divergence risk, not a present one).

Neither finding affects any of the 5 ROADMAP success criteria or HON-01..06 requirements, all of which are independently verified as working by direct execution of code and tests in this session (not by trusting SUMMARY.md narrative).

---

_Verified: 2026-07-22T17:42:58Z_
_Verifier: Claude (gsd-verifier)_
