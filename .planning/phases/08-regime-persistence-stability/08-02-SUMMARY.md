---
phase: 08-regime-persistence-stability
plan: 02
subsystem: platform/labeling (diagnostic only — no production code touched)
tags: [track-a, terminal-month, churn-vs-k, jump-model, zero-trial, PER-04, criterion-3]
status: complete
requires:
  - "07-11 (joint_lift_joint_l1only.parquet — the tracked curve the k=1 anchor reproduces)"
  - "scripts/run_joint_lift.py::build_inputs (reused unmodified; same frozen lists, same frames)"
provides:
  - "scripts/terminal_month_diagnostic.py — zero-trial L1-refit harness, k=1-anchored"
  - "outputs/reports/platform/track_a/terminal_month_labels.parquet — 588 x 25 (step x lag x classifier), a date per cell"
  - "outputs/reports/platform/track_a/terminal_month_churn.json — churn-vs-k, lambda/d, degraded counts, anchor, fixed-month revision"
  - "08-TRACK-A.md — the reading: FLAT in k, terminal-month edge artefact refuted"
affects:
  - "08-10 (criterion 7 re-measure: Track A is NOT expected to move; if state_1 churn != 246/587 something is wired wrong)"
  - "08-01 (churn.py is the consolidation point for this script's duplicated state_change_count)"
decisions:
  - "Verdict FLAT in k: classifier #1's iloc[-k] churn is 242..249 of 587 pairs across k=1..6; the single-month lambda-vs-2*lambda edge artefact is refuted."
  - "Scope recorded, not hedged: a trailing BLOCK of j months also costs one lambda, so k<=6 cannot exclude a longer block-edge effect; and lambda/d is NOT isolated by the #1/#2 contrast (they also differ in K, features, d). Isolating it needs a lambda sweep, which is not authorized."
  - "No monotonicity assertion anywhere; a structural test keeps it that way so the diagnostic can return its own refutation."
  - "Degrade semantics mirror joint_driver: a classifier #1 failure short-circuits #2 for that step (0 of 588 degraded in practice)."
tech-stack:
  added: []
  patterns: ["elementwise anchor against the tracked artifact before any derived number is reportable", "persist the full matrix so every derived number is recomputable; a test recomputes it"]
key-files:
  created:
    - scripts/terminal_month_diagnostic.py
    - tests/unit/test_platform_terminal_month_diagnostic.py
    - outputs/reports/platform/track_a/terminal_month_labels.parquet
    - outputs/reports/platform/track_a/terminal_month_churn.json
    - .planning/phases/08-regime-persistence-stability/08-TRACK-A.md
  modified: []
metrics:
  duration: "~45m active work, plus a ~33m full run; the wall clock also includes a session interruption (rate limit) between the run and Task 2's commit"
  completed: 2026-09-23
actuals:
  tokens: 14666
  tasks: 3
  commits: 4
---

# Phase 8 Plan 02: Track A terminal-month diagnostic Summary

**Zero-trial churn-vs-k diagnostic. The k=1 column reproduces the tracked `state_1`/`state_2`
elementwise at 246/24 changes. Classifier #1's churn is FLAT in k (242–249 of 587 pairs), so
the terminal-month edge artefact is refuted as the explanation for its 41.91% filtered churn.**

## Tasks

| task | status | commit |
|---|---|---|
| 1 — tracer: harness + anchor + 33 tests (3 failure fixtures, 4 anchor-failure fixtures, artifact tests, no-monotonicity guard) | complete | `c01cfc8` |
| 2 — full 588-step run; label matrix + churn JSON committed | complete | `214329f` |
| 3 — `08-TRACK-A.md` | complete | `b79f210` |

## The anchor

The harness's step index equals `joint_lift_joint_l1only.parquet`'s index **elementwise** (588
rows, 1972-01-31 → 2020-12-31). `c1_lag1_state` matches `state_1` with **0 of 588 mismatched**,
and `c2_lag1_state` matches `state_2` with **0 of 588 mismatched**. The derived k=1 churn is
exactly **246** and **24**. No tolerance was applied. A separate re-check outside the harness
confirmed every `c{N}_lag{k}_date == t − k months` at all 588 steps.

## The churn-vs-k table

The window is 588 steps, 1972-01-31 → 2020-12-31. Rates are `n_changes / n_pairs` with
n_pairs = 587. Degraded steps: **0 of 588** for each classifier. Date misalignments
(`iloc[-1]` date ≠ `train_index[-1]`): **0 of 588**.

| k | classifier #1 (K=6, λ=10.0, d=10, **λ/d=1.0**) | classifier #2 (K=5, λ=16.0, d=8, **λ/d=2.0**) |
|---|---|---|
| 1 | **246 / 587 = 41.91%** | **24 / 587 = 4.09%** |
| 2 | 242 / 587 = 41.23% | 24 / 587 = 4.09% |
| 3 | 245 / 587 = 41.74% | 25 / 587 = 4.26% |
| 4 | 248 / 587 = 42.25% | 24 / 587 = 4.09% |
| 5 | 247 / 587 = 42.08% | 24 / 587 = 4.09% |
| 6 | 249 / 587 = 42.42% | 24 / 587 = 4.09% |

The secondary panel re-reads the same matrix by calendar month: how often month *m*'s lag-1
label is revised by a later fit at lag k.

| k | classifier #1 | classifier #2 |
|---|---|---|
| 2 | 237 / 587 = 40.37% | 24 / 587 = 4.09% |
| 3 | 263 / 586 = 44.88% | 29 / 586 = 4.95% |
| 4 | 273 / 585 = 46.67% | 30 / 585 = 5.13% |
| 5 | 280 / 584 = 47.95% | 42 / 584 = 7.19% |
| 6 | 298 / 583 = 51.11% | 44 / 583 = 7.55% |

## Verdict (as written in 08-TRACK-A.md)

> **Flat in k. The terminal-month edge artefact is refuted for classifier #1:** reading the
> 6th-from-last month instead of the last gives 249 changes against 246 over the same 587
> pairs, and no lag from 2 to 6 falls below 242.

The record states both readings before the verdict. It also records two limits on what the
run tested:

1. **Trailing-block effect not excluded.** A trailing block of j months costs a single λ, so a
   longer block-edge effect cannot be excluded by k ≤ 6.
2. **λ/d not isolated.** The #1/#2 contrast also differs in K, feature set and d. Only a λ
   sweep could isolate λ/d, and a sweep is not authorized.

**Registry: 42 trials before, 42 after.** No trial was consumed, λ was not swept, and
(K, λ) is unchanged for both classifiers.

## Deviations from Plan

1. **[Measurement vs. plan estimate] Runtime ~33 min, not "two minutes and change".** The plan
   assumed ~0.11 s per refit. The measured rate was ~0.55 s/step early (40-step smoke: 22 s)
   and higher late, because the expanding window grows T from 120 to 707. Recorded as measured.
   Nothing was changed.
2. **[Sequencing] Task 2's artifact-level tests were written and committed in Task 1
   (`c01cfc8`)** with the synthetic tests. They skipped cleanly until the artifacts existed and
   pass with 0 skipped since `214329f`. Task 2's commit therefore contains only the two
   artifacts.
3. **[Interruption] The Task 2 run finished (artifacts written 2026-09-22 16:22) before a
   rate-limit interruption, and was not re-run.** Before committing I checked it independently:
   - 588 × 25, no NaN cells, `full_window_run: true`;
   - index and k=1 elementwise against the tracked curve, re-checked outside the harness;
   - every lag date equals *t* − k months;
   - the only sibling commits touching `labeling/` since the harness commit add a new
     `stability.py` and leave `standardize_features`, `_refit_l1` and `_refit_classifier2`
     unchanged;
   - the sibling's uncommitted `run_joint_lift.py` edit leaves `build_inputs` unchanged.
4. **[Mutation check, added] The artifact guard can fail.** I altered one `n_changes` in the
   JSON in place and ran the tests: 2 artifact tests failed. The original was then restored
   byte-identical (sha256 checked).
5. **[Anchor strengthening, added] `assert_anchor` also runs on `--limit-steps` smoke runs.**
   It compares the collected prefix elementwise and skips the full-window 246/24 count loudly.
   The 40-step smoke run matched **40 of 40** cells for both classifiers.
6. **[Coordination] STATE.md and ROADMAP.md not updated by this agent.** Sibling executors
   (08-03, 08-04, 08-05) commit only their SUMMARY, and four agents share this working tree.
   STATE and ROADMAP updates are left to the orchestrator to avoid concurrent writes to shared
   files. PER-04 is ready to be marked complete.

## Verification results

| check | result |
|---|---|
| `pytest tests/unit/test_platform_terminal_month_diagnostic.py -q` | **33 passed, 0 skipped** |
| smoke: `--limit-steps 40` | completed; prefix anchor 40/40 for both classifiers |
| `total_trial_count() == 42` | **42** (before Task 1, after Task 2, at plan end) |
| Task 2 verify (c1 anchor, 246, λ/d 1.0) | pass — rates `[(1,0.4191),(2,0.4123),(3,0.4174),(4,0.4225),(5,0.4208),(6,0.4242)]` |
| Task 2 verify (c2 24, λ/d 2.0) | pass — `c2 anchored` |
| Task 3 verify (section strings) | pass |
| Task 3 verify (every count appears) | pass. A stronger added check also passes: every `N / D = R%` cell matches the JSON exactly, and both readings come before the verdict |
| legacy-import ratchet | **11 passed** (ratchet 31, unchanged; this plan adds no legacy import) |
| `ruff check` + `flake8 --select=E9,F63,F7,F82` on both Python files | clean |
| `pytest tests/ -q` | **2132 passed, 14 failed.** All 14 are in `tests/unit/test_platform_joint_diagnostics_record.py`, which has **uncommitted sibling edits** (` M` in `git status`, not this plan's file). Every failure is a `KeyError` on `track` / `series_identity` / `walk_forward_nowcast`: the sibling's edited tests expect a regenerated `diagnostics_*.json` that is not yet on disk. None involves this plan's files. |

## Deferred Issues (out of scope, not fixed)

- `tests/unit/test_platform_joint_diagnostics_record.py`: 14 failures from a sibling's
  in-flight uncommitted work (see above). This belongs to 08-01's wave. The orchestrator should
  confirm the sibling commits the regenerated diagnostics JSON before merge.

## Known Stubs

None. `state_change_count` is an intentional duplicate of 08-01's `churn.py` rule, placed here
so the two same-wave plans don't depend on each other. Its docstring names 08-01 as the
consolidation point.

## Threat Flags

None. The script reads config and tracked checkpoints only. It writes only under `--out-dir`
and never touches the registry.

## Self-Check: PASSED

- FOUND: scripts/terminal_month_diagnostic.py
- FOUND: tests/unit/test_platform_terminal_month_diagnostic.py
- FOUND: outputs/reports/platform/track_a/terminal_month_labels.parquet
- FOUND: outputs/reports/platform/track_a/terminal_month_churn.json
- FOUND: .planning/phases/08-regime-persistence-stability/08-TRACK-A.md
- FOUND commits: c01cfc8, 214329f, b79f210
