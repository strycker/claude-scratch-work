---
phase: 08-regime-persistence-stability
plan: 04
subsystem: tests/platform-allocation
tags: [G6, adr-0001-condition-iv, unpooled-consumers, both-halves-rule, tamper-evidence]
status: complete
requires:
  - "ADR-0001 § RE-PIN 2026-09-18 condition (iv) — the partial-pooling requirement"
  - "07-10 (pool_low_n_regime_sharpe, low_n_regime_flags, blend_regime_tilts)"
provides:
  - "tests/unit/test_platform_pooling_consumers.py — the G6 pin, three consumers, one file"
  - ".planning/phases/08-regime-persistence-stability/08-G6.md — the G6 record"
affects:
  - "08-09 (edits driver.py + weekly.py; must find this test green UNMODIFIED)"
  - "08-10 (consumes 08-G6.md when restating the phase's measurements)"
key-files:
  created:
    - tests/unit/test_platform_pooling_consumers.py
    - .planning/phases/08-regime-persistence-stability/08-G6.md
  modified: []
decisions:
  - "G6 is written as a pin of the KNOWN non-compliance, not as a compliance assertion — the latter is a test that can only pass."
  - "The sub-floor regime in the contrast fixture must carry TWO positive-Sharpe assets, or pooling moves the Sharpes but not the weights and the arm is a tautology."
metrics:
  duration: ~35m
  completed: 2026-09-23
actuals:
  tokens: 7220
  tasks: 2
  commits: 2
---

# Phase 08 Plan 04: G6 — the unpooled-consumer pin

Three consumers of `vol_targeted_tilt` pinned in one test file: `driver.py:497` and
`report/weekly.py:249` receive the **unpooled** per-regime Sharpe table (a known
non-compliance with ADR-0001 condition (iv), pinned as such), while
`allocation/joint_tilt.py:319` pools and produces **measurably different portfolio weights**
on the same fixture.

## Tasks

| task | status | commit |
|---|---|---|
| 1 — driver arm (tracer) | complete | `2ea3436` |
| 2 — weekly arm, contrast arm, `08-G6.md` | complete | `8013d7d` |

## The three arms

| arm | test | pins |
|---|---|---|
| driver | `TestDriverConsumer::test_driver_tilt_call_receives_the_unpooled_returns_by_regime` | the frame `driver.py:497` hands the allocator is `returns_by_regime_stats`' raw output |
| weekly | `TestWeeklyConsumer::test_weekly_tilt_call_receives_the_unpooled_returns_by_regime` | the same, in the path Glenn reads every week |
| contrast | `TestJointTiltContrast::test_pooled_joint_tilt_produces_different_weights_than_the_unpooled_path` | `joint_tilt` pools, and the two paths produce **different weights** |
| summary | `TestG6Summary` (2 tests) | `G6_CONSUMER_TABLE` resolves to real arms; 3 distinct sites; 2 unpooled / 1 pooled |

`5 passed in 3.86s`, 0 skipped.

## The sub-floor fixture and the cells pooling moves

| arm | sub-floor regime | occupancy | credibility | cell | unpooled → pooled |
|---|---|---|---|---|---|
| driver | 2 | 3/51 = **5.8824 %** | 0.7353 | `(2, SPY)` | −4.984659 → −3.764478 (Δ −1.220180) |
| | | | | `(2, TLT)` | +15.014234 → +11.376312 (Δ +3.637922) |
| weekly / contrast | 2 | 14/234 = **5.9829 %** | 0.7479 | `(2, SPY)` | +0.043301 → +0.214833 |
| | | | | `(2, TLT)` | +3.464102 → +2.693807 |

Contrast arm's weight movement: **SPY 0.657673 → 0.669976, TLT 0.342327 → 0.330024** — a
1.2303-percentage-point shift in the portfolio, not just in a statistic.

## The both-halves rule held, and it earned its keep twice

Every arm asserts `low_n_regime_flags(unpooled)["low_n"].any()` **before** the pin, then asserts
equality with the unpooled table **and** inequality with
`pool_low_n_regime_sharpe`'s output, naming the differing cell. All three mutation-checked:

| mutation | result |
|---|---|
| feed the **pooled** frame to the pin (a consumer that started pooling) | **fails** half 1 ✓ |
| flat-occupancy fixture, no sub-floor regime | **fails** the precondition ✓ — and pooling is confirmed a strict no-op on it, which is exactly why half 1 alone would have passed |
| force `joint_tilt` not to pool (all-above-floor `occupancy_1`/`occupancy_2`) | contrast deltas become `{SPY: 0.0, TLT: 0.0}` → **fails** ✓ |

## Deviations from Plan

### 1. [Rule 1 — Bug in the plan's implied fixture] The contrast arm's first fixture made pooling a no-op **on the weights**

- **Found during:** Task 2, on the first run of the contrast arm (it failed, correctly).
- **Issue:** The sub-floor regime had one negative-Sharpe asset (SPY) and one positive (TLT).
  `_per_regime_tilt` clips negative Sharpes to zero and normalizes within the regime, so a single
  survivor normalizes to 1.0 whether or not its Sharpe was pooled. Pooling changed the Sharpe
  column (half 2 passed) but the **portfolio weights were byte-identical**: `{SPY: 0.0, TLT: 0.0}`.
  The contrast arm would have been exactly the defect the plan set out to avoid, one level down —
  a fixture on which the property under test is not observable.
- **Fix:** Rebuilt the fixture so the sub-floor regime carries **two positive-Sharpe assets**
  (`(2, SPY)` +0.0433, `(2, TLT)` +3.4641) whose ratio pooling changes. Weights now move by
  1.2303 pp. The reason is recorded in a comment at the fixture and in `08-G6.md` § 2, so the next
  person who "simplifies" the fixture learns why they cannot.
- **Files modified:** `tests/unit/test_platform_pooling_consumers.py` (fixture only).
- **Commit:** `8013d7d`.

### 2. [Not met] The plan's runtime target of "under a second"

The file runs in ~3.9 s unloaded. Almost all of it is import and pytest collection overhead for
`trading_crab_lib.platform` (the driver arm itself runs 12 walk-forward steps with faked L1/L2
refits). Nothing was trimmed to chase the target: the synthetic backtest is already minimal, and
shrinking it further would push the sub-floor regime's occupancy over the 8% floor in early
windows. Recorded, not fixed.

### 3. Nothing else

Otherwise the plan's structure was followed exactly. No `pool_low_n_regime_sharpe` call was added to
`driver.py` or `weekly.py`; no file under `src/` was touched.

## Out of scope, recorded as an exclusion not an omission

Condition (iv)'s **covariance clause** is not pinned. No per-regime covariance exists at L4-01 —
`tilt.py::portfolio_vol` is a linear-sum/EWMA estimate, marked `ponytail` in its own source — and
the clause falls to L3 (design §6.2, regime-conditional Ledoit–Wolf). `08-G6.md` § 3 states this
explicitly. The *correctness* of the non-compliance is likewise not pinned: that is an open design
question, and a test cannot settle it.

## What a red result means

From `08-G6.md` § 4, verbatim:

> **Someone made a consumer compliant. Move the pin deliberately — do not delete it, and do not
> "fix" the test.**

Because pooling a sub-floor regime's Sharpe changes the tilt, therefore the allocation weights,
therefore ROADMAP criterion 7's measured lift. Fixing G6 is not authorised by this phase.

## Tamper-evidence for 08-09

`tests/unit/test_platform_pooling_consumers.py` is deliberately **absent from 08-09's
`files_modified`**, while 08-09 edits `driver.py` and `report/weekly.py`. 08-09 must find this file
green **unmodified**. If wiring §5.3's hysteresis turns an arm red, that is a fact about 08-09's
change reaching the allocator's `returns_by_regime` argument — not a stale test.

## Verification

| check | result |
|---|---|
| `pytest tests/unit/test_platform_pooling_consumers.py -q` | **5 passed, 0 skipped**; 3.86 s unloaded, 21 s under four concurrent sibling agents |
| both-halves re-confirmation (2026-09-23, current tree) | driver + weekly fixtures: pooling **not** a no-op (regime 2 low-n in each); unpooled frame passes both halves; pooled frame **fails**; no-sub-floor fixture fails the precondition; contrast deltas real `{SPY: +0.012303, TLT: −0.012303}` vs non-pooling mutant `{0.0, 0.0}` |
| Task 1 structural verify (`pool_low_n_regime_sharpe` present, ≥4 asserts) | `both-halves structure present` |
| Task 2 `08-G6.md` string verify (`driver.py`, `weekly.py`, `joint_tilt.py`, `covariance clause`, `strict no-op`) | `ok` |
| `git show --name-only 2ea3436 8013d7d` | only `tests/unit/test_platform_pooling_consumers.py` and `08-G6.md` — **no file under `src/`** (a range diff to HEAD is not usable: sibling 08-03 commits interleave and touch `src/`) |
| `pytest tests/unit/test_platform_legacy_import_ratchet.py -q` | 11 passed — ratchet still 31 |
| `total_trial_count()` | **42**, unchanged |
| `ruff check` / `flake8 --select=E9,F63,F7,F82` | clean |
| full suite `pytest tests/ -q` | **2132 passed, 14 failed, 0 skipped** (2026-09-23, shared tree). All 14 failures are in `test_platform_joint_diagnostics_record.py::TestTheTwoChurnSeriesAreSeparatelyDenominated`, a file that was **uncommitted and mid-edit by a sibling plan** (08-01/02/08/10 own it) together with `scripts/run_joint_lift.py` and `scripts/joint_lift_diagnostics.py`. It does not reference any 08-04 artifact, and 08-04 changed no file under `src/` or `scripts/`, so it cannot reach it. The one failure in an earlier run (`test_platform_labeling_stability.py`, sibling 08-03 mid-work) has since cleared. The 2018 baseline cannot be compared directly while four plans are adding tests at once |
| holdout | synthetic-only; no checkpoint, network or 2021+ read anywhere in the file |

## Human-check (Task 2)

Self-verified against the plan's three questions: `08-G6.md` § 1's table names all three consumers
**with the test that establishes each**; § 4 tells a future reader to **move** the pin, not delete
it, and says why (criterion 7); § 3 lists the covariance clause as an **explicit exclusion** with
its reason (no per-regime covariance at L4-01, falls to L3 per design §6.2).
