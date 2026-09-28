---
phase: 08-regime-persistence-stability
plan: 10
subsystem: platform/backtest + scripts/run_joint_lift + docs
tags: [criterion-6, criterion-9, PER-07, PER-10, criterion-7, A11, dsr-gate, F-4, registry-ceiling, no-trade-band, honesty-framework]
status: complete
requires:
  - "08-05 (08-A11.md: b-promote-dsr; registry rows spent 0)"
  - "08-09 (08-A7.md: b-bounded-turnover, 5pp band; registry rows spent 2; band-off l1only byte-identical)"
  - "08-01 (churn.churn_rate — F-4's one rule); 08-08 (the filtered l2 leg)"
provides:
  - "joint_driver.quality_tier / annualized_sharpe / QUALITY_TIER_RULE; joint_lift_table reports the A11 gate per leg"
  - "run_joint_lift: top-level quality_tier verdict (governs only a decision-bearing run); no_trade_band and use_regime_filter carried in the record; F-4's second site fixed"
  - "measurement_l1only.json + curves: criterion 7 under the 5pp band, registry 42 -> 44"
  - "measurement_l2_observational.json + curves: filter + band, observational"
  - "tests/unit/test_docs_recorded_counts.py — four doc sites pinned to a live collection by equality"
  - "08-MEASUREMENTS.md — the phase's closing record"
affects:
  - "Registry at the ADR-0002 ceiling (44/44). Any further evaluated configuration needs an ADR-0002 amendment first."
  - "Growing the suite now turns test_docs_recorded_counts red until CLAUDE.md x2 and README.md x2 are updated in the same commit."
decisions:
  - "Decision-bearing branch: registered EVALUATION at exactly 2 rows (08-A7.md b-bounded-turnover is not inert on l1only), preceded by a band-OFF NO_REGISTRY control that reproduced git byte-for-byte."
  - "The A11 hurdle is computed at the count read AFTER the run's own rows land (44 for l1only), never a literal; the gate governs decision-bearing legs only."
  - "A leg with no defined Sharpe does not pass the gate (reported defined=False)."
  - "Decision-bearing trial tags: 08-10-c1-alone-L1only-notrade5pp / 08-10-joint-c1xc2-L1only-notrade5pp."
metrics:
  duration: "about 55 min of wall-clock, 2026-09-28T14:00Z to 14:56Z; three harness runs (control 21 min and l2 25 min in parallel, then l1 decision-bearing 7.6 min alone)"
  completed: 2026-09-28
actuals:
  tokens: 19000   # chars/4 over added lines of the non-parquet diff 66e3b7a..HEAD (67,377 chars) plus this summary
  tasks: 3
  commits: 6
---

# Phase 8 Plan 10: criterion 7 re-measured under the 5pp band — registry 42 → 44, the A11 gate fails both legs

**One-liner.** The A11 deflated-Sharpe gate landed in `joint_lift_table` first. Next came a
band-off control, which reproduced git byte-for-byte. Only then was the decision-bearing leg run,
once, with the band on. It moved `wealth_delta` from −0.123438 to **−0.125307** and `dd_delta`
from +0.024084 to **+0.026164**, over 588 steps 1972-01-31 → 2020-12-31. The registry went
**42 → 44**, and the gate **FAILED** both legs against a hurdle of 2.226891.

## Which branch, and why

- **Evaluation branch, 2 registry rows.** Glenn's 08-09 ruling (`b-bounded-turnover`, a 5pp
  no-trade band) bounds `w_t − w_{t−1}` whatever the probability vector looks like. It is
  therefore **not inert** on the l1only leg: `08-A7.md` §4 says so, and authorises 2 rows.
- **The reproduction branch did not apply.** Its equivalent was run as a **control**: the same
  source with the band forced off (`NO_REGISTRY`). That run makes any movement attributable to
  the band alone.

| step | what | result |
|---|---|---|
| 1 | A11 code consequence (`bbd88e2`) | `quality_tier` / `annualized_sharpe` in `joint_driver.py`; the gate fields in `joint_lift_table`; the record's `quality_tier` block. On the tracked pre-08-10 curves at n = 42 it reproduces the recorded DSRs **exactly** (2.28151091802503e-12, 1.4690427074211624e-11) |
| 1b | trial tags (`59ee219`) | `08-10-…-notrade5pp` |
| 2 | control: l1, band OFF, `NO_REGISTRY`, final source | `joint_lift_{joint,baseline}_l1only` and `joint_lift_probs_{1,2}_l1only` are **`cmp`-identical** to `git show 66e3b7a:`, and `assert_frame_equal(check_exact=True)` passes. `wealth_delta` −0.12343826162064975 and `dd_delta` +0.02408401236666291 match **exactly**. Registry 42 → 42 |
| 3 | observational l2 (filter + band), `NO_REGISTRY` | 588 steps, 100 degraded. Probability and belief matrices are **byte-identical** to git (max diff 0.0) and were not rewritten. `state_*` and `degraded` are identical; the curves changed and were regenerated. Registry 42 → 42 |
| 4 | **decision-bearing l1, band ON — run exactly once** (14:37:09 → 14:44:48 UTC, exit 0) | **2 rows appended, 42 → 44.** Checked from outside: `total_trial_count()` read 42 before and 44 after; the ledger went from 5 to 7 lines; the diff is `+2 −0`. First differing cell against git: **1972-03-31, `return`**. Only `return`, `turnover`, `cost` and `scale` moved; `state_1`, `state_2`, `active_regime` and `degraded` are identical |

## Criterion 7 — both legs, with windows

| | prior (07-11, band off, 42 trials) | **08-10, band on** | window |
|---|---|---|---|
| l1only `wealth_delta` | −0.123438 | **−0.125307** (−0.1253065774082902) | 588 steps, 1972-01-31 → 2020-12-31, 0 degraded |
| l1only `dd_delta` | +0.024084 | **+0.026164** (+0.026164042149829925) | same |
| l1only mean turnover, #1-alone / joint | 0.163251 / 0.119788 | **0.127412 / 0.080972** | 588 / 588 steps |
| l1only months with zero turnover, #1-alone / joint | 0 / 0 of 588 | **304 / 341 of 588** | same |
| l1only terminal log wealth, #1-alone / joint | 4.718723 / 4.595285 | **4.606331 / 4.481024** | same |
| l1only max drawdown, #1-alone / joint | −0.281442 / −0.257358 | **−0.293991 / −0.267826** | same |
| l2 `wealth_delta` / `dd_delta` (observational) | −0.134505 / −0.001137 (pre-filter, pre-band) | **−0.300608 / +0.061011** | 588 steps, same dates, **100 degraded** |
| l2 mean turnover, #1-alone / joint (observational) | 0.098907 / 0.060451 (pre-filter) | **0.141119 / 0.068775** | 588 / 588 steps |

The prior values are a comparison point, not a target. Read as measured:
- turnover and cost fell on both legs;
- the lift did not improve;
- both legs ended with lower terminal wealth and deeper drawdowns.

τ was not swept.

## The A11 gate (ADR-0003)

| leg | Sharpe | DSR | hurdle | verdict |
|---|---|---|---|---|
| l1only #1-alone | 0.896446 | **1.8159919159760753e-13** | `expected_max_sharpe(44, 1.0)` = **2.2268911497604993** | **FAILED** |
| l1only joint | 0.899378 | **4.1314825526463704e-12** | 2.226891 | **FAILED** |
| l2 #1-alone / joint | 1.263438 / 1.157769 | 6.52e-35 / 1.02e-45 | 2.208694 (read at 42, before the l1 rows) | NOT GOVERNING |

- The record's verdict: **"FAILED on 2 of 2 legs (baseline, joint)"**. Glenn accepted this in
  advance (`08-A11.md` §3.4).
- `sharpe_variance` is the **1.0 placeholder**.

## Registry

| read | count |
|---|---|
| before any 08-10 run | **42** |
| after the control and after l2 | 42, 42 |
| after the decision-bearing run | **44** |
| plan verify: `42 + spent(08-A11) 0 + spent(08-A7) 2` | **44**, which matches |
| headroom against the ADR-0002 ceiling of 44 | **0** |

The ledger commit is `4bfb81b`, the same commit as `measurement_l1only.json`.

## F-4's second site

`_leg_kpis` now routes `state_{1,2}_transition_rate` through `churn.churn_rate`
(`_n_transitions` is kept as a delegate to `churn.state_change_count`).
- Both records now say **246/587 = 0.419080** and **24/587 = 0.040886**, on both legs. The
  plan's verify accepts ÷587 and rejects ÷588.
- The measurement records are now equal to the diagnostics records, field for field (tested).
- The source fix, the records, the curves and the ledger are all in **one commit**
  (`4bfb81b`).

## Recorded test counts (Task 2)

- The live `pytest --collect-only` count is **2392**, measured after every other test was in.
- Four sites were updated from 1705 to 2392: `CLAUDE.md` (layout tree, current status) and
  `README.md` (badge **URL**, feature list).
- The pin asserts equality. A synthetic 1705 document fails and names the site, the stale value
  and the live one. An unmatched site is a failure, not a skip. A mutated 2391 badge turns
  exactly the badge arm red.

## Tasks and commits

| task | commit | what |
|---|---|---|
| 1, A11 gate (before any run) | `bbd88e2` | the quality tier in `joint_lift_table` and the record, plus 9 test arms |
| 1, tags (before any run) | `59ee219` | the decision-bearing trial pair renamed for the banded configuration |
| 1, measurement | **`4bfb81b`** | F-4 source fix, both measurement JSONs, all four l1only/l2 curves, `registry/trials.jsonl` (+2) and 26 record tests, in one commit |
| 2 | `b012187` | `test_docs_recorded_counts.py`, CLAUDE.md ×2, README.md ×2 |
| 3 | `c0a895d` | `08-MEASUREMENTS.md` |
| summary | this commit | |

## Verification

| check | result |
|---|---|
| plan verify: F-4 ÷587, not ÷588, both legs of both records | pass |
| plan verify: `n_steps_joint == n_steps_baseline == 588`, both records | pass |
| plan verify: registry `== 42 + spent` | 44 = 42 + 2 |
| plan verify: four doc sites agree, none 1705 | 2392 at all four |
| plan verify: 08-MEASUREMENTS tokens and ≥ 10 verdicts | ok, 14 verdicts |
| full suite, plan gate (`set -o pipefail` plus the no-skip/no-xfail grep) | **2392 passed, 0 failed, 0 skipped, 0 xfailed**, rc 0. The 5 warnings are the pre-existing seaborn deprecations |
| legacy-import ratchet | green, **31** |
| G6 pin `test_platform_pooling_consumers.py` | **unmodified** (last commit `94d5283`), passes |
| `git diff --stat 66e3b7a HEAD -- legacy/ gsd-scratch-work/ trading-crab-lib/` | empty |
| ruff and flake8 (E9,F63,F7,F82) | clean on every touched .py, before each commit |
| 2021+ holdout | not read; every window ends 2020-12-31 |

## Deviations from Plan

1. **[Rule 2] The records carry the configuration measured.** `no_trade_band` and
   `use_regime_filter` were added to the record. They are needed: without them a band-on record
   is indistinguishable from the pre-band one.
   - The first l2 run started before this edit. It was killed about 2 minutes in (it was
     `NO_REGISTRY` and had written nothing) and restarted.
   - The control's record, which is not committed, lacks the two fields.
2. **[Plan branch] The plan's 1e-12 reproduction test arm does not apply to the evaluation
   branch.** Its purpose is served in two ways:
   - by the control run (exact reproduction, stated above);
   - by 08-09's standing band-off sha256 unit pin.

   The record tests instead re-derive the lift and the gate from the committed curves, and read
   the tracked ledger for the two claimed rows.
3. **[Placement] The A11 gate tests live in `test_platform_joint_diagnostics_record.py`.** The
   plan lists that file; `test_platform_backtest_joint_driver.py` is not in `files_modified`.
4. **[Provenance, stated] The ledger rows carry `git_sha` `59ee219`.** The F-4 fix and the two
   record fields had to share the records' commit, so they were uncommitted working-tree edits
   when the run executed.
   - Those edits touch only `run_joint_lift.py`'s record fields.
   - They do not touch `joint_driver.py`, the backtest, or the rows' `config` and `metrics`,
     which are identical at `59ee219` and at `4bfb81b`.
5. **[Order] l2 ran before the decision-bearing leg, as the plan's ladder orders.** Its DSR is
   therefore read at 42 and the l1only DSR at 44. Both are recorded with their read timestamps.
6. **[Scope] All runs dumped to scratch; only in-scope files were copied in.** The l1only and
   l2 probability and belief matrices re-produced byte-identical and were not rewritten.
7. **[TDD] RED was not committed separately.** The A11 gate and its tests went in one commit.
   Their ability to fail was shown by mutation (`>` → `>=` turns the boundary arm red), and
   for the doc pin by a mutated badge.
8. **[Finding, not fixed] "0.418980" is a transcription error for 246/587 = 0.419080.** It
   appears at six sites: `08-01-SUMMARY`, `08-01-PLAN`, `08-10-PLAN`, `STATE.md`,
   `churn.py:93` and `test_platform_evaluation_churn.py:101`. All are out of scope, and the
   records were always right. This plan's own code comment uses 0.419080.
9. **[Not changed] CLAUDE.md's "(10 skipped: HDBSCAN + cssselect optional)"** beside the count
   is stale for this environment, which has 0 skipped. It is outside the four pinned sites and
   is left for a docs pass (recorded in 08-MEASUREMENTS §7).

## TDD Gate Compliance

- **RED was not committed separately** (see Deviation 7).
- **GREEN** is `bbd88e2` and `b012187`.
- The suite never went red in any commit.

## STATE-ready open items

`08-MEASUREMENTS.md` §7 carries the full, STATE-ready block. The headline items:
- **the registry is at 44 / 44**, so any new evaluated configuration needs an ADR-0002
  amendment first;
- **criterion 7 FAILED the A11 gate on both decision-bearing legs**, on the placeholder
  variance;
- the Phase 7 criterion-6 dependence verdict is still UNRESOLVED (`298b1bc`, no tie-break);
- G6's non-compliance is pinned and not fixed, and the covariance clause is unimplemented;
- AMENDMENT condition (i) is satisfied;
- a λ ruling is owed if Track A is ever to move;
- the vocabulary finding (35.7% / 41.3%) needs a ruling;
- the evaporated-flag and `run_stability` keying defects;
- the "0.418980" transcription error at six sites.

## Known Stubs

None.

## Threat Flags

None. The two registry rows were the planned, authorised spend.

## Self-Check: PASSED

- FOUND: `08-MEASUREMENTS.md`, `tests/unit/test_docs_recorded_counts.py`, both measurement
  JSONs, all four curves, and `registry/trials.jsonl` at 7 lines (44 trials).
- FOUND commits on `claude/keen-galileo-zqcml6-w5`: `bbd88e2`, `59ee219`, `4bfb81b`,
  `b012187`, `c0a895d`. Not pushed.
- STATE.md, ROADMAP.md and REQUIREMENTS.md were not touched (orchestrator instruction).
