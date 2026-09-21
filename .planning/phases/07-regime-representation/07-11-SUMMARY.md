---
phase: 07-regime-representation
plan: 11
subsystem: platform/backtest
tags: [criterion-7, joint-lift, adr-0002, deflated-sharpe, trial-registry, honesty-framework]
status: complete
requires:
  - "07-06 (deflated_sharpe.py, registry.total_trial_count)"
  - "07-08 (classifier2_config, freeze_classifier2_columns, ADR-0002 (e)/(f))"
  - "07-09 (criterion 6's verdict — UNRESOLVED)"
  - "07-10 (blend_regime_tilts; 07-BANDS.md §8's four confirmed dispositions)"
provides:
  - "platform/backtest/joint_driver.py — run_joint_backtest, JointStepRecord, joint_lift_table"
  - "scripts/run_joint_lift.py, scripts/joint_lift_diagnostics.py"
  - "07-JOINT-LIFT.md — criterion 7 measured"
  - "outputs/reports/platform/joint_lift/ — both legs' curves and measurement records"
  - "two append-only registry rows: 07-11-c1-alone-L1only, 07-11-joint-c1xc2-L1only"
affects:
  - "07-12 (ADR-0002 acceptance quotes 07-JOINT-LIFT.md)"
decisions:
  - "Degradation is symmetric: a step degrades if EITHER classifier fails, so the two legs share a degraded-step SET, not merely an index."
  - "No `sharpe` key written into registry metrics — two near-identical legs would collapse registry_sharpe_variance to ~1e-6 and silently disable the multiple-testing correction project-wide."
  - "Classifier #2 gets a local refit entry point (a sort_column seam), pinned by a parity test against _refit_l1, because driver.py must stay unmodified."
tech-stack:
  added: []
  patterns: ["one-parameter ablation from one harness", "window returned in the same mapping as the delta"]
key-files:
  created:
    - src/trading_crab_lib/platform/backtest/joint_driver.py
    - tests/unit/test_platform_backtest_joint_driver.py
    - scripts/run_joint_lift.py
    - scripts/joint_lift_diagnostics.py
    - .planning/phases/07-regime-representation/07-JOINT-LIFT.md
  modified:
    - registry/trials.jsonl
metrics:
  duration: "~1h05m"
  completed: 2026-09-21
actuals:
  tokens: 29015
  tasks: 3
  commits: 5
---

# 07-11 — Criterion 7: joint allocation lift, measured

## Tasks

| task | status | commits |
|---|---|---|
| 1 — `run_joint_backtest` + `joint_lift_table` + 28 tests | complete | `4f3de0e` (RED), `7410cab` (GREEN) |
| 2 — live runs, registry read live, DSR both legs | complete | `d5c3ac9`, `42161dd` |
| 3 — `07-JOINT-LIFT.md` | complete | `1ca7173` |

## The routing

**`L1_ONLY_LAST_FILTERED_STATE`, decision-bearing** (ADR-0002 (e)). Each classifier's
per-step probability vector is a degenerate one-hot on its own last filtered state, fit
on `train_index` only. The L2 leg was computed and reported under the `NO_REGISTRY`
sentinel — observational, firewalled, zero rows, nothing downstream changed on it.

## The measured numbers — every one with its window

**Decision-bearing legs, both over 588 steps, 1972-01-31 → 2020-12-31, 0 degraded:**

| | #1 alone (`bw1=1.00`) | joint (`bw1=0.50`) |
|---|---|---|
| terminal log wealth | 4.718723 | 4.595285 |
| max drawdown / underwater | −28.1442% / 47 mo | −25.7358% / 40 mo |
| annualized Sharpe (588 obs) | 0.917073 | 0.914903 |
| mean turnover / total cost | 0.163251 / 0.095992 | 0.119788 / 0.070435 |

**Lift:** `wealth_delta` **−0.123438** and `dd_delta` **+0.024084**, both over **588
steps, 1972-01-31 → 2020-12-31**. Indexes element-wise equal; 586 of 588 monthly
returns differ, so the blend took effect.

**Observational L2 legs (588 steps, 1972-01-31 → 2020-12-31, 100 degraded each —
87 classifier #1's nowcaster, 13 classifier #2's):** `wealth_delta` **−0.134505**,
`dd_delta` **−0.001137**. The routings agree in sign on wealth and **disagree in sign
on drawdown** (ADR-0002's named failure signature, half-fired, recorded not resolved).

## Deflated Sharpe — both legs

`sharpe_variance` = 1.0 (the declared placeholder; registry holds 0 Sharpe-bearing
rows), `n_trials` = **42** read live 2026-09-21T14:24:23.984933Z,
`expected_max_sharpe(42, 1.0) = 2.208694`.

| leg | DSR | verdict |
|---|---|---|
| #1 alone | 2.28151 × 10⁻¹² | "…does not clear the multiple-testing hurdle…" |
| joint | 1.46904 × 10⁻¹¹ | "…does not clear the multiple-testing hurdle…" |

**Neither clears it.** The joint leg's raw Sharpe is also lower than the baseline's, so
the blend did not help on this axis before deflation either.

## Bands

Governing (universal) tier: `abs(wealth_delta) < 15` — **held**; `dd_delta ∈ [−1, 1]`
— **held**. Advisory (domain) triggers (`≥ 5`, `≥ 0.5`) — **neither fired**, so no note
recorded. Band 3a (`n_resolved ≤ n_transitions ≤ n_label_transitions`) held for both
classifiers; band 3b's rate band held (3.597% and 1.724% against 10%); band 4 not
suspicious on any clause (`n_compared` = 588 of 588 expected for both).

## Trial arithmetic

`total_trial_count()` **40** (2026-09-21T14:18:34.285851Z) → **42**
(2026-09-21T14:24:23.984933Z). **2 rows added**, one per tagged run. ADR-0002's ceiling
is **44**; respected with 2 rows unspent. **6 sentinel `run_joint_backtest`
invocations** (one two-leg dry run, two two-leg L2 runs) added 0 rows.

## Suspicion signatures checked

1. **Implausible terminal log wealth** — in-script `abs(x) < 10` assertion ran on both
   legs; measured 4.72 / 4.60 (≈9.9% / 9.6% annualized over 49 years). Plausible.
2. **Differing leg indexes** — `index.equals` True, 588 = 588, degraded 0 = 0, deltas
   independently recomputed from the persisted parquet to the last digit.
3. **Transition count** — full-sample rates 3.597% / 1.724%, inside band 3b. Surfaced a
   finding the bands do not govern: classifier #1's *filtered* labeling changes state in
   **246 of 588 (41.84%)** decision months.
4. **Lift exactly zero** — `−0.123438`; 586 of 588 months differ; pinned by two tests.
5. **DSR exactly 0.0 or 1.0** — raw floats 2.28e-12 and 1.47e-11, strictly positive,
   distinct, denominator guard did not fire. Running on the declared 1.0 placeholder.

Two more the data raised: **oil's +134.57% month in 1974-01** (driving skew ~7.4 /
kurtosis ~126, and with 92 of 588 exactly-zero months — a step-function posted-price
series) and **gold absent before 1985-03**; both shared identically by the two legs so
neither biases the delta.

## Deviations from plan

**1. [Rule 3 — blocking] Classifier #2 cannot be refit through `driver.py::_refit_l1`.**
*Found during:* Task 1. *Issue:* `_refit_l1` calls `canonicalize_states` with its
default `sort_column="trailing_return_1m"`, a column classifier #2's disjoint feature
set does not contain, so it raises `ValueError` unconditionally. The plan's instruction
to use `_refit_l1` for both classifiers and its instruction not to modify `driver.py`
cannot both be satisfied. *Fix:* `_refit_classifier2` in `joint_driver.py` composes the
same public helpers with an explicit `sort_column`; `driver.py` stays byte-identical
and classifier #1 still goes through `_refit_l1`. A parity test asserts the local refit
reproduces `_refit_l1` exactly on classifier #1's own inputs, so it is a seam, not a
fork. *Commit:* `7410cab`.

**2. [Rule 2 — missing critical functionality] Two runner scripts added beyond the
plan's file list.** `scripts/run_joint_lift.py` and `scripts/joint_lift_diagnostics.py`.
A measurement nobody can re-run is the D-02-A defect class; wave 1 set the precedent
with `scripts/run_policy_trials.py`. The diagnostics script is deliberately separate so
it can never touch the registry. *Commit:* `d5c3ac9`.

**3. [Rule 1 — bug] The L2 routing's degrade counter attributed every L2-block failure
to classifier #2.** *Found during:* Task 2, on reading a `100 / 0 / 100` split. *Fix:*
catch each nowcaster separately; the observational leg was re-run under the sentinel
(0 rows). True split: 87 classifier #1, 13 classifier #2. *Commit:* `42161dd`.

**4. [Rule 1 — bug, in this plan's own diagnostics] A silent zero from a column-shape
mismatch.** `compute_sojourn_lag_headline` wants integer state columns;
`measure_label_disagreement` wants `state_{k}` strings. The first draft of the §8
diagnostics passed strings to both and got `n_resolved = 0`, `median_lag = NaN` with no
error — a "detection never happened" reading produced by the wrong matrix shape. Fixed
and pinned with an assertion that the matrix shares at least one state column with the
labeling. This is the project's signature defect class, fifth instance. *Commit:*
`d5c3ac9`.

## Extra scope taken

ADR-0002's routing decision (e) requires an L2 observational leg. The plan's Task 2
described two runs; four were performed (2 tagged L1-only + 2 sentinel L2), spending 2
registry rows as budgeted.

## Known Stubs

None. No placeholder value, hardcoded empty collection or "coming soon" string was
introduced by this plan. Every number in `07-JOINT-LIFT.md` is a live measurement.

## What is NOT closed

- **ADR-0002 is still `Proposed`.** Its acceptance is plan 07-12's job.
- **Criterion 6 remains UNRESOLVED.** Nothing here establishes a second independent
  axis, and no further dependence statistic was computed (the `298b1bc`
  pre-registration forbids tie-breaks).
- **§4.4 criterion 3's Hungarian subsample test has never been run** for either
  classifier.
- **Whether the trial registry should carry `sharpe` metrics at all** — deliberately
  left open as an ADR-0002 amendment question, with the arithmetic recorded.
- **The L2/L1 `dd_delta` sign disagreement is unexplained.**

## Test suite

**1977 passed, 0 skipped, 0 xfailed** (baseline 1949; +28 from
`tests/unit/test_platform_backtest_joint_driver.py`).
`src/trading_crab_lib/platform/backtest/driver.py` and `allocation/tilt.py` are
untouched in the diff; `MAX_LEGACY_IMPORT_SITES` is unedited at **31**.

## Self-Check: PASSED

All six created files and all five task commits verified present on disk and in `git log` (2026-09-21).
