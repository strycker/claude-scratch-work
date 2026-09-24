---
phase: 08-regime-persistence-stability
plan: 09
subsystem: platform/allocation + platform/backtest + platform/report
tags: [criterion-4, PER-05, A7, no-trade-band, bounded-turnover, hysteresis, one-hot-identity, honesty-framework]
status: complete
requires:
  - "08-08 (the filtered belief the hysteresis and the tilt consume; 08-CHURN.md's post-filter max-probability distribution)"
  - "08-04 (G6's pooling-consumer pin, which must pass unmodified)"
  - "08-05 (08-A11.md: registry rows spent 0)"
provides:
  - "allocation/hysteresis.py::execute_rebalance — the 5pp no-trade band, one implementation for all three call sites"
  - "allocation/hysteresis.py::no_trade_band_from_config / hysteresis_thresholds — validated reads; raise on a present invalid value"
  - "config allocation.no_trade_band: 0.05 (not swept); thresholds 0.70/0.40 kept with the ruling in a comment"
  - "report/weekly.py: assemble_weekly_report(active_regime=...) — the hysteresis output, no internal argmax; executed_weights checkpoint"
  - "08-A7.md — A7 closed by rewording; registry rows spent: 2 (authorised here, consumed by 08-10)"
  - "TestOneHotIdentity — F-2's identity across 36 admissible threshold pairs"
affects:
  - "08-10: criterion 7 must be re-measured with the band ON (live config has it). That run is decision-bearing, spends 2 rows, and takes the registry to 44/44. With the band off, the l1only curve is byte-identical to git, so any movement is the band's"
  - "08-10: the tracked l2 curves are still pre-filter AND pre-band; regenerating them is 08-10's"
  - "ROADMAP Phase 4 criterion 3: reworded text proposed in 08-A7.md §2 and NOT applied (ROADMAP is outside this plan's scope)"
decisions:
  - "RULING (Glenn, 2026-09-24): mechanism b-bounded-turnover — a 5-percentage-point no-trade band, NOT SWEPT; A7 closes by rewording; 08-10's criterion-7 re-measurement spends 2 rows, 42 -> 44"
  - "RULING (Glenn, 2026-09-24): hyst-input-per-classifier — the hysteresis stays on classifier #1's own belief; the one-classifier/two-classifier mismatch is declared out of remit; a joint R1 x R2 state space is future work"
  - "RULING (Glenn, 2026-09-24): keep-absolute — act 0.70 / unwind 0.40 for K=6 and K=5; no coefficient, no trial; a K-relative rule is to be adopted before L2 is ever decision-bearing"
  - "Reading taken (before any number): leading degraded steps execute nothing, so the first NON-degraded step is the band's first step and trades in full"
  - "Reading taken (before any number): at serve one band step is one month; a same-month re-run re-bands against the same held book"
  - "Band disabled = key absent or null; the live config value is pinned by a test so deleting it goes red"
tech-stack:
  added: []
  patterns: ["one helper resolved by object at all call sites", "band-off golden (sha256 from the pre-change commit)", "suppress-AND-allow fixtures", "mutation per wiring site", "NO_REGISTRY scratch runner with the band forced on or off"]
key-files:
  created:
    - .planning/phases/08-regime-persistence-stability/08-A7.md
  modified:
    - src/trading_crab_lib/platform/allocation/hysteresis.py
    - src/trading_crab_lib/platform/backtest/driver.py
    - src/trading_crab_lib/platform/backtest/joint_driver.py
    - src/trading_crab_lib/platform/report/weekly.py
    - config/platform_settings.yaml
    - tests/unit/test_platform_hysteresis.py
    - tests/unit/test_platform_report_weekly.py
    - tests/unit/test_platform_backtest_joint_driver.py
metrics:
  duration: "about 40 min, 2026-09-24T14:37Z to 15:18Z"
  completed: 2026-09-24
actuals:
  tokens: 20702   # chars/4 over the added lines of this plan's five commits (15947 before this summary)
  tasks: 3        # Tasks 1 and 2 were answered checkpoints (ruled at 36bbf28); Task 3 executed
  commits: 5
---

# Phase 8 Plan 09: A7 closed by rewording, with a 5pp no-trade band that moves the portfolio

**The outcome.** §5.3's bounded-turnover arm is now implemented once, as
`execute_rebalance`: a 5-percentage-point no-trade band on the executed book. The same
function object is called from `driver.py`, `joint_driver.py` and `report/weekly.py`.

- The weekly report now shows the hysteresis state machine's own output. It no longer
  recomputes an argmax.
- `active_regime` still gates no weight. That is what "closed by rewording" means for A7.
- F-2's one-hot identity is pinned across 36 admissible threshold pairs.
- With the band disabled, the real decision-bearing l1only curve is byte-identical to git.
- The registry reads 42 before and after.

## Glenn's rulings (2026-09-24), with their rationale

Tasks 1 and 2 were `checkpoint:decision` tasks. Glenn had already answered them before this
executor started, and the rulings are recorded verbatim in `08-09-PLAN.md` `<rulings>` at
`36bbf28`. They were not re-asked.

**Task 1(i): `b-bounded-turnover`.**
- **Why:** it is the only §5.3 mechanism that changes the decision-bearing leg. Options (a)
  and (c) are provably inert there.
- **Cost, accepted in advance:** criterion 7's re-measurement in 08-10 consumes **2 registry
  rows, 42 → 44**. That is the ADR-0002 ceiling exactly, and leaves zero headroom for the rest
  of v1.
- **What happens to A7:** it closes by **rewording**, chosen knowingly.

**Task 1(ii): `hyst-input-per-classifier`, with the mismatch DECLARED.** A "blended belief"
is not defined, because classifier #1 has K=6, #2 has K=5, and their state ids are
independent. The joint R1 × R2 state space is recorded as future work.

**τ, the band width: 5 percentage points, NOT SWEPT.** The value is the absolute half of the
practitioner "5/25" drift band. It was taken from outside this backtest and not chosen by
reading its turnover. The edge-case semantics were fixed in the ruling before any number was
read:
- the comparison is `<=`;
- cash is the residual;
- if the residual would go negative, the traded assets are scaled down pro rata;
- a zero target is sold only if the position is above 5pp;
- the first step trades in full;
- "held" means the last executed weight.

**Task 2: `keep-absolute`.** The option's own test is met: the post-filter belief clears 0.70
in **307/488** non-degraded months for #1 and **415/488** for #2. No coefficient and no trial.
A K-relative rule is named as the path, to be adopted before L2 is ever decision-bearing.

## Resolved thresholds

| classifier | K | act | unwind | x uniform (act) |
|---|---|---|---|---|
| #1 | 6 | **0.70** | **0.40** | 4.2x |
| #2 | 5 | **0.70** | **0.40** | 3.5x |

Both satisfy the invariant `0 < unwind <= act <= 1.0`, which is now enforced at read time by
`hysteresis_thresholds()`.

## Tasks and commits

| step | commit | what |
|---|---|---|
| Tasks 1, 2 | `36bbf28` (orchestrator) | rulings recorded. Not re-asked |
| Task 3: identity pin, landed first | `2f8eece` | `TestOneHotIdentity` over 36 pairs, plus the arms that show it can fail |
| Task 3: helper | `d4ac92d` | `execute_rebalance`, the band and threshold readers, config values and comments, band unit arms |
| Task 3: wiring | `036ae74` | the three call sites, weekly `active_regime` and `executed_weights`, call-site tests |
| Task 3: record | `54519aa` | `08-A7.md` |
| summary | this commit | |

## Evidence

### The identity pin, landed first (`2f8eece`)

**Result:** for a one-hot input, `update_active_regime` returns `probs.idxmax()` in every case.
The grid is:
- all **36** pairs with `0 < unwind <= act <= 1.0`, drawn from {1e-9, .05, .1667, .40, .50,
  .70, .90, 1.0};
- **72 cases per pair**: K ∈ {5, 6}, every hot state, the cold start, and every `prev_active`.

**The check can fail.** Three arms show it:
- a real posterior `{0: .55, 1: .45}` with `prev = 1` returns 1, not the argmax;
- `act = 1.01` returns `None` on a cold start;
- mutating the cold start's `>=` to `>` turns **8** grid cases red.

**Measured on the tracked curves** (588 steps, 1972-01-31 → 2020-12-31, both l1only legs):
- `active_regime == state_1` in **588/588** months;
- **246** changes;
- **0** `None`.

### The band suppresses AND allows, and the negative-residual branch fires

Unit arms, in `test_platform_hysteresis.py::TestNoTradeBand`:

| arm | what it shows |
|---|---|
| **suppress + allow in one step** | held SPY .50 / TLT .30, target .53 / .20. SPY is **held** at .50 (a 3pp move). TLT is **traded** to .20 (a 10pp move). Cash .30 |
| exactly-5pp boundary | 0 → .05 and .10 → .05 (both exact in binary) are **not** traded. 0 → .0500001 **is** traded |
| zero target | GLD .04 is held; TLT .20 is sold to 0 |
| **negative residual** | held A .64 plus traded B .20 and C .20 would total 1.04. A stays at exactly .64. B and C are scaled by .36/.40 = 0.9 to .18 each. Their ratio is preserved, the book sums to 1, cash is 0. The precondition "would exceed 1" is asserted first |
| no scale-down when not needed | residual ≥ 0 means `scale_down == 1.0` |
| cash never banded | target cash .48, executed cash .50 |
| held = last executed | two 3pp steps: .50 → (target .53, held) → (target .56, **traded**). Carrying the last *target* would have held |
| first step | `held=None` trades in full, including SPY .03, which lies inside the band from 0 |
| band off | returns the target objects themselves |

**Six mutations of the helper each go red:**
- `<` in place of `<=`;
- the scale-down removed;
- the scale-down applied to everything;
- never hold;
- always hold;
- first step banded.

**At the call sites:**

| site | suppress | allow | negative residual | `held` threading |
|---|---|---|---|---|
| `driver.py` (`TestDriverCallSite`) | ≥1 step | ≥1 step | (unit + serve) | `args[2] is prev_out["weights"]` across a forced degraded step; first call `None`; degraded step turnover 0 |
| `joint_driver.py` (`TestNoTradeBand`) | 57/60 steps held ≥1 asset | 28/60 traded ≥1 | **1/60 steps scaled** | `args[2] is prev_out["weights"]`; turnover equals book-to-book `compute_turnover` exactly; first call `None`, including after 8 leading degraded l2 steps |
| `weekly.py` (`TestNoTradeBandAtServe`) | SPY .26 held (2.5pp) | TLT .60 → .715 (11.5pp) | held SPY .32 plus TLT .715 = 1.035, so TLT is scaled to .68 and the book sums to 1 | same-month re-run re-bands against July's book; next month's held is August's execution |

**Five wiring mutations each go red:**
- the executed book ignored, at driver, joint_driver and weekly;
- `held` never threaded, at driver;
- same-month re-banding against its own output, at weekly.

`test_all_three_call_sites_resolve_the_one_band_helper` checks by resolved object.

**Why this is the gating pin under (b).** The plan's `<behavior>` gating spec was written for
(a) and (c): "hysteresis output ≠ argmax ⇒ weights differ". Under (b), `active_regime` gates
nothing by ruling. The pin that stops "pass it and ignore it" is therefore the one above: with
the band on, the weights **differ** from band-off, and the differences are the band's
suppressions. Band on/off leaves `active_regime` identical, which is asserted.

### The decision-bearing leg with the band disabled: byte-identical to git

`scripts/run_joint_lift.run("l1", dry_run=True)` was run through a scratch runner that sets
`allocation.no_trade_band = None` (NO_REGISTRY). It took 9.5 min.
- `joint_lift_{baseline,joint}_l1only.parquet` and `joint_lift_probs_{1,2}_l1only.parquet` are
  **`cmp`-identical** to `git show 36bbf28:` (content last written at `d5c3ac9`).
  `assert_frame_equal(check_exact=True)` passes on all four.
- Criterion 7 reproduces: `wealth_delta` **−0.12343826162064975**, `dd_delta`
  **+0.02408401236666291**, 588 steps, 1972-01-31 → 2020-12-31.
- The unit pin is the band-off l1only joint curve's sha256, generated at `36bbf28` before any
  change. It passes with the key absent and with it null.
- **The l1only leg with the band ON was not run.** That run is 08-10's, and it spends the two
  rows.

### Observational l2 readings (NO_REGISTRY, firewalled, not acted on)

**Setup.** Both runs used the current code (filter on), via
`run_joint_lift.run("l2", dry_run=True)` through the scratch runner, dumped to scratch rather
than to `outputs/` (see Deviations).
- Window: **588 steps, 1972-01-31 → 2020-12-31**, **100 degraded** in every run.
- `state_1` and `degraded` are identical to the tracked l2 curve.
- The belief matrices reproduce the tracked `joint_lift_belief_{1,2}_l2.parquet` with a
  max difference of **0.0**.

Change counts are per 587 adjacent pairs. "None-as-state" treats the neutral posture as a state
(the honest count). "NaN≠NaN" is F-2's definition, given only for comparability.

| | band off (before) | band on (after) |
|---|---|---|
| `active_regime` changes, None-as-state | **58 / 587** | **58 / 587** |
| `active_regime` changes, NaN≠NaN | 192 / 587 | 192 / 587 |
| `active_regime` None | **161 / 588** | **161 / 588** |
| mean monthly turnover, #1-alone (baseline) leg, all 588 steps | **0.173319** | **0.141119** |
| mean monthly turnover, joint leg, all 588 steps | **0.106643** | **0.068775** |
| mean monthly turnover, baseline leg, 488 non-degraded | 0.208835 | 0.170037 |
| mean monthly turnover, joint leg, 488 non-degraded | 0.128496 | 0.082868 |

- **`active_regime` is identical on and off, element for element, on both legs.** This is
  expected: the band acts after the hysteresis and feeds nothing back into it.
- **For context, the tracked pre-filter l2 curve** gives 122 / 587 (None-as-state), 462 / 587
  (NaN≠NaN), 387 / 588 None, and turnover 0.098907 (baseline) and 0.060451 (joint).
- **The l2 lift was also printed**, observational only: `wealth_delta` −0.205889 → −0.300608
  and `dd_delta` +0.042309 → +0.061011. It is reported and not acted on (ADR-0002 decision (e)).

### Registry

| read | count |
|---|---|
| before this plan's first commit | **42** |
| each of the three scratch runs (l1 off, l2 off, l2 on) | `rows_added` 0; before 42, after 42 |
| after this plan's last run | **42** |
| verify: `42 + 08-A11 spend (0)` | 42, which matches. Headroom 2, reserved for 08-10 |

### G6

`tests/unit/test_platform_pooling_consumers.py` is **unmodified**: `git diff HEAD` is empty,
and its last commit is still `94d5283`. It passes (5 tests). The weekly arm's fake config has
no `no_trade_band`, so the serve path does no held-book I/O there.

## Is the mechanism inert on the decision-bearing leg? Will 08-10 spend rows?

**It is not inert.** The band changes l1only weights by design. On the synthetic l1only
fixture, band-on differs from band-off and summed turnover drops. **08-10's criterion-7
re-measurement is a new configuration.** It spends **2 registry rows, 42 → 44**, and the
banded rows carry `no_trade_band: 0.05` in their registry config.

## Deviations from Plan

1. **[Ruling-driven] The gating pin was adapted to mechanism (b).** The plan's `<behavior>`
   described the gating pin for (a) and (c). Under (b), the discriminating pin is band on/off
   with suppress and allow, as described above. `active_regime` is asserted untouched.
2. **[Scope] The l2 readings were dumped to scratch, not `outputs/reports/platform/joint_lift`.**
   The plan's command would have overwritten tracked l2 artifacts, and those are outside
   `files_modified`. 08-08 already deferred their regeneration to 08-10.
3. **[Finding] F-2's "462 / 587 = 78.7%" counts NaN→NaN pairs as changes.** With None as a
   state, the tracked pre-filter curve shows **122 / 587**, so "flickers nearly twice the raw
   label" does not hold. This is recorded in `08-A7.md` §10. It did not bear on the ruling,
   which rests on (b) being the only non-inert mechanism and on the 387 all-cash months.
3a. **[Finding] The plan's quoted pre-band l1only turnovers are confirmed exactly** on the
   tracked curves: 0.119788 (joint) and 0.163251 (baseline).
4. **[Reading, stated] Leading degraded steps.** Nothing is executed during them, so the first
   non-degraded step trades in full. This does not affect l1only, which has 0 degraded steps.
   It is pinned in the joint-driver test.
5. **[Reading, stated] Weekly cadence.** One band step per month. A same-month re-run re-bands
   against the same held book.
6. **[Rule 2] New `executed_weights` checkpoint at serve.** It is the only way "held = last
   executed" exists at the weekly site. It is loaded before it is saved, and it is written only
   when a band is configured.
7. **[Rule 2] Banded runs are attributable.** `no_trade_band` is added to `trial_config` in
   both drivers only when set, so historical row shapes are unchanged.
8. **[Placement] The `driver.py` call-site tests live in `test_platform_hysteresis.py`,**
   because `test_platform_backtest_driver.py` is not in `files_modified`. That file still
   passes unmodified, including its pre-08-08 golden.
9. **[Scope] The reworded Phase 4 criterion 3 is proposed text only** (`08-A7.md` §2).
   ROADMAP.md was not edited, per instruction.

## Verification

| check | result |
|---|---|
| plan pytest verify (hysteresis, weekly, joint_driver) | green |
| `test_platform_pooling_consumers.py` (G6), unmodified | 5 passed |
| weekly argmax-assignment regex | "no longer recomputes its own argmax" |
| registry verify `total_trial_count()==42` | 42 |
| 08-A7.md grep ('stabilize a label, not a portfolio', 'identity', '588', 'not swept') | ok |
| registry-spend parse | `42 = 42 + {'08-A7.md': 2, '08-A11.md': 0}; headroom 2` |
| `pytest` on hysteresis, weekly, joint_driver, driver, pooling and ratchet | 195 passed |
| legacy-import ratchet | green, 31 |
| ruff and flake8 (E9,F63,F7,F82) | clean on every touched .py, before each commit |
| 2021+ holdout | not read; every run ends at 2020-12-31 |
| `pytest tests/ -q` | **2350 passed, 0 failed, 0 skipped** (baseline 2263 + 87: hysteresis +66, weekly +13, joint_driver +8). The 5 warnings are pre-existing seaborn deprecations in `test_plotting.py` |

## Known Stubs

None.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: new persisted state | src/trading_crab_lib/platform/report/weekly.py | New `executed_weights` checkpoint (parquet: asset, weight, basis, as_of) in the platform checkpoint namespace. A local file only, the same pattern as `regime_belief` and `hysteresis_state`. It feeds the served weights, so a stale or foreign file would move the recommendation. A checkpoint dated after `as_of` is refused with a WARNING |

## Self-Check: PASSED

- FOUND: `08-A7.md`, all eight modified files. The G6 test file is unchanged.
- FOUND commits on `claude/keen-galileo-zqcml6-w5`: `2f8eece`, `d4ac92d`, `036ae74`,
  `54519aa`. Not pushed.
- Registry 42. STATE.md, ROADMAP.md and REQUIREMENTS.md were not touched (orchestrator
  instruction).
