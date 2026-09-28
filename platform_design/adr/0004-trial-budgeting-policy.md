# ADR-0004: Trial Budgeting Policy — Per-Phase Pre-Registered Budgets Replace a Standing Ceiling

## Status

**Accepted, 2026-09-28**, by Glenn's ruling at the close of Phase 8's UAT
(`08-regime-persistence-stability`). Serves design D13 (trial registry, deflated Sharpe) and
Phase 7 decisions **D-16** and **D-17**.

**Amends ADR-0002 § *Trial ceiling*.** It does not supersede ADR-0002 as a whole. The 44 in that
section stands as a **closed, fully spent budget for Phases 7–8**: it is history, and nothing in
this document rewrites it. What this ADR removes is the reading of 44 as a **standing,
project-wide cap**. ADR-0002 never claimed that reading; it said *"Ceiling implied for the
remainder of this phase"*. Phase 8 nonetheless treated it that way.

**What this acceptance does not change:** the deflated-Sharpe denominator, the append-only
ledger, the `NO_REGISTRY` sentinel, the `independent_trial` flag, the locked 2021+ holdout, and
ADR-0003's quality-tier hurdle. See § *What stays exactly as it is*.

## Context

**Where 44 came from.** On 2026-09-10, Phase 7's discussion (`07-DISCUSSION-LOG.md` Q4, recorded
as D-17) asked whether the phase should declare a hard trial ceiling. Glenn chose *"Declare a
ceiling in the ADR — expected count written down before running; exceeding it requires an
explicit amendment. Makes silent search-creep visible."* The expected count at that time was
~5, putting the registry near 35. ADR-0002 then computed the actual ceiling on 2026-09-17 as
arithmetic:

- 38 prior genuine trials,
- plus 2 rows from the INV-01 invariant screen (07-07),
- plus 4 budgeted rows for criterion 7's runs (2 runs × 2 `append_trial` sites),
- **= 44.**

Glenn chose the mechanism. **No one chose the number 44.** It carries no statistical content:
it is not a dimensionality bound, a sample-size limit or an overfitting threshold. It is the
count at the time plus the planned runs.

**What the ceiling was for, and what it was not for.** The protection against overfitting from
multiple testing is **D-16's deflated Sharpe ratio**, not the ceiling. The DSR deflates the
headline Sharpe by the expected maximum Sharpe of N skill-less trials, where N is the whole
registry since project start. More trials raise the bar honestly; they do not break the method.
The ceiling addresses a different failure: **search that nobody declared** (forking paths, "one
more λ"). The registry can *record* such search but cannot *prevent* it. A number written down
before the runs is what makes a later overrun visible. ADR-0002 says so itself: *"the formula is
not a cap the code enforces; it is the arithmetic a reader uses to check whether search-creep
occurred silently between two `total_trial_count()` readings."*

**Why a decision is needed now.** 08-10's two authorised rows took the ledger to exactly 44 on
2026-09-28. Treated as a standing cap, 44 would block every future registry-bearing evaluation,
or force an ADR-0002 amendment for each one. That would make the amendment routine, and a
routine amendment protects nothing. Treated as scoped to Phase 7 (its literal text), there is
currently **no** rule governing Phase 9. Neither state is acceptable.

## Decision

1. **Every phase declares a trial budget before its first registry-bearing run.** The
   declaration lives in the phase's `CONTEXT.md` (written at discuss-phase), under a heading
   `## Trial budget`. A phase that will append no rows declares **`budget: 0`** explicitly.
   Silence is not a zero budget; it is a missing declaration. It contains:
   - **Opening count:** a live `total_trial_count()` reading, with its UTC timestamp. It is read
     at declaration time, never copied from a planning file.
   - **Budget as arithmetic, one line per component:** `rows_per_run × N_runs`, where
     `rows_per_run` is derived from the call sites the evaluation actually traverses and
     `N_runs` names each run. This is the same form as ADR-0002 § *Trial ceiling*. Components
     that write with `NO_REGISTRY` are listed as **0 rows** and give the reason.
   - **Phase ceiling:** `opening count + budget`.
   - **Hurdle at the ceiling:** `expected_max_sharpe(phase ceiling, sharpe_variance)`, with the
     `sharpe_variance` in force and whether it is still the ADR-0002 open-item-8 placeholder.
     This states what the budget costs before it is spent.

2. **Overruns are allowed, by amendment written before the run that would exceed the ceiling.**
   The amendment is appended under the phase's `## Trial budget` heading. It states the new
   ceiling, the arithmetic for the added rows, the reason, the date and a fresh live
   `total_trial_count()` reading. Amendments are **never retroactive.** A row appended over
   budget without a prior amendment is recorded as a **finding** in the phase's measurements
   record. It is never deleted: the ledger is append-only, and the row still counts toward the
   denominator.

3. **Unspent budget expires with its phase.** It does not roll forward. Rolling it forward would
   let a small declared budget quietly accumulate into a large undeclared one.

4. **No standing project-wide cap.** The deflated Sharpe prices the search at every count.
   Whether a new budget is worth its cost is judged per phase, from the hurdle line in item 1.
   Reference points at the current placeholder `sharpe_variance = 1.0`, computed with this
   project's `expected_max_sharpe`:

   | Registry trials | 44 | 50 | 60 | 75 | 100 | 150 | 200 | 500 |
   |---|---|---|---|---|---|---|---|---|
   | Hurdle (Sharpe units) | 2.2269 | 2.2763 | 2.3453 | 2.4277 | 2.5306 | 2.6701 | 2.7655 | 3.0525 |

   The hurdle grows roughly with √(2 ln N). Doubling the registry from 50 to 100 costs about
   0.25 Sharpe. Going from 44 to 500 costs about 0.83.

## What stays exactly as it is

- **D-16:** the DSR denominator is `total_trial_count()`, the **whole registry since project
  start**, including the provenance header's `prior_genuine_trials`. Budgets govern the
  *declaration* of search, never the *counting* of it.
- **`NO_REGISTRY`:** fits that select nothing write no row, and so spend no budget. Examples are
  the serving nowcaster refit on full history (approved 2026-09-28 for gap G-08-2) and
  observational legs (ADR-0002 decision (e)).
- **`independent_trial: false`:** ablation arms still count toward N and are still excluded
  from `registry_sharpe_variance`.
- **The 2021+ holdout** remains locked. No budget, however large, authorises a look at it before
  design freeze.
- **ADR-0003:** the quality-tier hurdle is `expected_max_sharpe(total_trial_count(), …)`. It
  reads the live count and is unaffected by how budgets are declared.

## Considered Options

- **Keep 44 as a standing project cap, amending ADR-0002 for each new run.** Rejected. It turns
  the amendment into routine paperwork attached to every run, which removes the signal an
  amendment is meant to carry. It also treats a Phase 7 budget as if it had statistical meaning.
- **Remove the ceiling entirely and rely on the DSR alone** (the option D-17 declined: "No ceiling, registry is enough").
  Rejected. The DSR prices *recorded* search. Only a prior declaration makes *unrecorded* or
  creeping search visible, which is D-17's reason, and nothing has changed it.
- **Raise the cap to a larger fixed number (e.g. 100).** Rejected. That is 44's defect with a
  different number: it has no derivation, and it would again become a wall or a routine
  amendment when reached.
- **A project-level hard stop at a hurdle value** (e.g. stop when the hurdle exceeds 2.5).
  Rejected for now. `sharpe_variance` is still the placeholder 1.0 (ADR-0002 open item 8), so a
  hard stop keyed to the hurdle would rest on an `[ASSUMED]` number. Revisit when a measured
  `sharpe_variance` exists. Until then, the per-phase hurdle line keeps the cost visible without
  gating on it.
- **Per-phase budgets declared before the runs, amendable in advance, non-rolling (chosen).**
  This keeps D-17's mechanism, scopes it the way ADR-0002's text already did, and leaves
  pricing to the DSR.

## Consequences

- **Phase 8 is closed at 44 against its authorised budget.** Its gap-closure plans (08-11
  onward) run at **budget 0**: the serving-nowcaster builder writes with `NO_REGISTRY`, and the
  test fixes append nothing.
- **Phase 9's discuss-phase must produce a `## Trial budget` section** before any
  registry-bearing run. Its opening count will be 44 unless something is appended first; under
  this ADR nothing may be appended first.
- **Enforcement sites that hard-code 44 must migrate before the first Phase 9 registry row.**
  Until then they are *correct*: they describe the closed Phase 7–8 budget, and the ledger reads
  44. Once a Phase 9 row is appended, they will fail, and the failure is the intended alarm, not
  a flake. The sites are:
  - `tests/unit/test_platform_gate_tiers.py`: `RECORDED_TRIAL_CEILING = 44` and the
    `live <= RECORDED_TRIAL_CEILING` assertion (≈ line 196).
  - `tests/unit/test_platform_joint_diagnostics_record.py`:
    `TestTheDecisionBearingRunSpentExactlyTwoRows::test_the_tracked_ledger_carries_the_two_rows_the_record_claims`
    asserts `total_trial_count() == 44` and that 08-10's rows are the ledger's last two.
  - `scripts/run_joint_lift.py`: `ADR_0002_CEILING = 44` and its `count_after <=` guard.

  The migration keeps the historical assertions (08-10 moved the ledger 42 → 44 and its rows
  exist) but checks the live count against the **current open phase's declared ceiling**, read
  from a machine-readable form of the `## Trial budget` declarations. It is a planned task, not
  part of this ADR.
- **Cost:** one short section per phase at discuss-phase time, and one amendment paragraph per
  overrun. That is intended. The paperwork scales with the search, which is the thing it exists
  to make visible.

## Standing warning

`30`, `34`, `35`, `38`, `40` and `42` are stale registry figures from earlier documents. After
this ADR, **`44` is also no longer a ceiling value**: it is Phase 7–8's closing count. Every
budget declaration reads `total_trial_count()` live.
