---
phase: 08-regime-persistence-stability
plan: 05
subsystem: platform/evaluation (decision record only — no src/ change)
tags: [A11, quality-gate, deflated-sharpe, reversal, honesty-framework, PER-08]
status: complete
requires:
  - "07-BANDS.md §§0.1, 8 (the prior reasoning this plan reverses)"
  - "07-JOINT-LIFT.md / measurement_l1only.json (the instance that reopened A11)"
  - "ADR-0002 open items 8 and 10"
provides:
  - "08-A11.md — A11 ANSWERED 2026-09-21, written as a reversal"
  - "platform_design/adr/0003-quality-gate-tier.md (Accepted) — the quality tier"
  - "tests/unit/test_platform_gate_tiers.py — the ruling pinned, 12 tests, 0 skipped"
  - "the exact phrase `registry rows spent: 0` for 08-09/08-10 budget reconciliation"
affects:
  - "08-10 (executes the code consequence BEFORE re-measuring criterion 7; parses the spend phrase)"
  - "08-09 (both remaining registry rows are left intact for its Task 1)"
  - "ROADMAP criterion 7's recorded MET — now FAILED as a result, still MET as a measurement"
key-files:
  created:
    - .planning/phases/08-regime-persistence-stability/08-A11.md
    - platform_design/adr/0003-quality-gate-tier.md
    - tests/unit/test_platform_gate_tiers.py
  modified:
    - platform_design/adr/0002-l1-second-classifier.md
decisions:
  - "A11 ANSWERED YES via option b-promote-dsr: the deflated-Sharpe hurdle is promoted to a governing quality tier."
  - "registry rows spent: 0 — both remaining rows of the 44 ceiling are reserved for 08-09."
  - "07-BANDS §8's [ASSUMED] objection is narrowed, not answered; the sharpe_variance = 1.0 placeholder is carried forward as a declared cost now load-bearing on a gate."
  - "TDD RED/GREEN executed but committed as one atomic commit — a transient red commit in a working tree shared by four concurrent agents would have broken their full-suite verification."
metrics:
  tasks: 3
  commits: 2
  completed: 2026-09-21
actuals:
  tokens: 10836
  tasks: 3
  commits: 2
---

# 08-05 — A11 answered, and written as the reversal it is

## Tasks

| task | status | commit |
|---|---|---|
| 1 — `checkpoint:decision`, A11 | **ANSWERED by Glenn** before execution; not re-asked | — |
| 2 — `08-A11.md`, the reversal record | complete | `208629c` |
| 3 — the pin + the ADR (tdd) | complete (RED observed, then GREEN) | `bc14480` |

## Task 1 — the option Glenn selected, and his rationale verbatim

**Selected option id: `b-promote-dsr`.** Answer YES to A11, using the deflated Sharpe hurdle as
the governing quality gate.

Glenn's rationale, as given:

> **Registry rows spent: 0.** Promotion of an already-computed quantity costs no new trials. This
> is load-bearing: plan 08-09's `b-bounded-turnover` needs both of the two remaining rows (42 of a
> 44 ceiling), and this ruling deliberately preserves them.
>
> **Accepted consequence, stated in advance:** criterion 7's already-recorded MET becomes FAILED
> retroactively on BOTH legs, at DSR 2.28e-12 and 1.47e-11 against the
> `expected_max_sharpe(42, 1.0)` = 2.208694 hurdle.
>
> **Rationale to record:** it is the only quality threshold in this project that is *derived*
> (Bailey–López de Prado multiple-testing correction, arithmetic on the trial count) rather than
> asserted by analogy. Options `a` and `c` were rejected — `c` decisively, because both domain
> flags are already recorded `False` on the actual result, so promoting them buys the appearance
> of a gate with none of the substance. Option `d` was rejected because it would spend from the
> two-row headroom that 08-09 needs, and a pre-registered number chosen from intuition is still an
> intuition.
>
> **Honest caveat that must appear in the record:** `sharpe_variance = 1.0` is a placeholder until
> 20 independent Sharpe-bearing trials exist, so the 2.208694 hurdle rests on a declared
> assumption. That is a weaker form of 07-BANDS §8's objection than the one against the four
> `[ASSUMED]` bands — conservative, documented, with a stated path to replacement — but it is the
> same species of objection and must not be papered over.
>
> **Ordering fact for the record:** this ruling was taken BEFORE 08-01 and 08-02 executed. No
> number produced by this phase existed at the moment of the ruling. The pre-registration claim
> therefore stands unqualified — and it is true because of the ordering, not because the wave
> graph enforced it.

## What the ruling does to criterion 7's already-recorded verdict

ROADMAP criterion 7 carries **`✅ MET 2026-09-21 as a measurement`**. Under this ruling it becomes
**FAILED on both decision-bearing legs**, retroactively:

| Leg | observed Sharpe | DSR | vs hurdle 2.2086935028832686 |
|---|---|---|---|
| `#1-alone` | 0.9170725133308871 | 2.28151091802503 × 10⁻¹² | FAILED |
| joint | 0.914903185594245 | 1.4690427074211624 × 10⁻¹¹ | FAILED |

Neither Sharpe is within a factor of two of the bar. **The measurement is not re-described:**
`wealth_delta` remains −0.123438 nats over 588 steps (1972-01-31 → 2020-12-31) and criterion 7
remains MET *as a measurement*. What changed is that "measured and reported honestly" is no longer
sufficient for a **quality** verdict. Criterion 7 is now **MET as a measurement and FAILED as a
result**, and the record says both.

## The pin — the value it rejects and the value it accepts

`tests/unit/test_platform_gate_tiers.py`, **12 tests, 0 skipped, 0 xfailed**.

- **REJECTED:** observed Sharpe **0.9170725133308871** and **0.914903185594245** (the recorded
  criterion-7 legs, skew/kurtosis/n_obs as measured), giving DSR **2.28151091802503e-12** and
  **1.4690427074211624e-11**. Both reproduce the recorded values to `rel=1e-9`, so the pin also
  catches a silent estimator change.
- **ACCEPTED:** observed Sharpe **2.60** on the #1-alone leg's *own* moments and track length —
  the only thing that differs from the rejected arm is the Sharpe itself — giving DSR
  **0.752687478118391**. Without this arm the gate could be a constant `False` and every other
  assertion would still be green.
- **Boundary:** at `observed_sharpe` exactly equal to the hurdle the DSR is exactly **0.5** (to
  1e-12), and the gate does **not** pass — pinning that `dsr > 0.5` and
  `observed_sharpe > expected_max_sharpe(...)` are the same statement, and that equalling the bar
  is not clearing it.
- **Provenance:** the hurdle rises with the trial count (2.2086935028832686 at 42 →
  2.2268911497604993 at 44), the live `total_trial_count()` is read and asserted against
  ADR-0002's ceiling, and the recorded legs are shown to **still fail at the ceiling** — so the
  consequence does not depend on the count being exactly 42.
- **The surviving assumption, pinned on purpose:** `DEGENERATE_SHARPE_VARIANCE == 1.0` and
  `registry_sharpe_variance(<empty ledger>)` returns it. The day 20 independent Sharpe-bearing
  trials exist, this goes red and the hurdle is re-dated deliberately rather than drifting.

## Which ADR carries the ruling

**`platform_design/adr/0003-quality-gate-tier.md`**, status **Accepted, 2026-09-21** — a new ADR,
not an amendment, because introducing a gate changes what a verdict means project-wide.
`platform_design/adr/0002-l1-second-classifier.md` is edited by exactly one cross-reference
section closing its open item 10.

**The Considered Options section names the three rejected answers descriptively, not by their
option ids.** That is deliberate: the plan's verification greps every ADR for all four ids and
asserts exactly one distinct id is present. Writing the rejected ids into an ADR would have made
that check match on four and stop discriminating. The ids live in `08-A11.md` §3.3, which the ADR
points at.

## The exact status line for `.planning/STATE.md`'s open-items list

> **audit item A11 ANSWERED 2026-09-21** — reversing the 2026-09-18 decision to leave it open:
> one gate now fails on a bad-but-working model, the deflated-Sharpe hurdle
> `expected_max_sharpe(total_trial_count(), sharpe_variance)`, chosen because it is the only
> quality threshold in this project that is derived rather than assumed; **registry rows spent:
> 0**; criterion 7's recorded MET is **FAILED retroactively on both legs** (DSR 2.28151 × 10⁻¹²
> and 1.46904 × 10⁻¹¹ against a 2.208694 hurdle); and `sharpe_variance = 1.0` remains a declared
> placeholder until 20 independent Sharpe-bearing trials exist, so the hurdle rests on an
> assumption that is carried forward, not closed.

## The ordering claim, verified rather than asserted

At the moment `08-A11.md` was written, this phase had produced **no numbers**: no
`08-*-SUMMARY.md` existed in the phase directory and no churn artifact existed under
`outputs/reports/platform/`. Both checked, not assumed. Independently corroborated by commit
order — the ruling commit `208629c` precedes sibling plan 08-02's harness commit `c01cfc8` on this
branch. **The pre-registration claim stands unqualified**, and it stands because of the ordering
that actually occurred: `08-01` and `08-02` are `autonomous: true` in the same wave with no
ordering constraint against this plan, so the dependency graph would have permitted either to
write first. The one edge the graph *does* enforce is 08-10 → 08-05.

## Handoff to 08-10, stated because it is a scope fence

This plan wrote **no `src/`** — `git diff --stat 208629c^..bc14480 -- src/` is empty. Surfacing
the quality-tier verdict alongside the existing `*_universal_ok` / `*_domain_note` flags in
`joint_lift_table` is executed by **plan 08-10, under this ruling, before criterion 7 is
re-measured**. `joint_driver.py` is owned by 08-01 in this wave and 08-10 in wave 5; editing its
band constants from here would have been a cross-plan write.

## Deviations from plan

**1. [Rule 3 — blocking] TDD RED/GREEN collapsed into one commit.**
- **Found during:** Task 3 (`tdd="true"`).
- **Issue:** the task's own constraint that this plan writes no `src/` means the gate predicate
  lives in the test file, so the gate arms cannot be genuinely red first. A real RED *was*
  available — the arms that read `0003-quality-gate-tier.md` — but committing it would have put a
  failing test into a working tree shared by **four concurrently-committing sibling agents**,
  breaking any sibling running `pytest tests/ -q` as its own verification.
- **Fix:** RED was executed and observed locally (**2 failed, 10 passed** — the two
  record-reading arms), then the ADRs were written and the file went green (**12 passed**), and
  both were committed as one atomic commit `bc14480`. The substance of TDD is preserved — the
  pin was observed failing before the artifact it pins existed — without poisoning a shared tree.

**2. [Rule 1 — record accuracy] Two DSR values were mislabelled in the ruling as given.**
- **Issue:** the ruling text labels `2.28e-12` and `1.47e-11` as "(l1only)" and "(l2)". They are
  in fact the **two legs of the L1-only routing** — `#1-alone` baseline and joint — per
  `measurement_l1only.json`. The L2-observational routing's own two legs report
  **3.254192e-30** and **6.144320e-33** over the same window.
- **Fix:** `08-A11.md` §2 carries the correction explicitly as a labelled parenthesis rather than
  silently applying it, and names the L2 figures for completeness while noting that routing is
  `NO_REGISTRY`-firewalled per ADR-0002 decision (e), so nothing rests on it. **The ruling's
  substance is unaffected** — "FAILED on both legs" is true of the two legs criterion 7's recorded
  MET actually rests on, and would be true of the L2 pair as well.

**3. [scope] `platform_design/adr/README.md`'s ADR index was NOT updated with a row for 0003.**
- `README.md` is outside this plan's `files_modified`, and four sibling agents are committing to
  the same tree. Recorded here as a follow-up rather than taken. One row is needed:
  `| 0003 | quality-gate-tier | Promote the deflated-Sharpe hurdle to a governing quality tier; A11 ANSWERED | Accepted, 2026-09-21 |`.

## Known gaps, stated

1. **The hurdle rests on a placeholder.** `sharpe_variance = 1.0` is `DEGENERATE_SHARPE_VARIANCE`,
   a declared assumption until 20 independent Sharpe-bearing trials exist. ADR-0002 open item 8
   therefore **escalates from governing a report to governing a gate** and remains open. This is
   named in the ADR, in `08-A11.md` §4, and pinned by a test.
2. **No code enforces the gate yet.** It is a decision and a pin; the implementation is 08-10's.
   Until then, `joint_lift_table` still emits only the two universal/advisory flag pairs.
3. **`platform_design.md` gets no cross-reference line** from this plan (outside `files_modified`).
   The ADR README convention asks for one pointing at 0003.

## Self-Check

### Verification results

| Check | Result |
|---|---|
| Task 2 verify 1: required strings in `08-A11.md` (`What was decided before`, `What changed`, `11.61`, `2.208694`, `ANSWERED`, `07-BANDS`) | **ok** |
| Task 2 verify 2: prior reasoning comes before the ruling | **holds** (offset 1306 < 5503) |
| `registry rows spent: 0` present in `08-A11.md` | **yes** (2 occurrences, §0 preamble and §3.5) |
| `pytest tests/unit/test_platform_gate_tiers.py -q` | **12 passed** |
| No-skip check on that file | **no skips, no xfails** |
| ADR check (selected id in an ADR git shows changed since merge-base `79391241`) | **pass**: `platform_design/adr/0003-quality-gate-tier.md -> b-promote-dsr`, exactly one distinct id across all ADRs |
| ADR check before the Task 3 commit | failed as designed: an untracked file is not in `git diff`. It passed once the file was committed, which shows the check can fail |
| `git diff --stat 208629c^..bc14480 -- src/` | **empty** |
| `total_trial_count()` | **42**, as expected for 0 rows spent |
| `pytest tests/ -q` (full suite, run against the tree at `bc14480` plus concurrent sibling work) | **2075 passed, 13 skipped, 0 failed** in 21m25s. The 13 skips are optional-dependency skips in other files; none are in this plan's test file |
| Lint: `ruff check tests/unit/test_platform_gate_tiers.py` | **clean** (fixed one import-order issue before committing) |

### Files and commits

- FOUND: `.planning/phases/08-regime-persistence-stability/08-A11.md`
- FOUND: `tests/unit/test_platform_gate_tiers.py`
- FOUND: `platform_design/adr/0003-quality-gate-tier.md`
- FOUND: `platform_design/adr/0002-l1-second-classifier.md` (one section appended)
- FOUND: commit `208629c`, commit `bc14480`

## Self-Check: PASSED
