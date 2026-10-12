# ADR-0005: Registry true-zero reset at the 08.5 clean slate

## Status

**Accepted, 2026-10-10**, by Glenn's explicit choice on 2026-10-05 (Phase 08.5, decision D-03).
Relates to ADR-0002's ledger reset (the P7W1 precedent: archive intact, restart with a
provenance header) and to ADR-0004 (per-phase trial budgets).

**What this acceptance does not change:** the append-only rule holds from the new epoch onward
(this header replacement is the one deliberate break, and the prior ledger is kept byte for byte
in the archive); the `total_trial_count()` code, which already sums header priors; the per-phase
trial budgets of ADR-0004; the locked 2021+ holdout; ADR-0003's deflated-Sharpe quality tier.

## Context

Phase 8.5 is a clean slate before the migration: every generated file leaves git and the
platform is rebuilt from a fresh clone. Glenn chose to restart the trial ledger at a true zero
rather than carry the 48 earlier trials as a prior, because those trials measured a pipeline that
the clean slate and the rebuild replace.

**The caveat, stated plainly.** The 48 archived trials ran on the same 1972-2020 development
data that every post-reset trial will use. The reset does not make that data unseen. The
deflated Sharpe ratio (D-16) deflates by the expected maximum Sharpe of N skill-less trials, so
a smaller N means a lower hurdle:

| N trials | `expected_max_sharpe(N, 1.0)` |
|---------:|------------------------------:|
| 0 or 1 | 0.0 |
| 2 | 0.5198 |
| 4 | 1.0521 |
| 10 | 1.5746 |
| 48 (the hurdle before the reset) | 2.2606 |

Right after the reset the hurdle is far below what the full search history would demand.

## Decision

1. **Archive.** The prior ledger is copied byte-identical (`git show 4e49652:registry/trials.jsonl`,
   11 lines, 48 genuine trials) to `registry/archive/trials-pre-8.5-reset.jsonl`. No row is
   deleted from history; the pre-reset tree is also tag `model/pre-reset-2026-10`.
2. **Header.** `registry/trials.jsonl` becomes exactly one provenance-header line:
   `trial_tag` `REGISTRY-RESET-8.5`, `record_type` `provenance_header`, `archived_to`,
   `archived_row_count` 11, `archived_genuine_trials` 48, `prior_genuine_trials` 0, and this ADR
   in `adr`.
3. **Count.** `total_trial_count()` reads 0. Trials since this epoch = `prior_genuine_trials`
   (0) plus the rows appended after the header.
4. **Disclosure rule.** The first post-reset DSR or hurdle line that the platform prints carries
   an epoch note naming this ADR and the 48 archived trials, so the lower hurdle is never read
   as a like-for-like successor of the pre-reset one.

## Consequences

- Hurdles printed right after the reset understate the multiple-testing burden. That is the
  price of the choice, accepted knowingly.
- The guard is the untouched 2021+ holdout. Development, tuning and model selection stay on data
  through 2020-12, and the holdout decides at design freeze whether any edge survives.
- Nothing prints a DSR line until the first budgeted run in Phase 9 (M3), which adds the note
  from decision 4.
- Phase 9 and later budgets (ADR-0004) count from this epoch.
- Undoing the reset later would mean merging post-reset rows onto the archive and restating every
  hurdle printed since; treat it as one-way.
