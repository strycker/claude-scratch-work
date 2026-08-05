# Deferred Items — 260805-570

Out-of-scope findings noted during execution but not fixed (per Scope Boundary rule
— only auto-fix issues directly caused by the current task's changes).

## Pre-existing ruff F541 warnings in `scripts/evaluate_momentum.py`

`python -m ruff check src/ scripts/` reports 9 `F541` (f-string without any
placeholders) warnings in `scripts/evaluate_momentum.py`, none of it touched by this
plan. Confirmed pre-existing via `git stash` before/after comparison: 11 errors before
this plan's changes, 9 after (the 2 fixed were a pre-existing `E501` line and an
`I001` import-order issue in `scripts/build_platform_data.py`, both in a file this
plan's Task 3 already modifies — those were fixed as blocking-issue auto-fixes,
Rule 3). The 9 remaining `F541` warnings in `evaluate_momentum.py` are untouched by
any of this plan's three tasks and are left for a future cleanup pass.
