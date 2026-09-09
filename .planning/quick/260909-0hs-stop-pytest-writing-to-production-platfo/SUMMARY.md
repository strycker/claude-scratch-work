---
id: 260909-0hs
slug: stop-pytest-writing-to-production-platfo
date: 2026-09-09
status: complete
---

# Summary

**A bug I introduced on 2026-09-08 was destroying the 2021+ holdout checkpoint on
every pytest run.** Found by the stop-hook flagging uncommitted changes I had not
made deliberately.

## What happened

Closing the holdout fence (commit `7f99548`) made `build_monthly_spine()` write
through `write_monthly_features_split()`, which saves to **two** namespaces:
`get_platform_checkpoint_manager()` and `get_holdout_checkpoint_manager()`.

`tests/unit/test_platform_transforms.py` has an autouse fixture redirecting
`PLATFORM_CHECKPOINT_DIR` to `tmp_path`. It knows nothing about
`HOLDOUT_CHECKPOINT_DIR` — that namespace was not a write target when the fixture was
written. So the dev side went to tmp and **the holdout side went to production**.

Every run of that file carved its 24-month synthetic frame at the 2020-12 boundary and
saved the empty post-cutoff side over the real checkpoint:

```
data/holdout/monthly_features.parquet   68 rows x 53 cols  ->  0 x 24
```

This is the worst possible file to lose. A dev-fenced rebuild carves from live data and
**cannot regenerate the holdout** — and it is the one dataset the entire honesty
framework exists to protect. It was recovered only because it happened to be committed
(`git checkout -- data/holdout/`, restored to 68 x 53).

It is also a repeat of a documented project pitfall (P20 / D5, "running pytest no
longer corrupts the checkpoint") reintroduced one namespace over.

## Fix

`tests/conftest.py`'s session-scoped `_isolated_checkpoint_dir` already redirected the
incumbent and platform namespaces — and carried a comment explaining this exact trap
for `PLATFORM_CHECKPOINT_DIR`. Added `HOLDOUT_CHECKPOINT_DIR` alongside them, with
teardown restore, so no individual test file has to remember.

## Verification

- 3 new tests in `TestProductionCheckpointIsolation` assert all three namespaces are
  redirected away from production. All three **fail** against the pre-fix conftest.
- Real-world reproduction: `pytest tests/unit/test_platform_transforms.py` with the
  holdout at 68 x 53 leaves it at 68 x 53.
- Full suite run with `git status` captured before and after: **no production data
  file touched**. 1382 passed (was 1379). ruff clean.

## Also corrected

The Phase 6 context amendment written minutes earlier claimed `regime_labels` existed
(372 x 1), citing it as evidence that D-18/D-19's premise was stale. It existed only
because a diagnostic `label_regimes()` call in the audit had created it moments before.
Claim withdrawn; §C now records that the `daily_raw` half is genuinely stale and the
`regime_labels` half **still holds**. Same error shape the audit is about — confirming
what you went looking for.
