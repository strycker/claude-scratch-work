# Phase 7 Plan 2 — Pre-Fix Evidence (captured before the D-02-A recompute)

**Captured:** 2026-09-14, immediately before `scripts/recompute_monthly_features.py` overwrites
`data/checkpoints/platform/monthly_features.parquet`.

**Purpose:** `07-CONTEXT.md` D-02-A supersedes D-02's "the frozen set is 9" — the checkpoint
`compute_lean_features` was compared against was stale (`oil` truncated to 1985-02 in
`monthly_features` despite `monthly_raw.oil` running full 1962-01+). This document reproduces
the pre-fix numbers **from the persisted artifacts**, not by quoting `BASELINE-v1-tracer-bullet.md`
or `UAT-AUDIT-2026-09-09.md`, so the eventual pre/post table (plan 07-04) has a reproducible left
column rather than a citation.

**Compound baseline warning (must not be forgotten downstream):** the numbers below were produced
under **two** pre-fix conditions simultaneously — (1) the pre-fix EXPANDING driver policy (A13's
original asymmetry, not yet frozen by plan 07-01) **and** (2) the stale 9-column checkpoint this
plan corrects. No single cause can be read off the eventual delta between this record and plan
07-02/07-04's post-fix numbers; the movement is a mix of the D-01 policy freeze and the D-02-A
data correction, and 07-04's table must say so in prose.

---

## 1. Reproduction recipe

The 82.8% disagreement figure's provenance was located this session at
`src/trading_crab_lib/platform/plotting/regime.py:499`, function `label_disagreement`. It was
**not** in `evaluation/` or `labeling/`, which is why earlier research passes reported it
unlocated.

```python
import pandas as pd
from trading_crab_lib.platform.plotting.regime import label_disagreement

# Reference labeling: the ONE full-sample smoothed jump-model fit.
full = pd.read_parquet("outputs/reports/platform/backtest_full_sample_states.parquet")["state"]

# Comparison labeling: the walk-forward's per-step filtered state probabilities.
# Persisted columns are `state_0`...`state_4` STRINGS (see report.py:665-700, where the
# integer state index is renamed to `state_{k}` before the parquet write).
probs = pd.read_parquet("outputs/reports/platform/backtest_filtered_state_probs.parquet")

# idxmax(axis=1) returns the STRING column label of the max-probability state per
# row, e.g. "state_3" — NOT an integer. label_disagreement runs
# pd.to_numeric(..., errors="coerce") on each series and then .dropna(); handing it
# the raw "state_3" strings makes to_numeric coerce every value to NaN, dropna()
# empties the frame, and the function returns {"n_compared": 0, ...} SILENTLY —
# no exception, no warning. This reads as "0% disagreement, fully resolved" while
# having compared nothing. The int() strip below is load-bearing.
comparison = probs.idxmax(axis=1).str.replace("state_", "", regex=False).astype(int)

result = label_disagreement(full, comparison)
```

Re-run against the artifacts as they stood at this commit and returned exactly the figures
in section 2 below — verified directly, not assumed.

## 2. Reproduced figures

| Field | Value |
|---|---|
| `n_compared` | **470** |
| `n_disagree` | **389** |
| `pct_disagree` | **0.8276595744680851** |
| `first_common_date` | **1974-02-28** |
| `last_common_date` | **2020-12-31** |

**Why `first_common_date` is 1974-02-28, not 1972-01-31 (the walk-forward's declared
`first_decision`):** `backtest_filtered_state_probs.parquet` has 470 rows against the
walk-forward's 588 decision steps — a gap of 118 rows. Early decision steps degraded (an
existing, documented `run_backtest` behavior excluding degraded steps from `per_step_metrics`)
and are absent from the filtered-probs artifact entirely. This gap is part of the honest
pre-fix baseline and must not be smoothed over in the pre/post table: a naive reader could
otherwise assume the comparison spans the full 1972+ decision window when it does not.

## 3. Per-state confusion (reference × comparison), verbatim

Rows are the full-sample smoothed reference state (0-4); columns are the walk-forward
filtered-probs argmax state (0-4). Cell values are counts out of 470 total.

```
comparison   0   1   2   3   4
reference
0            0   3   0   0   3
1            0  11  44  18   4
2            0   4  23  33  18
3           27  55  80  43  36
4            1  22  29  12   4
```

## 4. The nine pre-fix frozen columns

Reproduced by calling `report.py::_reference_label_columns(dev_features, lean_cols,
first_decision=1972-01-31)` against the monthly_features checkpoint as it stood **before**
this plan's recompute (the stale checkpoint, `oil` truncated to 1985-02):

1. `cape_shiller`
2. `credit_spread_baa_aaa`
3. `curve_10y3m`
4. `div_yield`
5. `real_rate_level`
6. `realized_vol_1m`
7. `realized_vol_3m`
8. `trailing_return_1m`
9. `trailing_return_3m`

(`curve_10y2y`, `gold`, `oil`, `fred_vix` were excluded — their pre-fix non-NaN coverage did
not span the full 1972-01 → 2020-12 decision range on the stale checkpoint.)

## 5. The seven pre-fix A13 change points

Reproduced from `tests/unit/test_platform_plotting_regime.py::EXPECTED_CHANGE_POINTS` as it
stood before this plan's Task 3 re-pin (this is the sequence the real-checkpoint test asserted
against, on the stale checkpoint):

| Decision date | Active feature count |
|---|---|
| 1972-01-31 | 4 |
| 1972-02-29 | 6 |
| 1972-04-30 | 8 |
| 1973-02-28 | 9 |
| 1986-06-30 | 10 |
| 1995-02-28 | 12 |
| 2000-01-31 | 13 |

## 6. Vintage of the underlying artifacts

- `HEAD` at capture time: `46557c1d33aeae97eaf4c048554484cc3960a877`
- Most recent commit touching `outputs/reports/platform/`:
  `27fe529f7d155ab177b002afd1bfc07b43e4af6b`, committed `2026-09-11T12:58:04-05:00`
  (`git log -1 --format='%H %cI' -- outputs/reports/platform/`)

The backtest artifacts on disk (`backtest_full_sample_states.parquet`,
`backtest_filtered_state_probs.parquet`) are therefore **three days older** than this capture
and predate plan 07-01's driver/reference freeze — consistent with the compound-baseline
warning above: these artifacts reflect the pre-freeze, pre-recompute state of the system in
both senses.

## 7. Compound-baseline statement (required by the plan)

These artifacts, and every figure derived from them in this document, were produced under:

- **the pre-fix EXPANDING driver feature-admission policy** (A13's original asymmetry between
  `driver.py::_window_active_features` and `report.py::_reference_label_columns`, not yet
  frozen — that freeze is plan 07-01's D-01, landed in commit `cc375db` on 2026-09-14 AFTER
  these artifacts were last written on 2026-09-11); **and**
- **the stale 9-column `monthly_features` checkpoint** this plan (07-02) corrects, in which
  `oil` was truncated to its 1985-02 start instead of the full 1962-01 history carried by
  `monthly_raw`.

Any comparison of these numbers against post-07-01 and/or post-07-02 figures reflects the
combined effect of both changes. Plan 07-04's pre/post table must carry this as a third,
explicitly labelled state (pre-fix / 9-col / expanding-driver) rather than attributing the
whole delta to either change alone — attributing it to one cause would repeat exactly the
"fooled by its own backtest" failure mode `UAT-AUDIT-2026-09-09.md` documents.
