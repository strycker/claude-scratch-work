---
phase: 08-regime-persistence-stability
reviewed: 2026-09-29T12:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - src/trading_crab_lib/platform/report/serving.py
  - src/trading_crab_lib/platform/report/weekly.py
  - src/trading_crab_lib/platform/backtest/driver.py
  - src/trading_crab_lib/platform/backtest/joint_driver.py
  - src/trading_crab_lib/platform/allocation/hysteresis.py
  - src/trading_crab_lib/platform/assets/returns.py
  - src/trading_crab_lib/platform/evaluation/churn.py
  - src/trading_crab_lib/platform/evaluation/report.py
  - src/trading_crab_lib/platform/evaluation/sojourn_lag.py
  - src/trading_crab_lib/platform/labeling/stability.py
  - src/trading_crab_lib/platform/prediction/regime_filter.py
  - scripts/run_joint_lift.py
  - scripts/joint_lift_diagnostics.py
  - scripts/run_subsample_stability.py
  - scripts/terminal_month_diagnostic.py
  - scripts/diagnose_s1_truncation.py
findings:
  critical: 6
  warning: 9
  info: 7
  total: 22
status: issues_found
---

# Phase 8: Code Review Report

**Reviewed:** 2026-09-29T12:00:00Z
**Depth:** standard (plus targeted cross-file checks where the honesty framework required them)
**Files Reviewed:** 16
**Status:** issues_found

## Summary

I reviewed the Phase 8 diff (`e27c1e3..HEAD`) in the context of each whole file. The areas were the Bayes filter (`regime_filter.py`) and its wiring into both drivers and the weekly report, the no-trade band, the serving builder, the subsample-stability machinery, the churn and sojourn metrics, and the five scripts.

The plumbing is careful. Load-before-save ordering holds. A same-month re-run is idempotent: I checked that `update_active_regime` satisfies `f(b, f(b, a)) == f(b, a)` on every branch. The staleness cap does what its constant says (exactly 3 month-ends serves, 4 refuses). The serving fit never opens a holdout path.

The problems are in the mathematics the plumbing carries, and in the honesty ledger:

- **The filter's likelihood inversion divides by the wrong prior.** On the real served model this flips the dominant state's evidence from "against" to "for". The live executed book is built on that belief (CR-01, reproduced below from the checkpoints).
- **The filter carries a belief across walk-forward steps whose state ids the phase's own record says are not comparable** (CR-02).
- **Two features in the served and backtested L2 feature set have a real publication lag** that the backtest does not model. 08-SERVING reads this as a train/serve difference, but it is look-ahead in the evaluated backtest (CR-03).
- **`run_stability` in the library still has the same-id occupancy bug** that the script's `summarize_subsample` was written to fix (CR-04).
- **Registry integrity:** `run_backtest` rows cannot say whether the filter was on (CR-05). `run_joint_lift.py` checks the trial ceiling only *after* appending to the append-only ledger (CR-06).

## Critical Issues

### CR-01: Likelihood ratio divides the posterior by the wrong class prior, which flips evidence on real data

**File:** `src/trading_crab_lib/platform/prediction/regime_filter.py:125-164` (called from `report/weekly.py:230,252`, `backtest/driver.py:582-584`, `backtest/joint_driver.py:354-357`)

**Issue:** `likelihood_ratio` computes `L(j) = posterior(j) / class_prior(j)`. The prior used is `unconditional_belief` over the **whole** label series. The posterior, however, comes from a model trained on a different set of rows: the D-01 embargoed, finite-row block, over its own `classes_` subset. The Bayes inversion is only valid with the training prior of the model that produced the posterior. The docstrings call the mismatch "slightly different priors". On the real serving data it is not slight:

- Whole-label prior (`regime_labels`, 695 months): `{0: .058, 1: .328, 2: .102, 3: .288, 4: .121, 5: .104}`
- Training-block prior (08-SERVING §2): 153 rows, `{0: 11, 3: 137, 4: 5}` → `{0: .072, 3: .895, 4: .033}`
- Served posterior: `{0: .418, 3: .564, 4: .018}`

| State | Current ratio (whole-label prior) | Correct ratio (training prior) |
|---|---|---|
| 0 | .418 / .058 = **7.2** | .418 / .072 = 5.8 |
| 3 | .564 / .288 = **1.96** (evidence FOR) | .564 / .895 = **0.63** (evidence AGAINST) |
| 4 | .018 / .121 = 0.15 | .018 / .033 = 0.55 |
| 1, 2, 5 | 1.0 | 1.0 |

I reproduced the belief recorded in 08-SERVING §3.2 (3: 36.9%, 0: 27.2%, 1: 21.3%, …) with the current ratios, so this is the live number. With the correct prior, state 3 falls to about 16% and state 0 rises to about 30%. The tilt consumes the belief, so the executed book Glenn trades is built on inverted evidence for the dominant state.

A second problem: states absent from `classes_` get ratio 1.0, while present states are inflated by `1 / Σ_present prior`. A flat posterior therefore pushes mass away from absent states on every step. That is the "evidence against a state the nowcaster never produced" which the docstring says it avoids. It matters most when a new regime is emerging.

The backtest is affected too. Its in-window labels span the whole window, while the rectangular finite block begins only when the latest active feature has 120 months of history.

**Fix:** Invert with the prior of the rows the model was actually fit on, restricted to `classes_`. Keep `unconditional_belief(labels)` only as π₀. For example, have `fit_l2_nowcaster` return its training `y`:

```python
model, active, y_train = fit_l2_nowcaster(...)          # y_train = rows fit_nowcaster kept
class_prior = y_train.value_counts(normalize=True)       # over model.classes_ only
belief = filter_step(start, A, posterior, class_prior)   # likelihood_ratio: absent states -> 1.0
```

Also renormalize so that a posterior equal to the training prior gives ratio 1.0 for every state. Changing the recipe changes evaluated curves, so under ADR-0004 it needs its own trial budget. That does not make the current served number correct.

### CR-02: Filter state is carried across steps whose state ids are not comparable

**File:** `src/trading_crab_lib/platform/backtest/driver.py:577-585`; `src/trading_crab_lib/platform/backtest/joint_driver.py:340-357, 597-609`; `src/trading_crab_lib/platform/report/weekly.py:233-246`

**Issue:** `prev_belief` is indexed by the state ids of step *t−1*'s L1 refit. At step *t*, `filter_step(prev_belief, A_t, posterior_t, prior_t)` multiplies it by a transition matrix and posterior indexed by step *t*'s refit. The only check is that the id *sets* match. 08-SERVING §1, item 7 states the problem directly: *"Each backtest step re-labels history with its own jump-model fit, so a state id at one step is not comparable with the same id at another step."*

`canonicalize_states` sorts on the `trailing_return_1m` centroid, so two regimes whose sort-column centroids cross between adjacent refits swap ids. The filter then moves the whole accumulated belief onto the wrong regime. With a sticky `A`, that mistake persists for months.

The same defect exists at serve. `weekly.py:233` reuses the persisted `regime_belief` (and `hysteresis_state`) after `serving.py` rebuilds the model and labels. Nothing fingerprints the belief to the labels it was built on.

**Fix:** Before each `filter_step`, map `prev_belief` from the previous step's ids to the current step's ids. Use a Hungarian match on the confusion matrix of `states_{t-1}` against `states_t` over their shared months, or on the de-standardized centroids (`labeling/stability.py::match_states` already exists). Then permute the vector. At serve, store a label fingerprint (for example a hash of `regime_labels["state"]` plus `model.classes_`) in the `regime_belief` and `executed_weights` checkpoints, and cold-start when it changes.

### CR-03: Backtest look-ahead from unmodelled publication lag in L2 model columns

**File:** `src/trading_crab_lib/platform/backtest/driver.py:550,562` (`feature_row = dev_features.loc[[t]]`); root cause in `config/platform_settings.yaml:65-70` (`M2SL`/`TOTALSL` `shift: false` in the "no meaningful lag" block) and the multpl `div_yield` series

**Issue:** 08-SERVING §1 item 8 and §3.2 show that on 2026-09-29 the 2026-08-31 row still lacks `fred_m2sl`, `fred_totalsl` and `div_yield`, and the 2026-07-31 row lacks `div_yield`. That proves these month-*t* values are not available at month-end *t*. M2SL (H.6) and TOTALSL (G.19) publish about 4–6 weeks after month end and are revised.

The backtest scores `dev_features.loc[[t]]` at decision date *t* with those columns populated. So every evaluated L2 posterior used month-*t* values that did not exist at *t*. That is textbook look-ahead, and it flows into the nowcaster that `fit_l2_nowcaster` serves. 08-SERVING §2.1 files this under "a stated difference between train and serve". Under the honesty framework it is an optimistic bias in every evaluated L2 number (and in the M2/credit invariant ratios feeding classifier #2's features, if those enter a fit at *t*).

**Fix:** Give these series a publication-lag `shift` (or ALFRED vintage alignment keyed by as-of date) so row *t* holds only values published by *t*. Then re-measure. Until then, label every L2-routed number as affected. Also correct the "never meaningfully lagged" comment at `platform_settings.yaml:15-18,49-64`.

### CR-04: `run_stability` reads occupancy, null and episodes by reference id, not matched partner, so evaporation can be misattributed

**File:** `src/trading_crab_lib/platform/labeling/stability.py:770-791`

**Issue:** For each **reference** state, the loop reads the subsample's `occupancy[state]`, `rows_destandardized.loc[sub_states == state]` and `episodes[state]`, all by the same numeric id. It only takes `matched_distance` from the Hungarian partner.

When the assignment is not the identity (which the docstring calls "a finding in its own right"), a row describes two different states at once. Take a reference state whose partner has 0 months (a frozen centroid, distance about 0) while the subsample state with the same id holds 30 months. That row gets `evaporated=False` and a small distance, and is scored stable. This is Trap B, the failure criterion 3 exists to catch.

`scripts/run_subsample_stability.py::summarize_subsample` fixed this by reading from `partner`, and its test says so: *"a same-id read would report sub state 0's months and call it alive"*. The public library function was left wrong. The test `test_runner_reproduces_run_stability_when_the_assignment_is_identity` pins equality only in the identity case, so it cannot catch the bug.

**Fix:**

```python
for state in range(K):
    partner = int(match["assignment"][state])
    months = int(sub_fit.occupancy[partner])
    null = split_half_null(sub_fit.rows_destandardized.loc[sub_states == partner], ...)
    ... n_episodes=episodes[partner]["n_episodes"], longest_episode=episodes[partner]["longest_episode"]
```

Better still, have the script call one library implementation instead of keeping two.

### CR-05: `run_backtest` registry rows cannot say whether the Bayes filter was on

**File:** `src/trading_crab_lib/platform/backtest/driver.py:651-672` (caller: `evaluation/report.py:857`)

**Issue:** `use_regime_filter` now defaults to `True` and changes the curve whenever `use_regime_tilt` is on. `trial_config` records `no_trade_band` ("a banded run is a different configuration") but not the filter. `run_full_backtest_evaluation` calls `run_backtest` without passing the flag. Its next registered run will write a row whose `config` (and `config_hash`) can equal a pre-08-08 row's while the curve differs. The ledger then cannot distinguish two evaluated configurations, and the append-only design makes that permanent.

**Fix:**

```python
trial_config["use_regime_filter"] = bool(apply_filter)
```

Do the same in `joint_driver.py`'s `trial_config` (lines 687-713). Enforcing `NO_REGISTRY` for L2 routing (WR-05) makes that path unreachable today, but not by construction.

### CR-06: Trial-ceiling guard runs after the rows are already in the append-only ledger

**File:** `scripts/run_joint_lift.py:98, 234-257`

**Issue:** Both `run_joint_backtest` calls append their registry rows first. Only then does `assert count_after <= ADR_0002_CEILING` run. The ledger is at 44 (the constant), and the tags are pinned to the 08-10 names. Any re-run of `--routing l1` without `--dry-run` therefore:

1. appends two more rows (46);
2. then raises;
3. leaves the ledger permanently over the ceiling with duplicate-tagged rows.

ADR-0004 §2 requires the amendment to be written *before* the run that would exceed the ceiling. The guards are also plain `assert`, which `python -O` strips. The same pattern applies to `_leg_kpis → _pair_rate → churn_rate`, which can raise (fewer than 2 non-null states) after the rows are written. `ADR_0002_CEILING = 44` is also stale: ADR-0004 replaced the standing cap with per-phase budgets.

**Fix:** Check before spending, and raise instead of asserting:

```python
count_before = total_trial_count()
if decision_bearing and count_before + 2 > phase_ceiling():   # ADR-0004 budget, read not hard-coded
    raise RuntimeError(f"would take the ledger to {count_before + 2} > {phase_ceiling()}; amend first")
...
if rows_added != expected_rows:
    raise RuntimeError(...)
```

## Warnings

### WR-01: Staleness cap never refuses when `monthly_features` itself is stale

**File:** `src/trading_crab_lib/platform/report/weekly.py:121, 508-524`

**Issue:** The lag is measured only against the newest `monthly_features` row. If step 1 (`build_platform_data.py`) has not been re-run for six months, the newest row and the newest complete row are both old, `lag` is 0, and the page says "(the newest row)". It then serves six-month-old guidance with no refusal and no warning. Glenn's words were "fail if the scored month is more than 3 months old". 08-SERVING §2.1 and §3.4 flag the data-relative reading as **awaiting his confirmation**, yet the module docstring and the constant's comment present it as the ruling.

**Fix:** Add a second, wall-clock check: `_months_between(as_of, pd.Timestamp.today().normalize() + MonthEnd(0))`. Also print the run date and the newest-row date on the page. At minimum, stop describing the data-relative reading as ruled until it is confirmed.

### WR-02: Distinct-posterior count uses exact float equality, which is not portable across platforms

**File:** `src/trading_crab_lib/platform/report/weekly.py:555-566`; `src/trading_crab_lib/platform/report/serving.py:136-137`

**Issue:** `np.unique(predict_proba(...), axis=0)` compares bit patterns. The model is `CalibratedClassifierCV(method="sigmoid")` over `LogisticRegression`. Its outputs pass through BLAS dot products whose last-ulp results differ between Accelerate (Glenn's Mac) and OpenBLAS (this container). If the "constant" posterior comes from near-zero, not exactly zero, slopes, the count can read 1 here and 231 on the Mac. The page would then silently drop the "does not depend on the features" sentence. The project lists non-portable float comparisons as a priority defect.

**Fix:** Also report the maximum absolute spread `np.ptp(P, axis=0).max()`, and key the sentence on a stated tolerance, for example 1e-12. Or record that the exact count is platform-dependent.

### WR-03: The serving builder writes the model before its companion artifacts, so a failure leaves a mixed serving set

**File:** `src/trading_crab_lib/platform/report/serving.py:164-190`

**Issue:** `cm.save_model(model, "nowcaster")` runs before `build_core_research_series`, `tradable_asset_returns`, `cm.save(asset_returns)` and `report_returns_by_regime`. A failure in steps 8–9 (a missing `splice` key, a missing `monthly_raw`, a parquet write error) leaves the new nowcaster beside the previous build's `returns_by_regime` and `asset_returns`, possibly keyed on different labels or K. `weekly.py` loads that mix without complaint.

**Fix:** Compute all three artifacts first, then save them together at the end. Alternatively, write a build id into each artifact and have `_load_serving_artifact` refuse a mismatch.

### WR-04: The "one filter rule" differs between `driver.py` and serve on missing observations

**File:** `src/trading_crab_lib/platform/backtest/driver.py:573-578` vs `report/weekly.py:243-246` and `joint_driver.py:359-382`

**Issue:** All three docstrings claim the same recursion "so the allocator consumes the same kind of object at train and serve". On a step without an observation, `driver.py` **holds** the belief. `joint_driver.py` and `weekly.py` advance it with `predict_only_step`. The served nowcaster is fit by `driver.py`'s recipe (`fit_l2_nowcaster`), so train and serve differ at exactly the ragged-edge months ruling q1-c introduced. `regime_filter.py`'s own docstring says holding "would assert 'the world did not move', a different and unwarranted claim".

**Fix:** In `driver.py`, advance by `predict_only_step` using the last available `A`. Keep the previous step's `states` for this purpose.

### WR-05: `independent_trial` is hard-coded False, and L2 routing does not enforce `NO_REGISTRY`

**File:** `src/trading_crab_lib/platform/backtest/joint_driver.py:402, 453-458, 713`

**Issue:** Every future run through this harness, including a genuinely new decision-bearing configuration (a new blend weight, K or feature set), will be marked `independent_trial: False` and excluded from `registry_sharpe_variance`. No parameter can express otherwise, so the multiple-testing variance estimate is quietly starved.

Separately, the docstring says L2 routing "MUST be paired with `registry_path=registry.NO_REGISTRY`". The default is `None`, which means the real ledger, and nothing checks it.

**Fix:** Add an `independent_trial: bool` keyword argument with no default, so callers must decide. Add `if routing == ROUTING_L2_NOWCAST and registry_path != registry.NO_REGISTRY: raise ValueError(...)`.

### WR-06: The split-half null is miscalibrated against the comparison it is used for, and biases toward "stable"

**File:** `src/trading_crab_lib/labeling/stability.py:340-385, 772-775` (and `scripts/run_subsample_stability.py:352`)

**Issue:** The null compares two **disjoint** halves of about n/2 rows each. The quantity it is read against is the distance between a full-sample centroid and a subsample centroid that share most of their rows (a decade drop keeps about 83% of them). The shared rows shrink the matched distance, while the null is inflated both by the halved n and by disjointness. So the matched distance falls "inside the null" even when a state has moved. The module says it gives no verdict, but a human will read these two numbers side by side.

**Fix:** Build the null under the scheme itself: refit the reference state's rows on resamples that keep the same overlap fraction. At minimum, report the overlap fraction on each row and document the direction of the bias.

### WR-07: `run_joint_lift.build_inputs` keeps a second copy of the "ONE" research-to-tradable mapping

**File:** `scripts/run_joint_lift.py:193-199`

**Issue:** `assets/returns.py::tradable_asset_returns` says it is "the ONE research-to-tradable mapping" with "two callers". The decision-bearing criterion-7 harness still inlines its own copy. If the helper changes (ordering, cash handling), the joint-lift universe and the served universe will drift apart, which is the train/serve skew the helper was extracted to prevent.

**Fix:** `asset_returns = tradable_asset_returns(returns, splice_cfg)`.

### WR-08: The truncation-invariance script hard-codes the repo path and hides NaN-versus-value mismatches

**File:** `scripts/diagnose_s1_truncation.py:23, 29, 66-69`

**Issue:**
- `REPO = Path("/home/user/claude-scratch-work")` fails on Glenn's Mac.
- `np.nanmax(np.abs(a - b))` ignores any cell where one side is NaN and the other is a value. `bit_identical` can therefore be True while the two belief paths differ. This script is the evidence behind the S-1 "no leak" ruling.
- The docstring says "T <= 2020-12-31", but nothing checks it.

**Fix:** Use `Path(__file__).resolve().parents[1]`. Compare with `pd.testing.assert_frame_equal(a, b, check_exact=True)` or check `(np.isnan(a) != np.isnan(b)).any()` explicitly. Add `if T > DEFAULT_HOLDOUT_CUTOFF: raise SystemExit(...)`.

### WR-09: Track B churn counts pairs across degraded gaps as adjacent months

**File:** `src/trading_crab_lib/platform/evaluation/churn.py:191-222`

**Issue:** The matrix holds only non-degraded steps (100 of 588 are degraded under l2). `argmax_churn` treats consecutive rows as adjacent months, so a change across a multi-month degraded gap counts as a single month-to-month change. The denominator is `n_rows − 1`, not the number of true adjacent pairs. The docstring asks callers to quote the degraded count, but the rate itself mixes gap pairs with real month-to-month pairs.

**Fix:** Count only pairs whose dates are exactly one month-end apart, and report the excluded gap pairs separately. Or reindex to the full step index and let `state_change_count` drop the NaN rows while keeping the gaps in view.

## Info

### IN-01: Diagnostics read the act threshold directly, bypassing validation

**File:** `scripts/joint_lift_diagnostics.py:173`
**Issue:** It uses `cfg.get(...).get("act_threshold", 0.70)` instead of `hysteresis_thresholds(cfg)`, so an invalid config is not rejected here.
**Fix:** `act_threshold, _ = hysteresis_thresholds(cfg)`.

### IN-02: The trajectory section uses a different transition matrix from the filter

**File:** `src/trading_crab_lib/platform/report/weekly.py:656`
**Issue:** The trajectory uses `empirical_transition_matrix`, while the filter uses `transition_matrix_for` (with row completion). For an unvisited "from" state, the page shows "no trajectory" even though the filter used a completed row.
**Fix:** Use `transition_matrix_for(regime_labels, state_index=state_index)` for both.

### IN-03: The band's exact-edge behaviour is asymmetric in floating point

**File:** `src/trading_crab_lib/platform/allocation/hysteresis.py:210`
**Issue:** With `<=` and no tolerance, `0.35 − 0.30 = 0.04999…` is held, while `0.55 − 0.50 = 0.050000000000000044` is traded. The behaviour is ruled and IEEE-deterministic, but "a move of exactly band" is not decided consistently. The same applies to `trades_implied` at `weekly.py:276-279`.
**Fix:** Document the asymmetry beside the ruling, or compare on rounded basis points.

### IN-04: `_training_block` re-derives the fit's row rule instead of receiving it from the fit

**File:** `src/trading_crab_lib/platform/report/serving.py:83-94`
**Issue:** The recorded `n_train_rows` and `class_counts` come from an independent re-derivation that can diverge silently from what `fit_nowcaster` actually kept.
**Fix:** Return the training `y` from `fit_l2_nowcaster`. This also fixes CR-01.

### IN-05: Unreachable guard in `argmax_churn`

**File:** `src/trading_crab_lib/platform/evaluation/churn.py:220-221`
**Issue:** The `if n_rows` guards can never be false: `churn_rate` has already raised for `n_rows < 2`.
**Fix:** Remove them.

### IN-06: The zero-variance rule differs from sklearn's relative constant test

**File:** `src/trading_crab_lib/platform/labeling/stability.py:165`
**Issue:** The code uses an absolute `10·eps` rule. StandardScaler uses `_is_constant_feature`, which is relative to the mean and n. For a near-constant column with a large mean, the two can disagree, and the inversion would no longer be exact. The pin test only covers ordinary data.
**Fix:** Mirror `sklearn.preprocessing._data._is_constant_feature`, or fit a real `StandardScaler` and read `scale_`.

### IN-07: Held weights for assets that leave the universe persist indefinitely

**File:** `src/trading_crab_lib/platform/allocation/hysteresis.py:206-211`; `report/weekly.py:647`
**Issue:** An asset in `executed_weights` that drops out of `tradable_asset_returns` (for example, gold becomes unavailable) has target 0. If its held weight is at or below the band, it is held forever, and `vol_targeted_tilt` never sized it.
**Fix:** Force-trade any held asset that is absent from the current universe, and log it.

---

_Reviewed: 2026-09-29T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
