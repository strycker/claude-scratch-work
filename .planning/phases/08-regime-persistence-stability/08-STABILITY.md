# Phase 8 Plan 07 — §4.4 Criterion 3 (Subsample Stability), Run for Both Classifiers

**Measured:** 2026-09-23, `scripts/run_subsample_stability.py --classifier both`, commits
`0f30661` (harness) and `d47d2de` (run and artifacts). Artifacts:
`outputs/reports/platform/stability/stability_record.json` (77 record rows),
`stability_rows.parquet` (8,883 per-refit rows), `stability_cost_matrices.parquet` (98,526 cost
entries: every K x K matrix for every refit, in two unit spaces).

**Read this with its scope in view.** Exactly one pass/fail is rendered in this document: §4.4
AMENDMENT condition (i) on classifier #1's state 0 (§6). Every other number is a reading. No
persistence threshold exists anywhere in the record, none was invented, and no result here re-pins
K or lambda.

---

## 1. What criterion 3 asks

§4.4 criterion 3, verbatim (`platform_design/platform_design.md`):

> **3. Stability:** re-estimate on subsamples (drop first decade / last decade / block bootstrap);
> states persist with matched emission parameters (match via Hungarian algorithm on distribution
> distances to defeat label switching). A "regime" that evaporates when 2008–09 is dropped is an
> *episode*, not a regime.

§4.4 AMENDMENT 2026-09-18, recurrence exemption, condition (i), verbatim:

> (i) it recurs in **at least three temporally separated episodes**, so that removing any one
> leaves the state intact — criterion 3 applied directly, not by proxy;

The AMENDMENT's own closing rule: *"A state failing any of (i)–(iv) is an episode, and criterion
3's verdict stands: it is merged."* Condition (i) binds only on the sub-floor state invoking the
exemption, which is classifier #1's state 0. Classifier #2 does not invoke the exemption.

## 2. What was run

| | Classifier #1 | Classifier #2 |
|---|---|---|
| pinned (K, lambda, n_restarts) | (6, 10.0, 10) — `labeling` | (5, 16.0, 10) — `labeling_2` |
| `sort_column` | `trailing_return_1m` (`canonicalize_states` default, as `label_regimes` uses it) | `rs_equities_bonds` |
| frozen columns (in order) | `cape_shiller, credit_spread_baa_aaa, curve_10y3m, div_yield, oil, real_rate_level, realized_vol_1m, realized_vol_3m, trailing_return_1m, trailing_return_3m` (10) | `rs_equities_bonds, rs_oil_equities, equities_tr_mom_12m, long_duration_tr_mom_12m, oil_mom_12m, corr_equities_tr_long_duration_tr_24m, cpi_acceleration, m2_gdp` (8) |
| **reference identity** (elementwise vs tracked checkpoint, no tolerance) | `regime_labels`: **695 months, 6 states, 1963-02-28 → 2020-12-31, 0 mismatches** | `regime_labels_2`: **696 months, 5 states, 1963-01-31 → 2020-12-31, 0 mismatches** |
| subsample refits | 808 = 2 decade drops + 4 x 200 bootstrap + 6 LOO | 807 = 2 + 800 + 5 |

- **Classifier #1 is fit on the ten frozen columns, not `lean_feature_set`'s 13.**
  `diagnostics.label_regimes` selects the lean set, but the tracked checkpoint reproduces
  elementwise from `_reference_label_columns`' ten (first decision 1972-01-31). The ten are used
  because they reproduce the checkpoint, and the record says so (`feature_set_note`).
- **Holdout carved first.** Both `monthly_features` and `monthly_raw` are split at 2020-12-31
  before any feature, frozen list or fit is derived. The elementwise identity above shows the carve
  changed nothing.
- **Four schemes.** Drop first decade (first 120 months), drop last decade (last 120), circular
  block bootstrap at L in {6, 12, 24, 48} with 200 replicates each, and leave-one-episode-out
  (drop each state's longest reference episode and refit).
- **Seeds.** Stability seed 20260921 (split-half null, and the bootstrap via
  `SeedSequence([seed, L, replicate])`). Every refit uses `fit_jump_model`'s production
  `random_state=42`, the same as the reference, so the data is the only thing that differs.
  `n_bootstrap = 200`, `n_null_reps = 200`.
- **Runtime.** 231 s wall on 4 workers for the committed run (166 s on an earlier identical run).
  The two runs are **bit-identical** in all three artifacts. Fits measured ~0.3–0.5 s each, not
  the 0.111 s `08-RESEARCH.md` §5.7 assumed. **Criterion 3 is cheap. There was never a cost
  reason it was not run.**
- **Registry: 42 before, 42 after.** Asserted inside the run and again by the record test.

**Keyed on the matched partner.** On every row, occupancy, the split-half null and the episode
count describe the Hungarian-matched partner, which is the subsample state whose distance is on
that row. `stability.run_stability` keys those quantities on the subsample state with the same
id, which agrees only when the assignment is the identity. For classifier #1 it is **never** the
identity (§4), so that convention would have put another state's occupancy beside every distance.
See §8 and the summary's deviations.

## 3. The pre-registered prediction (commit `2363752`, 2026-09-21 20:23 UTC — before any criterion-3 number existed)

Quoted verbatim from `08-RESEARCH.md` §5.1 at `2363752`:

> So the falsifiable prediction for criterion 3, stated before it is run:
>
> - **State 0 (crisis) will pass.** Drop-first-decade removes one 3-month episode; drop-last-decade
>   removes two (6 months). Six to seven temporally separated episodes remain either way, so
>   AMENDMENT condition (i) ("at least three temporally separated episodes") holds. D-05 expects
>   this test to adjudicate the crisis state; on the arithmetic, **it will exonerate it.**
> - **State 2 is the likely failure** — and *neither decade-drop touches it* (1996-2002 is
>   interior). The three named schemes are, by construction, poorly aimed at the actual failure
>   mode.
>
> [...] For state 2 the test is degenerate *and that is the answer*: a state with exactly one
> episode fails leave-one-episode-out by construction, which is the design's definition of an
> episode, reached with **no invented threshold**.

The first number produced by this harness was committed at `0f30661` (2026-09-23 18:29 UTC).

## 4. Results

**How to read a row.** `dist` is the Euclidean distance between the reference centroid and its
matched partner's centroid in de-standardized (winsorized) feature units, which is plan 08-03's
distance as built. `null` is the within-state split-half null **at that row's own subsample n**,
in the same units. The null is not zero, so a distance with no null beside it cannot be read.
`ref-SD` is the same pair of quantities with both sides divided by the reference fit's
per-column SD (see caveat 8.4: the winsorized distance barely sees 8 of classifier #1's 10
columns). `ref in sub` counts subsample months the reference gave this state. `overlap` counts
those months that also sit in the partner. `evap` is built from the partner's occupancy alone.

### 4.1 Full-sample reference labelings (reproduced, not assumed)

**Classifier #1** — 695 months, 1963-02-28 → 2020-12-31.

| state | months | occupancy | episodes | longest | episode spans (start, length in months) |
|---|---|---|---|---|---|
| 0 | 40 / 695 | 5.76% | 9 | 11 | 1970-05 (3), 1973-11 (11), 1981-09 (1), 1987-10 (4), 1990-08 (3), 2002-06 (4), 2008-09 (8), 2011-08 (3), 2020-03 (3) |
| 1 | 228 / 695 | 32.81% | 5 | 87 | 1963-02 (87), 1971-03 (32), 1976-05 (23), 1988-09 (23), 1991-04 (63) |
| 2 | 71 / 695 | 10.22% | 1 | 71 | 1996-07 (71) |
| 3 | 200 / 695 | 28.78% | 4 | 100 | 2002-10 (71), 2009-10 (22), 2011-11 (100), 2020-06 (7) |
| 4 | 84 / 695 | 12.09% | 3 | 72 | 1981-10 (72), 1988-02 (7), 2009-05 (5) |
| 5 | 72 / 695 | 10.36% | 4 | 41 | 1970-08 (7), 1974-10 (19), 1978-04 (41), 1990-11 (5) |

**Classifier #2** — 696 months, 1963-01-31 → 2020-12-31.

| state | months | occupancy | episodes | longest | episode spans (start, length in months) |
|---|---|---|---|---|---|
| 0 | 116 / 696 | 16.67% | 2 | 104 | 1974-01 (104), 1983-11 (12) |
| 1 | 158 / 696 | 22.70% | 4 | 95 | 1963-01 (95), 1971-12 (25), 1987-04 (25), 1994-03 (13) |
| 2 | 156 / 696 | 22.41% | 5 | 58 | 1970-12 (12), 1982-09 (14), 1984-11 (29), 1989-05 (58), 1995-04 (43) |
| 3 | 166 / 696 | 23.85% | 1 | 166 | 1998-11 (166) |
| 4 | 100 / 696 | 14.37% | 1 | 100 | 2012-09 (100) |

### 4.2 Contiguous schemes — drop first decade, drop last decade, leave-one-episode-out

Subsample sizes: decade drops 575 (#1) / 576 (#2) months. LOO = full sample minus the dropped
episode.

| # | scheme | ref state | partner (identity?) | dist | null median [p10, p90] at n | margin | occupancy (of subsample) | episodes | evap | ref in sub | overlap | ref-SD dist | ref-SD null | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | drop_first_decade | 0 | 0 (N) | 4.840 | 5.595 [1.674, 10.954] at n=21 | 0.583 | 21 / 575 (3.7%) | 5 | False | 37 | 18 | 1.818 | 1.605 |  |
| 1 | drop_first_decade | 1 | 3 (N) | 10.222 | 1.210 [0.479, 2.309] at n=148 | 0.760 | 148 / 575 (25.7%) | 6 | False | 118 | 96 | 1.076 | 0.350 |  |
| 1 | drop_first_decade | 2 | 4 (N) | 19.981 | 2.010 [0.729, 4.667] at n=180 | 1.219 | 180 / 575 (31.3%) | 4 | False | 71 | 62 | 1.303 | 0.318 |  |
| 1 | drop_first_decade | 3 | 2 (N) | 27.158 | 1.690 [0.532, 3.772] at n=75 | 1.101 | 75 / 575 (13.0%) | 2 | False | 200 | 70 | 1.261 | 0.465 |  |
| 1 | drop_first_decade | 4 | 5 (N) | 1.769 | 1.957 [0.693, 3.985] at n=69 | 0.171 | 69 / 575 (12.0%) | 2 | False | 84 | 69 | 0.345 | 0.555 |  |
| 1 | drop_first_decade | 5 | 1 (N) | 1.567 | 1.794 [0.690, 4.435] at n=82 | 0.170 | 82 / 575 (14.3%) | 2 | False | 65 | 58 | 0.903 | 0.635 |  |
| 1 | drop_last_decade | 0 | 0 (N) | 3.142 | 5.890 [2.312, 11.951] at n=25 | 0.354 | 25 / 575 (4.3%) | 9 | False | 34 | 24 | 1.185 | 1.336 |  |
| 1 | drop_last_decade | 1 | 1 (N) | 0.502 | 0.906 [0.402, 1.624] at n=245 | 0.033 | 245 / 575 (42.6%) | 5 | False | 228 | 223 | 0.173 | 0.254 |  |
| 1 | drop_last_decade | 2 | 3 (N) | 12.590 | 2.331 [0.828, 5.772] at n=133 | 0.626 | 133 / 575 (23.1%) | 3 | False | 71 | 70 | 0.903 | 0.411 |  |
| 1 | drop_last_decade | 3 | 2 (N) | 17.808 | 3.819 [1.163, 8.030] at n=35 | 0.553 | 35 / 575 (6.1%) | 2 | False | 86 | 28 | 1.685 | 0.914 |  |
| 1 | drop_last_decade | 4 | 5 (N) | 1.712 | 1.335 [0.511, 2.781] at n=72 | 0.272 | 72 / 575 (12.5%) | 1 | False | 84 | 72 | 0.253 | 0.555 |  |
| 1 | drop_last_decade | 5 | 4 (N) | 1.289 | 2.024 [0.755, 4.594] at n=65 | 0.148 | 65 / 575 (11.3%) | 3 | False | 72 | 59 | 0.529 | 0.680 |  |
| 1 | leave_one_episode_out | 0 | 0 (N) | 12.076 | 5.981 [2.231, 16.178] at n=34 | 1.562 | 34 / 684 (5.0%) | 9 | False | 29 | 28 | 1.232 | 1.114 | dropped 1973-11→1974-09 (11 mo of 9 episodes) |
| 1 | leave_one_episode_out | 1 | 2 (N) | 5.051 | 1.080 [0.465, 1.914] at n=156 | 0.343 | 156 / 608 (25.7%) | 3 | False | 141 | 126 | 0.810 | 0.323 | dropped 1963-02→1970-04 (87 mo of 5 episodes) |
| 1 | leave_one_episode_out | 2 | 2 (N) | 19.444 | 1.218 [0.478, 2.383] at n=107 | 0.934 | 107 / 624 (17.1%) | 5 | False | 0 | 0 | 3.118 | 0.339 | dropped 1996-07→2002-05 (71 mo of 1 episode); **DEGENERATE**; ref-SD Hungarian pairs it with 1 |
| 1 | leave_one_episode_out | 3 | 4 (N) | 4.692 | 3.078 [1.020, 8.548] at n=87 | 0.130 | 87 / 595 (14.6%) | 2 | False | 100 | 78 | 0.511 | 0.533 | dropped 2011-11→2020-02 (100 mo of 4 episodes) |
| 1 | leave_one_episode_out | 4 | 5 (N) | 8.210 | 4.267 [1.311, 9.914] at n=41 | 0.709 | 41 / 623 (6.6%) | 5 | False | 12 | 4 | 2.884 | 0.823 | dropped 1981-10→1987-09 (72 mo of 3 episodes) |
| 1 | leave_one_episode_out | 5 | 3 (N) | 9.411 | 1.057 [0.540, 2.011] at n=133 | 1.519 | 133 / 654 (20.3%) | 6 | False | 31 | 23 | 2.669 | 0.331 | dropped 1978-04→1981-08 (41 mo of 4 episodes) |
| 2 | drop_first_decade | 0 | 0 (Y) | 186.651 | 110.231 [16.716, 254.645] at n=24 | 2.756 | 24 / 576 (4.2%) | 2 | False | 116 | 24 | 3.124 | 0.751 |  |
| 2 | drop_first_decade | 1 | 1 (Y) | 40.749 | 69.097 [15.951, 163.299] at n=107 | 0.138 | 107 / 576 (18.6%) | 4 | False | 50 | 12 | 1.805 | 0.364 |  |
| 2 | drop_first_decade | 2 | 2 (Y) | 72.778 | 118.445 [17.872, 267.988] at n=179 | 0.050 | 179 / 576 (31.1%) | 2 | False | 144 | 141 | 0.474 | 0.268 |  |
| 2 | drop_first_decade | 3 | 3 (Y) | 0.000 | 171.168 [24.381, 403.889] at n=166 | 0.000 | 166 / 576 (28.8%) | 1 | False | 166 | 166 | 0.005 | 0.352 |  |
| 2 | drop_first_decade | 4 | 4 (Y) | 4.941 | 209.580 [34.834, 585.318] at n=100 | 0.002 | 100 / 576 (17.4%) | 1 | False | 100 | 100 | 0.009 | 0.349 |  |
| 2 | drop_last_decade | 0 | 0 (Y) | 219.012 | 129.697 [25.611, 294.388] at n=24 | 1.993 | 24 / 576 (4.2%) | 2 | False | 116 | 24 | 3.124 | 0.751 |  |
| 2 | drop_last_decade | 1 | 1 (Y) | 218.369 | 40.939 [3.909, 91.838] at n=227 | 0.667 | 227 / 576 (39.4%) | 4 | False | 158 | 120 | 0.902 | 0.225 |  |
| 2 | drop_last_decade | 2 | 2 (Y) | 1015.081 | 46.217 [10.035, 102.165] at n=38 | 3.049 | 38 / 576 (6.6%) | 2 | False | 156 | 37 | 1.903 | 0.556 |  |
| 2 | drop_last_decade | 3 | 3 (Y) | 1504.034 | 134.244 [29.303, 309.049] at n=139 | 9.916 | 139 / 576 (24.1%) | 1 | False | 146 | 0 | 2.111 | 0.284 | ref-SD Hungarian pairs it with 4 |
| 2 | drop_last_decade | 4 | 4 (Y) | 2284.632 | 159.631 [25.757, 447.363] at n=148 | 0.580 | 148 / 576 (25.7%) | 1 | False | 0 | 0 | 2.081 | 0.349 | ref-SD Hungarian pairs it with 3 |
| 2 | leave_one_episode_out | 0 | 0 (Y) | 96.047 | 45.779 [7.298, 127.452] at n=151 | 0.169 | 151 / 592 (25.5%) | 2 | False | 12 | 0 | 2.259 | 0.209 | dropped 1974-01→1982-08 (104 mo of 2 episodes); ref-SD Hungarian pairs it with 1 |
| 2 | leave_one_episode_out | 1 | 1 (Y) | 1200.689 | 109.830 [25.743, 240.046] at n=225 | 6.764 | 225 / 601 (37.4%) | 2 | False | 63 | 58 | 1.738 | 0.249 | dropped 1963-01→1970-11 (95 mo of 4 episodes); ref-SD Hungarian pairs it with 2 |
| 2 | leave_one_episode_out | 2 | 2 (Y) | 166.770 | 277.011 [26.699, 649.543] at n=70 | 0.135 | 70 / 638 (11.0%) | 4 | False | 98 | 70 | 0.918 | 0.405 | dropped 1989-05→1994-02 (58 mo of 5 episodes) |
| 2 | leave_one_episode_out | 3 | 3 (Y) | 1812.304 | 127.037 [18.775, 317.068] at n=158 | 0.742 | 158 / 530 (29.8%) | 5 | False | 0 | 0 | 2.735 | 0.273 | dropped 1998-11→2012-08 (166 mo of 1 episode); **DEGENERATE**; ref-SD Hungarian pairs it with 0 |
| 2 | leave_one_episode_out | 4 | 4 (N) | 2435.057 | 146.242 [19.451, 389.553] at n=168 | 0.589 | 168 / 596 (28.2%) | 1 | False | 0 | 0 | 2.000 | 0.341 | dropped 2012-09→2020-12 (100 mo of 1 episode); **DEGENERATE**; ref-SD Hungarian pairs it with 0 |

### 4.3 Circular block bootstrap — 200 replicates per block length

Summaries run across replicates. Distance and null quantiles use non-evaporated replicates only;
no replicate evaporated, so that restriction removed nothing. `occ` is the median occupancy, with
the minimum over the 200 replicates in parentheses. The null's own n equals that median. Each
replicate's null at its own n is in the parquet. `id frac` is the share of replicates whose whole
assignment was the identity.

| # | L | ref state | id frac | partner mode (share) | dist median [p10, p90] | null median [p10, p90] | margin | occ median (min) of n | episodes (median) | evap replicates | ref-SD dist | ref-SD null |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 6 | 0 | 0.01 | 0 (0.71) | 8.206 [3.093, 21.307] | 5.233 [2.257, 8.408] | 1.007 | 33 (13) of 695 | 10 | 0 / 200 | 2.240 | 1.147 |
| 1 | 6 | 1 | 0.01 | 1 (0.29) | 2.249 [0.748, 4.752] | 0.984 [0.825, 1.231] | 0.180 | 224.5 (30) of 695 | 23 | 0 / 200 | 0.419 | 0.250 |
| 1 | 6 | 2 | 0.01 | 4 (0.27) | 8.844 [1.644, 25.766] | 1.830 [1.344, 2.682] | 0.478 | 103 (19) of 695 | 13.5 | 0 / 200 | 1.220 | 0.417 |
| 1 | 6 | 3 | 0.01 | 2 (0.29) | 6.638 [1.607, 25.473] | 2.329 [1.838, 3.181] | 0.262 | 161.5 (19) of 695 | 20 | 0 / 200 | 0.495 | 0.315 |
| 1 | 6 | 4 | 0.01 | 5 (0.48) | 3.541 [1.435, 7.428] | 1.567 [0.919, 2.623] | 0.369 | 78 (18) of 695 | 12 | 0 / 200 | 0.838 | 0.529 |
| 1 | 6 | 5 | 0.01 | 1 (0.23) | 4.750 [1.600, 9.655] | 1.989 [1.349, 3.358] | 0.543 | 69.5 (16) of 695 | 11 | 0 / 200 | 1.328 | 0.631 |
| 1 | 12 | 0 | 0.01 | 0 (0.71) | 9.013 [3.729, 18.316] | 4.903 [1.822, 8.472] | 0.990 | 33.5 (11) of 695 | 8 | 0 / 200 | 2.538 | 1.171 |
| 1 | 12 | 1 | 0.01 | 1 (0.32) | 2.537 [1.144, 6.191] | 0.928 [0.743, 1.218] | 0.251 | 202.5 (46) of 695 | 13 | 0 / 200 | 0.573 | 0.254 |
| 1 | 12 | 2 | 0.01 | 4 (0.28) | 8.412 [1.984, 24.351] | 1.675 [1.215, 2.577] | 0.459 | 108 (13) of 695 | 9 | 0 / 200 | 1.128 | 0.392 |
| 1 | 12 | 3 | 0.01 | 2 (0.34) | 5.187 [1.449, 27.533] | 2.249 [1.714, 3.105] | 0.172 | 164 (30) of 695 | 12 | 0 / 200 | 0.563 | 0.301 |
| 1 | 12 | 4 | 0.01 | 5 (0.52) | 4.410 [1.812, 9.584] | 1.386 [0.683, 2.712] | 0.461 | 84 (28) of 695 | 7 | 0 / 200 | 0.910 | 0.498 |
| 1 | 12 | 5 | 0.01 | 1 (0.29) | 6.469 [2.131, 10.957] | 1.762 [1.019, 2.936] | 0.694 | 70 (22) of 695 | 7 | 0 / 200 | 1.719 | 0.624 |
| 1 | 24 | 0 | 0.00 | 0 (0.67) | 9.834 [3.698, 21.774] | 4.611 [1.498, 8.345] | 1.037 | 35 (12) of 695 | 8 | 0 / 200 | 2.635 | 1.109 |
| 1 | 24 | 1 | 0.00 | 1 (0.28) | 3.642 [1.654, 6.807] | 0.919 [0.743, 1.223] | 0.361 | 181 (43) of 695 | 8 | 0 / 200 | 0.772 | 0.259 |
| 1 | 24 | 2 | 0.00 | 4 (0.24) | 6.792 [1.888, 25.951] | 1.490 [1.059, 2.595] | 0.416 | 104.5 (23) of 695 | 6 | 0 / 200 | 0.990 | 0.409 |
| 1 | 24 | 3 | 0.00 | 3 (0.32) | 7.373 [1.705, 26.127] | 2.174 [1.585, 2.815] | 0.264 | 161 (36) of 695 | 7 | 0 / 200 | 0.586 | 0.304 |
| 1 | 24 | 4 | 0.00 | 5 (0.40) | 5.040 [1.847, 11.958] | 1.206 [0.611, 2.601] | 0.511 | 88 (22) of 695 | 5 | 0 / 200 | 1.004 | 0.468 |
| 1 | 24 | 5 | 0.00 | 5 (0.25) | 6.669 [2.040, 11.704] | 1.673 [0.889, 2.991] | 0.750 | 76.5 (23) of 695 | 6 | 0 / 200 | 1.780 | 0.585 |
| 1 | 48 | 0 | 0.03 | 0 (0.67) | 10.222 [4.683, 22.173] | 4.387 [1.299, 8.112] | 1.078 | 37.5 (10) of 695 | 7 | 0 / 200 | 2.675 | 1.022 |
| 1 | 48 | 1 | 0.03 | 1 (0.34) | 4.814 [1.763, 7.805] | 0.863 [0.529, 1.276] | 0.428 | 175.5 (32) of 695 | 6 | 0 / 200 | 0.799 | 0.257 |
| 1 | 48 | 2 | 0.03 | 4 (0.27) | 7.428 [1.852, 26.362] | 1.459 [0.914, 2.253] | 0.446 | 120.5 (14) of 695 | 4 | 0 / 200 | 1.181 | 0.380 |
| 1 | 48 | 3 | 0.03 | 3 (0.34) | 11.245 [2.443, 27.172] | 2.118 [1.350, 3.252] | 0.380 | 150.5 (19) of 695 | 6 | 0 / 200 | 0.747 | 0.309 |
| 1 | 48 | 4 | 0.03 | 5 (0.36) | 5.600 [2.073, 14.418] | 1.237 [0.654, 2.824] | 0.629 | 93 (17) of 695 | 5 | 0 / 200 | 1.101 | 0.478 |
| 1 | 48 | 5 | 0.03 | 5 (0.28) | 6.248 [2.077, 11.698] | 1.748 [0.929, 3.605] | 0.724 | 77 (11) of 695 | 5 | 0 / 200 | 1.898 | 0.587 |
| 2 | 6 | 0 | 0.86 | 0 (0.94) | 202.003 [32.346, 506.139] | 90.981 [56.838, 123.255] | 0.541 | 41 (8) of 696 | 7 | 0 / 200 | 2.629 | 0.612 |
| 2 | 6 | 1 | 0.86 | 1 (0.93) | 250.356 [48.481, 617.346] | 80.865 [48.878, 115.293] | 0.931 | 220.5 (16) of 696 | 18 | 0 / 200 | 0.850 | 0.240 |
| 2 | 6 | 2 | 0.86 | 2 (0.92) | 458.299 [129.539, 950.808] | 135.725 [66.803, 222.233] | 0.512 | 121.5 (24) of 696 | 13 | 0 / 200 | 0.862 | 0.320 |
| 2 | 6 | 3 | 0.86 | 3 (0.94) | 699.078 [148.584, 1806.703] | 187.389 [120.382, 275.214] | 0.525 | 125 (19) of 696 | 13 | 0 / 200 | 1.407 | 0.361 |
| 2 | 6 | 4 | 0.86 | 4 (1.00) | 931.246 [223.866, 1680.742] | 210.928 [165.274, 284.595] | 0.308 | 144.5 (30) of 696 | 16 | 0 / 200 | 1.064 | 0.323 |
| 2 | 12 | 0 | 0.85 | 0 (0.91) | 218.712 [35.522, 533.262] | 73.719 [48.433, 117.316] | 1.103 | 49.5 (9) of 696 | 5 | 0 / 200 | 2.529 | 0.508 |
| 2 | 12 | 1 | 0.85 | 1 (0.91) | 167.704 [30.886, 413.007] | 63.828 [41.335, 98.493] | 0.603 | 199 (22) of 696 | 12 | 0 / 200 | 0.884 | 0.238 |
| 2 | 12 | 2 | 0.85 | 2 (0.93) | 586.646 [93.897, 1096.302] | 128.792 [53.418, 197.100] | 0.622 | 122 (30) of 696 | 9 | 0 / 200 | 0.813 | 0.307 |
| 2 | 12 | 3 | 0.85 | 3 (0.94) | 778.460 [151.782, 1808.986] | 161.912 [109.263, 244.704] | 0.615 | 131 (22) of 696 | 8 | 0 / 200 | 1.520 | 0.333 |
| 2 | 12 | 4 | 0.85 | 4 (1.00) | 1099.310 [326.102, 1911.934] | 190.578 [147.706, 264.626] | 0.353 | 166.5 (29) of 696 | 10 | 0 / 200 | 1.123 | 0.307 |
| 2 | 24 | 0 | 0.83 | 0 (0.88) | 209.750 [32.134, 523.327] | 66.151 [37.645, 97.777] | 0.907 | 80.5 (12) of 696 | 4 | 0 / 200 | 2.001 | 0.415 |
| 2 | 24 | 1 | 0.83 | 1 (0.86) | 230.472 [37.931, 685.717] | 60.173 [34.574, 106.521] | 0.860 | 166 (17) of 696 | 7 | 0 / 200 | 1.155 | 0.246 |
| 2 | 24 | 2 | 0.83 | 2 (0.94) | 651.735 [96.685, 1150.089] | 110.678 [40.715, 199.614] | 0.661 | 120.5 (12) of 696 | 6 | 0 / 200 | 1.024 | 0.306 |
| 2 | 24 | 3 | 0.83 | 3 (0.95) | 574.314 [123.076, 1657.280] | 153.855 [93.041, 250.480] | 0.489 | 135 (16) of 696 | 5 | 0 / 200 | 1.328 | 0.320 |
| 2 | 24 | 4 | 0.83 | 4 (0.99) | 1030.319 [195.660, 1928.629] | 189.868 [128.413, 272.346] | 0.377 | 144.5 (32) of 696 | 6 | 0 / 200 | 1.018 | 0.311 |
| 2 | 48 | 0 | 0.70 | 0 (0.84) | 200.008 [35.220, 531.492] | 61.556 [30.097, 105.932] | 0.903 | 82 (9) of 696 | 3 | 0 / 200 | 2.185 | 0.415 |
| 2 | 48 | 1 | 0.70 | 1 (0.81) | 177.449 [27.589, 569.307] | 65.775 [32.253, 112.578] | 0.756 | 148 (19) of 696 | 5 | 0 / 200 | 1.579 | 0.267 |
| 2 | 48 | 2 | 0.70 | 2 (0.83) | 719.622 [184.795, 1270.051] | 102.315 [36.195, 200.092] | 0.654 | 124 (14) of 696 | 5 | 0 / 200 | 1.144 | 0.301 |
| 2 | 48 | 3 | 0.70 | 3 (0.86) | 773.586 [176.365, 2558.343] | 151.464 [73.392, 244.083] | 0.624 | 144.5 (12) of 696 | 4 | 0 / 200 | 1.471 | 0.320 |
| 2 | 48 | 4 | 0.70 | 4 (0.97) | 1140.231 [236.729, 2416.498] | 177.783 [113.540, 252.320] | 0.466 | 142 (15) of 696 | 4 | 0 / 200 | 1.045 | 0.311 |

### 4.4 Evaporation

**Zero of 8,883 per-refit rows evaporated.** Across every scheme and replicate, no matched partner
captured zero months. The smallest occupancy seen was 8 months (classifier #2, state 0, L=6).

**This is not evidence that every state persisted, and the flag is not the whole Trap-B story.**
A K-fixed refit almost always fills all K slots, so a reference state whose months are **gone**
from the subsample still gets a non-empty partner built from other months. Two rows show it
directly. In both, `evaporated = False` while **none** of the reference state's months are in the
subsample (`ref in sub = 0`, `overlap = 0`):

- classifier #1 state 2 under LOO (its only episode, 1996-07 → 2002-05, dropped);
- classifier #2 state 4 under **drop last decade** (its only episode, 2012-09 → 2020-12, lies
  entirely inside the dropped decade) and under LOO; classifier #2 state 3 under LOO.

The `overlap` column carries this reading. The evaporation flag does not. Read the two together.

## 5. Whether the prediction held

**State 0 — the verdict held; the arithmetic behind it did not.**

- Condition (i) holds under all three calendar-order schemes (§6).
- The predicted mechanism was "the reference labeling minus the dropped months". The refit is not
  that. Under drop-first-decade, 37 of the reference's 40 crisis months remain in the subsample
  (40 − the 3-month 1970-05 episode, as predicted), but the refit's crisis partner holds **21**
  months, only 18 of them reference-crisis months, in **5** episodes, not the predicted 8.
  Under drop-last-decade, 34 remain (40 − 6, as predicted). The partner holds 25 months in **9**
  episodes, more than the predicted 7, because it picks up months the reference never labeled
  crisis.
- The predicted count "six to seven" was wrong in both directions (5 and 9). The conclusion,
  at least three, held in both.

**State 2 — the structural prediction held exactly. Whether that makes it a failure is not this
document's call.**

- Neither decade drop touches it. `ref in sub` = 71 of 71 under both drops, as predicted.
- Leave-one-episode-out is **degenerate**: 1 episode, **71 months dropped, 1996-07 → 2002-05**,
  `ref in sub = 0`, `overlap = 0`. The refit's slot 2 is filled entirely by other months, at
  winsorized distance 19.44 against a same-n null of 1.22 (ref-SD 3.12 against 0.34), and the
  ref-SD reading assigns reference state 2 to a **different** partner (1). Per the plan, this
  degeneracy is the answer to the leave-one-episode-out question, reached with no threshold.
- Under both decade drops, where all 71 of its months remain, state 2's partner still sits far
  from the reference centroid (winsorized 19.98 against a null of 2.01, and 12.59 against 2.33;
  ref-SD 1.30 against 0.32, and 0.90 against 0.41). That is a reading, not a verdict (§7).

**Not predicted: classifier #2 has two one-episode states.** State 3 (166 months, one block,
1998-11 → 2012-08) and state 4 (100 months, one block, 2012-09 → 2020-12) are each a single
contiguous episode, so leave-one-episode-out is degenerate for both. State 4 also loses **all** of
its months under drop-last-decade. ADR-0002's re-pin comment (`config/platform_settings.yaml`,
`labeling_2`) re-pinned lambda because at 4n the K=5 fit produced *"five contiguous blocks ... with
no state ever recurring"* and records that at 2n *"states recur"*. Three of five do. Two do not.
This is recorded here as a structural fact about the labeling, not as a verdict on classifier #2.

**`08-RESEARCH.md` assumption A8** (*"state 2 is the likeliest criterion-3 failure"*): the
structure it rested on is confirmed. That state 2 **fails** is not something this phase may
declare (§7).

## 6. The one verdict this phase renders — AMENDMENT condition (i), classifier #1 state 0

Computed only on schemes that preserve calendar order, where "temporally separated episodes" is
defined. The count is the matched partner's episodes in the subsample. The minimum of three is
quoted from the AMENDMENT, not chosen here.

| scheme | partner episodes in subsample | ≥ 3 (quoted)? | partner occupancy | evaporated | dist vs own-n null | ref-SD dist vs null |
|---|---|---|---|---|---|---|
| drop_first_decade | **5** | **holds** | 21 / 575 | False | 4.840 vs 5.595 (n=21) | 1.818 vs 1.605 |
| drop_last_decade | **9** | **holds** | 25 / 575 | False | 3.142 vs 5.890 (n=25) | 1.185 vs 1.336 |
| leave_one_episode_out | **9** | **holds** | 34 / 684 | False | 12.076 vs 5.981 (n=34) | 1.232 vs 1.114 |
| circular_block_bootstrap (x4) | not evaluated: resampled order is not calendar order | — | see §4.3 | False in 800 / 800 | see §4.3 | see §4.3 |

**Condition (i) is SATISFIED under criterion 3 applied directly.** Dropping the first decade leaves
the crisis state with 5 temporally separated episodes. Dropping the last decade leaves 9. Dropping
its longest episode (1973-11 → 1974-09, 11 months) leaves 9. It did not evaporate under any scheme
or replicate. Its smallest bootstrap occupancy was 10 months (L=48).

Condition (i) is not in question, so nothing is carried into `.planning/STATE.md`'s open items
under D-05. Conditions (ii)–(v) are not evaluated here.

The distances beside that verdict are readings, not part of it. Under the decade drops the
crisis partner's winsorized distance is **below** its own-n null (4.84 vs 5.60; 3.14 vs 5.89).
Under LOO it is about twice the null (12.08 vs 5.98). In ref-SD units the distances sit near the
null: 1.82 vs 1.61, 1.19 vs 1.34, and 1.23 vs 1.11.

## 7. What is NOT a verdict

Every row other than §6's is a reading, and the reading is Glenn's. **No persistence threshold
exists, none was invented, and adding one now, after the numbers, would be the tie-break shape the
pre-registration at `298b1bc` forbids elsewhere, and no more honest here.** If a threshold is ever
wanted, pre-register it before re-running this harness. The run is about four minutes, so there
is no cost argument against doing it properly.

In particular, none of these is a verdict:

- classifier #1 state 2's degenerate LOO and its large decade-drop distances;
- classifier #2 states 3 and 4 being single episodes;
- any distance above or below its null;
- the bootstrap rows, which are the weakest evidence here (8.3).

Whether any state is merged, re-pinned or left alone is a decision, and it is not taken here.

## 8. Caveats that must not be buried

**8.1 An identity assignment is not persistence, and a non-identity assignment is a finding.**
An identity Hungarian assignment means the canonical ordering held. It does **not** mean the states
persisted; the distances carry that information. A non-identity assignment means the canonical
ordering itself flipped, and **every downstream occupancy, profile and lift number is keyed on those
ids.**

- **Classifier #1's assignment is non-identity under every contiguous scheme** (drop first, drop
  last, all six LOOs).
- It is non-identity in **97–100%** of bootstrap replicates (identity fraction 0.00–0.03 across the
  four block lengths).
- Sorting on the ascending `trailing_return_1m` centroid does not reproduce the same ids on any
  subsample tried.
- Classifier #2 holds identity under both decade drops and LOO for states 0–3 (the state-4 LOO
  flips), and in 70–86% of bootstrap replicates.

**8.2 Winsorized units, and per-subsample clip bounds.** De-standardizing recovers **winsorized**
units, and the 1%/99% clip bounds also differ per subsample. For centroids over 40 or more months
the effect is second-order, but it is not zero.

**8.3 Block bootstrap is the weakest of the four schemes** for a temporally penalized model. The
jump model's DP penalises state changes **in index order**, and a block-bootstrapped series contains
synthetic seams at which the penalty fires on artefacts. Seam counts, identical by construction
for both classifiers up to one or two:

| block length L | seams per replicate (median [min, max]) — #1 | — #2 |
|---|---|---|
| 6 | 116 [114, 119] | 116 [113, 119] |
| 12 | 58 [56, 61] | 58 [56, 61] |
| 24 | 29 [27, 33] | 29 [27, 33] |
| 48 | 15 [13, 18] | 15 [13, 18] |

The Politis–White anchor n^(1/3) ≈ **8.86** months (n = 695) is cited as a variance-estimation
anchor and is explicitly **not** obeyed. Criterion 3 needs blocks long enough to preserve the
persistence the states encode, and classifier #1's median sojourn is 9.5 months while classifier
#2's is 29.0. Bootstrap rows also resample months with repetition. Duplicated rows shrink a
split-half null, so a bootstrap null is optimistic relative to a contiguous one at the same n.
Bootstrap episode counts count resampling seams, which is why condition (i) is not evaluated on
them.

**8.4 The winsorized-unit distance is nearly one-dimensional.** This was found while running the
plan. The rest of §8 was anticipated. Each column's share of the reference's total squared
between-centroid distance:

| | largest winsorized-unit shares | ref-SD-unit shares (same fit) |
|---|---|---|
| Classifier #1 | `oil` **79.5%**, `cape_shiller` **19.2%**, `real_rate_level` 1.2%, all other 7 columns ≤ 0.1% | spread across all 10: 3.4% (`curve_10y3m`) to 18.7% (`cape_shiller`) |
| Classifier #2 | `rs_equities_bonds` **100.0%**; all other 7 columns 0.0% | 0.0% (`cpi_acceleration`) to 24.6% (`m2_gdp`) |

- The primary distance therefore cannot see the columns that define classifier #1's crisis state,
  which are credit spread, realized vol and trailing return. For classifier #2 it is a
  one-dimensional distance on the equity/bond ratio.
- The cause is scale: winsorized SDs range from 0.011 (`div_yield`) to 27.5 (`oil`) for #1, and
  from 0.001 (`rs_oil_equities`) to 2,371 (`rs_equities_bonds`) for #2.
- `08-RESEARCH.md` §5.2 chose centroid distance on a benchmark with **unit-scale** columns, so the
  choice presupposed comparable scales.
- The primary distance is reported **as built**. A companion in reference-SD units is reported
  beside it. It divides both de-standardized centroids, and the null's rows, by one common scale
  (the reference fit's winsorized SD), with no refit.
- The companion is a second reading, not a replacement. No verdict in §6 depends on either
  distance.
- Both cost matrices are persisted (`units` = `winsorized` / `reference_sd`).
- Whether 08-03's distance should change is a question for a later plan, not something to settle
  here after seeing the numbers.

## 9. What this does not license

- **No re-pin of K or lambda.** (K, lambda) were passed in pinned. Nothing was selected, and the
  registry is 42 before and after. Using any number above to choose a K or lambda would make this
  run an unlogged trial after the fact.
- **No dependence statistic of any kind.** Criterion 6 is UNRESOLVED, and the pre-registration
  at `298b1bc` forbids tie-breaks. No ARI, NMI or Cramér's V is computed, referenced or implied.
  `partner_overlap_months` compares a classifier's subsample fit with its **own** reference
  labeling. It is a count, not a dependence statistic, and it touches the other classifier
  nowhere.
- **No use of any 2021+ data.** Both frames are carved at 2020-12-31 before anything is derived,
  and both references end 2020-12-31, as the identity assertion shows.
