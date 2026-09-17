---
phase: 07-regime-representation
plan: 08
subsystem: platform/labeling
tags: [adr, classifier2, regime-labeling, disjointness, honesty-framework]
status: complete
requires:
  - "07-05 (canonicalize_states sort_column parameter; features/relative.py port)"
  - "07-07 (INV-01 named survivors m2_gdp/credit_gdp; monthly_raw with fred_m2sl/fred_totalsl)"
  - "07-06 (07-DSR-ESTIMATOR-NOTE.md; total_trial_count reading the provenance header)"
provides:
  - "platform_design/adr/0002-l1-second-classifier.md (status Proposed)"
  - "platform/labeling/classifier2.py — classifier2_config, freeze_classifier2_columns, label_leadership_regimes"
  - "config labeling_2 section + allocation.blend_weight_1"
  - "regime_labels_2 / regime_confidences_2 / regime_profiles_2 checkpoints"
  - "criterion 5 — disjointness and occupancy, tested"
affects:
  - "07-09 (criterion 6 dependence — consumes regime_labels_2)"
  - "07-10 (criterion 7 joint lift — consumes blend_weight_1 and the routing contract)"
  - "07-11 (amends ADR-0002 to Accepted)"
tech-stack:
  added: []
  patterns:
    - "freeze rule reused unmodified (_reference_label_columns), never reimplemented"
    - "additive config read via cfg.get(), not added to _REQUIRED_PLATFORM_SECTIONS"
    - "4n lambda formula enforced as a raising invariant, not a comment"
    - "ordering oracle test: a decoy column at position 0 negatively correlated with sort_column"
key-files:
  created:
    - platform_design/adr/0002-l1-second-classifier.md
    - src/trading_crab_lib/platform/labeling/classifier2.py
    - tests/unit/test_platform_labeling_classifier2.py
  modified:
    - config/platform_settings.yaml
    - tests/unit/test_platform_features_relative.py
    - platform_design/adr/README.md
decisions:
  - "Classifier #2's construction pinned by a human before the fit: Lean 8, K=3, lambda=32.0, sort_column=rs_equities_bonds, L1-decision-bearing/L2-observational routing, blend_weight_1=0.50"
  - "The criterion-7 routing is recorded as a fourth option, not relabelled as routing-l1-only: the L2 leg is computed via NO_REGISTRY and firewalled from D-16/D-17/DSR by a stated constraint"
  - "Holdout carve moved inside label_leadership_regimes — T-07-17's assumed mitigation does not hold for a frame derived from monthly_raw"
  - "Classifier #2's diagnostics artifact routed to its own classifier2/ subdirectory rather than overwriting classifier #1's"
  - "OUTPUT_DIR imported via platform.labeling.diagnostics rather than trading_crab_lib, to hold the legacy-import ratchet at 31 without widening coupling"
metrics:
  duration: "~1 session (interrupted by a rate limit, resumed)"
  completed: 2026-09-17
actuals:
  tokens: 74000
  tasks: 3
  commits: 3
---

# Phase 7 Plan 08: Classifier #2 — Pinned, Recorded, Fit Summary

Classifier #2 (the leadership/relative axis, K=3, λ=32.0 on eight columns disjoint from
classifier #1's 13) is pinned in ADR-0002 at status **Proposed** before a single fit, then fit
on the live dev spine: 696 months, 1963-01-31 → 2020-12-31, occupancy
**15.3736 / 46.1207 / 38.5057 %** summing to 1.0 with error exactly 0.0 and no state below
§4.4's five-percent floor.

## Task 1 — the decision checkpoint (resolved before this execution began)

Resolved by Glenn on 2026-09-17 at the blocking `checkpoint:decision` gate. Not re-opened and
not re-asked here. Full record with every value's failure signature:
`.planning/phases/07-regime-representation/07-DECISIONS-07-08.md`. The six values, verbatim:

| # | Value | Authority |
|---|---|---|
| (a) | `rs_equities_bonds`, `rs_oil_equities`, `equities_tr_mom_12m`, `long_duration_tr_mom_12m`, `oil_mom_12m`, `corr_equities_tr_long_duration_tr_24m`, `cpi_acceleration`, `m2_gdp` — the "Lean 8", n=8 | D-10, D-11 |
| (b) | K = 3 (three asset sleeves in the candidate set) | D-13 |
| (c) | λ = 32.0 = 4n (any other value REJECTED) | D-13 |
| (d) | `sort_column` = `rs_equities_bonds` | plan 07-05's parameter |
| (e) | **L1 decision-bearing / L2 observational and firewalled** — a fourth option, not one of the plan's three ids | amendment item 4 |
| (f) | `allocation.blend_weight_1` = 0.50 | D-14 |

**Re-verified live before Task 2, not assumed:** `set(candidates) & lean_feature_set(cfg) ==
set()` with the lean set at exactly 13; every candidate's first valid month is ≤ 1963-01-31,
clearing D-11's 1972+ freeze with nine years of margin.

On (e), the contract ADR-0002 now binds: the **L1-only leg is decision-bearing** (registered
with a `trial_tag`, counts toward D-16's denominator and D-17's ceiling); the **L2 leg is
observational** (appended via the `NO_REGISTRY` sentinel, contributes zero rows, excluded from
D-16, D-17 and the deflated Sharpe); both legs must state their measurement window inline with
the number; the L1-only leg is **not comparable** to wave 1's `wealth_delta` +0.377847 /
`dd_delta` −0.066124; and **nothing downstream in this phase may change on the basis of the L2
leg**. That last clause is written as a constraint rather than assumed, for the reason the
decision record gives: `NO_REGISTRY` keeps the arithmetic honest but cannot keep the reader
honest.

## Task 2 — ADR-0002 at Proposed

`platform_design/adr/0002-l1-second-classifier.md`, written **before** the fit because D-17
requires the trial ceiling to precede the runs. Contents beyond the six pinned values:

- The four premises the WAVE-2 OPENING AMENDMENT corrected, stated as context because every
  decision was written against them.
- **Nine considered options, each argued rather than listed**: the two rejected routings;
  `BCNSDODNS` (quarterly-native, 305 obs from 1945-10, needs the forward-fill `M2SL`/`TOTALSL`
  avoid); `TOTBKCR` (starts 1973, after the 1962 spine start); market-cap/GDP (the Buffett
  indicator — **stays blocked** per D-12, restated not worked around); gold/equity relative
  strength (D-10 admits it, D-11 excludes it — `gold` starts 1985-02 while `oil` runs 1962+);
  the product (state₁, state₂) space (D-14 — 15 cells over ~590 months, thin tail below the 5%
  floor); and empirical selection of the constants, which is the option Glenn argued for.
- **The L2-window decision in its own section**, with the comparability consequence and the
  firewall clause both foregrounded.
- The **deflated-Sharpe estimator** lifted verbatim from `07-DSR-ESTIMATOR-NOTE.md`, including
  its named limitation (the registry holds **zero** usable Sharpe-bearing rows today, so the
  estimator runs entirely on its logged `1.0` placeholder) and its honest claim boundary ("a
  deflated Sharpe ratio," not "the deflated Sharpe design §22 specifies").
- **Trial ceiling as arithmetic in three separate components**, never one number:
  | Component | Arithmetic | Rows |
  |---|---|---|
  | INV-01 screen (spent) | `len(INVARIANT_CANDIDATES) = 2` | 2 (38 → 40, measured by 07-07) |
  | Criterion 7 runs (not spent) | `2 rows/run × 2 runs` — the factor **re-derived** for this routing from `run_full_backtest_evaluation`'s two `append_trial` sites, not inherited from ADR-0001 | ≤ 4 |
  | L2 observational leg | `NO_REGISTRY` | 0 |
  **Live reading: `total_trial_count()` = 40, read 2026-09-17T16:42:52Z.** Implied ceiling for
  the rest of the phase: **44**. ADR-0001's standing warning restated and extended: 30, 34, 35,
  38 and 42 are all stale and none may be used as a ceiling value.
- **INV-01 coverage**: named survivors `["m2_gdp", "credit_gdp"]`, of which only `m2_gdp` enters
  the frozen eight (`credit_gdp` dropped at 0.957–0.969 correlation); R4 held structurally
  (`pca.transform()` is never called, so no component score exists to be mistaken for a
  feature); era-stability tolerance 0.15 over 10 eras — **with plan 07-07 §4's own caveat
  carried forward rather than upgraded**, that a PC1 loading of exactly 1/√2 on two standardized
  candidates is an arithmetic identity, not evidence of five-decade stability.
- **The two standing objections named where the pinning is justified**, so a reader sees that
  pinning is a current-capability constraint and not a claim that selection is wrong: ROADMAP
  **T0.9** (registered nested selection inside the walk-forward loop) against (a)–(c), and
  ROADMAP **T0.10** (learned regime-conditional allocation) against (f), including the finding
  that T0.10 cannot be expressed today because **there is no learner at L3 or L4** —
  `returns_by_regime_stats` is descriptive statistics and `regime_tilt_weights` is a
  probability-weighted average of it, with no coefficients anywhere.
- **The eight probe edges, all resolved, none deferred** — REG-01's three (each marked
  re-opened and re-resolved by classifier #2, `ordering` substantively so) and INV-01's five,
  each naming the plan and test that closed it.

Config: `labeling_2` and `allocation.blend_weight_1` added, both read via `cfg.get()`, neither
added to `_REQUIRED_PLATFORM_SECTIONS`. `labeling` and `taxonomy` untouched;
`lean_feature_set()` still returns exactly 13 and `validate_platform_config()` still passes.

## Task 3 — the fit

`src/trading_crab_lib/platform/labeling/classifier2.py`. Mirrors
`labeling/diagnostics.py::label_regimes`'s wiring order and **reuses
`evaluation/report.py::_reference_label_columns` unmodified** as D-11's freeze rule — a test
asserts the frozen list equals that function called directly on the same inputs, so a second
implementation cannot creep in unnoticed.

### Frozen column list, resolved live

All **eight** candidates survived; **zero** exclusions:

| Column | First valid | Survived the 1972-01-31 freeze |
|---|---|---|
| `rs_equities_bonds` | 1962-01-31 | yes |
| `rs_oil_equities` | 1962-01-31 | yes |
| `equities_tr_mom_12m` | 1963-01-31 | yes |
| `long_duration_tr_mom_12m` | 1963-01-31 | yes |
| `oil_mom_12m` | 1963-01-31 | yes |
| `corr_equities_tr_long_duration_tr_24m` | 1963-01-31 | yes |
| `cpi_acceleration` | 1962-04-30 | yes |
| `m2_gdp` | 1962-02-28 | yes |

**Intersection with classifier #1's 13 lean columns: 0** (empty set), with the lean set
independently asserted at exactly 13 members so the result cannot be obtained by shrinking the
denominator.

### Live run — every number with its window

- **Window: 1963-01-31 → 2020-12-31, 696 months.** First decision date **1972-01-31**, derived
  as `dev_features.index[min_train_months=120]`, matching classifier #1's own derivation.
- **68 post-2020-12-31 rows carved out of the fit** (HON-01/T-07-17), logged.
- **Occupancy (696 months, 1963-01 → 2020-12):** state 0 **15.3736 %**, state 1 **46.1207 %**,
  state 2 **38.5057 %**. Sum error `|Σ − 1.0| = 0.0`, exactly K=3 entries.
- **No state below §4.4's five-percent floor.**
- Median sojourns: 107.0 / 160.5 / 268.0 months. Auto-profiles: state 0 *high rs_oil_equities,
  low rs_equities_bonds, low m2_gdp*; state 1 *high corr_equities_tr_long_duration_tr_24m, low
  m2_gdp, low rs_equities_bonds*; state 2 *high m2_gdp, low
  corr_equities_tr_long_duration_tr_24m, high rs_equities_bonds*.

### ⚠ Finding: the labeling is far MORE persistent than the failure signature anticipated

Recorded, **not acted on**. Over 696 months classifier #2 produces **3 change points** — runs
of 132, 107, 189 and 268 months, transitioning at **1974-01-31**, **1982-12-31** and
**1998-09-30**. Four regimes in 58 years; the median sojourn is 8 to 22 years.

The decision record's failure signature for (a) predicted the opposite direction: *"the fitted
states will flip far more often than classifier #1's — visible as a change-point count well
above #1's."* The observed behaviour is the other extreme. That the prediction missed its
direction does not make the observation benign: a leadership label that changes roughly once
every fifteen years is of limited use to a monthly-rebalanced tilt, whatever its dependence or
lift turns out to be, and plans 07-09 and 07-10 should read their numbers with that in view.

**Why nothing was changed in response.** Re-fitting at a lower λ to get more transitions is
precisely the selection trial D-13 forbids, and λ = 4n is pinned by construction in ADR-0002 —
changing it is an amendment, not an edit. The observation is reported for plan 07-11's
acceptance decision to weigh.

**One comparison could not be made honestly.** Classifier #1's `regime_labels` checkpoint is
**not on disk** in this working tree, so its change-point count could not be recounted live for
a like-for-like comparison. ADR-0001's recorded §5.4 figure (7 transitions, 7 of 7 resolved) was
measured over the **narrowed 356-month window ending 2017-05**, not over 696 months, so quoting
"3 vs 7" as a comparison would be comparing two different windows — exactly the error the
routing decision exists to prevent. The count is reported on its own window and left
uncompared.

### Tests

**25 tests** in `tests/unit/test_platform_labeling_classifier2.py` and **4** in
`tests/unit/test_platform_features_relative.py::TestClassifier2Disjointness` (29 new).

**Every test was mutation-checked against a deliberately broken implementation.** This project's
documented core failure mode is the evidence-shape failure — a check that can only confirm and
never fail; criterion 8 in this very phase was a "Verified" exit check whose grep discarded all
31 real violations. So each guard was verified to fail:

| Mutation | Tests that failed | Caught? |
|---|---|---|
| `canonicalize_states(..., sort_column=frozen[0])` — the exact A14 centroid-column-0 fallback | `test_states_are_numbered_by_ascending_sort_column_centroid` only | yes, precisely |
| drop the `sort_column=` keyword entirely | 9 tests (default `trailing_return_1m` is absent → raises) | yes |
| `occupancy_and_sojourns(states)` — drop `n_states=K` | `test_never_occupied_state_is_a_zero_entry_not_a_missing_key`, `test_below_floor_state_warns_naming_that_state_and_still_returns` | yes, exactly those two |
| disable the `lambda != 4n` check | `test_lambda_not_four_times_feature_count_raises`, `test_lambda_off_by_one_column_raises` | yes |
| remove the holdout carve | `test_post_holdout_months_are_carved_before_the_fit` | yes |
| swap a candidate for `realized_vol_1m` in the config | both disjointness assertions, printing `collides ... on: ['realized_vol_1m']` | yes |

The ordering oracle is the load-bearing one: its feature list places a `decoy` column at
position 0 that is the exact negative of the sort column, so an implementation ordering on
centroid column 0 produces the reversed numbering and fails. The sub-floor fixture collapses the
fit through **constant data** rather than a λ override, so the pinned λ = 4n stays honest inside
the test rather than needing a test-only escape hatch.

## Deviations from Plan

**1. [Rule 2 — missing critical functionality] The holdout carve was moved inside
`label_leadership_regimes`.**
- **Found during:** Task 3, checking T-07-17's stated mitigation against the data.
- **Issue:** T-07-17's mitigation reads "the fit runs on the dev-side `monthly_features`
  checkpoint, which `honesty/holdout.py` already carves." That assumption does not hold.
  `monthly_features` (708 × 53) carries **none** of classifier #2's eight candidate columns —
  they are derived on demand from `monthly_raw`, which runs to **2026-08-31** on disk. A caller
  passing `add_relative_features(monthly_raw, cfg)` straight in would have fit on 68 post-cutoff
  months.
- **Fix:** `split_by_holdout_boundary(..., cutoff=DEFAULT_HOLDOUT_CUTOFF)` applied inside the
  function, dropped rows logged, pinned by `test_post_holdout_months_are_carved_before_the_fit`
  (which fails when the carve is removed).
- **Commit:** `351eee8`

**2. [Rule 1 — bug] The diagnostics artifact would have overwritten classifier #1's.**
- **Found during:** Task 3, reading `report_labeling_diagnostics`'s default path.
- **Issue:** the plan says to report through `report_labeling_diagnostics` so the five-percent
  floor WARNING comes from the same code path as classifier #1. Its default output directory
  writes one shared `labeling_diagnostics.parquet`, so calling it verbatim would have
  overwritten classifier #1's artifact — T-07-15's spoofing failure mode applied to the report
  rather than the checkpoint.
- **Fix:** same function, same WARNING path, routed to
  `outputs/reports/model_metrics/classifier2/`.
- **Commit:** `351eee8`

**3. [Rule 3 — blocking issue] `from trading_crab_lib import OUTPUT_DIR` raised the
legacy-import ratchet to 32.**
- **Found during:** Task 3 verification. The ratchet test failed naming the new site.
- **Fix:** import `OUTPUT_DIR` from `platform.labeling.diagnostics`, which already owns that
  seam as one of the 31 ratcheted sites. The coupling does not widen — this module already
  depends on `diagnostics.py` for four functions, so nothing new is reachable from `platform/`
  — and the module docstring states the manoeuvre explicitly rather than leaving it to be
  discovered. **Ratchet still 31 and unedited.**
- **Commit:** `351eee8`

**4. [Recorded, not a fix] Task 3's precondition was partly unmet, with a narrower path
available.**
- The precondition reads "the live `monthly_features` checkpoint carries classifier #2's
  candidate columns." It does not, and no plan in wave 2 wires them into it. Rather than treat
  that as a blocker, the fit was run on `add_relative_features(monthly_raw, cfg)` — the exact
  data path plan 07-05's own SUMMARY describes as "the raw candidate columns classifier #2's fit
  will consume." Recorded here so plan 07-09 knows where classifier #2's features come from and
  does not go looking for them in `monthly_features`.

**5. [Recorded] `outputs/reports/model_metrics/` left untracked, deliberately.**
- `.gitignore`'s `outputs/*` rules are all commented out and `git check-ignore -v` returns
  nothing for that path, so it is an ordinary untracked path, not an ignored one (verified
  directly, twice). It is left untracked because no plan claims it as an artifact and because
  classifier #1's own twin artifact at `outputs/reports/model_metrics/labeling_diagnostics.parquet`
  has never been tracked in this repo either — `git ls-files outputs/` lists only
  `outputs/reports/platform/*`. Committing #2's while #1's stays untracked would invent an
  inconsistency. The numbers it holds are in this SUMMARY and in the ADR.

No architectural deviation (Rule 4) arose; the six pinned values were not re-opened.

## Verification

| Check | Result |
|---|---|
| ADR-0002 exists at Proposed, all required tokens present | PASS |
| Probe-edge table: exactly 8 rows, none `deferred` | PASS (verified programmatically) |
| Trial ceiling: three arithmetic components + live reading with timestamp | PASS (40 @ 2026-09-17T16:42:52Z, ≥ 38) |
| `labeling_2.lambda == 4 × len(features)` | PASS (32.0 = 4 × 8) |
| `lean_feature_set(load_platform_config())` | 13 |
| `validate_platform_config()` | PASS |
| `pytest tests/unit/test_platform_labeling_classifier2.py -q` | 25 passed |
| `pytest tests/unit/test_platform_features_relative.py -k Disjoint -q` | 5 passed (4 new + 1 pre-existing) |
| Plan's verify command (4 files) | 99 passed |
| `MAX_LEGACY_IMPORT_SITES` | 31, unedited |
| Live occupancy sums to 1.0 with its window recorded | PASS (error 0.0, 1963-01 → 2020-12) |
| `ruff check` on all three changed/new source files | All checks passed |
| **Full suite** | **1873 passed, 0 skipped** (baseline 1844; +29 = 25 + 4 new tests) |

## Commits

| Commit | Task | What |
|---|---|---|
| `196cd6b` | 2 | ADR-0002 at Proposed; `labeling_2` + `allocation.blend_weight_1`; ADR index row |
| `ea83dc2` | 3 (RED) | 29 failing tests — collection fails, module absent |
| `351eee8` | 3 (GREEN) | `classifier2.py` + the live fit's three checkpoints |

## Next Phase Readiness

- Plan **07-09** can measure criterion 6's dependence: `regime_labels_2` is on disk (696 months,
  1963-01 → 2020-12) alongside the frozen column list. Note it must reindex onto the months
  both labelings share, and classifier #1's `regime_labels` checkpoint is **not currently on
  disk** — it needs regenerating or reading from the evaluation's smoothed reference.
- Plan **07-10** has `allocation.blend_weight_1 = 0.50` and the routing contract in writing,
  including the ≤ 4 registry rows its two runs are budgeted.
- Plan **07-11** amends ADR-0002 to Accepted. It should weigh the persistence finding above —
  3 change points in 696 months — as part of that decision.
- **Blocker for none of the above**, but carried forward: nothing in wave 2 wires classifier
  #2's candidate columns into `monthly_features`. Every consumer must derive them via
  `add_relative_features(monthly_raw, cfg)`.

---
*Phase: 07-regime-representation*
*Completed: 2026-09-17*

## Self-Check: PASSED

All 5 named artifacts verified present on disk (ADR-0002, classifier2.py, the new test file,
this SUMMARY, the regime_labels_2 checkpoint). All 3 task commits (`196cd6b`, `ea83dc2`,
`351eee8`) verified present in `git log --oneline --all`.
