---
phase: 08-regime-persistence-stability
plan: 03
subsystem: ml-platform
tags: [regime-stability, hungarian-matching, subsample-stability, honesty-framework, jump-model]

# Dependency graph
requires:
  - phase: 07-regime-representation
    provides: "canonicalize_states' no-fallback sort_column (07-05), the frozen ten-column
      feature policy (D-02-A), the report-only/no-gate posture (D-07), the legacy-import
      ratchet at 31"
  - phase: 03-labeling
    provides: "jump_model.standardize_features / _recompute_centroids / canonicalize_states,
      diagnostics.occupancy_and_sojourns"
provides:
  - "labeling/stability.py — standardization_params / destandardize_centroids /
    winsorized_frame / fit_for_stability: a common unit space in which a subsample
    centroid and a full-sample centroid are comparable at all (Trap A)"
  - "match_states — Hungarian assignment on de-standardized centroid distance, returning
    the full K x K cost matrix, per-state matched distance, per-state margin and
    is_identity"
  - "split_half_null — the within-state, n-matched yardstick computed from this project's
    own data; no synthetic constant is hard-coded"
  - "state_episodes / stability_row — episode spans and the seven-field record whose
    `evaporated` flag is built from occupancy alone and outranks the distance (Trap B)"
  - "four subsample schemes (drop first decade, drop last decade, circular block
    bootstrap with a required seam count, leave-one-episode-out) plus run_stability,
    which raises on any column-set difference (Trap C)"
affects: [08-07 (runs criterion 3 and writes the verdict), platform/labeling]

# Actuals (#2632)
actuals:
  tokens: 32000
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Round-trip identity pin: a re-derived inverse is asserted elementwise against the
      function it inverts, plus a companion test proving the pin CAN fail (a ddof=1
      inverse does not satisfy it). The pin doubles as a tripwire on the inverted
      function."
    - "Two-sided trap test: assert the correct path AND that the naive path is wrong,
      with the wrong value named in the assertion message. A test that only exercises
      the correct path passes equally on a broken implementation."
    - "Flag-before-verdict field ordering: a discriminator derived from a primary
      observable (occupancy) is constructed before any field derived from a secondary
      one (distance), so no consumer can read the secondary without the primary."
    - "Cross-check, don't duplicate: the episode scan is pinned against the existing
      occupancy_and_sojourns run-length machinery and raises on disagreement rather than
      becoming a second, silently divergent scanner."
    - "Mutation-verified tests: each named trap was re-checked by deliberately breaking
      the implementation and confirming the corresponding test fails."

key-files:
  created:
    - src/trading_crab_lib/platform/labeling/stability.py
    - tests/unit/test_platform_labeling_stability.py
  modified: []

key-decisions:
  - "match_states ships in Task 1's commit, not Task 2's. Task 1's Trap A test asserts on
    a MATCHED distance (the plan's own key_links end '-> destandardize_centroids -> a
    distance both sides share'), so the tracer slice is not end-to-end without it. The
    null, episodes and the row record remain Task 2."
  - "Centroid distance is the only distance implemented. Gaussian-W2 was left out
    entirely rather than shipped as a labelled secondary: it halves the separation ratio
    at n=40 (1.22x vs 1.70x) and needs a covariance that a 40-month state in d=10 cannot
    condition. wasserstein_distance_nd is named in the docstring with its three measured
    ratios and is never imported or called."
  - "EVAPORATED_OCCUPANCY_MONTHS = 0 — zero months exactly, not a fraction. A state
    holding 1-3 months in a subsample is not flagged; its occupancy is carried and the
    reader sees it. A fractional threshold would be a persistence threshold invented
    after the fact."
  - "The module holds exactly four constants (EVAPORATED_OCCUPANCY_MONTHS,
    BLOCK_LENGTH_LADDER, DEFAULT_STABILITY_SEED, DECADE_MONTHS) and a test enumerates
    them, so a fifth — e.g. a persistence threshold — fails CI rather than arriving
    unremarked."
  - "run_stability enforces the frozen column list at three points (input frame, the
    subsample's all-NaN columns, the fitted column list) rather than one, because Trap
    C's realistic channel is a column going entirely NaN over a window, not a caller
    handing over the wrong frame."
  - "standardize_features itself is NOT modified to return its scaler. Its signature is
    used by label_regimes, _refit_l1, _refit_classifier2 and label_leadership_regimes;
    the (center, scale) pair is recomputed here and held in step by the pin test."

requirements-completed: [PER-06]

coverage:
  - id: D1
    description: "standardization_params inverts standardize_features elementwise under a
      pin test that fails if either drifts; fit_for_stability returns both centroid
      spaces plus a K-length occupancy array including zeros."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestStandardizationInversePin (3 tests)"
        status: pass
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestFitForStability (4 tests)"
        status: pass
      - kind: other
        ref: "mutation: destandardize_centroids replaced with a no-op -> 2 failures"
        status: pass
    human_judgment: false
  - id: D2
    description: "Trap A has a test that fails on the trap: the de-standardized matched
      distance is asserted correct AND the naive standardized comparison is asserted
      wrong, with the wrong value named."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestTrapAUnitSpace"
        status: pass
    human_judgment: false
  - id: D3
    description: "Hungarian matching recovers a known non-identity permutation exactly;
      margins near 1.0 are surfaced even when matched distances are tiny."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestHungarianMatching (3 tests)"
        status: pass
      - kind: other
        ref: "mutation: linear_sum_assignment replaced with an identity -> permutation test fails"
        status: pass
    human_judgment: false
  - id: D4
    description: "The split-half null is materially non-zero at n=40 and falls as n rises;
      it is computed from project data with no hard-coded synthetic constant."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestSplitHalfNull (3 tests)"
        status: pass
      - kind: other
        ref: "mutation: null computed as a distance-to-self -> 3 failures"
        status: pass
    human_judgment: false
  - id: D5
    description: "Trap B: a zero-occupancy state is flagged evaporated=True while its
      matched distance is exactly 0.0, exercised through _recompute_centroids' own freeze
      rule; one WARNING is logged per evaporated state."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestTrapBEvaporation (3 tests)"
        status: pass
      - kind: other
        ref: "mutation: evaporated derived from matched_distance -> 2 failures"
        status: pass
    human_judgment: false
  - id: D6
    description: "Four subsample schemes exist; circular bootstrap preserves length n for
      every ladder value and returns a seam count pinned at both ends of the block-length
      range and against an independent calendar-adjacency formulation."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestContiguousSchemes, ::TestCircularBlockBootstrap, ::TestLeaveOneEpisodeOut (12 tests)"
        status: pass
      - kind: other
        ref: "mutation: n_seams = n_blocks; n_seams = 0; non-circular truncating draw; global RNG -> each fails"
        status: pass
    human_judgment: false
  - id: D7
    description: "Trap C: run_stability raises on any column-set difference, with the
      surviving-vs-expected counts in the message."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestTrapCFrozenColumns (2 tests)"
        status: pass
      - kind: other
        ref: "mutation: frozen-column check short-circuited -> 2 failures"
        status: pass
    human_judgment: false
  - id: D8
    description: "No threshold constant, no verdict field, no Wasserstein estimator in
      live code, and no artifact written."
    requirement: PER-06
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_labeling_stability.py::TestNoThresholdAndNoVerdict (2 tests)"
        status: pass
      - kind: other
        ref: "regex scan for an import of or a call to wasserstein_distance_nd -> none"
        status: pass
    human_judgment: false

duration: ~4min (commit-to-commit span; excludes reading/research and mutation-check time)
completed: 2026-09-22
status: complete
---

# Phase 8 Plan 03: Subsample Stability Machinery for §4.4 Criterion 3 Summary

**Built the Hungarian-matching, de-standardized-centroid, split-half-null and
four-subsample-scheme machinery criterion 3 needs — and built its two silent-failure
modes as tests that fail on the trap rather than as footnotes that describe it.**

## Performance

- **Duration:** ~4 min (span between first and last task commit)
- **Started:** 2026-09-22T16:08:35Z (first commit)
- **Completed:** 2026-09-22T16:10:00Z (last commit)
- **Tasks:** 3/3 completed
- **Files:** 2 created, 0 modified (824 + 672 lines)
- **Tests added:** 40, all passing, 0 skipped

## Public Surface

`src/trading_crab_lib/platform/labeling/stability.py` — pure functions of a fit; the fit
itself, `jump_model.py`, `label_regimes` and the pinned (K, lambda) are never touched.

| Function | Returns |
|---|---|
| `standardization_params(X)` | `{"winsor_lower", "winsor_upper", "center", "scale"}` — the three quantities `standardize_features` fits and then discards |
| `destandardize_centroids(centroids, params, columns)` | `pd.DataFrame` of centroids in winsorized feature units |
| `winsorized_frame(X, params)` | the de-standardized ROW space, so a centroid and the rows it averages are comparable |
| `fit_for_stability(X_df, *, K, lam, n_restarts, sort_column, random_state)` | `StabilityFit` — `states`, `centroids_standardized`, `centroids_destandardized`, `columns`, `params`, `occupancy` (length K, zeros included), `rows_destandardized` |
| `match_states(ref_destd, sub_destd)` | `assignment`, `cost_matrix` (full K x K, persisted), `matched_distance`, `margin`, `is_identity` |
| `split_half_null(rows, *, n_reps, seed)` | `median`, `p10`, `p90`, `n`, `n_reps` |
| `state_episodes(states, *, n_states)` | per state: `episodes` (start/end/length + index labels), `n_episodes`, `longest_episode`, `months` |
| `stability_row(...)` | the §5.5 seven-field record plus `evaporated` |
| `scheme_drop_first_decade(index, *, months)` | positional indices |
| `scheme_drop_last_decade(index, *, months)` | positional indices |
| `scheme_circular_block_bootstrap(index, block_length, *, seed)` | `(positions, n_seams)` |
| `scheme_leave_one_episode_out(states, state_id, *, n_states)` | `positions`, `mask`, `degenerate`, `n_episodes_before`, `n_months_dropped` |
| `run_stability(X_df, *, K, lam, n_restarts, sort_column, reference_fit, schemes, ...)` | `list[stability_row]` |

Module constants: `EVAPORATED_OCCUPANCY_MONTHS = 0`, `BLOCK_LENGTH_LADDER = (6, 12, 24, 48)`,
`DEFAULT_STABILITY_SEED = 20260921`, `DECADE_MONTHS = 120`. **There are exactly four**, and
`TestNoThresholdAndNoVerdict::test_the_module_defines_no_persistence_threshold_and_emits_no_verdict`
enumerates them — a fifth constant fails CI rather than arriving unremarked.

## Task Commits

Each task was committed atomically, scoped to the two files in `files_modified`:

1. **Task 1 (tracer): a common unit space, inverse pinned against the function it inverts** — `8e22c3f` (feat)
2. **Task 2: Hungarian matching, the split-half null, and the evaporation flag** — `12fc234` (feat)
3. **Task 3: four subsample schemes with the frozen column list held fixed** — `87f7f7d` (feat)

The tracer feedback gate was honoured: Task 1's `<verify>` (the inverse pin, the
round-trip and Trap A) was run and green before any Task 2 code was written, and the
module's `__main__` self-check ran end-to-end at that point.

## The Three Trap Tests, and What a Broken Implementation Would See

These are the numbers that make each test falsifiable rather than decorative. Every one
was confirmed by deliberately breaking the implementation and re-running.

### Trap A — the discarded scaler

`standardize_features` refits its winsorization bounds AND its `StandardScaler` on
whatever rows it is handed (`jump_model.py:112-124`), so two fits on differently-scaled
data have **identical** standardized centroids. The fixture is `X_sub = 0.5 * X_ref`:
the same data at half the scale, so the true centroid geometry differs by a factor of two.

| quantity | value |
|---|---|
| true centroid gap `g` (shift 3.0 on d=10) | **9.4868** |
| de-standardized matched distance for the high state | **4.7303** (= 0.5·g ✓) |
| **naive `centroids_standardized` matched distance** | **0.000e+00** |

A naive implementation reports **exactly zero** — "perfectly stable" — on two fits whose
geometry differs by 2x. The test asserts both halves, and names `0.000e+00` against
`4.7303` in the failure message. `test_inverse_pin_discriminates_a_wrong_ddof` additionally
proves the inverse pin itself can fail: a `ddof=1` inverse misses by >1e-6.

*Mutation check:* `destandardize_centroids` replaced with `return pd.DataFrame(arr)` →
`TestTrapAUnitSpace` and `TestDestandardizeCentroids` both fail.

### Trap B — the frozen zero-occupancy centroid

`_recompute_centroids` freezes a zero-occupancy state at its previous centroid
(`jump_model.py:126-138`). The test builds a decode in which state 2 captured zero months,
calls `_recompute_centroids` directly, asserts the freeze happened, then matches.

| quantity | value |
|---|---|
| matched distance for the evaporated state 2 | **0.0 exactly** |
| matched distances for all three states | `[0.0, 3.1391, 0.0]` |
| subsample occupancy for state 2 | **0 months** |

A reader — or an implementation — scoring "small distance ⇒ stable" marks state 2 as the
*most* stable state in the table, tied with state 0, while it has vanished entirely. The
test asserts `matched_distance < 1e-12` **and** `evaporated is True` **and**
`occupancy_months == 0`, with the message stating that the zero means the centroid was
frozen, not that the state persisted. `evaporated` is constructed from occupancy alone,
before any other field of the row, and a WARNING naming classifier/scheme/state is logged.

*Mutation check:* `evaporated = bool(matched_distance > 1.0)` → 2 failures.

### Trap C — feature-set churn measured in place of state stability

`run_stability` raises, with the counts in the message:

```
Trap C [input frame]: subsample column set differs from the reference fit's frozen list
— 9 surviving vs 10 expected (missing=['gold'], extra=[], reordered=False). ...
```

Without the raise, the run completes and every number in the table is a comparison
between a 10-column reference fit and a 9-column subsample fit — a measurement of the
feature set, not of the states. Two tests trigger it: a frame missing a column, and the
realistic channel of a column going entirely NaN within the subsample window.

*Mutation check:* the column check short-circuited to `return` → 2 failures.

### The null is not zero

| n | split-half null median (d=10, standard normal) |
|---|---|
| 40 | **0.8555** |
| 400 | **0.2959** |

A null computed as a distance-to-self returns ~0 and makes every matched distance look
significant. The tests assert `median > 0.3` at n=40 and assert the null **falls** from
n=40 to n=400 on the same distribution.

*Mutation check:* `b = arr[perm[:half]].mean(...)` (comparing a half to itself) → 3 failures.

### Seams, specifically

`n_seams` was the one place the first draft of the tests was **inadequate** and was
strengthened before commit. `assert n_seams in {0, 1}` for a full-length block and
`n_seams >= n-6` for unit blocks are both satisfied by the broken `n_seams = n_blocks`
(1 block → 1; 300 blocks → 300). The mutation check caught this. The tests now:

- pin the **zero** case: a full-length block whose draw starts at position 0 is the
  original series in order and must report **exactly 0** seams (`n_blocks` would say 1);
- cap unit blocks at `n-1`, the number of adjacent pairs that exist (`n_blocks` = `n` = 300 > 299);
- pin the count against an **independent** formulation — counting resampled month labels
  that are not consecutive months in the original calendar — at block lengths 3, 7 and 24.

*Mutation checks:* `n_seams = n_blocks` → 3 failures; `n_seams = 0` → 3 failures; a
non-circular truncating draw → 1 failure; a global RNG → the determinism test fails.

## The Block-Length Ladder

`BLOCK_LENGTH_LADDER = (6, 12, 24, 48)` months, with the rationale in the constant's own
comment:

- **The anchor is quoted, not obeyed.** Politis & White (2004), corrected by Patton,
  Politis & White (2009), give a data-driven optimal block length of `O(n^(1/3))` —
  ≈ **8.9 months** at n=695. It minimises the asymptotic MSE of a long-run-variance
  estimate. That is a different objective from preserving persistence.
- **Obeying it would destroy the thing being tested.** Classifier #1's median sojourn is
  **9.5 months**; classifier #2's is **29.0**. A ~9-month block sits on #1's median and
  under a third of #2's, so every state would "evaporate" for a reason that has nothing
  to do with whether it is a regime.
- **The ladder brackets both medians** (6 < 9.5 < 48 and 6 < 29.0 < 48) and is reported
  as a range. A result that holds across it is strong; one that flips across it is itself
  the finding. A test pins both bracketing facts.
- Block bootstrap is documented in `scheme_circular_block_bootstrap`'s own docstring as
  the **weakest** of the four schemes for a temporally penalized model specifically, and
  `n_seams` is a required return value so a reader can discount by how many synthetic
  seams the penalty fired on.

## No Threshold Constant Exists in the Module — Confirmed

- The module defines exactly four constants, enumerated by a test:
  `EVAPORATED_OCCUPANCY_MONTHS`, `BLOCK_LENGTH_LADDER`, `DEFAULT_STABILITY_SEED`,
  `DECADE_MONTHS`. None is a persistence threshold.
- `EVAPORATED_OCCUPANCY_MONTHS = 0` is **zero months exactly**, not a fraction of the
  subsample. A state with 1-3 months is not flagged; its occupancy is carried and the
  reader sees it. "Near-zero" is a judgement and this module makes none.
- `stability_row` emits no `verdict`, `passed`, `stable`, `persists` or `is_regime` field;
  a test asserts each of those keys is absent.
- No `outputs/` artifact is written, no record is made, and nothing is added to the trial
  ledger. Running criterion 3 and rendering the verdict are **08-07's**, deliberately, so
  the guards are reviewable before they produce a number anyone will cite.
- The only pass/fail the machinery supports is §4.4 AMENDMENT condition (i) — "at least
  three temporally separated episodes" — which a human applies in 08-07 by reading the
  `n_episodes` column. `state_episodes` and `scheme_leave_one_episode_out` supply that
  quantity; neither compares it to anything.

## Deviations from Plan

1. **Task 1's second `<verify>` was already revised in the plan** from
   `python -m trading_crab_lib.platform.labeling.stability` to an import-and-call check,
   per the orchestrator's note. The revised check was run and passes. `stability.py`
   nonetheless carries the `__main__` synthetic self-check footer Task 1's `<action>`
   asks for (it is a debugging aid, consistent with `diagnostics.py`'s own footer); no
   runnable entry point was added to a library module and nothing imports it as one.
2. **`match_states` ships in Task 1's commit rather than Task 2's.** Task 1's Trap A test
   asserts on a *matched distance*, and Task 1's own `key_links` end with "-> a distance
   both sides share", so the tracer slice is not end-to-end without it. Task 2 still owns
   the null, the episodes and the row record. No behaviour was moved, only the commit
   boundary.
3. **`winsorized_frame` is an extra public helper** not named in the plan. It exists
   because `split_half_null` must receive rows in the *same* de-standardized space as the
   centroids, and making that space a named function rather than an inline `.clip(...)`
   is what lets `TestDestandardizeCentroids` assert that a de-standardized centroid equals
   the mean of its state's de-standardized rows.
4. **The frozen-column check runs at three points, not one.** The plan specifies a raise
   on "any subsample's column set". Trap C's realistic channel is a column going entirely
   NaN over a window (a subsample cannot change the *frame's* columns), so the check also
   runs on the subsample's non-all-NaN columns and on the fitted column list.
5. **The seam tests were strengthened after a failed mutation check** (see above). This
   is not a deviation from the plan — the plan explicitly requires the test to fail if
   seams "are counted as blocks" — but the first draft did not meet that bar and is
   recorded here because the gap is exactly the defect shape this phase exists to catch.

No architectural changes, no package installs, no `legacy/` or submodule edits, no
post-2020-12 data read, no secret written.

## Issues Encountered

None blocking. Two self-inflicted issues, both caught before commit:

- The first draft of `state_episodes`' test asserted an exact dict equality that did not
  account for the carried `start_label`/`end_label` — fixed to assert the span tuple and
  the labels separately.
- The first draft of the seam tests was satisfied by `n_seams = n_blocks` (above).

## Verification Results

Plan `<verify>` blocks, all run:

```
# Task 1
pytest tests/unit/test_platform_labeling_stability.py -q -k "destandard or inverse or trap_a"
=> 6 passed, 34 deselected

python -c "<import-and-call public-surface check>"
=> stability imports clean; public surface: ['destandardize_centroids',
   'fit_for_stability', 'match_states', 'run_stability',
   'scheme_circular_block_bootstrap', 'scheme_drop_first_decade',
   'scheme_drop_last_decade', 'scheme_leave_one_episode_out', 'split_half_null',
   'stability_row', 'standardization_params', 'state_episodes', 'winsorized_frame']

# Task 2
pytest tests/unit/test_platform_labeling_stability.py -q -k "permutation or evaporat or null or margin"
=> 9 passed, 31 deselected

python -c "<regex scan for an import of / call to wasserstein_distance_nd>"
=> rejected estimator is named in prose only, never imported or called

# Task 3
pytest tests/unit/test_platform_labeling_stability.py -q
=> 40 passed

python -c "<BLOCK_LENGTH_LADDER length preservation at n=695>"
=> 6 695 / 12 695 / 24 695 / 48 695   (48 does not divide 695; length still n)
```

Phase `<verification>` block:

```
pytest tests/unit/test_platform_labeling_stability.py -q      => 40 passed, 0 skipped
pytest tests/unit/test_platform_legacy_import_ratchet.py -q   => 11 passed (ratchet still 31)
python -c "... total_trial_count()"                            => 42, unchanged
ruff check <both files>                                        => All checks passed!
flake8 --select=E9,F63,F7,F82 <both files>                     => clean
python -m trading_crab_lib.platform.labeling.stability         => self-check ok
```

**Legacy-import ratchet:** unchanged at **31**. `stability.py`'s only intra-project
imports are `trading_crab_lib.platform.labeling.jump_model` and
`trading_crab_lib.platform.labeling.diagnostics`; `scipy.optimize.linear_sum_assignment`
is a third-party import and is not counted by the AST scan
(`test_platform_legacy_import_ratchet.py:25-35`).

**Trial registry:** **42**, unchanged. Nothing here writes to the ledger; re-fitting at a
pinned (K, lambda) selects nothing.

**Full suite:** `pytest tests/ -q` against the shared working tree →
**2115 passed, 13 skipped, 0 failed** (1201 s). This run includes all 40 of this plan's
tests and the concurrent siblings' work as it stood when the run started. The 13 skips
are **not** from this plan — `test_platform_labeling_stability.py` is 40 passed, 0
skipped — but the phase baseline in `08-CONTEXT.md` reads "0 skipped", so whoever
closes the phase should attribute them.

A later `pytest tests/unit -k platform` run on the same tree gave **1226 passed, 14
failed, 13 skipped**. All 14 failures are in
`tests/unit/test_platform_joint_diagnostics_record.py`, which a sibling plan had modified
but **not yet committed**, alongside `scripts/joint_lift_diagnostics.py`,
`scripts/run_joint_lift.py` and an untracked `platform/evaluation/churn.py`. These
failures are **not attributable to this plan**, and the evidence is structural, not a
reassurance: (a) `stability.py` is imported by **nothing** — `grep` for
`labeling.stability` across `src/ tests/ scripts/ pipelines/` returns only the module
itself and its own test; (b) that test file does not mention `stability`; (c) this
plan's two files are byte-identical to `87f7f7d` (`git diff --quiet HEAD` passes); and (d)
the earlier full run, which also contained this plan's code, had 0 failures. The
sibling's record-pin tests are in flight. Flagged here so it is not lost at phase close.

## Next Phase Readiness

- Plan **08-07** can now run criterion 3 for both classifiers. It needs to: build a
  reference `fit_for_stability` at each classifier's pinned (K, lambda), assemble the
  four schemes (including `scheme_leave_one_episode_out` per state and
  `scheme_circular_block_bootstrap` across `BLOCK_LENGTH_LADDER`), call `run_stability`,
  and persist the rows plus every `cost_matrix`.
- The falsifiable predictions stated in `08-RESEARCH.md` §5.1 before the run are unchanged
  and this plan does not touch them: **state 0 (crisis, 9 episodes) is expected to pass**
  and thereby exonerate the §4.4 AMENDMENT's condition (i); **state 2 (10.22%, ONE
  contiguous 71-month episode, 1996-07 → 2002-05) is the likely failure**, and neither
  decade-drop touches it — which is why `scheme_leave_one_episode_out` exists and why
  `degenerate=True` is the answer there rather than an error.
- Runtime is not a constraint: `08-RESEARCH.md` §5.7 prices the whole of criterion 3 at
  ~1,620 fits ≈ 3 minutes.
- No blockers.

---
*Phase: 08-regime-persistence-stability*
*Completed: 2026-09-22*

## Self-Check: PASSED

- `src/trading_crab_lib/platform/labeling/stability.py` — FOUND (824 lines), identical to HEAD
- `tests/unit/test_platform_labeling_stability.py` — FOUND (672 lines, 40 tests), identical to HEAD
- `8e22c3f` — FOUND (`feat(08-03): pin the de-standardization inverse against standardize_features`)
- `12fc234` — FOUND (`feat(08-03): split-half null, episodes, and an evaporation flag that outranks distance`)
- `87f7f7d` — FOUND (`feat(08-03): four subsample schemes with the frozen column list held fixed`)
- All plan `<verify>` blocks re-run 2026-09-23 after the session resumed: green (see Verification Results)
