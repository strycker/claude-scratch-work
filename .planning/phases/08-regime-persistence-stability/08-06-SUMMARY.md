---
phase: 08-regime-persistence-stability
plan: 06
subsystem: platform/prediction + platform/evaluation
tags: [criterion-1, PER-02, bayes-filter, signed-offset, leakage-guard, D-03, S-1, S-3, honesty-framework]
status: complete
requires:
  - "08-01 (diagnostics_l1only.json headline 9.5 / 4.0 / 2.375 / 25 of 25; probability-matrix artifacts)"
  - "prediction/transition_matrix.py::empirical_transition_matrix (unchanged)"
  - "honesty/gap_lag.py::compute_detection_lag (unchanged; floors at zero, line 86)"
provides:
  - "prediction/regime_filter.py: unconditional_belief, likelihood_ratio, filter_step, predict_only_step, transition_matrix_for"
  - "evaluation/sojourn_lag.py: _transitions_by_state (one transition rule), _require_integer_state_columns (shared T0.12 guard), compute_signed_detection_offsets"
  - "tests/unit/test_platform_nowcaster_recursion.py: the S-1/S-3 guard with three live arms plus one measured caveat"
affects:
  - "08-08: wires the filter into the drivers and owns the real-data signed-offset number. It inherits the false-positive caveat and the class-prior note below"
decisions:
  - "The filter has zero free parameters. No tunable, no smoothing constant, nothing registered. Registry 42 before and 42 after."
  - "The class prior and the cold start come from one call, unconditional_belief, over the whole in-window label series."
  - "The missing-observation rule is predict-only (pi A, normalized), not a hold."
  - "A posterior state that is absent gets a likelihood ratio of exactly 1.0. A zero prior on a present state raises."
  - "An absent 'from' row in A is filled with the unconditional belief. It is never uniform and never NaN, and a WARNING is logged."
  - "The signed offset's forward branch calls compute_detection_lag and does not reimplement it. The backward walk is the only new arithmetic."
  - "The T0.12 guard was extracted along with the transition rule so both functions call one check, as the plan required ('do not duplicate the check')."
tech-stack:
  added: []
  patterns: ["reuse-by-call for the forward branch", "honest expectation as a callable, shown FAILING under pytest.raises", "mutation evidence for every pin"]
key-files:
  created:
    - src/trading_crab_lib/platform/prediction/regime_filter.py
    - tests/unit/test_platform_regime_filter.py
    - tests/unit/test_platform_nowcaster_recursion.py
  modified:
    - src/trading_crab_lib/platform/evaluation/sojourn_lag.py
    - tests/unit/test_platform_evaluation_sojourn_lag.py
metrics:
  duration: "about 20 min wall clock (2026-09-23 18:20Z to 18:40Z), plus a 4m43s full suite and the SUMMARY"
  completed: 2026-09-23
actuals:
  tokens: 14987   # chars/4 over the added lines of this plan's five files
  tasks: 3
  commits: 5
---

# Phase 8 Plan 06: The Bayes filter, a signed detection offset, and a guard shown to discriminate

This plan adds an explicit forward filter, `π_t ∝ [Σ_i π_{t−1}(i) A_ij] · L_t(j)`, with no feature column and no free parameter. It also adds a signed detection offset that can go negative, because `compute_detection_lag` cannot. The leakage guard's substituted-smoothed-label arm makes the honest expectation fail inside `pytest.raises(AssertionError)`. `nowcaster.py` and `gap_lag.py` are unchanged.

## Tasks and commits

| task | commit | what |
|---|---|---|
| 1 (tracer), RED | `49a0e49` | `test_platform_regime_filter.py`, 16 tests. Collection fails because the module does not exist yet |
| 1 (tracer), GREEN | `738f949` | `prediction/regime_filter.py` |
| 2, extraction | **`b679892`** | `_transitions_by_state` and `_require_integer_state_columns` extracted. Adds a real-input headline pin |
| 2, addition | **`daa57e6`** | `compute_signed_detection_offsets` plus 9 tests |
| 3 | `51e4381` | `test_platform_nowcaster_recursion.py`: three arms and one caveat test |

The extraction (`b679892`) and the addition (`daa57e6`) are separate commits, with the addition newer. The plan's commit-structure verify passed on them.
Sibling 08-07 commits (`0f30661`, `d47d2de`) are interleaved between them. None of them touches this plan's files.

**Tracer gate.** Config reports `auto_advance=false` and `_auto_chain_active=false`, which would mean a human-verify stop after the tracer. The orchestrator asked for the whole plan and the frontmatter says `autonomous: true`, so I ran the autonomous tracer gate instead. After the Task 1 commit I re-ran its `<verify>`: 16 passed, the self-check exited 0, and nowcaster was untouched. See Deviations.

## Task 1: the filter

Five public functions (the plan's output section says four; its behaviour list names five):

| function | role |
|---|---|
| `unconditional_belief(states, *, state_index)` | In-window class distribution, zero-filled, sums to 1. It is **both** π₀ and the class prior (one call). |
| `likelihood_ratio(posterior, class_prior, *, state_index)` | `posterior(j) / prior(j)`. An absent state gets **1.0**. A zero or missing prior on a *present* state raises, naming the state. |
| `filter_step(prior_belief, A, posterior, class_prior)` | Predict (`π A`), then update (× ratio), then normalize. Zero mass raises and never returns NaN. Pure. |
| `predict_only_step(prior_belief, A)` | **The missing-observation rule:** `π A`, normalized. On a degraded step there is no `L_t`. Holding π would claim "the world did not move", and a filter with no observation advances by the transition step alone. |
| `transition_matrix_for(states, *, state_index)` | Full K×K. An absent 'from' row is filled with the unconditional belief and a WARNING names the state. A 'to' column absent from the crosstab is an observed zero. |

The module docstring covers five things: the recursion with the §4.2 citation, the "second catch" (A comes from smoothed in-window labels), the four structurally-void pitfalls (§6 P1, P3, P4 and §2.5), Pitfall 6 (still live), and that `assert_causal_features` protects nothing (`gating.py:33,48`).
The `__main__` footer runs 12 steps, including one predict-only step and one strict-subset posterior.

**Evidence the Task 1 pins can fail.** I applied seven mutations to the module. Each turned at least one test red, and I restored the module after each one:

| mutation | tests failed |
|---|---|
| no normalization | 4 |
| predict/update transposed | 3 |
| predict dropped | 2 |
| update dropped | 4 |
| absent class → 0.0 | 2 |
| predict-only implemented as a hold | 1 |
| absent row → uniform | 2 |

One real bug surfaced in my own code during the run. The out-of-index error message printed `[np.int64(5)]`, and the message test caught it. Fixed before commit.

## Task 2: one transition rule, and a signed offset

**Definition (as implemented).** Take a transition at position *i* into state *s*. Let `c` be `filtered_probs_matrix[s]` reindexed onto the reference index; NaN never counts as at-or-above the threshold.

- If `c[i] ≥ act_threshold`, walk **back** to the start *j* of the contiguous at-or-above run *containing i*. The offset is `j − i`, which is ≤ 0.
- Otherwise, the offset is `compute_detection_lag`'s own answer for that transition (called, not reimplemented), so it is ≥ 1 or NaN.

**Run-boundary rule.** Only the run containing *i* counts. An earlier, separate excursion above the threshold is not a lead. A run truncated by the first observed row (NaN before it) stops the walk there.

**Headline unchanged.** The headline pin was recomputed from the **real tracked dev inputs** (`regime_labels` checkpoint split at 2020-12-31, 695 months 1963-02-28 → 2020-12-31, plus the one-hot of `state_1` from `joint_lift_joint_l1only.parquet`, 588 steps 1972-01-31 → 2020-12-31). It reads `median_sojourn` **9.5**, `median_lag` **4.0**, `ratio` **2.375**, `n_transitions` **25**, `n_resolved` **25**. That is true before the extraction, after it, and after the addition, and it equals `diagnostics_l1only.json` field by field.
The plan's own verify only re-reads the committed JSON, which cannot detect a refactor. So the new test recomputes the headline. A mutated target-state rule turns it red (`median_lag` 3.0, ratio 3.17).

**Signed offsets on the same real l1only inputs:** 25 transitions, 25 resolved, `min_offset` **1.0**, `median_offset` **4.0**, `n_negative` 0, `n_zero_or_negative` 0.
The offsets equal `compute_detection_lag` elementwise on all 25. No L1 filtered transition leads the full-sample reference. This is Track A (L1) under l1only, not the filter. The l2/filter number belongs to 08-08.

**Fixture pins:**

| fixture | result |
|---|---|
| three-month lead | **−3**, `n_negative == 1`; `compute_detection_lag` reads **0** on the same input |
| run boundary | **−1**, not −5 |
| crossing at *i* | 0 |
| NaN warm-up | −4 |
| no-lead agreement fixture | [2, 1, 3, NaN], elementwise equal to `compute_detection_lag` |
| unresolved | NaN, counted in `n_transitions`, excluded from the median |
| string columns | raise naming `compute_signed_detection_offsets` |

**Evidence the Task 2 pins can fail.** Four mutations, each red:

| mutation | tests failed |
|---|---|
| no backward walk | 3 |
| naive earliest crossing | 1 (run boundary) |
| forward off-by-one | 3, including the real-data agreement |
| guard call removed | 1 |

## Task 3: the guard, with exact measured offsets

The synthetic world has K=3, 111 months and 9 reference transitions at positions 14, 24, 36, 45, 56, 69, 79, 88 and 100. Three of those turns are *led* (14, 45, 69): the three months before each carry ambiguous evidence tilted toward the incoming state. There is no RNG. The per-month posterior is `c·e_t`, normalized, which is a function of month *t* only. The smoothed label is a Viterbi (two-sided DP) decode under the same A.

| arm | belief | offsets (chronological) | min | median | n_negative |
|---|---|---|---|---|---|
| 1 honest | `filter_step` path, empirical A | **[1, 2, 2, 1, 2, 1, 2, 2, 2]**: led turns +1, others +2 | 1 | 2 | 0 |
| 2 substituted | one-hot(Viterbi smoothed label) | **[−3, 0, 0, −3, 0, −3, 0, 0, 0]**: −3 exactly at the 3 led turns, 0 elsewhere | −3 | 0 | 3 |
| 3 collapse | `filter_step`, A = 0.995·I + 0.005·A_emp | [4, 6, 38, 0, 6, 5, 5, 6, 6] | 0 | **6** | 0 |

- **Arm 2 discriminates.** `_assert_honest_expectation`, the callable Arm 1 passes, is shown raising inside `pytest.raises(AssertionError)` on Arm 2's result.
- **Arm 2's lead comes from the future.** A Viterbi decode truncated at *i−1* keeps the ambiguous months in the outgoing state at all three led turns.
- **Arm 3 moves in both directions the plan named.** Median offset goes up, **2 → 6**. The §5.4 ratio goes down toward 1, **11/2 = 5.5 → 11/6 = 1.833**. Median sojourn is 11 in both runs, and all 9 transitions resolve in both. The real-data reference today is 2.375. No thresholds or churn targets are asserted.
- **Evidence the arms can fail.** Five mutations were each red:
  - Arm 2 fed the honest belief.
  - The library's backward walk removed, i.e. the `gap_lag.py:86` floor. Arm 2 then reads 0 and `n_negative` 0. Note that `_assert_honest_expectation` alone would still catch 0; the `n_negative ≥ 1` assertion is what requires the signed function.
  - Arm 1 fed the smoothed one-hot.
  - Arm 3's A reverted to the empirical A.
  - `filter_step` with the update dropped: 3 red.
- The Task 3 verify blocks all pass: 4 passed, no skips or xfails, and `pytest.raises(AssertionError)` is present.

## Found: S-1's "even one strictly negative offset is proof" is not true as stated

I measured this during Arm 3 and pinned it as a fourth, non-arm test (`test_caveat_an_honest_filter_that_MISSES_a_short_regime_reads_as_a_lead`).
The filter was honest and causal, with A = 0.999·I + 0.001·A_emp. It never registers the 9-month state-0 run at positions 36–44: max belief on 0 there stays below 0.70. So its belief on state 2 stays above threshold straight through that run, and the **return to state 2 at 45 reads −14**.
Probed at other diagonals, the offset at 45 was: 0.995 → 0, 0.997 → −1, 0.998 → −2, 0.999 → −14. So a strictly negative offset from an honest filter is reachable in two ways:
- a missed intervening regime;
- a sticky prior plus genuinely leading data.

**For 08-08:** a real-data `n_negative > 0` is not proof of leakage by itself. Each negative offset has to be read against the reference run it spans. If the belief never registered that run, it is a miss, not a lead.
I did not change the plan's run-boundary definition to exclude this. That would be a redefinition, and the exclusion rule is itself a judgment 08-08 should make with real data in hand.

## Found, not fixed

- **Class prior vs. the nowcaster's effective training prior (08-08).** `unconditional_belief` runs over the whole in-window label series. The nowcaster actually trains on the D-01-embargoed subset (the trailing 12 months are excluded) and drops non-finite rows. So `posterior / prior` divides by a prior slightly different from the one the classifier learned.
  I kept the full-window prior. It is the plan's rule, and because it is a superset it cannot fire the zero-prior raise on a state the nowcaster did see. The difference should be stated by 08-08 when it wires the filter, not silently absorbed.
- **The forward lag search is unbounded.** It inherits `compute_detection_lag`'s convention. In Arm 3 the turn at 36 "resolves" at +38, in a later run of the same state. On real data, the first three reference transitions (1970-05, 1970-08, 1971-03) predate the first decision date (1972-01-31) and carry lags of 44, 54 and 28. They are in the committed 25-of-25 headline. This is pre-existing and unchanged here; it bears on how the 4.0 median is read.
- **The S-2 blend limit.** A smoothed/filtered blend too light ever to cross the threshold early is invisible to S-1 as well as S-2. This is stated in the guard's module docstring and not tested (no fixture was constructed to pass it).

## Deviations from Plan

1. **[Tracer gate mode]** I applied the autonomous tracer gate (re-verify, halt on failure) although config reports auto mode off. The orchestrator's explicit instruction and `autonomous: true` govern. The re-verify passed.
2. **[Rule 2 – scope]** I added a fourth test in Task 3's file beyond the three arms: the measured false-positive caveat above. The plan says "three arms", and these remain three. The caveat is labelled as not an arm.
3. **[Rule 2 – evidence strength]** The Task 2 no-change pin recomputes the headline from tracked real inputs rather than only re-reading the JSON the plan's verify reads. It also asserts equality with the JSON. Both inputs are tracked, so nothing skips in CI.
4. **The T0.12 guard was extracted** into `_require_integer_state_columns(caller=...)` in the extraction commit, to satisfy "do not duplicate the check". The headline's message text is byte-identical because `caller="compute_sojourn_lag_headline"`.
5. **Extra keys in the signed-offset dict:** `positions` and `target_states` (parallel, chronological). Additive.
6. **Task 2's addition was committed together with its tests** (`daa57e6`), not as a separate RED commit. A RED commit would have made the whole sojourn test file uncollectable, including the headline pin, in a working tree shared with 08-07. The ability of the pins to fail is shown by mutation instead.
7. **Arm 1's expected offsets.** My first draft comment guessed [2]×9. The measurement was [1, 2, 2, 1, 2, 1, 2, 2, 2], and the test pins the measured values. Arm 3 originally asserted the full honest expectation on the sticky run. It measured one offset of 0, a same-month crossing, so the assertion is now `n_negative == 0`, the causal invariant. That is the caveat above.

## Verification

| check | result |
|---|---|
| `pytest` on the three plan files | **42 passed** (16 + 22 + 4), 0 skipped |
| `python -m trading_crab_lib.platform.prediction.regime_filter` | exits 0; 12-step belief path printed |
| nowcaster untouched | `git diff` empty; no commit since `b93e558` touches `nowcaster.py` or `gap_lag.py` |
| Task 2 JSON pin verify | "headline pin values unchanged by the extraction" |
| Task 2 commit-structure verify | pass: `['feat(08-06): signed detection offset…', 'refactor(08-06): extract _transitions_by_state…']` |
| Task 3 verifies | 4 passed; "three live arms"; "Arm 2 proves the guard discriminates" |
| `pytest tests/ -q` | **2212 passed**, 0 failed, 0 skipped. The tree included 08-07's concurrent uncommitted edits. |
| legacy-import ratchet | 11 passed, `MAX_LEGACY_IMPORT_SITES = 31` |
| `total_trial_count()` | **42**, unchanged |
| ruff + flake8 (E9,F63,F7,F82) | clean on all five touched files, before every commit |
| 2021+ holdout | not read. The real-input loader splits at `DEFAULT_HOLDOUT_CUTOFF`, and the test asserts max date ≤ 2020-12-31. |

## Known Stubs

None.

## Threat Flags

None. There is no new I/O, network or schema surface. The real-input pin reads two tracked local files.

## Self-Check: PASSED

- FOUND: `src/trading_crab_lib/platform/prediction/regime_filter.py`, `tests/unit/test_platform_regime_filter.py`, `tests/unit/test_platform_nowcaster_recursion.py`. The modified `sojourn_lag.py` and its test file are present.
- FOUND commits on `claude/keen-galileo-zqcml6-w5`: `49a0e49`, `738f949`, `b679892`, `daa57e6`, `51e4381`. Not pushed.
