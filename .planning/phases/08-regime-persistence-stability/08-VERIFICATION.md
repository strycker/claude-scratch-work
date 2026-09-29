---
phase: 08-regime-persistence-stability
verified: 2026-09-29T17:58:58Z
verified_at_commit: 2210b73
status: gaps_found
score: 11/12 must-haves verified (10 ROADMAP success criteria 0-9 + 2 UAT gap truths G-08-1, G-08-2)
behavior_unverified: 0
overrides_applied: 0
re_verification: false
registry:
  sha256_prefix_before: c957e8fdb360
  sha256_prefix_after: c957e8fdb360
  total_trial_count_before: 44
  total_trial_count_after: 44
suite:
  live_collect_only: 2459
  full_run: "2459 passed, 0 failed, 0 skipped, 0 xfailed (380.65s, run once)"
gaps:
  - truth: "Criterion 1 (PER-02): an EXPLICIT BAYES FILTER, pi_t proportional to [sum pi_{t-1} A] * L_t, where L_t is the nowcaster's likelihood"
    status: partial
    reason: >-
      The belief propagates and cannot leak (both re-derived live, below). But L_t is computed as
      posterior / whole-window label prior (regime_filter.py::likelihood_ratio, fed by
      unconditional_belief over every in-window label). The nowcaster's posterior is calibrated to its
      own TRAINING class distribution. That distribution is the embargoed, complete-row block, not the
      window's labels. Bayes inversion is only valid with the training prior. On the real served model
      the training prior is {0: 0.072, 3: 0.895, 4: 0.033} (153 rows, 2007-04-30 -> 2019-12-31). The
      prior used is {0: 0.058, 3: 0.288, 4: 0.121, 1: 0.328, 2: 0.102, 5: 0.104} (695 labels,
      1963-02-28 -> 2020-12-31). The implemented L_t for state 3 is 1.96, i.e. evidence FOR it. The
      training-prior L_t is 0.63, i.e. evidence AGAINST it. After one filter step from cold start, the
      served belief (reproduced exactly: 3: 0.369, 0: 0.272, 1: 0.213, 2: 0.067, 5: 0.067, 4: 0.012) puts
      state 3 on top. With the training prior it would be 0: 0.300, 1: 0.293, 3: 0.163, so the top state
      changes. That belief is what the weekly executed book consumes. The plan's must-have literally said
      "divided by the in-window class prior", so the code matches the plan. The plan's
      operationalisation is what departs from the ROADMAP's "explicit Bayes filter". The defect is also
      present on the observational l2 backtest leg (B1 = 81/487 and 30/487 were measured through it).
      It is NOT on the decision-bearing l1only leg, where the filter is gated off.
      Independently reproduced from the code-review finding CR-01 in 08-REVIEW.md.
    artifacts:
      - path: "src/trading_crab_lib/platform/prediction/regime_filter.py"
        issue: "likelihood_ratio divides by unconditional_belief(window labels), not by the class distribution the nowcaster was trained/calibrated on"
      - path: "src/trading_crab_lib/platform/report/weekly.py"
        issue: "advance_regime_belief (:214-252) passes unconditional_belief(regime_labels) as class_prior; the served executed book is built on the resulting belief"
      - path: "src/trading_crab_lib/platform/backtest/driver.py"
        issue: "same class_prior at :582-584 (l2 filter)"
      - path: "src/trading_crab_lib/platform/backtest/joint_driver.py"
        issue: "same class_prior at :354-357 (l2 filter)"
    missing:
      - "A decision (Glenn), not a planner choice: divide by the nowcaster's training class distribution (e.g. returned by fit_l2_nowcaster beside the model, so train and serve share it), or accept the current prior as a stated approximation"
      - "If changed: a discriminating test whose fixture has training prior != window prior, and which FAILS on the current whole-window division"
      - "If changed: B1 and the l2 observational lift re-measured (NO_REGISTRY, observational), and the served belief re-recorded in 08-SERVING"
      - "Registry cost of any change must be priced under ADR-0004 before it is made (Phase 8 gap closure runs at budget 0)"
human_verification:
  - test: "On Glenn's Mac: git pull, reinstall both packages, `pytest tests/ -q`"
    expected: "2459 passed, 0 failed, 0 skipped. In particular test_the_record_is_re_derivable_from_its_own_curves[l1only] and [l2] pass"
    why_human: "G-08-1 is a macOS libm/SIMD last-bit effect. Linux can only simulate it (the arms that inject the UAT's exact macOS pair pass here); only the Mac proves it"
  - test: "On Glenn's Mac: `python scripts/build_platform_data.py`, then `python -m trading_crab_lib.platform.report.serving`, then `python -m trading_crab_lib.platform.report.weekly` twice"
    expected: "Both weekly runs exit 0 and show the as-of month, the lagging columns, the 1-distinct-vector disclosure, 'active regime: ... none (neutral posture)', the A7 sentence and the 5.0% band sentence. The second run's report is byte-identical"
    why_human: "Step 1 needs the network and a FRED key, and was not run by 08-14 or by this verification. A fresh build may move the ragged edge and so the scored month"
  - test: "Glenn confirms the staleness cap's data-relative reading (MAX_SCORING_LAG_MONTHS = 3 month-ends behind the newest monthly_features row, not the wall clock)"
    expected: "A yes or no. If he means the wall clock, change the one comparison in weekly._scored_row"
    why_human: "An open interpretation of Glenn's own ruling q1-c (08-SERVING.md §2.1, §3.4). This is an already-accepted open item and is not re-opened here"
  - test: "Read the weekly report's 'Per-Asset Signals (Returns by Regime)' section in neutral posture"
    expected: "Glenn decides whether 24 rows (4 assets x 6 regimes) that carry no regime id are acceptable on the trading page"
    why_human: "Since 036ae74 (08-09), active_regime None means no filter, and the rows print without their regime. On today's data the page lists SPY four times with contradictory signals and no label. The 'regime N' prefix is missing. This is a readability judgement, not a must-have"
---

# Phase 8: Regime Persistence & Stability — Verification Report

**Phase goal:** The real-time (filtered) regime labeling stops flickering, allocation responds
through the anti-flicker machinery the design already specifies, and the one §4.4 acceptance
criterion that has never been run is run.
**Verified:** 2026-09-29T17:58:58Z, at HEAD `2210b73`.
**Status:** gaps_found. One gap: the Bayes filter's likelihood normalisation (criterion 1). Four
human items.
**Re-verification:** No. This is the initial verification. It covers all 14 plans, including gap
closure 08-11..08-14.

**Integrity of this run.**
- The registry was untouched. `registry/trials.jsonl` sha256 prefix was `c957e8fdb360` before and
  after every command below. `total_trial_count()` read 44 before and after.
- Every real-data run used scratch copies under
  `/tmp/claude-0/-home-user-claude-scratch-work/64f0b80b-f0fb-5914-ae2e-935a933024c0/scratchpad`, via
  `TC_DATA_DIR` / `TC_OUTPUT_DIR` or a `--out` dir.
- `git status --porcelain -- data outputs registry` was empty at the end.
- No source, test, registry, data or output file was modified.
- Mutation checks ran on copies of test files in the scratch directory.

## Goal Achievement

### Observable Truths

Each row names how its check could fail.

| # | Truth | Status | Evidence (number, denominator, window), and how the check can fail |
|---|---|---|---|
| 0 | Per-step probability matrix persisted | ✓ VERIFIED | **Files:** `joint_lift_probs_{1,2}_l1only.parquet` hold **588 rows**, 1972-01-31 → 2020-12-31. `_l2` holds **488 rows**, 1974-02-28 → 2020-12-31, with **100 of 588 degraded** in `joint_lift_joint_l2`. Row sums are 1 ± 3e-16. **Live:** `scripts/diagnose_s1_truncation.py 1982-08-31` ran the current code (post-08-13 refactor) and wrote the belief matrices through `write_probability_matrix`. The **66 rows ≤ T** were **bit-identical** (max abs diff 0.0) to the committed `joint_lift_belief_{1,2}_l2.parquet`, for both classifiers. **Could fail:** any code drift or a leak changes a row. |
| 1 | A prior-state belief propagates, cannot leak — via an explicit Bayes filter | ✗ **PARTIAL (gap)** | **Propagation (verified).** Belief argmax ≠ posterior argmax in **273 / 488** (#1) and **155 / 488** (#2), l2, 1974-02 → 2020-12. **No leak (verified).** The live truncation re-run above was bit-identical at T = 1982-08-31. On the synthetic suite, the honest filter was invariant at all 111 cuts, the smoothed arm broke at 20 and the look-ahead arm broke at every changing month; all three pass in the full run. The filter is `filter_step = normalize((π_{t−1}·A) · L)`, parameter-free, and gated off under l1only (`joint_driver.py:597`). **Bayes-correct likelihood (FAILED).** L_t divides by the whole-window label prior, not by the nowcaster's training prior. On the served model this flips state 3's evidence from 0.63 to 1.96 and changes the top belief state (3 → 0). See `gaps`. |
| 2 | Both churn series reported, neither masquerades | ✓ VERIFIED | **Re-derived from the parquet.** **Track A:** `state_1` **246 / 587**, `state_2` **24 / 587**, 588 steps, identical under both routings. **B0:** argmax of the raw posterior, **221 / 487** and **66 / 487**. **B1:** argmax of the belief, **81 / 487** and **30 / 487**, 488 rows, 100 degraded. **l1only identity:** argmax(probs) = state_N in **588 / 588** for both. **Under l2:** **184** and **276 of 488** mismatched. **Could fail:** any wiring that fed one series from the other's object. |
| 3 | Track A diagnosed before changed (zero-trial, k = 1…6) | ✓ VERIFIED | **Recomputed** from `terminal_month_labels.parquet` (588 × 25). **#1:** **246 / 242 / 245 / 248 / 247 / 249** of 587. **#2:** 24 / 24 / 25 / 24 / 24 / 24. **k = 1** equals `state_1` / `state_2` elementwise, **0 / 588** mismatches. The curve is flat in k, so the edge artefact is refuted. **No λ sweep:** the registry is unchanged. |
| 4 | §5.3 anti-flicker gates allocation, closing A7; the mechanism is Glenn's | ✓ VERIFIED (under Glenn's ruling, 2026-09-24) | **Mechanism.** The bounded-turnover arm, named in criterion 4 itself: a 5pp no-trade band in one helper, `execute_rebalance`, at all 3 call sites (`driver.py:607`, `joint_driver.py:634`, `weekly.py:648`). **Re-derived band-off (git `d5c3ac9`) vs band-on turnover:** baseline **0.163251 → 0.127412**, joint **0.119788 → 0.080972**. No-trade months went from **0 / 588 → 304 / 588** and **0 / 588 → 341 / 588**. `state_*`, `active_regime` and `degraded` are identical between the band-off and band-on curves. `active_regime == state_1` in **588 / 588**, which confirms the identity. **Disclosed:** `active_regime` gates no weight. The reworded ROADMAP Phase 4 criterion 3 is still unapplied (stale, below). |
| 5 | §4.4 criterion 3 RUN, both classifiers, four schemes | ✓ VERIFIED (as a measurement) | **8,883 rows** = #1 (4,800 bootstrap + 6 + 6 + 36 LOO) + #2 (4,000 + 5 + 5 + 25). **AMENDMENT condition (i)** for #1 state 0: **5 / 9 / 9** episodes (drop first decade / drop last / LOO). **#1 state 2** under LOO: `degenerate = True`, 0 reference months in the subsample. **`evaporated`** fired 0 times while 36 rows have `reference_months_in_subsample == 0`, a disclosed limitation. The full-sample reference reproduction test runs live in the suite and passes. |
| 6 | Criterion 7 re-measured, both legs, one harness, window inline | ✓ VERIFIED | Re-derived with my own formulas (log1p sum, wealth max drawdown, Sharpe with ddof = 1 × √12), independent of `joint_lift_table`. **l1only:** `wealth_delta` **−0.1253065774082902**, `dd_delta` **+0.026164**. **l2:** **−0.300608 / +0.061011**. Sharpe **0.896446 / 0.899378**. Window: 588 steps, 1972-01-31 → 2020-12-31, all matching the record. |
| 7 | A11 answered, written as the reversal | ✓ VERIFIED | **Hurdle, computed independently** (Bailey–López de Prado E[max], γ = 0.5772): `expected_max_sharpe(44, 1.0)` = **2.226891**, `(42, 1.0)` = **2.208694**. These match the record. Observed l1only Sharpe 0.896 / 0.899 < 2.227 gives **FAILED 2 of 2**, independent of the DSR's magnitude. The gate is wired in `joint_lift_table` (`joint_driver.py:874-915`) and the hurdle is read live. ADR-0003 is Accepted and indexed. UAT test 5 was passed by Glenn. |
| 8 | G6 pinned as the known non-compliance | ✓ VERIFIED | **Mutation:** patching `driver.returns_by_regime_stats` to return the pooled table makes `TestDriverConsumer` raise `AssertionError`, so the pin can fail. Both halves are asserted. **Census warning below:** there are 5 `vol_targeted_tilt` call sites in `src`, not 3. |
| 9 | Recorded counts match reality; F-4 fixed | ✓ VERIFIED | **Live `pytest --collect-only`: 2459.** CLAUDE.md:128 and :606, and README.md:5 and :477, all say **2459**. The full run gave **2459 passed, 0 skipped**. **F-4:** `diagnostics_*.json` and `measurement_l1only.json` carry **0.4190800681431005** = 246/587. The "0.418980" transcription is gone from every source and doc site. |
| G-08-1 | Criterion-7 re-derivation is portable across platforms and still discriminates | ✓ VERIFIED (Linux; Mac run is a human item) | Three mutations were run on a scratch copy. **(a) Exact `==`:** 14 portability arms plus the UAT macOS-pair arm went red, so Linux now reproduces G-08-1. **(b) `rel = 1e-9` with no `abs = 0.0`:** 4 one-ppm DSR arms and 3 DSR-zero arms went red. **(c) `rel = 1e-1`:** the band-off-record arm and all 14 one-ppm arms went red. The unmutated file passes. All arms route through the single `_record_mismatches`. |
| G-08-2 | The weekly report runs end to end from supported commands | ✓ VERIFIED on tracked data (Mac step 1 is a human item) | **Live, scratch copies:** `serving` exit 0 (55 cols, classes [0, 3, 4], no registry row). **weekly** ran twice, exit 0 both times. The report is **byte-identical** across runs and scored as of **2026-06-30**, 2 month-ends behind 2026-08-31. Lagging columns are named. Executed book: TLT 0.390654 / SPY 0.330723 / USO 0.144729 / IAU 0.133895. **Disclosure:** "1 distinct posterior vector across 231 complete months (2007-04-30 → 2026-06-30)". This was independently re-derived: `np.unique` gives 1, and even a 1e6 input returns the same vector. **Refusals, live:** with no `nowcaster.pkl` the run raises `FileNotFoundError` naming the build command and writes 0 files. With the newest row pushed to lag 4, it raises `ValueError` naming the cap before any save; the checkpoint count stayed 24 → 24. |

**Score:** 11 / 12 truths verified. Criterion 1 is partial. 0 are present-but-behavior-unverified.

### Known, already-accepted limitations (confirmed disclosed, not re-opened)

| Limitation | Where disclosed | Confirmed |
|---|---|---|
| The served L2 posterior is input-independent. Glenn's ruling q2-ii: disclose on the page; the recipe fix is Phase 9 | 08-SERVING §2.2 / §3.4; printed on the page directly under the distribution | ✓ The live page carries the sentence; the count was re-derived as 1 |
| The staleness cap's data-relative reading awaits Glenn's confirmation | 08-SERVING §2.1 and §3.4 | ✓ Code comment `weekly.py:498-499`; human item 3 |

### Required Artifacts

| Artifact | Status | Details |
|---|---|---|
| `platform/evaluation/churn.py` | ✓ | Holds the two-series churn and `write_probability_matrix` (used live above) |
| `platform/prediction/regime_filter.py` | ⚠ substantive, wired | Formula and wiring are correct; the class prior used for L_t is the gap |
| `platform/labeling/stability.py` + `scripts/run_subsample_stability.py` | ✓ | `run_stability` same-id keying defect is disclosed, not fixed (the script is keyed correctly) |
| `platform/allocation/hysteresis.py::execute_rebalance` | ✓ | Called at 3 sites |
| `platform/report/serving.py` | ✓ | Uses `NO_REGISTRY` hard-coded; `fit_l2_nowcaster` is shared with `_refit_l2` (`driver.py:307-375`) |
| `platform/report/weekly.py` | ✓ | Q1-c branch, Q2-ii branch, `feature_names_in_` scoring, no `idxmax` recomputation |
| `outputs/.../joint_lift/*`, `track_a/*`, `stability/*` | ✓ | Every number above was recomputed from these files |
| 08-TRACK-A / 08-STABILITY / 08-CHURN / 08-A7 / 08-A11 / 08-G6 / 08-MEASUREMENTS / 08-SERVING | ✓ | Present; numbers match the re-derivations |
| ADR-0003, ADR-0004 | ✓ | Both Accepted; both indexed in `adr/README.md` |

### Key Link Verification

| From | To | Status |
|---|---|---|
| `run_joint_backtest` per-step probs | `write_probability_matrix` → parquet | WIRED (live re-run) |
| `states_1` (train_index) → `transition_matrix_for` / `unconditional_belief` → `filter_step` | allocator + hysteresis (l2 only) | WIRED (`joint_driver.py:597-640`) |
| `fit_l2_nowcaster` | `_refit_l2` and `serving.build_serving_artifacts` | WIRED (one function, two callers) |
| serving `nowcaster.pkl` (`feature_names_in_`) | `weekly._scored_row` → predict → filter → tilt → band | WIRED (live run) |
| `registry.total_trial_count()` | `joint_lift_table` quality tier | WIRED (read live) |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data? | Status |
|---|---|---|---|---|
| weekly report distribution | served posterior | `nowcaster.pkl` on the 2026-06-30 row | Real, but constant (disclosed) | FLOWING, input-independent (accepted q2-ii) |
| weekly filtered belief / executed book | belief | `advance_regime_belief` | Real; reproduced exactly | FLOWING — with a mis-normalised L_t (gap) |

### Behavioral Spot-Checks and Probes

| Behavior | Command | Result | Status |
|---|---|---|---|
| Full suite, once | `pytest tests/ -q` | 2459 passed, 0 skipped, 380.65s | ✓ PASS |
| Serving, then weekly ×2 on real tracked data | `python -m ...report.serving`; `python -m ...report.weekly` ×2 | exit 0 ×3; byte-identical report | ✓ PASS |
| Staleness cap refuses at lag 4 | weekly on a scratch holdout with 2 appended incomplete rows | ValueError, 0 files written | ✓ PASS |
| Missing artifact names its fix | weekly with no serving artifacts | FileNotFoundError names `report.serving` | ✓ PASS |
| Real-data causal invariance | `scripts/diagnose_s1_truncation.py 1982-08-31` | 66 / 66 rows bit-identical, both classifiers | ✓ PASS |
| G-08-1 tolerance discriminates | 3 mutations of `_floats_agree` | each turns its target arms red | ✓ PASS |
| G6 pin can fail | pooled-stats monkeypatch | AssertionError | ✓ PASS |

No `scripts/*/tests/probe-*.sh` exist, and no plan declares one.

### Requirements Coverage

| Req | Plans | Status | Evidence |
|---|---|---|---|
| PER-01 | 08-01 | ✓ SATISFIED | Criterion 0 |
| PER-02 | 08-06, 08-08, 08-12, 08-13, 08-14 | ⚠ PARTIAL | Criterion 1: propagation and no-leak hold; the Bayes likelihood normalisation is the gap |
| PER-03 | 08-01, 08-08 | ✓ SATISFIED | Criterion 2 |
| PER-04 | 08-02 | ✓ SATISFIED | Criterion 3 |
| PER-05 | 08-09, 08-12, 08-13, 08-14 | ✓ SATISFIED | Criterion 4 (ruled mechanism) |
| PER-06 | 08-03, 08-07 | ✓ SATISFIED | Criterion 5 |
| PER-07 | 08-10, 08-11 | ✓ SATISFIED | Criterion 6 and G-08-1 |
| PER-08 | 08-05 | ✓ SATISFIED | Criterion 7 |
| PER-09 | 08-04 | ✓ SATISFIED | Criterion 8 |
| PER-10 | 08-01, 08-10, 08-11, 08-13, 08-14 | ✓ SATISFIED | Criterion 9 (live 2459) |

Every ID PER-01..PER-10 is claimed by at least one plan. No orphaned requirement is mapped to Phase 8.

### Anti-Patterns and Warnings

| File | Line | Finding | Severity |
|---|---|---|---|
| `prediction/regime_filter.py` | 125-164 | L_t uses the wrong class prior (the gap) | 🛑 Gap |
| `report/weekly.py` | 392-400 | In neutral posture, all `returns_by_regime` rows print with no regime id. This regressed with 08-09 (`036ae74`), when `active_regime` stopped being `idxmax` | ⚠ Warning (human item 4) |
| `08-G6.md` §1 | — | Says "the three call sites that feed `vol_targeted_tilt`". `src` has **5**: `driver.py:599`, `weekly.py:636`, `evaluation/report.py:557` (smoothed oracle), `plotting/allocation.py:323`, and `joint_tilt.py` (which pools). The extra two are unpooled diagnostics. The line refs `driver.py:497` and `weekly.py:249` are stale | ⚠ Warning (census incomplete; the pin still holds) |
| `config/platform_settings.yaml` | 65-70 | `fred_m2sl` and `fred_totalsl` have `shift: false`, but each is released weeks after month end. The ragged edge (08-SERVING item 8) shows they lag in practice, and `div_yield` (in #1's frozen set) lags too. 08-SERVING §2.1 discloses "the backtest never modelled publication lag". Whether that is look-ahead in the evaluated backtest (the review's CR-03) is **pre-existing, Phase 1/7 scope, and not adjudicated here** | ⚠ Warning, out of scope, for a decision |
| `backtest/joint_driver.py` | 597-610 | The belief is carried across refits whose state ids are not comparable (the vocabulary finding; the review's CR-02) | ⚠ Warning, observational leg only |
| `joint_driver.py::quality_tier` | 774-816 | The annualised Sharpe is fed to the PSR with monthly `n_obs`, which is a units mix. The FAILED verdict is robust (SR < SR0 in any units); the DSR magnitudes (1e-13) are not interpretable as stated | ℹ Info |

A debt-marker scan (TBD / FIXME / XXX) over the phase-modified `src` files found no unreferenced marker.

### Stale claims in planning docs (not gaps; they need a docs pass)

1. **`.planning/REQUIREMENTS.md:250`**: PER-10 says "pinned to a live collection, **2392**". Live is **2459**.
2. **`.planning/STATE.md`** says "10 of 10 plans", "Suite **2392** passed", "Current focus: Phase 7" and `last_activity` 2026-09-23. It has no mention of 08-11..08-14, G-08-1/G-08-2 closure or ADR-0004.
3. **`.planning/ROADMAP.md`**:
   - 08-11..08-14 are still `[ ]`, although all four have SUMMARYs and commits.
   - The reworded Phase 4 criterion 3 (08-A7 §2) is still unapplied: it reads "with hysteresis bands … so the target mix doesn't flip".
4. **`08-MEASUREMENTS.md` §7** lists three items since closed or superseded:
   - "ADR index lacks 0003": now indexed.
   - "0.418980 at six sites": now corrected.
   - "any further configuration needs an ADR-0002 amendment first": superseded by ADR-0004.
   - Its suite figures (2392) are historical to 08-10.
5. **`CLAUDE.md:607`** says "(10 skipped: HDBSCAN + cssselect optional)". The live run has **0 skipped**.
6. **`08-G6.md`**: the census and line references (above).
7. **`08-UAT.md`** frontmatter still says `status: diagnosed` and "2393". Gap closure has not been re-UAT'd (human items 1-2).

## Gaps Summary

The phase goal is substantially achieved.
- **Flicker:** filtered-belief churn is 81 / 487 against a raw posterior's 221 / 487 (#1). This is observational.
- **Allocation:** it now moves through the ruled 5pp band; decision-bearing no-trade months went from 0 to 304 / 588.
- **§4.4 criterion 3:** run, 8,883 rows.
- **Other criteria:** every other number reproduces from the committed artifacts or from a live run, and the suite is green at the recorded 2459.

**One gap.** Criterion 1 requires an *explicit Bayes filter*. Its likelihood is formed with a class
prior that is not the nowcaster's training prior. On today's served model this reverses the
evidence for the dominant state and changes the top belief state. The weekly executed book is
built on that belief. It does not touch the decision-bearing l1only leg. A fix is a recipe change
that needs Glenn's decision and an ADR-0004 budget.

**If Glenn accepts the current prior as a stated approximation,** carry the fix to Phase 9 with
this override in the frontmatter:

```yaml
overrides:
  - must_have: "Criterion 1: an explicit Bayes filter pi_t proportional to [sum pi_{t-1} A] * L_t"
    reason: "L_t normalised by the window label prior rather than the nowcaster's training prior; accepted as a stated approximation, fix scheduled for Phase 9 under its ADR-0004 budget"
    accepted_by: "Glenn"
    accepted_at: "<ISO timestamp>"
```

---

_Verified: 2026-09-29T17:58:58Z_
_Verifier: Claude (gsd-verifier)_
