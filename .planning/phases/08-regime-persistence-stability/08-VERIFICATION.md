---
phase: 08-regime-persistence-stability
verified: 2026-09-29T21:40:00Z
verified_at_commit: a64b0e7
status: human_needed
score: 12/12 must-haves verified (10 ROADMAP success criteria 0-9 + 2 UAT gap truths G-08-1, G-08-2); round-2 plan must-haves 08-15..08-19 verified
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 11/12
  previous_verified_at: 2026-09-29T17:58:58Z
  previous_commit: 2210b73
  gaps_closed:
    - "Criterion 1 (PER-02): the explicit Bayes filter's likelihood now divides by the nowcaster's training class prior (CR-01), at serve and in both backtest drivers"
  gaps_remaining: []
  regressions: []
registry:
  sha256_prefix_before: c957e8fdb360
  sha256_prefix_after: c957e8fdb360
  total_trial_count_before: 44
  total_trial_count_after: 44
suite:
  live_collect_only: 2494
  full_run: "2494 passed, 0 failed, 0 skipped, 0 xfailed (1132.89s, run once, under heavy shared load)"
accepted_not_reopened:
  - "CR-02 (belief carried across unaligned refits) and CR-03 (unmodelled publication lag) deferred to Phase 8.1 (Glenn, 2026-09-29); every re-measured L2 number carries that qualification (08-MEASUREMENTS §11.5, 08-SERVING §4.6)"
  - "Input-independent served posterior (ruling q2-ii); disclosed on the live page"
  - "CR-04 (run_stability keying) is a tracked STATE todo"
  - "The two test-side hard-coded-44 sites (test_platform_gate_tiers.py, test_platform_joint_diagnostics_record.py) are Phase 8.1 work"
  - "Staleness cap data-relative reading CONFIRMED by Glenn 2026-09-29 (08-SERVING §2.1 line 134); previous human item 3 closed"
human_verification:
  - test: "On Glenn's Mac: git pull, reinstall both packages, `pytest tests/ -q`"
    expected: "2494 passed, 0 failed, 0 skipped. In particular test_the_record_is_re_derivable_from_its_own_curves[l1only] and [l2] pass (the [l2] arm now reads the regenerated CR-01 l2 record)"
    why_human: "G-08-1 is a macOS libm/SIMD last-bit effect. Linux passes the arms that inject the UAT's macOS pair; only the Mac proves it"
  - test: "On Glenn's Mac: `python scripts/build_platform_data.py`, then `python -m trading_crab_lib.platform.report.serving`, then `python -m trading_crab_lib.platform.report.weekly` twice"
    expected: "All exit 0. serving writes nowcaster AND nowcaster_class_prior. The page shows the as-of month, the lagging columns, the 1-distinct-vector disclosure, the filtered belief with regime 0 on top on the tracked data (a fresh build may move it), 'active regime: ... none (neutral posture)', the neutral per-asset sentence (no rows), the A7 sentence and the 5.0% band sentence. The second run's report is byte-identical"
    why_human: "Step 1 needs the network and a FRED key and was not run by 08-19 or by this verification. A fresh build may move the ragged edge, the scored month and the served belief"
  - test: "Read the weekly page's 'Per-Asset Signals (Returns by Regime)' section in neutral posture (previous human item 4, now fixed by 08-15)"
    expected: "Glenn accepts the planner's choice: one sentence and no rows in neutral posture; every printed row names its regime. Or he asks for the alternative 08-15 did NOT implement: all regimes' rows, grouped and labelled"
    why_human: "A readability ruling on the trading page, recorded as an open question in 08-15-SUMMARY; the implemented behaviour is verified below"
---

# Phase 8: Regime Persistence & Stability — Verification Report (re-verification after gap closure round 2)

**Phase goal:** The real-time (filtered) regime labeling stops flickering, allocation responds
through the anti-flicker machinery the design already specifies, and the one §4.4 acceptance
criterion that has never been run is run.
**Verified:** 2026-09-29T21:40:00Z, at HEAD `a64b0e7`.
**Status:** human_needed. The one previous gap (criterion 1, CR-01) is closed. No regression. Three
human items remain (Mac suite, Mac end-to-end run, the neutral-posture display ruling).
**Re-verification:** Yes. Previous verdict gaps_found 11/12 at `2210b73` (preserved below).
Round 2 was plans 08-15..08-19.

**Integrity of this run.**
- The registry was untouched. `registry/trials.jsonl` sha256 prefix was `c957e8fdb360` at the
  start, after the full suite, after every real-data run and at the end. `total_trial_count()`
  read **44** each time.
- Every real-data run used scratch copies of `data/checkpoints` and `data/holdout` extracted from
  `git archive HEAD` into
  `/tmp/claude-0/-home-user-claude-scratch-work/64f0b80b-f0fb-5914-ae2e-935a933024c0/scratchpad`
  (`ver2_*`, `v2`, `mut`, `trunc`), via `TC_DATA_DIR` / `TC_OUTPUT_DIR`, `--out` or `--dump-curves`.
- `git status --porcelain -- data outputs registry` was empty at the end. No source, test,
  registry, data or output file was modified. Mutations ran on a sandbox copy of
  `src/ tests/ scripts/ config/ data/ registry/ outputs/` in the scratch directory, and each
  mutated file was restored and diffed clean afterwards.
- One untracked file, `.planning/phases/07-regime-representation/07-VERIFICATION-WAVE2.md`,
  appeared during this run. It is not this verifier's; a concurrent agent shared the machine and
  the scratch directory.

## Step 0: the previous gap, re-verified first

**Previous gap (criterion 1, PER-02, CR-01):** L_t divided the posterior by the whole-window label
prior, not by the nowcaster's training prior.

**Now: ✓ CLOSED.** Evidence, each with how it could fail:

1. **The code.**
   - `driver.training_class_prior` (`driver.py:308-340`) is computed inside
     `fit_l2_nowcaster` (`:343-389`) from the rows the model is actually fit on. Those are the
     embargoed, CV-active, finite rows, pre-dropped at `:383-385`. It is returned as the third value.
   - `_refit_l2` returns `(posterior, class_prior)` (`:392-422`).
   - `run_backtest` passes it to `filter_step` (`:615, :642`).
   - `joint_driver` passes each classifier's own step prior (`:588, :598, :618-619` →
     `_filtered_belief` → `filter_step`, `:363`).
   - `serving.py` persists it as `nowcaster_class_prior` (`:138, :182-187`).
   - `weekly._served_class_prior` loads and validates it against `classes_` before any state is
     loaded or saved (`:492-518, :648`). `advance_regime_belief` requires `class_prior`, with no
     default (`:233-276`).
   - `likelihood_ratio` refuses a prior whose support is not the posterior's states, or that does
     not sum to 1 (`regime_filter.py:175-205`).
   - π_0 and A stay on the labels, over all K states: "two roles, two rules", stated in every
     docstring.
2. **Live, real tracked data (scratch).** `serving` exit 0 (classes [0, 3, 4], 55 columns).
   `weekly` ×2, exit 0 both times, report **byte-identical** (`cmp`).
   - **Persisted prior:** `nowcaster_class_prior` equals `[11/153, 137/153, 5/153]` exactly (float `==`).
   - **Independent re-derivation of the training block:** `build_nowcaster_training_set` →
     `model.feature_names_in_` columns → finite rows, counted with my own `value_counts`. Result:
     **153 rows, 2007-04-30 → 2019-12-31, {0: 11, 3: 137, 4: 5}**.
   - **Could fail:** if the fit trained on different rows than the prior's, or if serve persisted a
     different object.
3. **Hand-derived belief, independent of `filter_step`.**
   - Method: numpy π_0 = label frequencies (695 labels, {0: 40, 1: 228, 2: 71, 3: 200, 4: 84, 5: 72}).
     A = row-normalised transition counts. L = post / prior on classes_, 1.0 elsewhere.
     Normalise((π_0 A) · L).
   - **Training prior:** gives (0: 0.299992, 1: 0.292707, 3: 0.163236, 5: 0.092841, 2: 0.091552,
     4: 0.059672), argmax 0. This equals the persisted `regime_belief` checkpoint, and **equals
     08-SERVING §4.3's recorded exact floats** (dict `==`, all six).
   - **Old label prior:** gives 3: 0.369157, argmax 3.
   - **Could fail:** the check discriminates, because the two priors give different top states.
4. **Executed book (scratch `executed_weights`):** TLT 0.433693, SPY 0.320510, USO 0.135247,
   IAU 0.110549. This equals 08-SERVING §4.3.
5. **Backtest path is on the new prior, proven on real data.**
   - I re-ran `scripts/diagnose_s1_truncation.py 1982-08-31` on the current source (834.5 s).
   - It is **bit-identical** to the committed CR-01 belief matrices on all 66 rows ≤ T, for both
     classifiers. This holds by the script's check (max abs diff 0.0) **and** by my NaN-aware
     `assert_frame_equal(check_exact=True)` (answering WR-08's nanmax blind spot).
   - The same truncated output **differs** from the pre-CR-01 belief at `b7193fd`, for both
     classifiers.
   - **Could fail:** it shows both that the committed l2 artifacts came from the current code, and
     that the comparison can tell old from new.
6. **Mutation proof (sandbox copy; every mutation turned its target tests red; unmutated control
   passed):**

| # | Mutation | Tests that went red |
|---|---|---|
| M1 | weekly divides by the restricted label prior | `test_weekly_divides_by_the_served_training_prior_not_the_label_prior`, `TestTheServedModelOnTheTrackedData::test_the_training_prior_flips_state_3s_evidence_and_the_served_belief`, `test_cold_start_calls_the_shared_helper_and_the_tilt_gets_the_belief` |
| M2 | `fit_l2_nowcaster` returns the window-label prior | 4 serving tests incl. `TestTrainingClassPrior`, the persistence test and the real-data test |
| M3 | joint driver passes the window prior | `test_the_likelihood_prior_is_the_step_fits_training_prior`, `test_cold_start_is_unconditional_belief_on_the_steps_own_labels` |
| M3b | `run_backtest` passes the window prior | `test_the_likelihood_divides_by_the_fits_training_prior`, `test_cold_start_is_the_shared_unconditional_belief` |
| M4 | delete `likelihood_ratio`'s support check | `test_a_prior_with_mass_on_a_state_the_posterior_lacks_is_refused` |

## Goal Achievement

### Observable Truths

Each row names how its check could fail.

| # | Truth | Status | Evidence (this run) and how the check can fail |
|---|---|---|---|
| 0 | Per-step probability matrix persisted | ✓ VERIFIED (regression) | **l2 files:** `joint_lift_probs_{1,2}_l2.parquet` are unchanged since 08-19's control (dated 2026-09-22; not in `git diff b7193fd HEAD`). The belief matrices hold **488 rows, 1974-02-28 → 2020-12-31**. **Live:** the truncation re-run wrote belief matrices through `write_probability_matrix`, and they were bit-identical to the committed ones (above). **l1only:** the dry run (row 6) reproduced `joint_lift_probs_{1,2}_l1only.parquet` `cmp`-identically. **Could fail:** any drift changes a byte |
| 1 | A prior-state belief propagates, cannot leak — via an explicit Bayes filter | ✓ VERIFIED (**gap closed**) | **Bayes-correct likelihood:** Step 0 above. **Propagation:** re-derived from the committed parquet. argmax(belief) ≠ argmax(posterior) in **380 / 488** (#1) and **178 / 488** (#2). **No leak:** truncation at 1982-08-31, 66 / 66 rows bit-identical under two comparators. 08-19's `s1_truncation_invariance.json` records 3 of 3 cuts. The full suite (incl. the substituted-arm leakage guard) passes. **Zero train/serve skew on the prior:** one helper, returned by the one fit recipe, persisted, identity-tested at all three call sites; M1-M3b show each site's test bites. Gated off under l1only (`joint_driver.py:610`) |
| 2 | Both churn series reported, neither masquerades | ✓ VERIFIED (re-measured) | **From the committed parquet, my own argmax:** B1 **64 / 487** (#1) and **27 / 487** (#2); B0 **221 / 487** and **66 / 487** (unchanged); 488 rows, 1974-02-28 → 2020-12-31, 100 degraded. Belief max-prob ≥ 0.70: **273 / 488** and **408 / 488**. All match 08-MEASUREMENTS §11.3. **Track A** (246 / 587, 24 / 587) is unchanged: the l1only curves are `cmp`-identical. **Could fail:** a mismatch with the record, or B1 derived from the posterior |
| 3 | Track A diagnosed before changed | ✓ VERIFIED (regression) | Previous recomputation stands. Round 2 did not touch `terminal_month_labels.parquet` or `track_a/` (`git diff be1d7ba HEAD` lists no such file). The l1only `state_*` columns are byte-identical (dry run) |
| 4 | §5.3 anti-flicker gates allocation (ruled mechanism) | ✓ VERIFIED (regression) | `execute_rebalance` is still called at the 3 sites. The band pins pass in the full run. The live page carries the 5.0% band sentence. The decision-bearing l1only curves are `cmp`-identical to the committed band-on curves |
| 5 | §4.4 criterion 3 RUN, both classifiers, four schemes | ✓ VERIFIED (regression) | Round 2 touched no stability file. The full-sample reference reproduction test passes in the full run |
| 6 | Criterion 7 re-measured, both legs, one harness, window inline | ✓ VERIFIED (**live on the final source**) | I ran `scripts/run_joint_lift.py --routing l1 --dry-run` myself (NO_REGISTRY, scratch data). **All four l1only parquets `cmp`-identical** to the committed files. The JSON record matches the committed `measurement_l1only.json` field by field (`lift` 28 keys, both leg KPIs, both DSR values), except the by-construction fields: `decision_bearing`, `dry_run`, `registry_row_written`, `quality_tier.governs` / `verdict`, `n_trials_read_at`, and the registry block. The registry block's differences are the 42 → 44 read, 0 rows, NO_REGISTRY tags, and CR-06's `adr_0002_ceiling` → `declared_ceiling`. **l2 (observational):** re-derived from the committed curves with my own log1p / drawdown / Sharpe: `wealth_delta` **−0.2967834422163289**, `dd_delta` **+0.063258**, Sharpe **1.178896 / 1.159271**, turnover 0.139896 / 0.070970, 588 steps 1972-01-31 → 2020-12-31, 100 degraded. All equal §11.3 |
| 7 | A11 answered, written as the reversal | ✓ VERIFIED (regression) | The dry run's DSR blocks equal the committed ones, at hurdle 2.2268911497604993 (44 trials). The l2 record reads `quality_tier_ok` False on both legs |
| 8 | G6 pinned as the known non-compliance | ✓ VERIFIED (regression) | `TestDriverConsumer` passes in the full run. 08-17 changed the pooling fake to a tuple return, and the pin still asserts the unpooled table (08-17-SUMMARY) |
| 9 | Recorded counts match reality; F-4 fixed | ✓ VERIFIED | **Live `--collect-only`: 2494.** CLAUDE.md:128 and :606, and README.md:5 and :477, all say **2494**. The full run: **2494 passed, 0 skipped**. F-4's 0.4190800681431005 is in the unchanged l1only records |
| G-08-1 | Criterion-7 re-derivation portable and discriminating | ✓ VERIFIED (Linux; Mac is human item 1) | The full run passes every portability arm. `_floats_agree` / `_record_mismatches` are untouched by round 2 (not in the diff). The previous 3-mutation proof stands |
| G-08-2 | Weekly runs end to end from supported commands | ✓ VERIFIED on tracked data (Mac step 1 is human item 2) | **Live:** serving → weekly ×2, exit 0, byte-identical, scored as of 2026-06-30, with the 1-distinct-vector disclosure and the neutral sentence. **Refusals, live:** with no serving artifacts, `FileNotFoundError` names `nowcaster` and the build command, and the checkpoint count stays 19 → 19. **New:** with `nowcaster_class_prior` deleted, `FileNotFoundError` names `nowcaster_class_prior` and the build command, 24 → 24, and no belief / hysteresis / executed checkpoint is written |

**Score:** 12 / 12 truths verified. 0 are present-but-behavior-unverified.

### Round-2 plan must-haves (08-15..08-19)

| Plan | Must-have | Status | Evidence and how it could fail |
|---|---|---|---|
| 08-15 | Neutral posture: no per-asset row, one sentence. Active posture: only the active regime's rows, each naming its regime. No asset twice | ✓ | **Live page:** 0 rows plus the neutral sentence. **M5** (neutral prints all rows) turned 3 tests red, incl. `test_no_asset_appears_twice_in_any_posture[None]` and the `main()` tracer. Code: `weekly.py:414-425` |
| 08-16 | Training prior from the fit's own rows; persisted; weekly divides by it; the real-data test fails under the old prior | ✓ | Step 0 items 1-4; M1 and M2 |
| 08-16 | Served posterior bit-for-bit unchanged | ✓ | The live posterior is [0.41800356506238856, 0.5639928698752229, 0.018003565062388593], equal to the 08-SERVING §1 item 6 floats. `test_refit_l2_equals_the_inline_backtest_recipe` passes |
| 08-17 | Both drivers divide by the step fit's prior; `likelihood_ratio` refuses the old shape; l1only pins unmodified | ✓ | M3, M3b, M4. The l1only real record reproduced `cmp`-identically (row 6). No loosened assertion found: 08-17-SUMMARY lists every re-derived expectation, old → new. One assertion (`start == class_prior`) was dropped and replaced by the stronger identity tests M3 bites |
| 08-18 | CR-06: pre-flight before any append; `--declared-ceiling` required; no `assert` | ✓ | `grep -E "^\s*assert\b" scripts/run_joint_lift.py` returns nothing. `ADR_0002_CEILING` is absent. `preflight_trial_budget` runs before `load_platform_config` / `build_inputs` (`:307-314`). **M6** (pre-flight no-op) turned 4 tests red, incl. the `python -O` subprocess arm and the "1 input build and 2 leg calls BEFORE" sequence check |
| 08-18 | CR-05: the effective `use_regime_filter` on rows; `harness_plan` replaces `plan` | ✓ | `driver.py:727` and `joint_driver.py:704, :722`. **M7** / **M8** (nominal flag) each turned the effective-flag test red. **Convention verified from both ledgers:** every `run_backtest`-shaped row (42 archived) predates `ec354b1` (2026-09-23T20:10:50Z), and every joint row in the live ledger is `L1_ONLY_LAST_FILTERED_STATE` |
| 08-19 | l1only byte-unchanged; l2 re-measured behind controls; S-1 adjudicated; old → new amendments; served belief re-recorded | ✓ | `git diff --quiet b7193fd HEAD` over the six l1only files is clean, and my dry run reproduced them. l2, B1 and the served belief were re-derived (rows 1, 2, 6; Step 0). 08-MEASUREMENTS §11 and 08-SERVING §4 exist with old → new tables. §1-§10 / §1-§3 are not overwritten |

### Known, already-accepted limitations (confirmed disclosed, not re-opened)

| Limitation | Where disclosed | Confirmed |
|---|---|---|
| CR-02 / CR-03, deferred to Phase 8.1 | 08-MEASUREMENTS §11.5; 08-SERVING §4.6; STATE:575-581, :758; ROADMAP Phase 8.1 | ✓ Every re-measured L2 number is qualified in both amendments |
| Input-independent served posterior (q2-ii) | 08-SERVING §2.2 / §3.4; on the page | ✓ The live page says "1 distinct posterior vector across 231 complete months" |
| CR-04 (`run_stability` keying) | STATE:580 | ✓ Tracked todo |
| Test-side hard-coded-44 sites | 08-18-SUMMARY "affects"; STATE:581 | ✓ Phase 8.1 |
| Staleness cap data-relative | 08-SERVING §2.1:134, §3.4 | ✓ CONFIRMED by Glenn 2026-09-29. Previous human item 3 closed |
| Page shows no per-account trade rows (no `report.accounts` configured) | 08-SERVING §3.4 | ✓ Pre-existing; the executed book is in the checkpoint |

### Key Link Verification

| From | To | Status |
|---|---|---|
| `fit_l2_nowcaster` (fit rows) → `training_class_prior` | `_refit_l2` → `filter_step` (both drivers) | WIRED (identity-tested; M3 / M3b) |
| `fit_l2_nowcaster` → serving `nowcaster_class_prior` | `weekly._served_class_prior` → `advance_regime_belief` → `filter_step` | WIRED (live; M1 / M2) |
| `run_joint_lift` CLI `--declared-ceiling` | `preflight_trial_budget` before `build_inputs` | WIRED (M6) |
| `run_backtest` / `run_joint_backtest` effective flag | `trial_config["use_regime_filter"]` → `append_trial` | WIRED (M7 / M8) |
| hysteresis `active_regime` (None) | §2 Trajectory and §3 Per-Asset Signals | WIRED (live page; M5) |

### Data-Flow Trace (Level 4)

| Artifact | Data | Source | Real data? | Status |
|---|---|---|---|---|
| weekly distribution | served posterior | `nowcaster.pkl` | Real, constant (accepted q2-ii) | FLOWING, input-independent |
| weekly filtered belief / executed book | belief | `advance_regime_belief` with the persisted training prior | Real; hand-derived exactly | FLOWING (gap closed) |
| l2 records | belief matrices | `run_joint_backtest` (l2) | Real; truncation-reproduced | FLOWING (observational; CR-02 / CR-03 qualified) |

### Behavioral Spot-Checks and Probes

| Behavior | Command | Result | Status |
|---|---|---|---|
| Full suite, once | `pytest tests/ -q` | 2494 passed, 0 skipped, 1132.89 s | ✓ PASS |
| Live collection | `pytest --collect-only -q` | 2494 | ✓ PASS |
| Serving + weekly ×2, real tracked data | `report.serving`; `report.weekly` ×2 | exit 0 ×3; byte-identical; belief = §4.3 exact floats | ✓ PASS |
| Missing serving artifacts | weekly, no artifacts / no class prior | FileNotFoundError naming the artifact and command; 0 files written | ✓ PASS |
| Real-data causal invariance on the new path | `diagnose_s1_truncation.py 1982-08-31` + NaN-aware compare | 66 / 66 bit-identical (both); differs from the pre-CR-01 belief | ✓ PASS |
| l1only record reproducible on the final source | `run_joint_lift.py --routing l1 --dry-run` | 4 / 4 parquets `cmp`-identical; record equal except by-construction fields | ✓ PASS |
| Nine mutations | sandbox copy | each turned its target tests red | ✓ PASS |

No `scripts/*/tests/probe-*.sh` exist, and no plan declares one.

### Requirements Coverage

| Req | Plans (incl. round 2) | Status | Evidence |
|---|---|---|---|
| PER-01 | 08-01 | ✓ SATISFIED | Criterion 0 |
| PER-02 | 08-06, 08-08, 08-12..08-14, **08-16, 08-17, 08-18** | ✓ SATISFIED (was PARTIAL) | Criterion 1; gap closed |
| PER-03 | 08-01, 08-08, **08-17, 08-19** | ✓ SATISFIED | Criterion 2 (B1 re-measured 64 / 487, 27 / 487) |
| PER-04 | 08-02 | ✓ SATISFIED | Criterion 3 |
| PER-05 | 08-09, 08-12..08-14, **08-15** | ✓ SATISFIED | Criterion 4; neutral-posture display fixed |
| PER-06 | 08-03, 08-07 | ✓ SATISFIED | Criterion 5 |
| PER-07 | 08-10, 08-11, **08-18, 08-19** | ✓ SATISFIED | Criterion 6 (live dry-run reproduction) |
| PER-08 | 08-05 | ✓ SATISFIED | Criterion 7 |
| PER-09 | 08-04 | ✓ SATISFIED | Criterion 8 |
| PER-10 | 08-01, 08-10, 08-11, 08-13..08-19 | ✓ SATISFIED | Criterion 9 (live 2494) |

No orphaned requirement.

### Anti-Patterns and Warnings

| File | Finding | Severity |
|---|---|---|
| round-2 `src` / `scripts` files (6) | TBD / FIXME / XXX: none. No new TODO / HACK / PLACEHOLDER lines in the diff | — |
| `backtest/driver.py` vs `joint_driver.py` | WR-04 (open, out of scope): on a degraded step `run_backtest` HOLDS the belief, while `joint_driver` and serve advance by predict-only. "One rule" is not literally true on missing observations | ⚠ Warning (carried from 08-REVIEW; not a round-2 must-have) |
| `scripts/diagnose_s1_truncation.py` | WR-08: `nanmax` hides NaN-vs-value cells and REPO is hard-coded. 08-19 and this run add a NaN-aware second comparator, but the script itself is unchanged | ⚠ Warning |
| `joint_driver.py::quality_tier` | Annualised Sharpe fed to the PSR with monthly `n_obs` (units mix). The FAILED verdict is robust; the DSR magnitudes (1e-46, 1e-48) are not interpretable as stated | ℹ Info (carried) |
| `08-G6.md` §1 | "Three call sites" and stale line refs (`driver.py:497`, `weekly.py:249`). `src` has 5 `vol_targeted_tilt` call sites; the tilt call is now at `driver.py:657`, `weekly.py:692` and `evaluation/report.py:557` | ⚠ Warning (carried; the pin still holds) |

### Stale claims in planning docs (for the orchestrator to fix)

1. **`.planning/STATE.md`:**
   - Frontmatter: `stopped_at` "Phase 8 EXECUTED (10/10 plans)" and `last_activity` 2026-09-23
     ("Suite 2150"). `total_plans: 62` / `completed_plans: 61` do not reflect 19 Phase 8 plans.
   - "Current focus: Phase 7".
   - Body: "PHASE 8 EXECUTED — 10 of 10 plans" and "Suite **2392** passed".
   - No record of 08-11..08-19, CR-01 / CR-05 / CR-06 closure or the new B1.
   - `:575-577` still lists CR-01, CR-05 and CR-06 as open review findings.
   - `:55` states B1 = 81/487 and 30/487 as current.
2. **`.planning/ROADMAP.md`:**
   - `:30`: Phase 8 is `[ ]`.
   - `:657-668`: 08-11..08-19 are all `[ ]`, although every one has a SUMMARY and commits.
   - `:619`: "**Plans**: 10 plans across 5 execution waves" (now 19).
   - Phase 4 criterion 3 (`:16-17`) still reads "hysteresis bands … so the target mix doesn't
     flip" (the 08-A7 §2 rewording is unapplied; carried from the previous verification).
3. **`.planning/REQUIREMENTS.md:250`**: PER-10 "pinned to a live collection, **2392**". Live is **2494**.
   PER-02 (`:242`) could note the CR-01 closure.
4. **`08-UAT.md`**: frontmatter `status: diagnosed`. Test 1 expects "2393 passed" (live is 2494).
   G-08-1 and G-08-2 have not been re-UAT'd since closure (human items 1-2).
5. **`CLAUDE.md:607`**: "(10 skipped: HDBSCAN + cssselect optional)". The live run has **0 skipped**
   (carried).
6. **`08-MEASUREMENTS.md` §1 row 2 (`:33`) and §2 (`:47`), and `08-CHURN.md:270`**: they state
   B1 81 / 487 and 30 / 487 with no forward pointer to §11. The old numbers should stay (never
   overwrite), but a "superseded by §11 (CR-01)" pointer is missing, so a reader of §1 sees a
   pre-CR-01 number as current. The same applies to 08-MEASUREMENTS §7's closed items (carried).
7. **`08-G6.md`**: the census and line refs (above).

### Open questions for Glenn (recorded by the executors; not verification items)

- **08-19 / 08-MEASUREMENTS §11.4:** the belief-clears-0.70 counts behind the keep-absolute
  threshold ruling moved 307 → 273 and 415 → 408 of 488. Does he want to revisit the ruling? No
  quantity moves unless he does.
- **08-15:** in neutral posture, one sentence and no rows (implemented), or all regimes' rows
  grouped and labelled? (Human item 3.)
- **08-18:** the joint half of CR-05 was included on the planner's reading of the review's fix
  text; Glenn's summary named `run_backtest` only.

## Gaps Summary

**No gaps.**
- **CR-01:** the previous gap, closed on every path.
  - **Code:** one helper and one object, with train/serve identity at three call sites.
  - **Live real data:** the served belief was hand-derived exactly, and it discriminates (old top
    state 3, new 0).
  - **Backtest:** the committed l2 artifacts were reproduced by the current source under
    truncation, and they differ from the pre-fix ones.
  - **Tests:** mutation-proven at every site.
- **Decision-bearing l1only record:** reproduced byte-for-byte on the final source by a live dry run.
- **Suite and registry:** the suite is green at the recorded 2494; the registry is byte-identical at 44.
- **What remains is human:** the Mac suite, the Mac networked build → serving → weekly, and
  Glenn's display ruling.
- **Qualifications:** every L2-routed number stays observational and carries the accepted CR-02 /
  CR-03 (Phase 8.1) qualification.

---

## Previous verification (2026-09-29T17:58:58Z, HEAD `2210b73`) — preserved

**Verdict then:** `gaps_found`, **11 / 12**. Full run 2459 passed, 0 skipped (380.65 s). Registry
c957e8fdb360 / 44 before and after.

**Its one gap (now closed above).** Criterion 1 (PER-02) was partial: *"L_t is computed as
posterior / whole-window label prior (regime_filter.py::likelihood_ratio, fed by
unconditional_belief over every in-window label). The nowcaster's posterior is calibrated to its
own TRAINING class distribution … On the real served model the training prior is {0: 0.072, 3:
0.895, 4: 0.033} (153 rows, 2007-04-30 -> 2019-12-31). The prior used is {0: 0.058, 3: 0.288, 4:
0.121, 1: 0.328, 2: 0.102, 5: 0.104} (695 labels …). The implemented L_t for state 3 is 1.96 …
The training-prior L_t is 0.63 … the top state changes. That belief is what the weekly executed
book consumes. … also present on the observational l2 backtest leg (B1 = 81/487 and 30/487 were
measured through it). It is NOT on the decision-bearing l1only leg."* Artifacts named:
- `regime_filter.py::likelihood_ratio`
- `weekly.py::advance_regime_belief`
- `driver.py:582-584`
- `joint_driver.py:354-357`

"Missing": a Glenn decision, a discriminating test, re-measurement of B1 and the l2 lift, a
re-recorded served belief, and ADR-0004 pricing. **Round 2 delivered each:** Glenn's ruling
"fix small ones, then close"; the 08-16 / 08-17 tests; the 08-19 §11 / §4 amendments; budget 0.

**Its truths table (numbers as measured then; superseded values noted):**

| # | Then | Evidence then |
|---|---|---|
| 0 | ✓ | probs l1only 588 rows 1972-01-31 → 2020-12-31; l2 488 rows 1974-02-28 → 2020-12-31, 100 / 588 degraded; truncation at 1982-08-31: 66 rows bit-identical to the (pre-CR-01) belief |
| 1 | ✗ PARTIAL | propagation 273 / 488 and 155 / 488 (now 380 / 178); no leak; wrong likelihood prior (the gap) |
| 2 | ✓ | Track A 246 / 587, 24 / 587; B0 221 / 487, 66 / 487; B1 81 / 487, 30 / 487 (now 64 / 27); l1only identity 588 / 588; l2 mismatches 184 and 276 / 488 |
| 3 | ✓ | #1 246 / 242 / 245 / 248 / 247 / 249 of 587 for k = 1..6; #2 24 / 24 / 25 / 24 / 24 / 24; k = 1 anchored 0 / 588 mismatches |
| 4 | ✓ | Band-off → band-on turnover 0.163251 → 0.127412 (baseline), 0.119788 → 0.080972 (joint); no-trade months 0 → 304 / 588 and 0 → 341 / 588; `active_regime == state_1` 588 / 588 |
| 5 | ✓ | 8,883 rows; #1 state 0 episodes 5 / 9 / 9; #1 state 2 LOO degenerate; `evaporated` 0 fires vs 36 zero-reference rows (disclosed) |
| 6 | ✓ | l1only `wealth_delta` −0.1253065774082902, `dd_delta` +0.026164; l2 −0.300608 / +0.061011 (now −0.296783 / +0.063258); Sharpe 0.896446 / 0.899378 |
| 7 | ✓ | `expected_max_sharpe(44, 1.0)` = 2.226891, `(42, 1.0)` = 2.208694; FAILED 2 of 2 |
| 8 | ✓ | the pooled-stats monkeypatch made `TestDriverConsumer` fail |
| 9 | ✓ | 2459 live and at four sites (now 2494); F-4 0.4190800681431005 |
| G-08-1 | ✓ | 3 mutations of `_floats_agree` each turned their target arms red |
| G-08-2 | ✓ | serving + weekly ×2 byte-identical as of 2026-06-30; book then TLT 0.390654 / SPY 0.330723 / USO 0.144729 / IAU 0.133895 (now 0.433693 / 0.320510 / 0.135247 / 0.110549); refusals at lag 4 and with no `nowcaster.pkl` |

**Its human items and their fate:**
1. Mac full suite: carried, now at 2494.
2. Mac build → serving → weekly: carried.
3. Staleness cap reading: **closed**, confirmed by Glenn 2026-09-29.
4. Neutral-posture per-asset rows (24 unlabelled rows): **fixed by 08-15**. The display-form
   ruling is carried as the new human item 3.

**Its warnings (all carried unless noted):**
- `regime_filter.py` wrong prior: **closed**.
- `weekly.py:392-400` neutral-posture rows: **closed** (08-15).
- The `08-G6.md` census.
- `platform_settings.yaml` `fred_m2sl` / `fred_totalsl` `shift: false` and the `div_yield` lag:
  now **CR-03 → Phase 8.1**.
- `joint_driver.py:597-610` belief across non-comparable refits: now **CR-02 → Phase 8.1**.
- The `quality_tier` units mix (info).

**Its stale-claims list:**
- REQUIREMENTS:250 "2392": still stale, now vs 2494.
- STATE: still stale.
- ROADMAP 08-11..08-14 unchecked and the Phase 4 criterion 3 rewording unapplied: still stale.
- 08-MEASUREMENTS §7: still stale.
- CLAUDE.md:607 "10 skipped": still stale.
- 08-G6: still stale.
- 08-UAT "diagnosed" / "2393": still stale.

---

_Verified: 2026-09-29T21:40:00Z_
_Verifier: Claude (gsd-verifier)_

## Orchestrator note — 2026-09-29

Human item 3 (neutral-posture display) is **CLOSED**: Glenn ruled "One sentence, no rows" — the
behaviour 08-15 built stands. Items 1–2 (Mac full suite; Mac build → serving → weekly) remain open.
