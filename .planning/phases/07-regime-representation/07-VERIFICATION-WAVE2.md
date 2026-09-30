---
phase: 07-regime-representation
scope: phase-wave 2 ONLY — plans 07-05..07-12; ROADMAP criteria 5, 6, 7, INV-01, REG-01's wave-2 clauses, and criterion 8 as wave 2 touched it
verified: 2026-09-29T21:44:27Z
status: human_needed
score: 17/18 wave-2 truths verified (1 UNCERTAIN; 0 FAILED)
behavior_unverified: 0
overrides_applied: 0
re_verification: false   # first verification of wave 2; wave 1 is 07-VERIFICATION.md
wave1_resolution:
  file: 07-VERIFICATION.md
  recorded_status: human_needed   # STALE header; see "Stale claims" below
  resolved_by: 07-UAT.md UAT-2 (signed 2026-09-15, verdict accept-with-caveats)
  reopened_here: false
registry:
  sha256_prefix_before: c957e8fdb360
  sha256_prefix_after: c957e8fdb360
  total_trial_count: 44          # 38 header + 6 rows (2 x 07-07, 2 x 07-11, 2 x 08-10)
  rows_written_by_this_verification: 0
qualifications:
  - id: CR-03
    owner: Phase 08.1 (Point-in-Time Data Audit)
    summary: "fred_m2sl (classifier #2's m2_gdp) and div_yield (classifier #1's lean set, and the dividend leg of equities_tr) enter unshifted. Every wave-2 number that consumes them is conditional on unlagged inputs. Absolute-performance verdicts (DSR does not clear; A11 FAILED) stand in direction. The SIGN of wealth_delta / dd_delta is NOT direction-guaranteed. Criterion 5's band pass does NOT survive a 1-month M2SL lag (measured below)."
human_verification:
  - test: "Re-confirm the 07-UAT-WAVE2 UAT-3 acceptance of REG-01 as PARTIAL now that (a) A11 (2026-09-21) turned criterion 7 into FAILED-as-a-result and (b) CR-03 shows criterion 5's pass is conditional on unlagged M2SL"
    expected: "Glenn states whether the wave-2 closure still stands as signed, or whether criterion 5 should be re-labelled 'MET, conditional on 08.1'"
    why_human: "Acceptance of an open REG-01 clause is a human decision already taken once; the inputs to that decision have since changed"
  - test: "Decide whether ROADMAP criterion 7 and REQUIREMENTS REG-01 should carry the A11 retroactive FAILED-as-a-result annotation"
    expected: "Criterion 7 reads 'MET as a measurement (2026-09-21); FAILED as a result under ADR-0003 (A11)', matching 08-A11.md §3.4 ('the record now says both')"
    why_human: "08-A11.md says the record now says both; the Phase 7 ROADMAP block still says only ✅ MET. Editing the record is the orchestrator's/owner's call, not the verifier's"
  - test: "Decide whether Phase 08.1's framing 'Look-ahead can only flatter, so Phase 7–8's negative verdicts stand in direction' should be narrowed"
    expected: "It holds for the DSR / A11 verdicts (absolute performance of each leg). It does not hold for the sign of wealth_delta or dd_delta, which are differences of two differently exposed legs"
    why_human: "Scope wording of a future phase"
---

# Phase 7 — Wave 2 Verification (Regime Representation: the leadership classifier)

**Phase goal (ROADMAP.md):** decide what the regime labeler should see, prove that decision
walk-forward, and add a **second, independent labeler on relative/leadership features** so the
platform can inform allocation during ordinary markets, not only crisis avoidance.

**Scope of this pass:** phase-wave 2 only: plans 07-05..07-12, criteria 5, 6 and 7, INV-01,
REG-01's wave-2 clauses, and criterion 8 as far as wave 2 touched it.
Wave 1 (criteria 1–4) is **not** re-opened.

**Verified:** 2026-09-29T21:44:27Z · **Status:** `human_needed` · **Re-verification:** no. This is the
first verification of wave 2.

**Method.** Every criterion below is backed by a **live re-derivation** from the dev checkpoints
or by a **check that can fail**, and each one names how it could fail. Nothing here is taken from
a SUMMARY. The rules this pass followed:

- It ran in the repo venv and wrote **no registry rows**. Every evaluation run used the
  `NO_REGISTRY` sentinel.
- It read no 2021+ holdout rows. `monthly_raw` in the dev checkpoint tree runs to 2026-08-31, so
  every read of it pushed a `date <= 2020-12-31` predicate down into pyarrow. No post-cutoff row
  was materialised, and a loader guard raised if one ever was.
- It wrote no repo files except this report. All scratch output went to the session scratchpad.
- It did not modify source, tests, data or outputs, and it did not commit.

---

## 0. Wave 1's human_needed item: resolved, not re-opened

`07-VERIFICATION.md` (2026-09-15) returned `status: human_needed` on one item: whether the
blanket claim "`platform/` imports nothing from the legacy library" was acceptable.
**07-UAT.md UAT-2 (signed 2026-09-15) resolved it by correcting the claim.** The old grep ended in
`| grep -v platform` and so could not fail. An AST scan found 31 real sites, all older than
Phase 7. The ratchet test replaced the grep. Wave-1 verdict: **accept-with-caveats** (UAT-3).

That resolution is confirmed here by live measurement. An independent AST scan (§5, T9) counts
**31** sites today and **0** in the seven modules wave 2 created.
**`07-VERIFICATION.md`'s header still reads `status: human_needed`.** That is stale; see §9.

---

## 1. Observable truths

| # | Truth (wave-2 scope) | Status | Evidence: live re-derivation or a check that can fail, and **how it could fail** |
|---|---|---|---|
| T1 | **Criterion 5:** classifier #2 exists, is fit unsupervised on a feature set disjoint from classifier #1's 13, a test asserts disjointness, occupancy sums to 1.0, and no state sits outside §4.4 crit. 1 unmarked | ✓ VERIFIED (as-of Phase 7 inputs; **CR-03-conditional**, see §6) | The live refit (`label_leadership_regimes` on the carved spine, scratch checkpoint dir) is **byte-identical** to the tracked `regime_labels_2`: same 696-month index (1963-01-31 → 2020-12-31) and `np.array_equal`. Occupancy recomputed independently from the states is **16.6667 / 22.7011 / 22.4138 / 23.8506 / 14.3678 %**, sum error **0.0**, 12 transitions. Two-sided band [8%, 35%]: **0 violations**. `set(frozen_8) & lean_feature_set(cfg)` = **∅**, with `lean_feature_set` = 13. *Could fail:* the refit differs from the checkpoint (code or data drift); the intersection is non-empty; a state falls outside [0.08, 0.35]. `test_resolved_frozen_list_is_disjoint_from_the_lean_set` and `test_live_occupancy_within_design_44_band` pass (433-test run, §4) |
| T2 | **Criterion 6:** dependence between the two labelings is measured and reported; high dependence would be recorded as a failure to add an axis | ✓ VERIFIED **as a measurement**. The verdict it produced is INCONCLUSIVE, and the phase's own reading (UNRESOLVED, "criterion NOT satisfied") is reproduced, not upgraded | `measure_labeling_dependence(regime_labels, regime_labels_2)` gives ARI **0.354841**, NMI **0.464088**, Cramér's V **0.589748**, `n_compared` **695** (1963-02-28 → 2020-12-31), not suspicious. The pre-registered block-permutation control was **reproduced at the same seed** (20260918, 2000 resamples; blocks 26 / 13): NMI p95 **0.449202**, p99 **0.501195**, observed at the **96.60th** percentile. The rule gives **INCONCLUSIVE**. Every null percentile in 07-DEPENDENCE.md § RESULT matches to the printed digit. *Could fail:* any statistic or percentile differs; the verdict lands in another bucket. Classifier #1's input was also re-derived live (`label_regimes` on the 10 frozen columns), byte-identical to the tracked `regime_labels`. Occupancy 5.7554 / 32.8058 / 10.2158 / 28.7770 / 12.0863 / 10.3597 %. This is a **same-seed reproduction**: it computes no new statistic, no new null and no new seed, so the `298b1bc` pre-registration's no-tie-break clause is respected |
| T3 | **Criterion 7:** joint (#1 × #2) lift measured walk-forward against #1 alone; every configuration registry-logged; deflated Sharpe applied for the full count | ✓ VERIFIED **as a measurement, as-of Phase 7**. **Superseded twice:** by A11 (2026-09-21) → FAILED as a result; and by 08-10 (5pp band) → re-measured. See §3 | (a) **Live re-run on HEAD code, band off, `NO_REGISTRY`:** both legs re-run through `run_joint_backtest` on the carved dev checkpoints with `allocation.no_trade_band = None`, `ROUTING_L1_ONLY`, `NO_REGISTRY`. Result: 588 steps each, 1972-01-31 → 2020-12-31, 0 degraded; **both equity curves are bit-identical, in every column, to the archived Phase-7 curves** (max |Δreturn| = 0.0); terminal log wealth **4.718722841 / 4.595284580**, `wealth_delta` **−0.123438**, `dd_delta` **+0.024084**, Sharpe 0.917073 / 0.914903. This also proves Phase 8's 356-line `joint_driver.py` change is inert with the band off. (b) An independent recompute from the **git-archived Phase-7 curves** (`d5c3ac9`) using its own KPI formulas (Σlog1p, drawdown from cumulative wealth, Sharpe with ddof=1 and ×√12) gives `wealth_delta` **−0.123438**, `dd_delta` **+0.024084**, 588 = 588 steps, 1972-01-31 → 2020-12-31, index equal, 586/588 months differ. (c) The registry holds exactly the two tagged rows (`07-11-c1-alone-L1only` 4.718722841, `07-11-joint-c1xc2-L1only` 4.595284580), whose difference is −0.123438. (d) DSR recomputed at `n_trials` 42 and variance 1.0: **2.28151e-12 / 1.46904e-11**, hurdle 2.208694. An **independent** Bailey–López de Prado implementation gives 1.78e-12 / 1.17e-11 (the gap is only the skew/kurtosis estimator). Both are ≪ 0.5. *Could fail:* the live rerun, the archived recompute or the registry rows disagree with the record; the DSR crosses 0.5 |
| T4 | INV-01 (1): named candidates constructed and screened, with PCA as a discovery tool only | ✓ VERIFIED | Live `screen_invariant_candidates(..., NO_REGISTRY)`: the count stays 44 → 44, survivors are `['m2_gdp', 'credit_gdp']` by name, and the first admissible month for both is 1962-02-28. An AST scan of `invariants.py` finds **zero** `.transform` / `.fit_transform` / `.inverse_transform` calls. *Could fail:* any such call; a survivor named by a component label (also pinned by `test_rejects_integer_or_component_label_index`) |
| T5 | INV-01 (2): loading stability tested across eras, walk-forward | ✓ VERIFIED as executed. ⚠ **Weak evidence**, WARNING in §7 | 10 expanding eras ending 1972-01-31 → 2017-01-31; PC1 loadings 0.70710678 ± 2e-16 for both candidates; `stable`. **Can it fail?** Yes, but not the way the tolerance suggests. With exactly two standardized, positively correlated candidates the PC1 loading is identically 1/√2, so the 0.15 range tolerance **can never bind**. Only the sign clause (the correlation turns non-positive) or the coverage clause (only one candidate is computable in some era) can fire. Synthetic probe: mirroring `credit_gdp` after 1990 → both **`unstable`**; NaN-ing it before 1985 → both **`unstable`**. A real-data probe (mirrored TOTALSL level) stayed `stable`, because expanding-window correlation stayed positive. ADR-0002 already reads this clause "at plan 07-07 §4's own strength" |
| T6 | INV-01 (3): every candidate registry-logged | ✓ VERIFIED | The live ledger has 2 rows tagged `07-07-inv01-screen` (`m2_gdp`, `credit_gdp`, git `0ea91ec`, 2026-09-17T15:09:27Z). `total_trial_count()` = 38 (header) + 6 rows = **44**. *Could fail:* a missing or untagged row, or a count ≠ header + rows |
| T7 | INV-01 (4): survivors admitted as **named** features, never PCs (R4) | ✓ VERIFIED | `m2_gdp` is in classifier #2's resolved frozen eight. `credit_gdp` was dropped for collinearity and the drop is recorded. The live 10-era correlation is **0.9515–0.9899** (the record's "0.957–0.969 across eras" is 3 windows; see §9); the drop rationale holds |
| T8 | **Goal clause:** a second, **independent** labeler (REG-01's orthogonality clause) | ? UNCERTAIN (WARNING; human decision requested, §8) | Independence is **not established**, and the phase says so (ADR-0002 § Requirement coverage; ROADMAP criterion 6 "NOT satisfied"; REG-01 `[~]` Partial). A pre-registered INCONCLUSIVE cannot be broken by a tie-break. This is an accepted open item (07-UAT-WAVE2 UAT-3), not a hidden gap. It is surfaced because two inputs to that acceptance changed after signing: A11, and CR-03 |
| T9 | Criterion 8 as touched by wave 2: no new legacy imports; guard extended | ✓ VERIFIED | Independent AST scan of `src/trading_crab_lib/platform/**`: **31** legacy import sites (`trading_crab_lib` ×16, `.checkpoints` ×7, `.ingestion.http` ×3, `.ingestion.browser` ×2, `.ingestion` ×1, `.ingestion.assets` ×1, `.email` ×1), matching the 07-12 re-measure. **0** in `features/relative.py`, `features/invariants.py`, `labeling/classifier2.py`, `evaluation/dependence.py`, `evaluation/deflated_sharpe.py`, `allocation/joint_tilt.py` and `backtest/joint_driver.py`. `MAX_LEGACY_IMPORT_SITES = 31`, and the ratchet test passes. *Could fail:* count > 31, or any hit in a wave-2 module. (The `OUTPUT_DIR` via `labeling/diagnostics` indirection in `classifier2.py` is disclosed in its docstring.) |
| T10 | 07-05: `canonicalize_states` raises instead of silently ordering on centroid column 0 | ✓ VERIFIED (behavioral probe) | A direct call with `sort_column='rs_equities_bonds'` absent from `feature_names` raises `ValueError`. The default `trailing_return_1m` absent also raises. Classifier #2's call sites (`classifier2.py:386`, `joint_driver.py:313`) pass `sort_column` explicitly. *Could fail:* a restored fallback returns instead of raising |
| T11 | 07-06: the DSR denominator is the whole registry (header + post-header rows), never the raw row count | ✓ VERIFIED | Live `total_trial_count()` = 44 against 7 physical lines = 1 header (`prior_genuine_trials` 38) + 6 rows. *Could fail:* it returns 7 or 6 |
| T12 | 07-06/07-11: a DSR ≤ 0.5 is reported plainly as not clearing | ✓ VERIFIED. ⚠ Unit convention undocumented (§7) | `format_dsr_verdict` quoted verbatim in ADR-0002 and 07-JOINT-LIFT. The verdict is robust to convention: it holds with annualised SR and monthly T (the module's choice), with per-month SR against the same hurdle, and with per-month SR against hurdle/√12 (DSR 1.3e-16) |
| T13 | 07-08 (D-13/D-17): constants pinned and the trial ceiling written **before** the fit and before any evaluation run | ✓ VERIFIED (commit ordering, which can fail) | `56d4506` decisions 2026-09-17 16:39Z → `196cd6b` ADR-0002 Proposed with ceiling 16:46Z → `351eee8` first fit 20:18Z → `f6a42fb` re-pin (a §4.4 structural gate, 11 fits recorded, 0 lift reads) 09-18 16:44Z → lift runs 09-21 14:18Z |
| T14 | 07-10: the four load-bearing bands confirmed by a human **before** any lift number existed | ✓ VERIFIED (commit ordering) | `9b93cd8` band dispositions 2026-09-18 22:59Z < the first tagged lift read 2026-09-21 14:18:34Z. Both governing bands hold at the recorded and at the live values (`abs(−0.123438) < 15`; `+0.024084 ∈ [−1, 1]`) |
| T15 | 07-10: two probability vectors blended at the **weight** level, never a product state space; output contract preserved | ✓ VERIFIED | `blend_regime_tilts` takes `probs_1` and `probs_2` separately. `test_platform_allocation_joint_tilt.py` passes, including `test_pooling_changes_the_portfolio_weights_not_just_a_flag_field`. No product index is constructed |
| T16 | 07-11: one harness, one window: the #1-alone leg is `blend_weight_1 = 1.0` of the same loop | ✓ VERIFIED | Archived curves: `index.equals` True, 588 = 588, 0 degraded on both legs. The `state_1` and `state_2` paths are identical across legs (246 and 24 changes out of 587 on both). Live rerun: indexes equal, and `state_1` / `state_2` identical across the two live legs (246 / 24 changes of 587), matching the archive. *Could fail:* different indexes, or different state paths across legs |
| T17 | 07-09: pre-registered decision rule committed **before** the control code existed | ✓ VERIFIED (commit ordering) | `298b1bc` rule 2026-09-18 18:58:22Z (1 file, docs only) → classifier #1 re-pin labels 19:06Z → `12635d0` control code and result 19:28:54Z. The re-pin was driven by §4.4 criterion 1 (ADR-0001 § RE-PIN), and the rule explicitly anticipated the re-run ("the rule above survives that re-run unchanged") |
| T18 | 07-12: ADR-0002 Accepted carrying the unfavourable results as measured; ratchet re-measured at close | ✓ VERIFIED | ADR-0002 Status is Accepted 2026-09-21, scoped to the decision and explicitly not to an axis finding. It carries −0.123438 in nats and as the 11.61% shortfall, the INCONCLUSIVE verdict and both DSRs, each with its window inline. Ratchet: see T9 |

**Score:** 17 / 18 verified. T8 is UNCERTAIN (accepted open item, re-surfaced). **0 FAILED.**
**0 present-but-behavior-unverified.** Every behavior-dependent truth (T10, T15, T16) has a
behavioral probe or a passing behavioral test.

---

## 2. Criterion-by-criterion verdict (wave-2 roadmap scope)

| ROADMAP criterion | As recorded at Phase 7 close (2026-09-21) | This verification | Later supersession (not a Phase 7 defect) |
|---|---|---|---|
| **5** classifier #2, disjoint, occupancy | ✅ MET | **Reproduced exactly** (T1) | **CR-03:** does not survive a 1-month M2SL lag (§6.2). Owner: 08.1 |
| **6** dependence measured and reported | ⚠ MEASURED; verdict UNRESOLVED; "NOT satisfied" | **Reproduced exactly**, same seed (T2); REG-01 orthogonality stays UNCERTAIN (T8) | Phase 8 did not touch it. CR-03 exposes both inputs; direction unknowable (§6) |
| **7** joint lift, registry, DSR | ✅ MET as a measurement; wealth sign negative | **Reproduced** (T3) | **A11 / ADR-0003 (2026-09-21):** FAILED as a result on both legs. **08-10 (2026-09-28):** re-measured under the 5pp no-trade band, `wealth_delta` −0.125307, `dd_delta` +0.026164, DSR 1.82e-13 / 4.13e-12 against hurdle 2.226891 at 44 trials. All four figures were re-derived here from the archived 08-10 curves (`4bfb81b`) |
| **INV-01** | ✅ IN FULL | **Verified**, with the evidential-weight WARNING on clause 2 (T5) | Low CR-03 exposure (§6) |
| **REG-01** (wave-2 clauses) | `[~]` PARTIAL: criterion 6 open | **PARTIAL confirmed.** Three of four clauses are verified; orthogonality is unestablished | n/a |

---

## 3. Criterion 7: as-of Phase 7, and where Phase 8 superseded it

| | Phase 7 (07-11, 2026-09-21, band off) | Phase 8 (08-10, 2026-09-28, 5pp no-trade band) |
|---|---|---|
| Window | 588 steps, 1972-01-31 → 2020-12-31, 0 degraded (both legs) | same |
| Terminal log wealth, #1 alone / joint | 4.718723 / 4.595285 | 4.606331 / 4.481024 |
| `wealth_delta` | **−0.123438** nats (≈ 0.8839×, an 11.61% shortfall) | **−0.125307** |
| `dd_delta` | **+0.024084** (−25.74% vs −28.14%) | **+0.026164** |
| Sharpe, #1 alone / joint | 0.917073 / 0.914903 | 0.896446 / 0.899378 |
| Mean monthly turnover, #1 alone / joint | 0.163251 / 0.119788 | 0.127412 / 0.080972 |
| Filtered state changes, #1 / #2 | 246 / 24 **of 587 transitions** (recorded as "of 588" = 41.84%; see §9) | 246 / 24 of 587 |
| DSR (n_trials, variance 1.0) | 2.28e-12 / 1.47e-11 (42) | 1.82e-13 / 4.13e-12 (44) |
| Verdict | MET **as a measurement** (D-06) | A11 gate FAILED 2/2 |

Every number in both columns was re-derived in this pass. The Phase-7 column comes from the
archived curves, the registry rows and a live band-off rerun. The Phase-8 column comes from the
archived curves and the registry rows. **A11's retroactive FAILED is a change of rule, not a
change of measurement** (08-A11.md §3.4). It is recorded here as a supersession and not as a
Phase 7 defect.

---

## 4. Behavioral spot-checks and test runs

| Check | Command (scratchpad scripts) | Result | Status |
|---|---|---|---|
| Wave-2 test files, minus the slow joint-driver file (combined with the next row: **480 passed, 0 failed, 0 skipped** across all 14 wave-2 test files) | `pytest` on relative, invariants, classifier2, dependence, deflated_sharpe, joint_tilt, joint_diagnostics_record, legacy_import_ratchet, honesty_registry, registry, macro_ingest, labeling, tilt | **433 passed**, 0 failed, 0 skipped (34.4 s) | ✓ PASS |
| `test_platform_backtest_joint_driver.py` | `pytest --durations=8` | **47 passed**, 0 failed (737.5 s; the slowest tests are real-harness runs, 35–110 s each) | ✓ PASS |
| Criteria 5 + 6 live re-derivation | `c5c6.py` | Byte-identical labelings for both classifiers; statistics and same-seed null reproduced | ✓ PASS |
| Criterion 7 live rerun (band off, `NO_REGISTRY`) | `c7.py band_off {baseline,joint}` | Bit-identical to the archived `d5c3ac9` curves (both legs, every column); −0.123438 / +0.024084 reproduced | ✓ PASS |
| INV-01 screen + falsification probe | `inv01.py` (+ synthetic probe) | Survivors named, 0 transform calls; the stability check fails on a synthetic decorrelation | ✓ PASS |
| canonicalize no-fallback | direct call | Raises `ValueError` on an absent sort column | ✓ PASS |
| `classifier2.py` `__main__` self-check | `python -m trading_crab_lib.platform.labeling.classifier2` | **Raises `ValueError`** (it still builds λ = 4n = 32 against the re-pinned 2n rule) | ✗ stale self-check (WARNING, §7) |

Step 7c (probes): no `scripts/*/tests/probe-*.sh` exists and no wave-2 plan declares one, so there
was nothing to execute.
The full workspace suite was **not** run: a concurrent session's full-suite run was already
occupying the machine.

---

## 5. Required artifacts and key links

| Artifact | Exists | Substantive | Wired | Data flows |
|---|---|---|---|---|
| `platform/features/relative.py` (07-05) | ✓ | ✓ 358 lines | ✓ `add_relative_features` → classifier #2 and `run_joint_lift.build_inputs` | ✓ real `monthly_raw` columns |
| `platform/features/invariants.py` (07-07) | ✓ | ✓ | ✓ consumes `compute_invariant_ratios`, `expanding_steps`, `append_trial` | ✓ |
| `platform/labeling/classifier2.py` (07-08) | ✓ | ✓ | ✓ `canonicalize_states(sort_column=...)`; `_reference_label_columns` reused unmodified | ✓ reproduces `regime_labels_2` |
| `platform/evaluation/dependence.py` (07-09) | ✓ | ✓ | ✓ reuses `plotting/regime.py::label_disagreement` alignment | ✓ |
| `platform/evaluation/deflated_sharpe.py` (07-06) | ✓ | ✓ | ✓ `total_trial_count()` → `n_trials` in `run_joint_lift` / `quality_tier` | ✓ |
| `platform/allocation/joint_tilt.py` (07-10) | ✓ | ✓ | ✓ called per step by `joint_driver` | ✓ |
| `platform/backtest/joint_driver.py` + `scripts/run_joint_lift.py` (07-11) | ✓ | ✓ | ✓ | ✓ reproduces the recorded curves |
| ADR-0002 (07-08 / 07-12) | ✓ | ✓ Accepted | cross-referenced from ROADMAP, REQUIREMENTS and ADR-0003 | n/a |
| `config/platform_settings.yaml` `fred_monthly` M2SL/TOTALSL, `labeling_2`, `allocation.blend_weight_1` | ✓ | ✓ | ✓ `classifier2_config` raises if λ ≠ 2n (proven by the `__main__` failure above) | ⚠ `shift: false` (CR-03) |

---

## 6. CR-03 qualification: publication lag (owner: Phase 08.1; **not** a new wave-2 gap)

Phase 8's code review (CR-03, orchestrator-verified 2026-09-29) found three series entering the
features without publication-lag handling:

- `fred_m2sl` (about 1 month)
- `fred_totalsl` (about 2 months)
- multpl `div_yield` (2–3 months)

### 6.1 Where it enters wave 2's inputs (traced in code)

| Input | Path into wave 2 | Timing under the decision-bearing L1-only routing |
|---|---|---|
| `fred_m2sl` | `m2_gdp` = `fred_m2sl / fred_gdp`, one of classifier #2's frozen eight | The refit at decision *t* trains on months ≤ *t−1* (`expanding_steps`), and the tilt uses the label of month *t−1*. M2SL(*t−1*) is published around the end of *t*, so this is marginal to about 1 month of look-ahead on the decisive label |
| `div_yield` | (i) directly, one of classifier #1's ten frozen columns; (ii) `equities_tr` = price return + `div_yield`/12 (`splice.py:280-295`), which feeds classifier #1's `trailing_return_1m/3m` and `realized_vol_1m/3m` and classifier #2's `rs_equities_bonds`, `rs_oil_equities`, `equities_tr_mom_12m` and `corr_equities_tr_long_duration_tr_24m` | div_yield(*t−1*) is unpublished at *t* by 1–2 months. Channel (ii) is second-order: only the accrual term moves |
| `fred_totalsl` | `credit_gdp`, **not** in classifier #2's frozen eight; it does enter the firewalled L2 leg through `monthly_features` | not decision-bearing in wave 2 |

**Wave-2 origin of half of CR-03.** Plan 07-05 added `M2SL` and `TOTALSL` to the `fred_monthly`
block. That block's own header says its series are "never revised … none of these have
meaningful lag". Both are revised agency series, and the project constraint says "ALFRED for
point-in-time agency data". The 07-05 must-have ("ingested through the existing config-driven
FRED path") is met to the letter. The PIT treatment was not applied. This is recorded as the
origin; the fix belongs to 08.1.

### 6.2 Exposure and direction, per wave-2 criterion and number

| Number | Exposed? | Direction of the bias |
|---|---|---|
| **Criterion 5** occupancy (a full-sample fit) | Yes: `m2_gdp` and the `equities_tr` dividend leg | Not a flattery bias; a full-sample labeling is hindsight by design. **But the pass is fragile, measured here.** This is a diagnostic only: no registry, no performance number and no #1-vs-#2 statistic. With `fred_m2sl` shifted **1 month** and everything else pinned (K=5, λ=16), classifier #2's occupancy becomes **3.4483 / 13.7931 / 22.7011 / 21.8391 / 38.2184 %**, which **breaches §4.4 crit. 1 at both ends**. Only 17.82% of months keep their state id, with 14 transitions. With a **2-month** shift it returns to within the band (16.67 / 24.57 / 20.55 / 23.85 / 14.37 %, 98.13% same id). **Criterion 5's MET is conditional on unlagged M2SL.** 08.1 must re-run it, and the re-pinned constants may not survive |
| **Criterion 6** ARI / NMI / V and the INCONCLUSIVE verdict | Yes: both labelings' inputs | **Not signable.** A PIT-corrected labeling is a different labeling. Given the fragility above, the verdict could land in any bucket. The `298b1bc` pre-registration governs *these* labelings; a PIT re-measure needs its own pre-registration in 08.1 |
| **Criterion 7** absolute legs (terminal wealth, Sharpe, DSR) | Yes: both legs | **Flattering in expectation.** The verdicts "neither DSR clears" and "A11 FAILED 2/2" **stand in direction** |
| **Criterion 7** `wealth_delta` −0.123438 and `dd_delta` +0.024084 | Yes, **with unequal exposure**: the #1 channel is weighted 1.0 against 0.5; the #2 channel (M2) is only in the joint leg | **Sign NOT guaranteed.** A difference of two differently flattered legs can move either way. The **favourable** `dd_delta` has no robustness guarantee, and the negative `wealth_delta` is not guaranteed to stay negative. 08.1's blanket "negative verdicts stand in direction" is correct for the DSR/A11 verdicts only |
| L2 observational leg (Sharpe 1.204395, `wealth_delta` −0.134505) | Yes, **at month *t* itself**: CR-03's primary case; `fred_m2sl`, `fred_totalsl` and `div_yield` are all `monthly_features` columns the L2 nowcaster admits | Flattering; the most exposed number in the wave. Firewalled, and nothing was decided on it |
| Filtered churn 246/587; classifier #2 §5.4 ratio 1.074 (lag 27.0 against sojourn 29.0 mo) | Yes (labels at *t−1*) | Not signable |
| INV-01 screen | Uses unshifted ratios | **Negligible.** For n = 2 the loadings are an identity; the correlations (0.95–0.99) are insensitive to a 1-month shift of a slow level ratio |

---

## 7. Anti-patterns and warnings

| File / artifact | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| `src/trading_crab_lib/platform/labeling/classifier2.py` | `__main__` block (443-472) | Self-check builds `lambda = 4.0 * len(...)` and K = 3; `classifier2_config` raises (confirmed by running it) | ⚠ WARNING | The module's "one runnable check" has been broken since the 2026-09-18 re-pin |
| `classifier2.py` | module docstring (28-33); `CLASSIFIER2_K` comment (129-131); `label_leadership_regimes` docstring NOTE 2026-09-17 | Stale prose: "K = 3 … lambda = 4n = 32.0", "three leadership states" beside `= 5`, "pending a re-pin" | ⚠ WARNING | Misleads a reader. Code behavior is correct (T1) |
| `evaluation/deflated_sharpe.py`, `backtest/joint_driver.py::quality_tier` | `deflated_sharpe_ratio` z-formula; `annualized_sharpe` | **Unit convention undocumented:** an annualised SR (×√12) is paired with monthly `n_obs` in the PSR z and in the non-normality denominator, and compared to a hurdle whose unit depends on the undeclared unit of the 1.0 placeholder variance. `07-DSR-ESTIMATOR-NOTE.md` never mentions annualisation or frequency | ⚠ WARNING | The wave-2 verdict is robust under every convention tested (T12). A future leg near the hurdle would not be, and this now governs the ADR-0003 gate. Flag for 08.1 or the gate owner |
| INV-01 clause 2 | — | With two candidates the 0.15 loading-range tolerance cannot bind. The check fails only on a sign flip or a coverage loss | ⚠ WARNING (evidential weight) | "Era-stable" is evidenced only as "correlation stayed positive in every expanding era". ADR-0002 already concedes this |
| Debt markers (`TBD` / `FIXME` / `XXX` / `TODO` / `HACK`) in the 10 wave-2 source and script files | — | none found | — | — |

---

## 8. Human verification required

1. **Does REG-01 PARTIAL still stand as signed?** (07-UAT-WAVE2 UAT-3). *Test:* re-read the
   acceptance against A11 (criterion 7 is now FAILED as a result) and against §6.2 (criterion 5
   does not survive a 1-month M2SL lag). *Expected:* either "stands" or "criterion 5 re-labelled
   conditional on 08.1". *Why human:* this re-opens an already-signed human acceptance, and its
   inputs have changed.
2. **Annotate criterion 7 with A11.** *Test:* compare ROADMAP criterion 7 ("✅ MET 2026-09-21")
   with 08-A11.md §3.4 ("the record now says both"). *Expected:* the Phase-7 block states "MET as
   a measurement; FAILED as a result (ADR-0003)". *Why human:* the owner decides record edits.
3. **Narrow 08.1's direction sentence.** *Test:* read §6.2's criterion-7 delta row. *Expected:*
   "stand in direction" is scoped to the absolute-performance verdicts. *Why human:* it scopes a
   future phase.

---

## 9. Stale claims in Phase 7 artifacts (report-only; nothing edited)

1. **`07-VERIFICATION.md` header `status: human_needed`.** Stale. It was resolved by 07-UAT.md
   UAT-2 and UAT-3 on 2026-09-15 (accept-with-caveats).
2. **ROADMAP Phase 7, criterion 7 "✅ MET".** There is no A11 annotation. ADR-0002's own
   § A11 RULING and 08-A11.md §3.4 say criterion 7 is FAILED as a result on both legs.
   REQUIREMENTS.md REG-01's delivery note is also silent.
3. **ROADMAP Phase 7 open items say "audit item A11 open by deliberate choice".** A11 was
   answered 2026-09-21 (ADR-0003). The date is inconsistent across records: 08-A11.md and ADR-0002
   say 2026-09-21, while STATE.md says "(Glenn, 2026-09-22)". 07-UAT-WAVE2 also lists A11 as open.
4. **ROADMAP / ADR-0002 / 07-JOINT-LIFT say "§4.4 criterion 3 … never run for either
   classifier".** True at 2026-09-21. Superseded: 08-07 ran it (08-STABILITY.md).
5. **"246 of 588 (41.84%)" filtered churn** (ROADMAP, ADR-0002, 07-JOINT-LIFT, 07-UAT-WAVE2).
   The denominator is wrong: 588 steps give 587 transitions, so it is 246/587 = **41.91%**.
   This was corrected by Phase 8's F-4 fix and is re-derived here (246 and 24 of 587). The
   Phase 7 artifacts still carry the old figure.
6. **"`credit_gdp` correlation 0.957–0.969 across eras"** (07-DECISIONS-07-08, 07-08-SUMMARY,
   ADR-0002). This summarises 3 windows (ending 1972-01, 1997-01, 2020-12). The live 10-era
   range is **0.9515–0.9899**. The conclusion (near-collinear) is unaffected.
7. **ADR-0001 § RE-PIN 2026-09-18 table: classifier #1 state 4 median sojourn "12.0 mo".** The
   labeling it describes, byte-identical to the tracked `regime_labels`, has state-4 runs
   [5, 7, 72], so the median is **7.0 mo**. The same labeling's **state 2 occurs in exactly one
   71-month run.** That is a one-episode state in classifier #1, not disclosed in ADR-0001's
   re-pin; ADR-0002 does disclose classifier #2's states 3 and 4 as one-episode. These are
   inputs to criterion 6.
8. **`07-VALIDATION.md`.** The frontmatter says `status: validated` but `nyquist_compliant: false`.
   Every sign-off box is unchecked, and "Approval: pending — see the 2026-09-21 audit" is never
   closed.
9. **ROADMAP Phase 7 plan list.** The warning "All four plans below are entirely inside
   phase-wave 1 … No plan below touches classifier #2" now sits above a list that also contains
   the eight wave-2 plans (under their own subheading).
10. **`classifier2.py` docstrings and self-check.** See §7.
11. **Wave-1 recorded numbers (context only; wave 1 is not re-opened).** 07-DEPENDENCE §0.3
    already records that 07-MEASUREMENTS' frozen-column occupancy "cannot be reproduced today"
    after the `oil` splice fix (`dbd3fcc`) and classifier #1's re-pin (K 5→6, λ 52→10). The same
    applies to criterion 4's +0.377847 / −0.066124. They remain valid as-of-2026-09-14 records only.

---

## 10. Gaps summary

**No BLOCKER.** Nothing wave 2 claims is a stub, an unwired artifact or a number that fails to
reproduce.

- Criteria 5 and 6 reproduce **byte-for-byte** from live data.
- Criterion 7 reproduces from three independent sources: the archived curves, the registry rows,
  and a live band-off rerun that is bit-identical to them.
- INV-01 re-runs with the same survivors.
- The ratchet holds at 31, with 0 sites in any wave-2 module.

The status is `human_needed` rather than `passed` for three reasons:

- **T8.** REG-01's independence clause is unestablished. This is known and signed, but two
  inputs to that signature changed afterwards.
- **CR-03 (§6).** Criterion 5's pass is conditional on unlagged M2SL, and the sign of criterion
  7's deltas is not direction-guaranteed. Phase 08.1 owns this.
- **§9.** The Phase 7 record has not absorbed A11, F-4 or Phase 8's criterion-3 run.

Registry before and after this pass: `c957e8fdb360` → `c957e8fdb360`, with `total_trial_count()`
= 44 both times. **This verification wrote 0 rows.**

---

_Verified: 2026-09-29T21:44:27Z_
_Verifier: Claude (gsd-verifier), retroactive goal-backward pass over phase-wave 2_
