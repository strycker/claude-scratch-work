# Phase 7 Plan 3 — Wave-1 Measurements (criteria 2, 3, 4)

**Measured:** 2026-09-14, two real full 588-step L1+L2 walk-forward evaluations
(`scripts/run_policy_trials.py --variant frozen` then `--variant impute`), against the
D-02-A-recomputed dev `monthly_features` checkpoint (708 × 53, `oil` at 708 non-NaN months —
precondition asserted and passed in both runs).

**Read this document with D-04 in view: every number below is reported, never used to select
or re-run the policy.** The frozen ten-column policy was chosen before either run in this
document executed, on the structural requirement that the driver and the reference fit on the
same feature space (07-CONTEXT.md D-01/D-02-A). Nothing here sends the phase back to try
another policy — not a bad `dd_delta`, not a low ratio, not a high disagreement percentage.

---

## 1. Three labelled states, not two

Every affected number below carries **three** columns, per D-05/D-02-A:

- **pre-fix (9-col, stale)** — sourced from `.planning/BASELINE-v1-tracer-bullet.md`'s
  "current reference run" table and `07-PREFIX-EVIDENCE.md`. Produced under BOTH the
  pre-D-01 EXPANDING driver policy AND the stale 9-column `monthly_features` checkpoint
  (`oil` truncated to 1985-02). **Compound baseline — see §6.** (Also see §5 for a distinct,
  additional limitation: the frozen-vs-pre-fix comparison windows differ in size and end date.)
- **post-fix, frozen (10-col)** — this plan's Variant A, `trial_tag=P7-W1-frozen-10col`,
  the published run under `outputs/reports/platform/*`.
- **post-fix, impute (13-col, REJECTED)** — this plan's Variant B (D-03), `trial_tag=
  P7-W1-impute-13col-REJECTED`, isolated under `outputs/reports/platform/trials/impute-13col/`.
  Recorded so the ADR's rejection is evidence-backed, **never a candidate policy**.

## 2. Main pre/post table (D-05)

**⚠ Read `pct_disagree` and the §5.4 ratio rows below together with their sample windows,
not as bare percentages.** `n_compared` differs (470 vs. 356) AND the comparison windows have
different end dates (2020-12 vs. 2017-05) — see the "sample window" column added below and §5
for the full mechanism. **82.77% → 80.90% is NOT a clean 1.9-point improvement measured over the
same population; it is two different-sized, different-dated samples that happen to both be
high.** The same caveat applies to the ratio's `n_resolved`/`n_transitions` denominator.

| Quantity | Pre-fix (9-col, stale) | Post-fix, frozen (10-col) | Sample window (pre-fix vs. post-fix) | Universal bound | Domain band | Verdict |
|---|---|---|---|---|---|---|
| `pct_disagree` (with `n_compared`) | 0.8276595744680851 (389/470) | **0.8089887640449438** (288/356) | 470 months, 1974-02→2020-12 **vs. 356 months, 1974-02→2017-05** — **different populations, not directly comparable as a delta** (§5) | `x ∈ [0, 1]` — passed | `< 0.02` suspicious **[ASSUMED]** | in-band; not suspicious (`suspicious=False`); **window-narrowing limitation applies — see callout above and §5** |
| `median_sojourn` (months) | 97.0 | **83.0** | same window caveat as above (both derived from the same walk-forward run) | `∈ [0, 588]` — passed | none (no target, D-04) | in-band |
| `median_lag` (months) | 164.0 | **138.0** | same window caveat as above | `∈ [0, 588]` — passed | none | in-band |
| §5.4 ratio | 0.591 | **0.6014492753623188** | resolved-transition window narrower for frozen (see `n_resolved`/`n_transitions` row) | `> 0` — passed | no numeric target (D-04) | in-band; **window-narrowing limitation applies** |
| `n_resolved` of `n_transitions` | 4 of 6 | **7 of 7**, but resolved only within the 356-month, 1974-02→2017-05 window | `n_resolved ≤ n_transitions` — passed | `n_transitions > 30` implausible **[ASSUMED]** | in-band (7 ≤ 30); **"7 of 7" is 7-of-7-available, not 7-of-7-over-the-full-588-step decision range — see §5** |
| strategy `terminal_log_wealth` | 4.0265 | **4.025085120485386** | `abs(x) < 10.0` — passed | `[-3, 12]` per leg [CITED] | in-band |
| strategy `max_drawdown` | −21.24% (33 mo) | **−26.42% (58 mo)** | `x ∈ [-1, 0]` — passed | trend `> -0.50` [CITED] | in-band |
| ablation `terminal_log_wealth` | 3.6472 (derived: 4.0265 − 0.379267) | **3.647237762337872** | `abs(x) < 10.0` — passed | `[-3, 12]` per leg | in-band |
| ablation `max_drawdown` | ≈ −19.80% (derived: −0.2124 − (−0.014364)) | **−19.81% (35 mo)** | `x ∈ [-1, 0]` — passed | — | in-band |
| `wealth_delta` | +0.379267 | **+0.3778473581475139** | `abs(x) < 15` (arithmetic) — passed | `abs(x) < 5` **[ASSUMED]** | in-band on both bounds |
| `dd_delta` | −0.014364 | **−0.0661242048614149** | `x ∈ [-2, 2]` — passed | `abs(x) < 0.5` **[ASSUMED]** | in-band on both bounds |
| Multiclass Brier | 0.2087 (n_steps not recorded in the source doc) | **0.20717126722358387** (n_steps=356) | `x ∈ [0, 1]` — passed | no-skill floor `0.16` [CITED] | in-band; **expected to move — see §4, labels changed, not the nowcaster** |
| Regime occupancy (smoothed reference, states 0–4) | 1.6% / 14.0% / 31.9% / 40.6% / 11.9% | **11.51% / 9.06% / 35.54% / 31.51% / 12.37%** (states 0/1/2/3/4) | each `∈ [0,1]`, `Σ=1.0` — passed (Σ=1.0 to 1e-9) | soft warning `< 0.08`, `> 0.35` (§4.4 crit. 1; **corrected 2026-09-17** from a fabricated `0.05` floor) | floor: all pass. cap: **state 2 at 35.54% is a 0.54pp overshoot of the ~35% cap — marginal, within tolerance of an approximate bound**. Pre-fix variant failed BOTH ends (1.6%, 40.6%) |
| Mean monthly turnover | 0.0734 | **0.052170468675799005** | — | — | recorded |
| CVaR(5%) | −0.0463 | **−0.04356810003646968** | — | — | recorded |
| Registry rows, this evaluation | n/a (predates `trial_tag`) | **+2** (34 → 36) | exactly `2` — passed | — | in-band |

All bands marked **[ASSUMED]** are provisional per `07-VALIDATION.md` and are read here as
advisory flags only (D-07) — none gated this run, none is a target, and a breach (none
occurred) would have been recorded as a flagged verdict, not a halt.

## 3. D-03 rejected alternative — 13-feature + back-fill imputation (evidence, not a candidate)

Run once, logged once, isolated from the published artifacts by construction
(`output_dir=outputs/reports/platform/trials/impute-13col/`).

| Quantity | Post-fix, frozen (10-col) — published | Post-fix, impute (13-col, REJECTED) | Verdict |
|---|---|---|---|
| Frozen L1 feature list | `cape_shiller, credit_spread_baa_aaa, curve_10y3m, div_yield, oil, real_rate_level, realized_vol_1m, realized_vol_3m, trailing_return_1m, trailing_return_3m` (10) | `cape_shiller, credit_spread_baa_aaa, curve_10y2y, curve_10y3m, div_yield, fred_vix, gold, oil, real_rate_level, realized_vol_1m, realized_vol_3m, trailing_return_1m, trailing_return_3m` (13 — all lean columns, since the back-fill removes every remaining NaN in the decision window) | as designed |
| strategy `terminal_log_wealth` | 4.025085120485386 | 4.048185034686693 | recorded |
| strategy `max_drawdown` | −26.42% (58 mo) | −17.91% (44 mo) | recorded |
| `wealth_delta` | +0.3778473581475139 | +0.40094727234882077 | recorded |
| `dd_delta` | −0.0661242048614149 | +0.01898849550676429 | recorded |
| §5.4 ratio (`n_resolved` of `n_transitions`) | 0.6014 (7 of 7) | 1.2686567164179106 (3 of 6) | recorded — the impute variant's ratio rests on **3** resolved transitions, indicative only |
| `pct_disagree` (`n_compared`) | 0.8089887640449438 (n=356) | 0.8319148936170213 (n=470) | recorded |
| Regime occupancy | every state within §4.4 crit. 1's ~8%–~35% band | **state 0 at 1.29%** — below §4.4's real ~8% floor (**corrected 2026-09-17**: cited as a 5% floor, a threshold absent from §4.4; 1.29% fails either way) | **flagged** — noted here, not gated (D-07) |
| Registry rows, this evaluation | +2 (34 → 36) | +2 (36 → 38) | in-band |
| Published artifacts touched? | **yes** (`outputs/reports/platform/*`) | **no** — confirmed: `outputs/reports/platform/backtest_kpi_table.parquet`'s `strategy` row reads `4.025085` (frozen's value), not `4.048185` (impute's value) | **variant isolation confirmed** |

**D-03's substantive rejection reason, restated:** the imputation is non-causal by
construction — for each of `curve_10y2y`, `gold`, `fred_vix` (see note below on `oil`), the
pre-start gap is back-filled with that column's own first observed value. There is no history
before a series begins; fabricating pre-1990 VIX levels would place an invented stress feature
inside a crisis classifier. The alternative is retained here purely as evidence that the
rejection was tested, not merely argued.

**Note on `oil` in the imputation list:** `scripts/run_policy_trials.py`'s
`LATE_START_COLUMNS` names the historical four excluded columns
(`curve_10y2y, gold, oil, fred_vix`) from D-02's original (pre-D-02-A) 9-column set. After the
07-02 recompute, `oil` already has full 1962+ coverage in the dev checkpoint (0 leading NaNs),
so the back-fill is a **documented no-op** for `oil` specifically — the imputation variant's
effective policy imputes only `curve_10y2y`, `gold`, and `fred_vix`. This is noted, not hidden:
the resulting 13-column set is identical either way, since `oil` was already going to qualify
for the frozen reference set without imputation (07-02).

## 4. Why the Brier and confusion tables move — a mechanical fact, not model improvement

`full_sample_states` is reindexed onto the walk-forward's decision dates as `y_true`
(`report.py`, step (e)). Both the feature-space correction (D-02-A) and the driver freeze
(D-01) change WHICH smoothed labeling gets reindexed, so the **labels** the nowcaster is
scored against changed. Brier moving from 0.2087 (pre-fix) to 0.20717 (frozen, post-fix) is
this mechanical relabeling — it is not evidence the nowcaster's calibration improved or
worsened, per D-05.

## 5. Named limitation on criterion 3: the window-narrowing (human sign-off condition, 07-04 MUST carry this)

**This is a named limitation, not a footnote.** `07-04`'s ADR and its own copy of the D-05
pre/post table MUST state the differing sample size and end date at the point the numbers are
shown (inline, per §2's table above), not only in a discussion section — the same requirement
this section satisfies here.

**Mechanism, verified vs. inferred.** Verified directly against `driver.py`: `frozen_l1_features`
is threaded into `run_backtest`'s per-step loop and reaches ONLY `_refit_l1` (`driver.py`
lines 461 and 473, both call sites: `_refit_l1(train_features, cfg,
frozen_features=frozen_l1_features)`). `_refit_l2`'s signature (`driver.py` lines 266-271:
`_refit_l2(train_features, train_states, feature_row, cfg)`) takes NO
`frozen_features`/`frozen_l1_features` parameter at all — L2 can
only be affected through `train_states`, i.e. through L1's changed OUTPUT LABELS, never
directly by which columns L1 was frozen to. This is consistent with (not proof of) the observed
occupancy shift (smoothed-reference state 0: 1.6% pre-fix → 11.51% frozen post-fix) and with
the degrade-count shift (232/588 frozen vs. 118/588 pre-fix/impute) — a different L1 label
sequence changes which windows present the L2 CV split with only one class present. **The
mechanism beyond "L1's changed labels are the only channel through which L2 could be
affected" is INFERRED, not measured** — this record does not claim to have traced the specific
per-window class-imbalance chain from feature-set to degrade; it states only that the code path
makes label-mediation the only possible channel, and leaves the rest as a plausible, unproven
explanation.

**Criterion 3 is satisfied as worded**, per the human sign-off: post-fix disagreement (80.90%,
288/356) was measured by the located `label_disagreement` methodology (Task 1, delegation
proven exact against the 389/470 baseline) and is reported against the 82.8% baseline with **no
target set** — criterion 3 does not require the two numbers to be measured over identical
windows, only that the same methodology produced both. **The window-narrowing limitation
qualifies how the 82.77%→80.90% comparison should be READ (not as a clean same-population
delta); it does not fail the criterion.**

**D-04 holds, unchanged by this finding.** The degrade/window-narrowing finding is NOT grounds
to reopen the frozen-policy decision or to prefer the imputed (rejected) variant, even though
the imputed variant's window happens to match the pre-fix baseline's more closely. The policy
was chosen on the structural driver/reference equivalence requirement (D-01/D-02-A), declared
before either run in this document executed — this finding is additional reported evidence,
never a re-selection input.

## 6. Compound-baseline statement (required, D-02-A / D-05)

**The comparison basis moved twice, and this record cannot separate the two causes.** The
pre-fix numbers above were produced under BOTH:

1. the pre-fix EXPANDING driver feature-admission policy (A13's original asymmetry between
   `driver.py::_window_active_features` and `report.py::_reference_label_columns`, frozen only
   by plan 07-01's D-01, landed in commit `cc375db`); **and**
2. the stale 9-column `monthly_features` checkpoint (07-02's D-02-A correction, `oil`
   truncated to 1985-02 instead of the full 1962-01 history).

Any movement between the pre-fix column and the post-fix (frozen) column above is a mix of
the D-01 policy freeze and the D-02-A data correction. **This record cannot attribute a given
delta to one cause alone**, and does not attempt to. Attributing the whole delta to the policy
change would repeat exactly the "fooled by its own backtest" failure mode
`UAT-AUDIT-2026-09-09.md` documents.

**A related, independently observed fact worth naming plainly (not attributed to a single
cause):** the frozen 10-column policy's walk-forward produced **232 of 588** L2-degraded
steps (T-05-05, "at least 2 classes" solver failures), leaving only **356** non-degraded
steps to compare — spanning 1974-02-28 through **2017-05-31**, not the full window. The
imputed 13-column variant produced only **118** degraded steps (470 non-degraded, spanning
through 2020-08-31) — the SAME degraded-step count as the pre-fix baseline's 118-row gap
(`07-PREFIX-EVIDENCE.md` §1). This means the frozen policy's `pct_disagree` and §5.4 ratio
rest on a **smaller and differently-dated** sample than either the pre-fix or the impute
comparison. This is reported as a finding — a narrower, differently-timed comparison window —
not as a reason to prefer the imputed alternative (D-04 forbids that inference).

## 7. D-04 restated

None of the numbers in this record were used to select the frozen ten-column policy. The
policy was chosen on the structural requirement that the driver and the reference fit on the
same feature space, declared before either run in this document executed (07-CONTEXT.md D-01,
07-01-PLAN.md, landed prior to this plan). The disagreement percentage, the §5.4 ratio, and
both ablation deltas measured above are reported findings. In particular:

- `dd_delta` came back **more negative** for the frozen policy (−0.0661) than the pre-fix
  baseline (−0.014364) — the regime layer's drawdown cost under the frozen policy is larger
  than under the pre-fix (compound) baseline. This is recorded honestly and is **not** grounds
  to revisit the policy (D-04) — gate semantics are D-06: wave 2 proceeds whatever `dd_delta`
  turns out to be, provided it is honestly measured.
- The imputed (rejected) alternative's `dd_delta` (+0.0190) and `wealth_delta` (+0.4009) both
  look nominally more favorable than the frozen policy's. This is exactly the situation D-04
  exists to guard against: a nicer-looking number on the rejected alternative is not a reason
  to switch to it. The rejection stands on its substantive, non-causal-imputation reasoning
  (§3 above), decided before either run executed.

## 8. Registry accounting (T-07-15)

Read live via `read_trials()` immediately before/after each of the two `run_full_backtest_evaluation`
calls (never quoted from a prior session):

| Point | Count |
|---|---|
| Before either run (before Variant A) | **34** |
| After Variant A (frozen) | **36** (+2) |
| Before Variant B (impute) | 36 |
| After Variant B (impute) | **38** (+2) |
| **Net delta across both runs** | **+4** (exactly `2 × N_full_evaluation_runs`, N=2) |

The last 4 rows of `registry/trials.jsonl`, in order, carry `config.trial_tag`:
`P7-W1-frozen-10col`, `P7-W1-frozen-10col`, `P7-W1-impute-13col-REJECTED`,
`P7-W1-impute-13col-REJECTED` — every row attributable to the run that produced it.

## 9. Variant isolation (T-07-14)

Confirmed directly: `outputs/reports/platform/backtest_kpi_table.parquet`'s `strategy` row
holds `4.025085120485386` (the frozen variant's number). The impute variant's
`4.048185034686693` appears only under `outputs/reports/platform/trials/impute-13col/
backtest_kpi_table.parquet`. The published artifacts were never touched by Variant B.

## 10. Task 1's silent-zero trap — confirmed not triggered in either real run

Both runs asserted `not (pct_disagree == 0.0 and n_compared == 0)` — the coercion-bug
signature `07-PREFIX-EVIDENCE.md` documents — and both passed with real, non-zero
`n_compared` (356 for frozen, 470 for impute). `measure_label_disagreement`
(`platform/evaluation/disagreement.py`, this plan's Task 1) performed the `state_N`-string
coercion correctly in both real runs.

---

## Sign-off: APPROVED

**Task 3's human-verify checkpoint was reviewed and approved.** No number above was used to
select or revise the policy (D-04). The human's decision settled two points explicitly:

1. **Criterion 3 is satisfied as worded** — measured and reported against the 82.8% baseline,
   no target set. The window-narrowing limitation (§5) qualifies the INTERPRETATION of the
   82.77%→80.90% comparison; it does not fail the criterion.
2. **D-04 holds** — the degrade/window-narrowing finding is not grounds to reopen the policy
   decision or to prefer the imputed (rejected) variant.

**Binding condition carried into `07-04`:** the window-narrowing must be foregrounded as a
named limitation at the point the numbers are shown (§2's table, inline) — not left as a later
discussion-only footnote. `07-04`'s ADR and its own copy of the D-05 pre/post table MUST repeat
this treatment. See §5 above and `.planning/phases/07-regime-representation/07-03-SUMMARY.md`'s
"Binding condition for 07-04" section for the full requirement text.
