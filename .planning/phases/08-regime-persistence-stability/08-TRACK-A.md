# Phase 8 — Track A: the terminal-month diagnostic (plan 08-02, PER-04, ROADMAP criterion 3)

**Measured:** 2026-09-22, one full run of `scripts/terminal_month_diagnostic.py` (commit
`c01cfc8`), artifacts committed at `214329f`:

- `outputs/reports/platform/track_a/terminal_month_labels.parquet` — the 588 × 25
  (step × lag × classifier) label matrix, a **date per cell**
- `outputs/reports/platform/track_a/terminal_month_churn.json` — the derived record

Every number below can be recomputed from the parquet. `tests/unit/test_platform_terminal_month_diagnostic.py::TestPersistedRecord::test_churn_recomputes_from_the_matrix`
does exactly that, and a mutation check showed it fails when the JSON is altered.

**Registry trials consumed: 0.** `total_trial_count()` is **42** before and after (ceiling 44,
ADR-0002). Both classifiers were fit at their **pinned** (K, λ) only. No configuration was
varied, so no configuration was evaluated.

---

## 1. What was measured, and on what

**The question.** The L1 DP minimises `Σ_t d[t, s_t] + λ · Σ_t 1[s_t != s_{t−1}]`
(`labeling/jump_model.py:43-45`). The terminal month has no right-hand neighbour, so a
single-month deviation there costs **λ once**. The same deviation at an interior month costs
**2λ** (jump in, jump out). The filtered labeling reads exactly that month on every step
(`joint_driver.py:502`, `state_1 = states_1.iloc[-1]`). The objective implies the asymmetry
exists. This record measures how big it is.

**The method.** At each walk-forward step *t*, the step's **own** L1 fit reports the label it
gives to the k-th-from-last month of its **own** training window, for k = 1 … 6. Each of the six
`iloc[-k]` series is then churned across steps with the dropna-then-compare rule, and each rate
is divided by the number of adjacent pairs.

| | classifier #1 | classifier #2 |
|---|---|---|
| pinned K | 6 | 5 |
| pinned λ | 10.0 | 16.0 |
| resolved frozen columns *d* | **10**: `cape_shiller, credit_spread_baa_aaa, curve_10y3m, div_yield, oil, real_rate_level, realized_vol_1m, realized_vol_3m, trailing_return_1m, trailing_return_3m` | **8**: `rs_equities_bonds, rs_oil_equities, equities_tr_mom_12m, long_duration_tr_mom_12m, oil_mom_12m, corr_equities_tr_long_duration_tr_24m, cpi_acceleration, m2_gdp` |
| **λ / d** (from live config ÷ resolved list, not literals) | **1.0** (10.0 / 10) | **2.0** (16.0 / 8) |
| steps | 588, 1972-01-31 → 2020-12-31 | 588, 1972-01-31 → 2020-12-31 |
| degraded steps | **0 of 588** | **0 of 588** |
| steps where the fit's terminal month ≠ `train_index[-1]` | **0 of 588** | **0 of 588** |
| lag cells recorded (non-NaN) | 588 per lag, all 6 lags | 588 per lag, all 6 lags |

**The anchor (T-08-07).** The harness's step index equals the index of
`outputs/reports/platform/joint_lift/joint_lift_joint_l1only.parquet` elementwise. The k = 1
column equals that file's `state_1` / `state_2` elementwise with **0 of 588 cells mismatched**
for each classifier, and the derived k = 1 churn is exactly **246** and **24**. No tolerance was
applied. This harness therefore fits the same classifier that produced every recorded number,
and the k > 1 columns below are comparable to it.

An independent re-check ran after the run, outside the harness's own assertion. It confirmed
that every `c{N}_lag{k}_date` equals *t* − k months at all 588 steps: each lag reads the month
its name says it reads.

## 2. The churn-vs-k table

Every rate is `n_changes / n_pairs`, where `n_pairs = 587` (588 steps − 1). Window: 588 steps,
1972-01-31 → 2020-12-31. 0 degraded.

| k | month read | classifier #1 (λ/d = 1.0) | classifier #2 (λ/d = 2.0) |
|---|---|---|---|
| 1 | `iloc[-1]`, today's `state_N` | **246 / 587 = 41.91%** | **24 / 587 = 4.09%** |
| 2 | `iloc[-2]` | 242 / 587 = 41.23% | 24 / 587 = 4.09% |
| 3 | `iloc[-3]` | 245 / 587 = 41.74% | 25 / 587 = 4.26% |
| 4 | `iloc[-4]` | 248 / 587 = 42.25% | 24 / 587 = 4.09% |
| 5 | `iloc[-5]` | 247 / 587 = 42.08% | 24 / 587 = 4.09% |
| 6 | `iloc[-6]` | 249 / 587 = 42.42% | 24 / 587 = 4.09% |

Classifier #1's six values span **242 → 249** changes over 587 pairs, a range of 7 changes
(1.19 pp). k = 6 exceeds k = 1 by 3 changes. Classifier #2's span **24 → 25**.

### Secondary panel — the same matrix read down the other axis

This panel re-reads the same cells and is **not** a second experiment. For each calendar month
*m*, it compares the label *m* received when it was the terminal month (lag 1) with the label a
later fit gave it at lag k.

| k | classifier #1: labels of month *m* revised vs. its lag-1 label | classifier #2 |
|---|---|---|
| 2 | 237 / 587 = 40.37% | 24 / 587 = 4.09% |
| 3 | 263 / 586 = 44.88% | 29 / 586 = 4.95% |
| 4 | 273 / 585 = 46.67% | 30 / 585 = 5.13% |
| 5 | 280 / 584 = 47.95% | 42 / 584 = 7.19% |
| 6 | 298 / 583 = 51.11% | 44 / 583 = 7.55% |

The denominator falls by one per k because the lag-k series starts k − 1 months earlier than
the lag-1 series and ends k − 1 months earlier, so the two share 588 − (k − 1) calendar months.

## 3. The two readings, written before the verdict

Both readings are taken from ROADMAP criterion 3 and `08-RESEARCH.md` F-3, which were written
before this run. The verdict in §4 picks one of them. No threshold separating "falling" from
"flat" existed before the numbers, and none is invented here.

- **Falling in k.** Churn at k = 2 … 6 is materially lower than at k = 1. Then a material share
  of the 41.91% is an edge artefact of reading `iloc[-1]`. The honest description of Track A
  would change from "the labeling is memoryless" to "the filtered read samples the DP's
  cheapest-to-deviate month."
- **Flat in k.** Churn at k = 2 … 6 is about the same as at k = 1. Then the terminal-month edge
  story is **refuted**, λ/d is left as the explanation, and classifier #1's labeling is unstable
  at λ/d = 1.0 across the months measured, not only at the edge.

## 4. Verdict

**Flat in k. The terminal-month edge artefact is refuted for classifier #1:** reading the
6th-from-last month instead of the last gives 249 changes against 246 over the same 587 pairs,
and no lag from 2 to 6 falls below 242.

**Scope of the refutation.** The following limits what this run tested. It does not weaken the
verdict.

- **What k = 1 … 6 tests is the single-month λ-vs-2λ asymmetry.** That asymmetry is refuted.
- **What it cannot exclude is a longer trailing-block effect.** Relabelling the trailing *block*
  of the last j months as one run also costs a single λ, for any j. Flat churn out to k = 6
  shows the instability is not confined to the last month. It does not show that it continues
  past six months. The secondary panel points the same way and is no stronger: a fixed month's
  label keeps being revised more often the further it recedes from the edge (40.37% at k = 2,
  51.11% at k = 6). That is what an unstable labeling looks like. It is not what a one-month
  edge artefact looks like.
- **What it does not establish is that λ/d is the cause.** It removes the edge explanation. The
  contrast between the two classifiers (41.91% at λ/d = 1.0 against 4.09% at λ/d = 2.0) is
  consistent with λ/d. But the two classifiers also differ in K (6 vs 5), in feature set
  (disjoint by D-10) and in *d*, so λ/d is **not isolated** by this comparison. Only varying λ
  on one classifier would isolate it, which is a λ sweep, and §5 rules that out.

**Classifier #2** is flat as well (24 → 25, over 587 pairs). It has no edge artefact to find,
which matches its 2.4× filtered ÷ full-sample ratio in F-3.

The honest description of Track A is therefore unchanged in substance from `08-CONTEXT.md`
D-01. D-01 recorded 246/588 = 41.84%; the pair-denominated figure per `08-RESEARCH.md` F-4 is
246/587 = 41.91%. **Classifier #1's filtered labeling churns at 41.91% (246/587), and that
churn is not an artefact of which of the last six months the filtered read samples.**

## 5. What this does NOT license

- **A lambda sweep is not authorized** (`08-CONTEXT.md` Q-7; ROADMAP criterion 3). Each value
  costs one registry trial against 42 of the 44 allowed by ADR-0002, and a sweep needs its own
  ruling. The finding that λ/d is not isolated (§4) is **recorded, not acted on**.
- **No re-pin of K or λ follows from this record.** Classifier #1 stays at K = 6 / λ = 10.0 and
  classifier #2 at K = 5 / λ = 16.0. This plan changed no production code: `driver.py`,
  `joint_driver.py` and `labeling/` are untouched.
- **Track A's number is not a target, and none is declared.** 41.91% is a measurement. This
  record declares no churn level as acceptable or unacceptable.
- **Nothing in §5.1 or §5.3 can move Track A, and this record is not evidence about either.**
  §5.1's Bayes filter changes L2 (Track B). §5.3's hysteresis changes the allocation response.
  Neither touches `state_1`, which is L1's terminal-month label. Refuting the edge story does
  not transfer to either track.
- **No 2021+ data was read.** The holdout carve mirrors `run_joint_backtest` exactly, and the
  anchor against the tracked curve's index confirms the window ends at 2020-12-31.

## 6. Forward pointer

Plan **08-10** re-measures criterion 7 under the changed labeling. This record is what tells a
reader of that measurement **whether Track A was expected to move**: it was not. Track A is L1,
and nothing in this phase changes L1's (K, λ). So if 08-10 reports `state_1` churn different
from 246/587, the §5.1/§5.3 work did not cause it, and something is wired wrong (ROADMAP
criterion 2: "A §5.1-style change must NOT move it; if it appears to, something is wired
wrong").
