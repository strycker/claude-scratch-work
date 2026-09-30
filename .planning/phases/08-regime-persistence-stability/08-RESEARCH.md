---
phase: 8
slug: regime-persistence-stability
kind: research
created: 2026-09-21
branch: claude/keen-galileo-zqcml6-w5
source_commit: b1c3519
reads_only: true
---

# Phase 8 — Regime Persistence & Stability: research

**Researched:** 2026-09-21
**Domain:** recursive/autoregressive features in walk-forward classifiers; latent-state
stability testing under subsampling; hysteresis in a probability-driven allocator
**Confidence:** HIGH on the code-and-measurement findings (every number below was re-derived
this session from tracked artifacts); MEDIUM on the literature framing; explicitly LOW where
the design and the literature do not settle a choice and a human must.

---

## 0. Read this first — three measured findings that change the shape of the phase

These are not opinions. Each was re-derived this session from git-tracked artifacts at
`b1c3519`, and each is falsifiable by re-running the one-line computation given beside it.

### F-1. The 41.84% is an **L1** quantity. Adding a prior-state feature to **L2** cannot move it.

`08-CONTEXT.md` D-02 attributes the churn to `nowcaster.py::build_nowcaster_training_set`
having no prior-state feature. The recorded 41.84% is not computed from the nowcaster's
output. It is computed from the `state_1` column of the joint-lift curve, and `state_1` is
set here:

```python
state_1=(None if states_1.empty else states_1.iloc[-1]),
```
[VERIFIED: `src/trading_crab_lib/platform/backtest/joint_driver.py:502`]

`states_1` is `_refit_l1(...)`'s return — the **jump model's** label for the last month of
the training window. `scripts/joint_lift_diagnostics.py:110,138-140` reads exactly that
column:

```python
filtered = joint[curve_col]
...
"walk_forward_filtered": {
    "n_transitions": _n_transitions(filtered),
```
[VERIFIED: `scripts/joint_lift_diagnostics.py:110,138-140`]

And the suite already pins that `state_1` is **identical across both routings** —
`test_the_two_routings_share_the_state_path_exactly`
[VERIFIED: `tests/unit/test_platform_joint_diagnostics_record.py:186-190`]. Routing
`l1only` never calls `_refit_l2` at all; it feeds the tilt
`_last_state_one_hot(states_1)`.

**Consequence.** Success criterion 2 ("filtered state-change rate re-measured over the same
588 steps, reported before and after") will, if it re-measures the same column, report
**246 → 246, unchanged, bit for bit**, whether the §5.1 fix works perfectly or not at all.
That is the exact "success-shaped failure" the caller asked about, arriving through a door
nobody was watching — and it is *guaranteed*, not merely possible.

**Recommendation (falsifiable).** Criterion 2 must name **which series** it measures. Two
distinct churn series exist and both should be reported, side by side, before and after:

| series | what it is | today | can §5.1 move it? |
|---|---|---|---|
| `state_1` churn | L1 jump model's terminal-month label, per refit | **246/587 = 41.91%** (recorded as 246/588 = 41.84%, see F-4) | **No.** L1 is untouched by an L2 feature. |
| `argmax(regime_probs)` churn | the L2 nowcaster's filtered prediction | **never measured anywhere** | Yes — this is the object §5.1 changes. |

Re-derivation: `pd.read_parquet("outputs/reports/platform/joint_lift/joint_lift_joint_l1only.parquet")["state_1"].pipe(lambda s: (s!=s.shift()).iloc[1:].sum())` → `246`.
[VERIFIED: measured this session]

### F-2. The hysteresis is not merely unwired. Under the decision-bearing routing it is a **provable identity**; under the other routing it makes things **worse**.

`update_active_regime`'s three branches [VERIFIED: `allocation/hysteresis.py:82-94`]:

```python
    if prev_active is None:
        top_regime = probs.idxmax()
        return top_regime if probs[top_regime] >= act_threshold else None
    if probs.get(prev_active, 0.0) >= unwind_threshold:
        return prev_active
    qualifying = probs[probs >= act_threshold]
    return qualifying.idxmax() if not qualifying.empty else None
```

Under `ROUTING_L1_ONLY` the input is one-hot:

```python
    return pd.Series({states.iloc[-1]: 1.0}, dtype=float)
```
[VERIFIED: `backtest/joint_driver.py:251`]

With a one-hot input and `0 < unwind_threshold ≤ act_threshold ≤ 1.0`, branch 2 fires iff
`prev_active == argmax`, and otherwise branch 3 selects the single state at p=1.0. The
function is therefore **the identity on argmax**, for every threshold pair the config can
hold. Measured confirmation on the tracked curve: `active_regime` is elementwise equal to
`state_1` in all 588 months, and changes 246 times — the same 246.
[VERIFIED: measured this session, `joint_lift_joint_l1only.parquet`]

Under `l2` routing, where real calibrated probabilities reach it:

| quantity (`joint_lift_joint_l2.parquet`, 588 steps) | value |
|---|---|
| `active_regime` changes | **462 / 587 = 78.7%** |
| `state_1` changes (same rows) | 246 |
| `active_regime` is **None** (neutral / all-cash) | **387 / 588 = 65.8%** |
| None among the 488 **non-degraded** steps | **309 / 488 = 63.3%** |
| degraded steps (nowcaster refit failed) | 100 / 588 |

[VERIFIED: measured this session]

The anti-flicker state machine, fed genuine probabilities, **flickers nearly twice as much
as the raw label it was meant to stabilise**, by oscillating in and out of neutral. The
mechanism is forced: `active_regime` can only be `None` on a non-degraded step if no regime
reached `act_threshold`. So at least **309 of 488 non-degraded steps had max calibrated
probability below 0.70**. That is a lower bound derived from the code's own branch
structure, not an estimate.

`act_threshold: 0.70` / `unwind_threshold: 0.40`
[VERIFIED: `config/platform_settings.yaml:453-454`, verbatim] were written against design
§4.2's HMM-filter intuition, where a forward-backward posterior is sharp. A *calibrated*
multinomial logistic posterior at **K=6** is not. 0.70 is 4.2× the uniform prior 1/6.

**This is why D-04's ordering decision (§5.1 before §5.3) is right, and for a stronger
reason than D-04 gives.** §5.3 cannot be evaluated at all until the probability vector it
consumes stops being degenerate.

### F-3. Classifier #2 churns at **4.08%** through the *same* code path. The architecture is not the cause.

| | classifier #1 | classifier #2 |
|---|---|---|
| filtered (`state_N`) transitions | 246 / 588 = **41.84%** | 24 / 588 = **4.08%** |
| full-sample transitions | 25 / 695 = 3.60% | 12 / 696 = 1.72% |
| filtered ÷ full-sample | **11.6×** | 2.4× |
| K | 6 | 5 |
| fitted feature count *d* | **10** (frozen) | **8** |
| λ | **10.0** | **16.0** |
| **λ / d** | **1.0** | **2.0** |

[VERIFIED: `outputs/reports/platform/joint_lift/diagnostics_l1only.json`; K/λ from
`config/platform_settings.yaml` `labeling:` and `labeling_2:`, re-read via
`load_platform_config()` this session; d=10 recomputed this session via
`_reference_label_columns` →
`['cape_shiller','credit_spread_baa_aaa','curve_10y3m','div_yield','oil','real_rate_level','realized_vol_1m','realized_vol_3m','trailing_return_1m','trailing_return_3m']`]

Both classifiers run through the identical `_refit_l1`/`_last_state_one_hot`/terminal-month
read. One churns ten times more than the other. Whatever "memoryless" means, it is not a
property of the walk-forward architecture — it is a property of classifier #1's
**(K, λ) configuration**, and specifically of λ relative to the feature dimension.

The mechanism is arithmetic and checkable by eye. The DP minimises

> `Σ_t d[t, s_t] + λ · Σ_t 1[s_t != s_{t-1}]`

[VERIFIED: `labeling/jump_model.py:43-45`, verbatim from the docstring] where `d[t,k]` is a
**squared** Euclidean distance in standardized space, so `E[d]` scales with *d* (the number
of columns). At λ/d = 1.0 a jump costs about one standardized-unit² of misfit spread across
all ten dimensions; at λ/d = 2.0 it costs twice that per dimension. Classifier #1's λ was
re-pinned **down from 52.0 to 10.0** on 2026-09-18 — coefficient 4 → coefficient 1 — while
classifier #2 was re-pinned to coefficient 2:

> `lambda is now a formula of the FITTED column count: 10.0 = 1 x n(frozen) = 10.`
[VERIFIED: `config/platform_settings.yaml:329`, verbatim]

There is a further, additive mechanism specific to the *filtered* read: the terminal month
of a DP decode is the only month with no right-hand neighbour, so deviating there costs
**λ once**, where deviating at an interior month costs **2λ** (jump in and jump out). The
filtered labeling reads exactly that month, every step. This is a structural property of
the objective quoted above. [VERIFIED: `labeling/jump_model.py:43-45` + `joint_driver.py:502`
— the arithmetic follows from the two together; the *magnitude* of the effect is `[ASSUMED]`
until measured.]

**Falsifiable prediction, cheap to test, and it costs no selection trial if recorded as a
diagnostic rather than a choice.** For each of the 588 steps, record the label the step-*t*
fit assigns to month *t−1* (`iloc[-1]`, today's `state_1`) **and** to months *t−2 … t−6*
(`iloc[-2:-6]`). Churn the `iloc[-k]` series across steps for k = 1…6. If churn falls
sharply with k, the terminal-boundary effect is confirmed and a large share of the 41.84% is
an **edge artefact of reading `iloc[-1]`**, not a statement about the nowcaster. If churn is
flat in k, the effect is refuted and λ/d is the whole story.

**This does not relitigate D-02.** D-02 quotes design §5.1 correctly and §5.1's fix is worth
building. F-3 says something narrower and checkable: **the fix will not move the recorded
number**, because the recorded number does not measure the thing being fixed. Both can be
true at once, and the phase should say so out loud rather than discover it at UAT.

⚠️ **λ is a tuning dimension.** A λ sweep is a selection exercise: 42 of 44 registry trials
are used (ADR-0002 ceiling). The diagnostic above (churn vs. k, at fixed λ) costs **zero**
trials. A λ sweep costs one per value. **That is Glenn's call, not this document's.**

### F-4. The recorded 41.84% has an off-by-one denominator.

`_n_transitions(filtered)` compares 587 adjacent pairs over 588 rows; the rate divides by
`max(1, len(joint))` = 588.
[VERIFIED: `scripts/joint_lift_diagnostics.py:139-141`]
So `246/588 = 0.41837` is recorded where `246/587 = 0.41908` is the transition rate.
Trivial in magnitude (0.07pp) — but it is exactly the shape of recorded number `08-CONTEXT.md`
D-07 says has misled this project three times. Fix it or state it; do not leave it.

---

<user_constraints>
## User Constraints (from 08-CONTEXT.md)

`08-CONTEXT.md` is a decisions-and-evidence document rather than a
`## Decisions` / `## Claude's Discretion` / `## Deferred Ideas` CONTEXT. Its binding
content is reproduced verbatim below.

### Locked decisions (D-01 … D-07, verbatim extracts)

- **D-03 — the trap this phase must not fall into.** *"The honest feature is the prior
  **predicted (filtered)** distribution, produced recursively within each walk-forward step.
  The prior **smoothed label** is built from future data. Substituting the smoothed label is
  **P1** … **A guard test must FAIL if the smoothed label is substituted.** A test that
  merely checks the feature exists is the evidence-shape failure this project has now
  recorded five times."*
- **D-04 — ordering.** *"Ordering decision (Glenn, 2026-09-21): §5.1 first, then §5.3."* …
  *"The roadmap note is explicit: **do not** address this by smoothing the reported number,
  since §5.4 says to report that quantity prominently."*
- **D-05 — criterion 3 has never been run**, for either classifier. *"This is the test that
  decides whether classifier #1's crisis state is a regime or an episode."* … *"Whichever
  way the result falls, it is recorded."*
- **D-06 — A11 is reopened and must be written as the reversal it is.**
- **D-07 — validation gap G6** (no test pins that `driver.py:497` receives the **unpooled**
  per-regime estimate) and **stale recorded counts** (1705 vs 2018).

### Scope fences (verbatim)

- *"No migration work. That is Phase 9."*
- *"**No dependence statistic of any kind.** Criterion 6 is UNRESOLVED and the
  pre-registration at `298b1bc` forbids tie-breaks."*
- *"No 2021+ holdout use for any selection decision."*
- *"`legacy/` and the reference submodules are untouched."*
- *"The legacy-import ratchet is 31 and may only decrease."*
- *"ADR-0001 condition (iv)'s covariance clause is NOT in scope."*

### Baselines that must not regress (verbatim)

| | |
|---|---|
| suite | **2018 passed, 0 skipped, 0 xfailed** |
| legacy-import ratchet | **31** |
| criterion 7 (L1-only, 588 steps 1972-01-31 → 2020-12-31) | `wealth_delta` −0.123438, `dd_delta` +0.024084 |
| registry | 42 trials, ceiling 44 (ADR-0002) |
| DSR hurdle | `expected_max_sharpe(42, 1.0)` = 2.208694 |

*"Criterion 7 is **re-measured** under the changed labeling. … **No target is pre-declared
for the churn reduction.**"*
</user_constraints>

<phase_requirements>
## Phase Requirements

Requirement IDs are *(to be assigned at planning)* per `ROADMAP.md`. The eight success
criteria are treated as the requirement set.

| # (ROADMAP) | Criterion | Research support |
|---|---|---|
| 1 | §5.1 recursive prior-state feature exists and is honest; guard test **fails** on smoothed substitution | §2 (options + tradeoffs), §3 (guard-test design, four signatures), §6 Pitfalls 1-6 |
| 2 | Churn measurably lower on the same 588-step window | **F-1** — the currently-measured series cannot move; §2.6 names the series that can |
| 3 | §5.3 hysteresis gates allocation, closing A7 | **F-2**, §4 (three mechanisms, what each breaks, measured consequences) |
| 4 | §4.4 criterion 3 RUN for both classifiers | §5 (distance choice measured, Hungarian, subsample schemes, block length, verdict framing) |
| 5 | Criterion 7 re-measured | §4.4 (every §5.3 mechanism changes it), §7 Validation Architecture |
| 6 | A11 revisited and recorded as a reversal | Out of this document's three questions — see §9 Open Questions Q-6 |
| 7 | G6 pinned: `driver.py:497` receives the **unpooled** estimate | §4.5 — seam located and verified |
| 8 | Recorded counts match reality (1705 → 2018) | §9 Q-8, plus **F-4** (a second stale number, found this session) |
</phase_requirements>

## Summary

Three questions were asked. The short answers:

**1. How do you add a recursive prior-state feature without leaking?** The honest options are
(A) a one-pass expanding-window *filtered history* generator, (B) in-step recursive
regeneration, (F) an explicit Bayes filter with no training column at all. (A) costs ~1
minute and carries a named train/serve inconsistency that can be iterated away; (B) costs
**~5 hours** by arithmetic given below and is internally consistent; (F) costs nothing, has
*zero* train/serve skew because it has no training-time analogue, and would require
rewording criterion 1. Teacher forcing on the smoothed label (C) and scheduled sampling (D)
are both rejected, C because it is P1 and D because it is partially P1 *and* adds a
hyperparameter this project cannot afford. **The leakage here is structurally invisible to
every honesty guard the platform owns** — it is confined inside `train_index`, so no
holdout, no expanding window, and no purge/embargo can see it. That is precisely why D-03
demands a guard test that fails.

**2. §4.4 criterion 3.** Use **centroid distance in de-standardized feature units** as the
primary cost, `scipy.optimize.linear_sum_assignment` for the Hungarian matching (scipy 1.17.1
is installed and adds no legacy import). Measured this session: at d=10, n=40, centroid
distance separates signal from null at 1.70×, Gaussian-closed-form W₂ at 1.22×, and empirical
multivariate Wasserstein at **1.05× — unusable**. Do **not** invent a persistence threshold.
Report the matched distance against two data-derived yardsticks (a within-state split-half
null at the same n, and the next-best alternative distance) plus subsample occupancy and
episode count. One *existing, human-authorised* pass/fail already applies to the crisis
state: AMENDMENT condition (i)'s "at least three temporally separated episodes". Everything
else is Glenn's read.

**3. §5.3 wiring.** Given F-2, a literal hard gate would put the book in **100% cash roughly
two-thirds of the time** and violates §5.1's "consume the probabilities, never the argmax".
A bounded-turnover smoothing of the weight path preserves the probability contract but closes
A7 by *rewording* — which A7 explicitly permits. A third mechanism — use `active_regime` to
scale the tilt's *magnitude* while probabilities keep setting its *direction* — is the only
one that satisfies A7's literal wording and §5.1's contract simultaneously.

**Primary recommendation:** before anything else, split criterion 2's churn metric into the
two series in **F-1**, and pin the split with a test. Every other decision in this phase is
downstream of measuring the right object.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|---|---|---|---|
| Prior-state feature construction | L2 (`prediction/nowcaster.py`) | L1 (labels are its target) | §5.1 names the nowcaster. But see F-1: the *measured* churn is L1's. |
| Recursion plumbing across steps | Backtest driver (`backtest/driver.py`, `joint_driver.py`) | L2 | The loop owns step-to-step state; the model is stateless per step. |
| Cold-start policy | Backtest driver + `report/weekly.py` | L2 | Must be **identical** at train and serve, so it belongs at the one place both go through. |
| Filtered-churn measurement | `scripts/joint_lift_diagnostics.py` + `evaluation/` | — | Already the measurement seam; F-1's split belongs here. |
| Subsample stability (criterion 3) | **new** `labeling/stability.py` | `labeling/jump_model.py` | Pure function of a fit; must not touch the fit itself. |
| Hungarian matching | same new module (scipy) | — | No new dependency; adds no legacy import. |
| Hysteresis → allocation | L4 (`allocation/`) + drivers + `report/weekly.py` | — | A7's three call sites. All three must change together or diverge. |
| Turnover bounding (if chosen) | L4 (`allocation/`) | — | A property of the weight path, not of the model. |

---

## 1. User-visible constraints this research honours

- No dependence statistic was computed, referenced or re-derived. Nothing here touches
  criterion 6.
- No 2021+ data was read. Every measurement above is on the dev split
  (`split_by_holdout_boundary(..., cutoff=DEFAULT_HOLDOUT_CUTOFF)`), and the parquet artifacts
  read end at `2020-12-31` by construction.
- No source, test or planning file was modified. This document is the only write.
- No new legacy import is proposed. `scipy`, `numpy`, `pandas`, `sklearn` are already
  platform-internal dependencies; the ratchet at 31 counts only
  `from trading_crab_lib.<non-platform>` sites
  [VERIFIED: `tests/unit/test_platform_legacy_import_ratchet.py:25-35`].

---

## 2. Question 1 — the recursive prior-state feature

### 2.1 What is actually there today

```python
    cutoff = labels.index.max() - pd.DateOffset(months=embargo_months)
    eligible_labels = labels.loc[labels.index <= cutoff]
    common = features_df.index.intersection(eligible_labels.index)
    X = features_df.loc[common]
    y = eligible_labels.loc[common]
```
[VERIFIED: `platform/prediction/nowcaster.py:78-82`, verbatim]

No prior-state column, exactly as D-02 says. The prediction row is scored at
`driver.py:_refit_l2` as `model.predict_proba(feature_row[active])`
[VERIFIED: `backtest/driver.py:303`].

### 2.2 The five options, with their real costs

Notation: at outer walk-forward step *t*, `train_index = index[:i]` (strictly before *t*)
[VERIFIED: `honesty/walkforward.py:48-49`]. Let `f_m` denote a *filtered* predicted
distribution for month *m*, i.e. one produced by a model fit only on data strictly before *m*.
Let `y_m` denote L1's label for month *m* from a fit over the whole training window.

**The single fact that makes this hard:** `y_m` is **not causal within the window**. The
module says so itself:

> *"The labeler is intentionally non-causal at the batch level: the DP jointly optimizes over
> the full time axis, so a label at month t is influenced by the global fit including months
> after t."*
[VERIFIED: `labeling/jump_model.py:15-18`, verbatim]

So `y_{m-1}` at training row *m* carries information from months after *m−1*. Crucially, all
of that information is still **inside `train_index`, strictly before *t***. Therefore:

> **No holdout boundary, no expanding-window guard, no purge and no embargo can detect this
> leak.** It is not leakage across the decision date. It is leakage across the *causality
> direction within the training window*, and the platform has no guard for that shape.

That is the whole reason D-03 demands a test that can fail.

---

#### Option A — one-pass expanding-window filtered-history generator  ✅ **recommended if criterion 1 is read literally**

Run **one** expanding pass over the dev index producing `f_m` for every month *m* from some
warm-up start onward: at each *m*, fit a nowcaster on `(X_{<m}, y_{<m})` and predict row *m*.
The prior-state feature at row *m* is then `f_{m−1}`, which is a causal function of data
strictly before *m*. Use that same column inside every outer step's training matrix, and use
`f_{t−1}` at serve time.

- **Leakage:** none across *t*. `f_{m−1}` depends only on months < *m* ≤ *t*.
- **Cost:** measured L2 fit ≈ **0.109 s/step** [VERIFIED: `07-RESEARCH-WAVE1.md:249`], so a
  588-step pass ≈ **64 s**; with a 1967 warm-up start, ~700 fits ≈ **76 s**.
- **The honest wart, named:** the column is produced by the *one-pass* model family, while
  serving is done by the *step-t* model. Pass-2's own predictions differ from pass-1's, so at
  serve time the feature is pass-1's prediction, not pass-2's. That is a residual train/serve
  inconsistency — milder than teacher forcing, but real.
- **It can be iterated away, cheaply and falsifiably.** Pass *k*+1 uses pass *k*'s filtered
  history. At ~2 min per full walk-forward run, five passes cost ~10 minutes. **Report whether
  the filtered churn and the prior-state column converge across passes.** If they do, the
  fixed point is the honest object. If they oscillate, that is a finding, and it is the kind
  of finding this phase exists to surface.
- **Second wart, also named:** `f_m` was trained against the label vintage from the refit at
  *m*; the target `y` at outer step *t* is the vintage from the refit at *t*. Given F-1 and
  F-3, L1 labels *are* revised between refits — so the feature and the target come from
  different label vintages. Not leakage. Worth stating in the module docstring.

#### Option B — in-step recursive regeneration  (honest, and priced)

At each outer step *t*, rebuild the prior-state column for all training rows using models fit
only within *t*'s own window. Internally consistent: one label vintage, one model family.

**The arithmetic.** Outer steps run *i* = 120…707 over the 708-month dev index; the inner
expanding loop at step *t* needs ≈ (*i* − warmup) fits. Σ ≈ 588²/2 ≈ **172,900 nowcaster
fits**. At 0.109 s each: **≈ 5.2 hours** per walk-forward run. Two legs (joint + baseline) and
two routings multiply it. Against a current ~2-minute full run
[VERIFIED: `07-RESEARCH-WAVE1.md:249`, `07-01-PLAN.md:390`], this is a ~150× regression in the
phase's core feedback loop.

That cost is not fatal on its own — but it removes the "cheap, run it twice" property that
`07-RESEARCH-WAVE1.md:486` explicitly relies on, and the honesty framework depends on being
able to re-run a measurement rather than patch an artifact
(`07-03-PLAN.md:483`, threat T-07-17). **Recommend against**, and say why in the ADR rather
than leaving it looking unconsidered.

#### Option C — teacher forcing on the prior smoothed label  ❌ **forbidden (D-03)**

`X["prior_state"] = y.shift(1)`. This is P1, in the form D-03 names. It is the trap; it is
also the implementation a hurried reader will reach for, because it is one line and it makes
CV accuracy beautiful. §3 below is the guard.

#### Option D — scheduled sampling  ❌ **reject, on two independent grounds**

Mix `y_{m−1}` and `f_{m−1}` with probability ε, decayed on a schedule.

- It uses the smoothed label *part of the time*. It is therefore **partially C** — partially
  P1 — and there is no principled ε at which it stops being so.
- The literature's own caveats cut against it here: pure sampling (ε→0) is reported to fail to
  converge, and scheduled sampling *"can cause the model to neglect probability mass on gold
  prefixes, degrading performance when the prefix at inference is correct"*
  [CITED: https://www.emergentmind.com/topics/scheduled-sampling]. The schedule is a
  hyperparameter, and **every evaluated configuration is a registry trial** — 42 of 44 are
  spent.
- Scheduled sampling exists to fix *exposure bias* in long autoregressive rollouts where
  errors compound over many steps. Here the rollout is **one step** (the prior month), and
  Option A already gives an exact, causal, non-teacher-forced column. The problem scheduled
  sampling solves is not the problem we have.

#### Option F — explicit Bayes filter; **no training column at all**  ⭐ **strongest on the merits; needs criterion 1 reworded**

Keep the nowcaster exactly as it is — no new feature, no new column, no train/serve question
— and apply the recursion **at inference only**, as design §4.2's forward recursion:

```
π_t(j)  ∝  [ Σ_i π_{t−1}(i) · A_ij ] · L_t(j)
```

where `A` is the empirical transition matrix estimated from the in-window L1 labels
(`prediction/transition_matrix.py::empirical_transition_matrix` already exists
[VERIFIED: `platform/prediction/transition_matrix.py:26`]) and `L_t(j)` is the nowcaster's
calibrated posterior divided by the in-window class prior (Bayes inversion to a
class-conditional likelihood).

- **Train/serve skew: structurally zero.** There is no training-time analogue of the feature,
  so there is nothing to skew.
- **Leakage: structurally zero.** `A` is estimated from in-window labels; `π_{t−1}` is the
  previous step's own output.
- **Free parameters: zero.** `A` and the prior come from the same in-window labels the
  nowcaster already trains on. No λ, no ε, no schedule, no registry trial.
- **It is what design §5.1's own first sentence describes:** *"A discriminative replacement
  for the HMM filter."* [VERIFIED: `platform_design/platform_design.md` §5.1, verbatim] —
  §4.2 then supplies the filter recursion verbatim.
- **Cold start is free:** `π_0` = the in-window unconditional occupancy.
- **Cost:** a matrix multiply per step. Negligible.
- **The catch:** ROADMAP criterion 1 says *"The nowcaster's **feature set includes** the prior
  predicted (filtered) state distribution"*. Option F does not put it in the feature set. It
  would satisfy §5.1's *intent* ("Without it, persistence is discarded and predictions
  flicker") while failing its *letter*. **That is Glenn's call and nobody else's** —
  recorded here as Q-1 in §9, not decided.
- **Second catch, stated plainly:** `A` estimated from the *smoothed* in-window labels
  encodes a transition structure that was itself fitted non-causally within the window. This
  is a weaker cousin of the C-problem. It is defensible (the same in-window labels are already
  the nowcaster's training target, so no *new* information class enters) but it is not
  nothing, and it should be written down rather than glossed.

### 2.3 Where the recursion lives (sub-question 2)

`_refit_l1` and `_refit_l2` are called once per step inside the loop
[VERIFIED: `backtest/driver.py:472-476`; `joint_driver.py:411,449,455`]. The placement
question resolves cleanly by option:

| option | where the column is built | cost | risk |
|---|---|---|---|
| A | **once**, before the loop, by a dedicated expanding-pass generator | ~76 s | model-family mismatch (§2.2), iterable to a fixed point |
| B | **inside** each step | ~5.2 h | none of A's; loses the cheap-rerun property the honesty framework leans on |
| F | **nowhere** — the recursion is a per-step inference update carried in a loop variable | ~0 | `A` from smoothed in-window labels |

For A, do **not** build the column inside `build_nowcaster_training_set`. That function's
contract is the D-01 structural label embargo and its docstring is explicit that the two
embargo concepts *"must never be merged"*
[VERIFIED: `prediction/nowcaster.py:11-13`, verbatim]. Adding a third concern there is how the
next reader conflates all three. Build it in its own module and pass the augmented frame in.

### 2.4 Cold start (sub-question 3)

Three candidates, and one of them is not available:

- **Drop the row.** Viable for *training* rows only. Not viable at serve: the first
  walk-forward step must emit a prediction. And restricting training to rows with a genuine
  `f_{m−1}` is fatal at the early steps — at *t* = 1972-01 the training window is
  1962-01…1971-12, and if the generator also starts at 1972 there are **zero** eligible
  training rows. **Reject** unless the generator warms up earlier than the decision spine
  (see below).
- **Uniform 1/K.** The maximally honest "no information" fill, and exactly `π_0` for a Bayes
  filter. At K=6 that is 0.1667 per state.
- **In-window unconditional class distribution.** Causal (uses only `train_index`), sharper
  than uniform, and computable *identically* at train time and serve time — which is the
  property that matters, because a cold-start rule that differs between the two **is itself
  train/serve skew**.

**Recommendation.** Warm the generator earlier than the decision spine (e.g. `min_train=60`
→ first `f_m` at 1967-01) so that by the 1972-01 first decision every training row from 1967
onward carries a genuine value; fill 1962-01…1966-12 with the **in-window unconditional class
distribution**, and use the *same* rule on a genuine cold start at serve. The generator's own
`min_train` is a new parameter; set it to a stated value with a stated reason and do not
sweep it (a swept warm-up is a selection dimension).

The literature does not settle this. Sequence models conventionally use a learned `<BOS>`
embedding, which has no analogue here. **`[ASSUMED]`** — flagged as Q-2.

### 2.5 Interaction with `PurgedEmbargoedKFold` (sub-question 4)

The splitter [VERIFIED: `honesty/cv.py:58-73`, verbatim]:

```python
            test_start, test_end = test_idx[0], test_idx[-1]
            purge_start = max(0, test_start - self.label_horizon)
            embargo_end = min(n, test_end + 1 + self.embargo)
            train_mask = np.ones(n, dtype=bool)
            train_mask[purge_start:embargo_end] = False
```

Called with `label_horizon=12, embargo=1` [VERIFIED: `prediction/nowcaster.py:90-91,125`].
Four things change when a recursive feature is added, and only one of them is about leakage:

1. **Embargo is the *right* guard, and it is sized at 1.** The embargo exists precisely for
   this channel: *"Embargo drops a fixed buffer of training rows right after the test fold;
   their features are still serially correlated with the test period, so they leak even when
   their labels do not"* [CITED: https://quantmemo.com/concepts/purged-embargoed-cv]. A
   lag-1 prior-state feature at row *m*+2 encodes month *m*+1, which may be in the test fold;
   `embargo=1` removes exactly that one row. **It is exactly sufficient for lag-1 and
   insufficient for anything wider.** If the feature is ever smoothed (an EWMA of past
   predictions, a multi-lag stack), the embargo must widen to match, and the current call site
   passes a literal `1`. Write that dependency into the docstring, or the next person who adds
   a second lag will silently break it.
2. **Purging becomes largely irrelevant here and should not be mistaken for protection.** The
   12-month purge targets the label horizon; it does nothing about a feature that is
   autocorrelated *by design*.
3. **CV accuracy will rise, and the rise means nothing.** Under a labeling with median sojourn
   9.5 months [VERIFIED: `diagnostics_l1only.json`], the prior state equals the current state
   in roughly 9 of 10 months. A model given that feature scores high for free. §5.1 says so:
   *"Persistence lets a trivial classifier score ~90% overall; the value is at the turns."*
   **CV accuracy therefore stops being a usable selection signal the moment this feature is
   added.** `transition_window_accuracy` already returns all three keys for exactly this
   reason [VERIFIED: `prediction/nowcaster.py:146-177`] — use those, not the headline.
4. **The folds' *calibration* meaning changes.** `PurgedEmbargoedKFold` is passed to
   `CalibratedClassifierCV(base, method="sigmoid", cv=cv)`
   [VERIFIED: `prediction/nowcaster.py:135`] — these folds calibrate, they do not select. If
   the training column comes from a one-pass generator (Option A) and serving uses the step-*t*
   model's own recursion, the calibration folds see a *different* prior-state distribution than
   serve time does. The resulting miscalibration points in a predictable direction:
   **overconfident persistence**. That is checkable — compare the reliability curve on the
   in-fold predictions against the reliability of the realised filtered path.

### 2.6 What criterion 2 should measure, concretely

Given F-1, add a filtered-churn series for the object §5.1 actually changes. The driver
already accumulates it:

```python
            per_step_metrics["proba"].append(regime_probs.values)
            per_step_metrics["classes"].append(list(regime_probs.index))
```
[VERIFIED: `backtest/driver.py:530-532`] — but it is **not persisted to disk**; only the
equity curve is (`outputs/reports/platform/joint_lift/` contains four parquets and four
JSONs, no probability artifact) [VERIFIED: directory listing this session]. Persisting the
per-step probability matrix is a prerequisite for measuring anything about the nowcaster's
filtered path — including the churn, the max-probability distribution that F-2 needs, and the
reliability curve §2.5(4) needs. **This is a small, boring change and everything else depends
on it.**

---

## 3. Question 1.5 — how leakage would SHOW UP

The caller is right that this matters more than anything else. Four signatures, in order of
decisiveness. Two of them can be written as tests that **fail** on a substituted smoothed
label, which is D-03's requirement.

### S-1. The lead/lag signature — decisive, and there is already machinery for it ⭐

A causal filtered estimate **cannot lead a transition**. A smoothed label **can**, and
routinely does: the DP decode places a switch using evidence from both sides.

Measure, over the 25 full-sample transitions of classifier #1
[VERIFIED: `diagnostics_l1only.json`], the **signed offset** between the month the
prior-state column changes and the reference transition month.

- Causal filtered feature → offset ≥ +1 month, always. Today's measured
  `median_lag = 4.0` months [VERIFIED: `diagnostics_l1only.json`] is the scale.
- Smoothed label substituted → offsets at **0 or negative**. Even one strictly negative offset
  is proof.

**Write it as a test that fails.** Construct the smoothed-substitution case explicitly, assert
the diagnostic reports a non-positive offset, and assert the honest case does not. A test that
only exercises the honest path is the evidence-shape failure D-03 names.
`evaluation/sojourn_lag.py::compute_sojourn_lag_headline` already computes transition-relative
lags [VERIFIED: `platform/evaluation/sojourn_lag.py` exists and is called at
`scripts/joint_lift_diagnostics.py:114`] — reuse it; a second implementation of the lag rule
is the divergence class ADR-0002's D-11 was written to prevent.

### S-2. The agreement-with-`y.shift(1)` signature — cheap, and catches the one-liner

If the column were `y.shift(1)`, then `argmax(prior_state_column)` equals `y.shift(1)`
**exactly**, on every row. The honest column cannot: it is a filtered estimate with a real
error rate.

Assert `accuracy(argmax(prior_col), y.shift(1)) < 1.0` — and, more usefully, assert it is
**below the filtered nowcaster's own accuracy against `y`**, which is the ceiling a causal
one-step-lagged estimate can reach. State the number measured; do not invent a band. The
degenerate `== 1.0` case is the falsifier.

⚠️ **S-2 alone is not sufficient and must not be shipped alone.** A *blend* of smoothed and
filtered (Option D) scores strictly below 1.0 and passes S-2 while still being partially P1.
S-1 catches that; S-2 does not.

### S-3. The persistence-classifier signature — catches "honest but useless"

Leakage is not the only success-shaped failure. The model can be perfectly honest and still
collapse into "repeat the prior state", which would drive churn to ~0 and look like a triumph.
Three readings distinguish it:

| reading | genuine persistence | collapsed into a persistence classifier |
|---|---|---|
| `overall_accuracy` | ↑ | ↑↑ |
| `transition_accuracy` | holds or ↑ | **↓** |
| `median_lag` (§5.4 detection lag) | ≈ 4 months, stable | **↑ toward median sojourn** |
| `median_sojourn / median_lag` (today **2.375**) | stays ≥ ~2 | **→ 1 or below** |
| coefficient mass on prior-state columns ÷ total | a share | ≈ all of it |

§5.4 names the ratio and says *"report it prominently"*
[VERIFIED: `platform_design.md` §5.4, verbatim]. It is the single best one-number summary of
"did persistence cost us the turns", and it is already computed
(`sojourn_lag.ratio = 2.375` today).

### S-4. The routing-consistency signature — catches F-1's trap

Report both churn series from F-1 in the same table. If the `state_1` churn is unchanged at
246 **and** the `argmax(regime_probs)` churn falls, the fix worked and the L1 number was never
the target. If both are unchanged, nothing happened. If `state_1` moved, something touched L1
that was not supposed to, and that is a bug, not a result.

### What distinguishes "genuinely added persistence" from "leaked the answer" — one sentence

**Leakage improves accuracy *at and before* the turns (S-1 offset ≤ 0); genuine persistence
improves accuracy *between* the turns while leaving `median_lag` and `transition_accuracy`
roughly where they were (S-3).** Leakage makes the model look *prescient*; persistence makes
it look *calm*. They are distinguishable precisely because one of them cheats on timing and
the other does not.

---

## 4. Question 3 — §5.3 hysteresis wiring, and A7

### 4.1 The contract, verified

`vol_targeted_tilt` **already accepts both shapes**:

```python
    if isinstance(regime_or_probs, (pd.Series, dict)):
        probs = pd.Series(regime_or_probs, dtype=float)
        regime = probs.idxmax() if not probs.empty else None
    else:
        regime = regime_or_probs
        probs = pd.Series({regime: 1.0}, dtype=float) if regime is not None else pd.Series(dtype=float)
```
[VERIFIED: `allocation/tilt.py:189-194`, verbatim]

So "pass the hard label instead" is a one-line change that type-checks. The question is not
*can* you; it is *what happens*.

And note the second half of that branch: `regime = probs.idxmax()` — `vol_targeted_tilt`
**already computes an argmax internally** and hands it to `regime_tilt_weights` as the
fallback regime. There is already an implicit "active regime" in the allocator that has
nothing to do with the hysteresis state machine.

### 4.2 A7, restated more strongly than A7 states it

A7 says the thresholds *"stabilize a label, not a portfolio"*. F-2 shows something harder:

- Under `l1only` (the decision-bearing routing, ADR-0002 decision (e)), `update_active_regime`
  is **the identity on argmax** for any admissible threshold pair. Wiring it into allocation
  would change **nothing**, measurably — `active_regime` is already elementwise equal to
  `state_1` in all 588 months.
- Under `l2`, wiring it in as-is would put the portfolio in `regime_tilt_weights` →
  `pd.Series(dtype=float)` → `{"weights": empty, "cash": 1.0, "scale": 0.0}`
  [VERIFIED: `allocation/tilt.py:196-198`] on the **387 of 588 months (65.8%)** where
  `active_regime` is None. **A 66%-cash strategy.**

Both of those are measurable predictions. Both should be *measured* in this phase rather than
argued, because criterion 5 re-measures criterion 7 anyway and the marginal cost is one more
run at ~2 minutes.

### 4.3 There is a third inconsistency, not in A7

`report/weekly.py` computes the hysteresis state, persists it, and then **does not use it**:

```python
    prev_active = load_active_regime(cm)  # load BEFORE save (Pitfall 3)
    active_regime = update_active_regime(...)
    save_active_regime(active_regime, cm)  # save AFTER load
    tilt = vol_targeted_tilt(regime_probs, ...)
```
[VERIFIED: `report/weekly.py:240-249`]

`assemble_weekly_report` never receives it, and recomputes its own:

```python
    active_regime = probs.idxmax() if not probs.empty else None
```
[VERIFIED: `report/weekly.py:117`]

…and then, further down, **prints an explanation of the hysteresis cold-start rule** while
displaying the argmax [VERIFIED: `report/weekly.py:134-139`]. The report narrates a state
machine whose output it is not showing. Any §5.3 work must fix this or it will diverge
further.

### 4.4 The three mechanisms, and what each breaks

§5.3 verbatim: *"Allocation responds through **hysteresis bands** (act when P crosses ~0.7;
unwind below ~0.4; hold in between) **and/or** smoothed allocation response with bounded
turnover."* [VERIFIED: `platform_design.md` §5.3, verbatim]

| | (a) hard gate on `active_regime` | (b) bounded-turnover weight path | (c) hysteresis scales the tilt's *magnitude* |
|---|---|---|---|
| what changes | `vol_targeted_tilt(active_regime, …)` | `w_t = w_{t−1} + clip(w*_t − w_{t−1}, ±τ)` (or a no-trade band) | `probs` still set direction; `active_regime` sets how far to lean |
| §5.1's "consume probabilities, never argmax" | **violated** | honoured | honoured |
| A7 "allocation acts on it" | literally satisfied | **not** satisfied → A7 closes by **rewording** | literally satisfied |
| measured consequence today | 66% all-cash (§4.2) | directly attacks 16.3% / 12.0% monthly turnover | shrinks the tilt when confidence is low; never forces cash |
| new free parameter | none (reuses 0.70/0.40) | **τ** — a selection dimension | **shrink floor** — a selection dimension |
| existing call sites changed | 3 | 3 (drivers + weekly) | 3 |
| effect on criterion 7 | large, must be measured | moderate, must be measured | moderate, must be measured |

Measured turnover today, from the tracked curves [VERIFIED: measured this session]:

| leg / routing | mean monthly turnover |
|---|---|
| baseline (classifier #1 alone), l1only | **0.163251** |
| joint, l1only | 0.119788 |
| baseline, l2 | 0.098907 |
| joint, l2 | 0.060451 |

16.3% monthly is ~196% annualised on a taxable account that rebalances monthly. That is the
quantity (b) and (c) attack and (a) does not — (a) reduces turnover by going to cash, which is
a different thing from reducing churn.

**Recommendation: (c), with (b) as the fallback if (c)'s shrink floor is judged too costly in
trials.** (c) is the only mechanism that satisfies A7's literal wording *and* §5.1's
probability contract simultaneously. But the choice is Glenn's — see Q-3.

**Regardless of mechanism, one prerequisite is unavoidable:** the 0.70/0.40 pair is mismatched
to a calibrated K=6 posterior (F-2). Two ways out, neither free:
- **Re-pin the thresholds.** Each value evaluated is a registry trial. Expensive at 42/44.
- **Define them relative to K.** E.g. act at *c* × (1/K). At K=6, today's 0.70 is 4.2× uniform;
  at K=5 (classifier #2) it is 3.5×. A K-relative rule is *derivable* rather than tuned, and
  it makes the two classifiers comparable — but the coefficient *c* is still a choice, so this
  buys principle, not freedom. **`[ASSUMED]`** that a K-relative form is preferable; Q-4.

### 4.5 G6 — the unpooled-consumer pin (criterion 7)

The seam is located and the asymmetry is real:

- `driver.py:497` calls `vol_targeted_tilt(regime_probs, stats, …)` where
  `stats = returns_by_regime_stats(dev_asset_returns.loc[train_index], states)` at line 490 —
  **raw, unpooled** [VERIFIED: `backtest/driver.py:490,497`].
- `joint_tilt.py` applies ADR-0001 condition (iv)'s partial pooling before tilting, and says so
  itself: *"condition (iv)'s partial pooling, which `vol_targeted_tilt` does not."*
  [VERIFIED: `allocation/joint_tilt.py:357`, verbatim; pooling applied at line 319]

So the G6 test should pin the **known non-compliance**, not assert compliance: a test that
`driver.py`'s tilt receives a `returns_by_regime` frame whose per-regime Sharpes are
byte-identical to `returns_by_regime_stats`' output (i.e. `pool_low_n_regime_sharpe` was **not**
applied). Writing it the other way round would be a test that can only pass.

---

## 5. Question 2 — §4.4 criterion 3 (subsample stability)

Criterion 3 verbatim:

> **3. Stability:** re-estimate on subsamples (drop first decade / last decade / block
> bootstrap); states persist with matched emission parameters (match via Hungarian algorithm
> on distribution distances to defeat label switching). A "regime" that evaporates when
> 2008–09 is dropped is an *episode*, not a regime.
[VERIFIED: `platform_design/platform_design.md` §4.4, verbatim]

### 5.1 What the two labelings actually look like (re-derived this session)

Classifier #1, K=6, λ=10.0, 10 frozen columns, 695 months 1963-02 → 2020-12
[VERIFIED: measured this session from `data/checkpoints/platform/regime_labels.parquet`]:

| state | occupancy | months | **episodes** | episode spans |
|---|---|---|---|---|
| 0 | 5.76% | 40 | **9** | 1970-05(3m), 1973-11(11m), 1981-09(1m), 1987-10(4m), 1990-08(3m), 2002-06(4m), 2008-09(8m), 2011-08(3m), 2020-03(3m) |
| 1 | 32.81% | 228 | 5 | 1963-02(87m), 1971-03(32m), 1976-05(23m), 1988-09(23m), 1991-04(63m) |
| **2** | **10.22%** | **71** | **1** | **1996-07 → 2002-05 (71m) — once, contiguous** |
| 3 | 28.78% | 200 | 4 | 2002-10(71m), 2009-10(22m), 2011-11(100m), 2020-06(7m) |
| 4 | 12.09% | 84 | 3 | 1981-10(**72m**), 1988-02(7m), 2009-05(5m) |
| 5 | 10.36% | 72 | 4 | 1970-08(7m), 1974-10(19m), 1978-04(41m), 1990-11(5m) |

Classifier #2, K=5, λ=16.0, 696 months: occupancies 16.7 / 22.7 / 22.4 / 23.9 / 14.4 % — all
comfortably inside criterion 1's band [VERIFIED: measured this session].

**The finding this table forces.** The AMENDMENT of 2026-09-18 says, verbatim:

> *"A 5.76% state recurring across nine crises spanning fifty years is more evidently a regime
> than a 20% state appearing once as a contiguous block — and criterion 1, being blind to
> arrangement, scores the second higher."*
[VERIFIED: `platform_design.md` §4.4 AMENDMENT, verbatim]

**Classifier #1's state 2 is that state.** 10.22%, one contiguous 71-month block
(1996-07 → 2002-05 — the late-90s melt-up and dotcom unwind), never recurring. It sits
*inside* criterion 1's band, so the recurrence exemption — which applies only to **sub-floor**
states — does not reach it, and nothing in the current suite looks at arrangement at all.
State 4 is a weaker version of the same shape: 3 episodes, but 72 of its 84 months in one.

So the falsifiable prediction for criterion 3, stated before it is run:

- **State 0 (crisis) will pass.** Drop-first-decade removes one 3-month episode; drop-last-decade
  removes two (6 months). Six to seven temporally separated episodes remain either way, so
  AMENDMENT condition (i) ("at least three temporally separated episodes") holds. D-05 expects
  this test to adjudicate the crisis state; on the arithmetic, **it will exonerate it.**
- **State 2 is the likely failure** — and *neither decade-drop touches it* (1996-2002 is
  interior). The three named schemes are, by construction, poorly aimed at the actual failure
  mode.

**Recommendation.** Add a fourth scheme — **leave-one-episode-out**: for each state, drop the
months of its longest episode and refit; does the state survive on its remaining episodes?
This is the design's own "drop 2008-09" example, generalised. For state 2 the test is
degenerate *and that is the answer*: a state with exactly one episode fails
leave-one-episode-out by construction, which is the design's definition of an episode, reached
with **no invented threshold**. Adding a fourth scheme goes beyond ROADMAP criterion 4's three
— Q-5.

### 5.2 Which distance — measured, not asserted

The jump model has **no emission distribution**. It has centroids, and squared-Euclidean
distance to them:

```python
            d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
```
[VERIFIED: `labeling/jump_model.py:179`, verbatim]

That is equivalent to a spherical Gaussian with common variance. "Emission parameters" here
means, at minimum, the centroid; at most, the centroid plus the empirical within-state
covariance.

I benchmarked the three candidates at this project's actual dimension (d=10) and actual state
sizes, 20 replications each, signal = a 0.3-per-dimension mean shift (true centroid distance
0.949), null = two independent draws from the *same* distribution
[VERIFIED: measured this session, `numpy` + `scipy.linalg.sqrtm` +
`scipy.stats.wasserstein_distance_nd`]:

| n | centroid: signal / null | Gaussian W₂ (closed form): signal / null | empirical W_nd: signal / null |
|---|---|---|---|
| **40** (state 0) | 1.202 / **0.706** → **1.70×** | 1.682 / 1.378 → 1.22× | 3.02 / **2.88** → **1.05×** |
| **71** (state 2) | 1.054 / 0.548 → **1.92×** | 1.377 / 1.016 → 1.36× | — |
| 100 | 1.074 / 0.459 → 2.34× | 1.300 / 0.888 → 1.46× | 2.74 / 2.55 |
| 230 | 1.025 / 0.281 → 3.65× | 1.140 / 0.566 → 2.01× | 2.50 / 2.37 (1.9 s) |
| 700 | 0.948 / 0.163 → 5.82× | 0.985 / 0.325 → 3.03× | — / 2.08 |

Three conclusions, each falsifiable by re-running the benchmark:

1. **Empirical multivariate Wasserstein is unusable here.** At n=40, d=10 its sampling bias
   (2.88) is three times the true signal (0.949) and swamps the difference. This is the known
   `n^{-1/d}` convergence of empirical optimal transport, arriving exactly where this project
   lives. **Reject `wasserstein_distance_nd`**, notwithstanding that it is available in the
   installed scipy 1.17.1.
2. **The covariance term in Gaussian W₂ is net noise below n≈230.** It halves the separation
   ratio at n=40 relative to the plain centroid. Report it as a *secondary* if at all; do not
   lead with it. KL / JS on fitted Gaussians is strictly worse still — it needs `Σ⁻¹`, and a
   40-month state in 10 dimensions gives a badly conditioned covariance, which then needs a
   shrinkage constant, which is a free parameter, which is a selection dimension.
3. **Centroid distance is the right primary.** It is faithful to the model's own objective,
   has zero free parameters, and has the best signal-to-null ratio at every sample size that
   matters here.

**The null is not zero.** At n=40 the null centroid distance is **0.706** — 74% of a 0.949
true signal. Any "states persist if matched distance < X" rule is therefore meaningless without
an n-matched reference. §5.5 handles that without inventing anything.

### 5.3 Two traps that will silently produce meaningless numbers

**Trap A — the standardization is refit per subsample.**

```python
    winsorized = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
    return StandardScaler().fit_transform(winsorized)
```
[VERIFIED: `labeling/jump_model.py:122-123`, verbatim]

Both the winsorization quantiles and the scaler are fit **on whatever rows are passed in**. A
subsample-fitted centroid therefore lives in a *different* standardized space than the
full-sample centroid, and comparing the two coordinate-wise compares apples to oranges. A naive
implementation will do exactly this and produce a distance table that looks fine.

Fix: **de-standardize both sides using each fit's own scaler before computing any distance.**
Note that `standardize_features` *discards* the scaler (it returns
`StandardScaler().fit_transform(...)` and keeps no handle), so the caller must either recompute
`winsorized.mean()` / `winsorized.std()` at the call site or add a scaler-returning variant.
Residual caveat to state, not hide: de-standardizing recovers **winsorized** units, and the 1%/99%
clip bounds also differ per subsample. For centroids (means over ≥40 months) that is
second-order, but it is not zero.

The alternative — standardize every subsample with the *full-sample* scaler — is defensible for
a stability test (which is not a causality test) but injects full-sample information into the
subsample fit. **Prefer de-standardization; if the full-sample-scaler route is taken, say so.**

**Trap B — a zero-occupancy state still returns a centroid.**

```python
def _recompute_centroids(X, states, K, prev_centroids):
    """Per-state mean of X, freezing any zero-occupancy state at its previous
    centroid (Pitfall 2 — prevents NaN from poisoning subsequent DP iterations)."""
```
[VERIFIED: `labeling/jump_model.py:126-130`, verbatim]

A subsample fit at fixed K **always** yields K centroids, even for a state that captured zero
months. That centroid would match cleanly against its full-sample partner and the state would
score as *stable* while having **evaporated** — which is the precise failure criterion 3 exists
to catch. **Every row of the criterion-3 output table must carry the subsample occupancy count,
and a state with 0 (or near-0) months must be flagged regardless of its matched distance.**

**Trap C — re-running the freeze rule changes the model, not the data.** Dropping the first
decade moves `first_decision`, and `_reference_label_columns` admits a column only if it is
non-NaN from `first_decision` onward [VERIFIED: `evaluation/report.py`, function body quoted in
§5.6]. Re-running it on a subsample can change the feature set, at which point criterion 3
measures feature-set churn rather than state stability. **Hold the frozen column list fixed at
the full-sample list across every subsample.** The `sort_column` is guaranteed present because
it is in that list.

### 5.4 Hungarian matching and the label-switching subtlety

`scipy.optimize.linear_sum_assignment` is the right implementation. Verified working this
session on a 6×6 cost matrix; scipy 1.17.1, numpy 2.4.6, pandas 3.0.5, sklearn 1.9.1 are
installed [VERIFIED: measured this session]. It adds no legacy import. K is 5 or 6, so cost is
irrelevant.

**The subtlety:** `canonicalize_states` **already** defeats most label switching by sorting
states on ascending `trailing_return_1m` centroid (classifier #1) or `rs_equities_bonds`
(classifier #2) [VERIFIED: `labeling/jump_model.py:247-251`; `labeling/classifier2.py:127`].
So:

- Run the matching on **canonicalized** outputs — that is what the pipeline produces.
- If the Hungarian assignment is the **identity**, that tells you the canonical ordering held;
  it does **not** tell you the states persisted. The informative output is the matched
  *distances*, not the permutation.
- If it is **not** the identity, the canonical ordering itself flipped between subsamples. That
  is a finding in its own right and should be reported, because every downstream
  occupancy/profile/lift number is keyed on those ids.

### 5.5 What "states persist" can mean operationally — without inventing a threshold

The design gives no threshold. D-15 and D-07 in Phase 7 handled that by reporting the quantity
and letting a human read it; do the same. Per (classifier × scheme × state), report:

1. **Matched partner id** and whether the permutation is the identity.
2. **Matched distance** (centroid, de-standardized units).
3. **Within-state split-half null at the same n** — split that state's own months at random into
   two halves, compute the centroid distance between them, repeat ~200 times, report the median.
   This is a data-derived "how far apart do two samples of *the same* state land at this n"
   yardstick, with **zero invented parameters**. §5.2's table shows why it is indispensable: at
   n=40 the null is 0.706, not 0.
4. **Margin** — matched distance ÷ next-best alternative distance in that row. Near 1.0 means the
   assignment is arbitrary regardless of how small the distance is.
5. **Subsample occupancy** (months and %). Zero ⇒ evaporated, whatever (2) says (Trap B).
6. **Episode count in the subsample** — the recurrence quantity.
7. **The full K×K distance matrix, persisted**, so a later reader can re-judge without a re-run.

**The one pass/fail that is not invented.** AMENDMENT condition (i) is an existing,
human-authorised threshold, quoted verbatim:

> *"(i) it recurs in **at least three temporally separated episodes**, so that removing any one
> leaves the state intact — criterion 3 applied directly, not by proxy"*
[VERIFIED: `platform_design.md` §4.4 AMENDMENT, verbatim]

It binds only on the sub-floor state invoking the exemption — classifier #1's state 0. For that
one state, criterion 3 has a verdict this phase may render. For every other state, the table is
the output and Glenn's reading is the verdict. **Do not add a threshold after seeing the
numbers**; that is the tie-break shape the `298b1bc` pre-registration forbids elsewhere, and it
would be no more honest here. If a threshold is ever wanted, pre-register it.

### 5.6 Block bootstrap — block length, and a caveat the criterion does not anticipate

**The literature anchor.** Politis & White (2004), corrected by Patton, Politis & White (2009),
give a data-driven optimal block length from the correlogram, minimising the asymptotic MSE of
the zero-frequency spectral-density (long-run variance) estimate; the optimal length is
`O(n^{1/3})` [CITED: https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf;
https://www.tandfonline.com/doi/abs/10.1081/ETC-120028836]. At n = 695 labeled months,
n^{1/3} ≈ **8.9 months**; at n = 588 decision months, **8.4**.

**Why that anchor is the wrong objective here, and should be quoted rather than obeyed.** It
targets *variance estimation*. Criterion 3 needs blocks long enough to preserve the persistence
the states encode. Classifier #1's median sojourn is **9.5 months**
[VERIFIED: `diagnostics_l1only.json`]; classifier #2's is **29.0**. A ~9-month block is right at
#1's median and less than a third of #2's — resampling at that length destroys exactly the
structure being tested, and every state would "evaporate" for a reason that has nothing to do
with whether it is a regime.

**Recommendation.** Do not pick one block length. **Report the result across a stated range —
e.g. 6, 12, 24, 48 months** — quoting the Politis–White n^{1/3} ≈ 8–9 as the variance-estimation
anchor *and* saying that it is not designed for this purpose. A result that holds across the
range is strong; one that flips is itself the finding. `[ASSUMED]` that {6,12,24,48} is the right
ladder — it brackets both medians; Q-5.

**The caveat criterion 3 does not anticipate.** The two decade-drops are contiguous and preserve
time ordering. **Block bootstrap does not.** The jump model penalises state changes *in index
order*, so a block-bootstrapped series contains synthetic seams at which the penalty fires on
artefacts. The object being fit is not the same object. Mitigations: use the **circular** block
bootstrap (keeps length n), and **report the seam count** so the reader can discount. Even then,
block bootstrap is the weakest of the three schemes **for a temporally penalized model
specifically**, and the write-up should say so rather than presenting three schemes as equals.

### 5.7 Runtime

One L1 fit ≈ **0.111 s** [VERIFIED: `07-RESEARCH-WAVE1.md:249`]. Criterion 3 needs:
2 classifiers × (2 decade-drops + 4 block lengths × B bootstrap replicates + K
leave-one-episode-out refits). At B = 200 that is 2 × (2 + 800 + ~6) ≈ **1,620 fits ≈ 3
minutes**, plus ~200 split-half nulls per state (pure arithmetic on cached rows, seconds).
**Criterion 3 is cheap. There is no cost reason it has never been run.**

---

## 6. Common Pitfalls

### Pitfall 1 — the prior-state column is silently dropped by the existing feature-admission logic

```python
        active.remove(max(active, key=lambda c: _first_valid_position(X, c)))
```
[VERIFIED: `backtest/driver.py:166`, verbatim]

`_cv_safe_active_features` drops the **newest-starting** column repeatedly until every class has
`>= n_splits` examples. A prior-state column that starts in 1967 or 1972 while every other
feature starts in 1962 is, by construction, the newest — so it is **the first column dropped**,
in exactly the early windows where the CV is tightest. The run would complete, log nothing
unusual, and the feature would simply not be there for a large share of steps.
**Warning sign:** measure and report, per step, whether the prior-state columns survived
admission. A run where they survive 400 of 588 steps is not the experiment anyone thinks it is.

### Pitfall 2 — `assert_causal_features` gives zero protection here

```python
FORBIDDEN_CENTERED_SUFFIXES: tuple[str, ...] = ("_centered", "_c5", "_zerophase")
```
[VERIFIED: `honesty/gating.py:33`, verbatim] and
`offending = [str(c) for c in columns if str(c).endswith(FORBIDDEN_CENTERED_SUFFIXES)]`
[VERIFIED: `honesty/gating.py:48`]. It is a **name-suffix** scan. `prior_state_p0` passes
trivially. Do not let the presence of a causal guard in `fit_nowcaster`'s first line create the
impression that this feature is checked. It is not. §3's tests are the only guard.

### Pitfall 3 — the K prior-state columns are exactly collinear

A K-way distribution sums to 1, so the K columns are rank K−1. `LogisticRegression` defaults to
L2 with `C=1.0` so it will not blow up, and `StandardScaler` sits in front
[VERIFIED: `prediction/nowcaster.py:131-134`] — but the fitted coefficients are then not
interpretable individually, which matters for S-3's "coefficient mass on prior-state columns"
reading. Drop one column, or report the *group* norm rather than per-column coefficients.

### Pitfall 4 — the two embargo concepts, and now a third

The module docstring already warns that `build_nowcaster_training_set`'s `embargo_months` and
`fit_nowcaster`'s `PurgedEmbargoedKFold` embargo *"must never be merged"*
[VERIFIED: `prediction/nowcaster.py:11-13`]. A recursive feature introduces a **third** lag
concept — the feature's own lag — which must stay ≤ the per-fold `embargo` (§2.5). Three
concepts named "embargo/lag" in one module is a comprehension hazard. Name the new one
something that is not "embargo".

### Pitfall 5 — measuring the wrong churn series

This is **F-1**, restated as a pitfall because it is the one most likely to be walked into: the
metric that exists measures L1's terminal-month label; the fix changes L2. Re-measuring the
existing metric produces 246 either way.

### Pitfall 6 — 100 of 588 steps degrade under l2 routing

[VERIFIED: measured this session, `joint_lift_joint_l2.parquet`]. A degraded step holds previous
weights and is excluded from `per_step_metrics`, which **dampens measured churn**. Any churn
number from the l2 routing must be quoted with its degraded count, and adding K more columns to
the nowcaster can only make `_cv_safe_active_features` degrade *more* often. Report the degraded
count before and after; a churn improvement bought by more degraded steps is not an improvement.

### Pitfall 7 — comparing centroids across differently-standardized fits

Trap A in §5.3. Listed again here because it produces a plausible-looking table rather than an
error, which is this project's named worst failure mode.

### Pitfall 8 — a state that evaporates still returns a centroid

Trap B in §5.3.

---

## 7. Validation Architecture

`workflow.nyquist_validation` is absent from `.planning/config.json`'s `workflow` block in the
explicit-false sense — it is present and `true` [VERIFIED: `.planning/config.json`].

### Test Framework

| Property | Value |
|---|---|
| Framework | pytest 8.0+ (`[tool.pytest.ini_options]` in root `pyproject.toml`) |
| Config file | `pyproject.toml` |
| Quick run | `pytest tests/unit/test_platform_nowcaster.py tests/unit/test_platform_hysteresis.py -x -q` |
| Full suite | `pytest tests/ -q` — baseline **2018 passed, 0 skipped, 0 xfailed** |

### Phase requirements → test map

| Req | Behavior | Type | Command | Exists? |
|---|---|---|---|---|
| C1 | prior-state feature present in the nowcaster's fitted columns | unit | `pytest tests/unit/test_platform_nowcaster.py -k prior_state` | ❌ Wave 0 |
| C1 | **guard fails** on smoothed substitution — S-1 lead/lag | unit | `pytest tests/unit/test_platform_nowcaster_recursion.py -k lead_lag` | ❌ Wave 0 |
| C1 | **guard fails** on smoothed substitution — S-2 exact agreement with `y.shift(1)` | unit | same file, `-k shift_one` | ❌ Wave 0 |
| C1 | prior-state column survives `_cv_safe_active_features` (Pitfall 1) | unit | `test_platform_backtest_driver.py -k prior_state_admission` | ❌ Wave 0 |
| C2 | the two churn series are measured **separately** (F-1) | unit | `test_platform_joint_diagnostics_record.py -k churn_series` | ❌ Wave 0 |
| C2 | per-step probability matrix is persisted (§2.6) | unit | `test_platform_backtest_driver.py -k proba_persisted` | ❌ Wave 0 |
| C3 | hysteresis output actually reaches the allocator | unit | `test_platform_hysteresis.py -k gates_allocation` | ❌ Wave 0 |
| C3 | one-hot input ⇒ `update_active_regime` is the identity (F-2, pinned) | unit | `test_platform_hysteresis.py -k one_hot_identity` | ❌ Wave 0 |
| C4 | Hungarian matching defeats a **deliberately permuted** labeling | unit | `test_platform_labeling_stability.py -k permutation` | ❌ Wave 0 |
| C4 | zero-occupancy state is flagged despite a small matched distance (Trap B) | unit | same file, `-k evaporated` | ❌ Wave 0 |
| C4 | distances are computed in a **common** unit space (Trap A) | unit | same file, `-k destandardized` | ❌ Wave 0 |
| C4 | frozen column list is held fixed across subsamples (Trap C) | unit | same file, `-k frozen_columns_fixed` | ❌ Wave 0 |
| C7 | `driver.py:497` receives the **unpooled** estimate (G6) | unit | `test_platform_backtest_driver.py -k unpooled` | ❌ Wave 0 |
| C8 | recorded test counts match `--collect-only` | unit | `test_docs_counts.py` | ❌ Wave 0 |

### Sampling rate

- Per task commit: the quick run above.
- Per wave merge: `pytest tests/ -q`.
- Phase gate: full suite green, ratchet still 31, before `/gsd-verify-work`.

### Wave 0 gaps

- [ ] `tests/unit/test_platform_nowcaster_recursion.py` — S-1 and S-2, each written so the
      **smoothed-substitution case fails**. Covers C1.
- [ ] `tests/unit/test_platform_labeling_stability.py` — permutation, evaporation, unit-space,
      frozen-columns. Covers C4.
- [ ] Extend `tests/unit/test_platform_backtest_driver.py` — admission, proba persistence,
      unpooled pin. Covers C1/C2/C7.
- [ ] Extend `tests/unit/test_platform_hysteresis.py` — one-hot identity, allocation gating.
      Covers C3.
- No framework install needed.

**A note the planner should carry forward:** every one of these must be written so that it
*can* fail on the thing it names. The repeated defect this project records — five times per
D-03 — is a test whose shape can only confirm.

---

## 8. Security Domain

`workflow.security_enforcement` is `true`, `security_asvs_level` 1
[VERIFIED: `.planning/config.json`].

### Applicable ASVS categories

| Category | Applies | Control |
|---|---|---|
| V2 Authentication | no | no auth surface; local batch CLI |
| V3 Session Management | no | no sessions |
| V4 Access Control | no | single local user |
| V5 Input Validation | **yes** | config validated at load (`platform/config.py` required-section check); new subsample/hysteresis parameters must be validated at read time, mirroring `classifier2_config`'s raise-on-drift pattern [VERIFIED: `labeling/classifier2.py:167-208`] |
| V6 Cryptography | no | none used |

### Threat patterns relevant to this phase

| Pattern | STRIDE | Mitigation |
|---|---|---|
| In-place patching of a persisted artifact instead of a re-run (T-07-17's class) | Tampering | Full re-runs are ~2 min; Option B's ~5.2 h would *create* the incentive this control relies on being absent. A reason to prefer A or F. |
| Silent degradation masquerading as a result (100/588 degraded steps) | Repudiation | Pitfall 6 — report degraded counts alongside every churn number. |
| A metric that cannot fail (F-1, and the five recorded evidence-shape failures) | Repudiation | §7's "must be able to fail" rule; §3's guard tests. |
| `joblib`/pickle model load (P27 class) | Tampering | Unchanged by this phase; `save_model` already uses joblib. |

No network surface, no user input, no secrets touched. ASVS L1 is satisfied by the existing
controls; the phase adds no new attack surface.

---

## 9. Open Questions — what a human must decide

| # | Question | What we know | What's unclear | Recommendation |
|---|---|---|---|---|
| **Q-1** | Does Option F (explicit Bayes filter, no training column) satisfy criterion 1, or must the prior state literally be a **feature**? | F has zero train/serve skew, zero free parameters, zero leakage surface, and implements §4.2's recursion verbatim. Criterion 1 says *"feature set includes"*. | Whether Glenn wants the letter or the intent. | **Ask before planning.** If the letter binds, go Option A. If the intent binds, F is strictly better on every axis this research can measure. |
| **Q-2** | Cold-start fill: uniform 1/K, or the in-window unconditional class distribution? | Both are causal. Dropping rows is fatal at the early steps. The rule must be *identical* at train and serve. | The literature offers no analogue (`<BOS>` is learned). `[ASSUMED]` that unconditional beats uniform. | Unconditional, with a warm-up start earlier than the decision spine. Record as a decision, not a default. |
| **Q-3** | §5.3 mechanism: (a) hard gate, (b) bounded turnover, (c) hysteresis scales magnitude? | (a) measured → 66% cash and violates §5.1's probability contract; (b) closes A7 by rewording; (c) satisfies both but adds a shrink parameter. | Whether Glenn accepts closing A7 by rewording. A7 explicitly offers it. | **(c)**, (b) as fallback. Measure criterion 7 under whichever is chosen. |
| **Q-4** | Re-pin 0.70/0.40, or redefine act/unwind relative to 1/K? | At least 309/488 non-degraded steps had max p < 0.70. 0.70 is 4.2× uniform at K=6, 3.5× at K=5. Every re-pinned value is a registry trial (42/44 used). | Whether a K-relative coefficient is "derived" (free) or "tuned" (a trial). | Propose K-relative; let Glenn rule on whether the coefficient is a trial. |
| **Q-5** | Criterion 3: add a **leave-one-episode-out** scheme, and report block bootstrap across {6,12,24,48} months rather than one length? | ROADMAP criterion 4 names three schemes. Neither decade-drop touches state 2's 1996-2002 block. Politis–White n^{1/3}≈8.9 targets variance estimation, not persistence. | Whether adding a fourth scheme is in scope. | Add it. It is the design's own "drop 2008-09" example generalised, and without it criterion 3 will miss the state most likely to fail. |
| **Q-6** | A11 (criterion 6) — should any gate fail on a bad-but-working model? | `07-BANDS.md` §8: promoting the domain tier means gating on `[ASSUMED]` numbers no measurement supports. Glenn reopened it 2026-09-21. | The decision itself. | **Out of this research's three questions.** Flagged so it is not lost. Whatever is decided, D-06 requires it be written as the reversal it is. |
| **Q-7** | Is the λ/d diagnostic (§F-3, churn vs. `iloc[-k]`) in scope? | It costs zero registry trials at fixed λ and would tell you how much of the 41.84% is an edge artefact. A λ **sweep** costs one trial per value. | Whether even a recorded-not-selected diagnostic is acceptable this close to the ceiling. | Run the zero-trial version (churn vs. k). Do **not** sweep λ without an explicit ruling. |
| **Q-8** | F-4's off-by-one (246/588 vs 246/587) — fix or annotate? | 0.07pp. But criterion 8 is about exactly this class of recorded number. | Whether changing the recorded rate breaks the pin in `test_platform_joint_diagnostics_record.py:133`. | Fix the denominator **and** update the pin in the same commit, with the reason in the message. |

### What could not be resolved

- **The magnitude of the terminal-month boundary effect (F-3).** The *existence* of the λ-vs-2λ
  asymmetry follows from the objective quoted at `jump_model.py:43-45`. How much of the 41.84%
  it accounts for cannot be known without the churn-vs-k measurement, which is Q-7.
- **The distribution of the nowcaster's max calibrated probability.** Only a lower bound is
  derivable (≥309/488 non-degraded steps below 0.70) because the per-step probability matrix is
  not persisted (§2.6). Persist it and this becomes a direct measurement.
- **Whether Option A's iterated passes converge.** Predicted to; not tested. Cheap to test
  (~10 minutes) and worth testing before committing to A.
- **How much `assert_causal_features`' blindness (Pitfall 2) has already cost elsewhere.** Out of
  scope; noted.

---

## 10. Environment Availability

| Dependency | Required by | Available | Version | Fallback |
|---|---|---|---|---|
| `scipy.optimize.linear_sum_assignment` | criterion 3 Hungarian matching | ✓ | scipy 1.17.1 | none needed |
| `scipy.linalg.sqrtm` | Gaussian W₂ secondary distance | ✓ | scipy 1.17.1 | drop the secondary |
| `scipy.stats.wasserstein_distance_nd` | (evaluated and **rejected**, §5.2) | ✓ | scipy 1.17.1 | n/a |
| `numpy` / `pandas` / `sklearn` | everything | ✓ | 2.4.6 / 3.0.5 / 1.9.1 | — |
| tracked parquet artifacts under `outputs/reports/platform/joint_lift/` | before/after comparison | ✓ | 4 parquets + 4 JSONs at `b1c3519` | re-run (~2 min/leg) |
| `data/checkpoints/platform/regime_labels{,_2}.parquet` | criterion 3 reference labeling | ✓ | 695 / 696 months | re-run `label_regimes` |

[VERIFIED: measured this session]

**Missing dependencies with no fallback:** none.
**New packages required:** none. The phase adds no dependency and no legacy import.

---

## 11. Package Legitimacy Audit

**No external packages are installed by this phase.** Every library named above
(`scipy`, `numpy`, `pandas`, `scikit-learn`) is a pre-existing pinned dependency in
`pyproject.toml` and was verified present and importable in the live environment this session.
The Package Legitimacy Gate is therefore not applicable; no package is `[ASSUMED]`, none is
`[SUS]`, none is `[SLOP]`, and no `checkpoint:human-verify` install gate is required.

---

## 12. Assumptions Log

| # | Claim | Section | Risk if wrong |
|---|---|---|---|
| A1 | The magnitude of the DP terminal-month boundary effect is material to the 41.84% | F-3 | Q-7's diagnostic is wasted effort; the λ/d explanation still stands on classifier #2's 4.08% |
| A2 | Option A's iterated passes converge to a fixed point | §2.2 | A carries a permanent, unresolvable train/serve inconsistency; Option F becomes clearly preferable |
| A3 | The in-window unconditional class distribution is a better cold start than uniform 1/K | §2.4 | Marginal; affects the first ~60 rows of the training column only |
| A4 | A K-relative act threshold is more principled than a re-pinned absolute | §4.4 | Both remain choices; neither is free |
| A5 | {6, 12, 24, 48} months is the right block-length ladder | §5.6 | A different ladder is used; the range-reporting recommendation is unaffected |
| A6 | Mechanism (c) (hysteresis scales tilt magnitude) fits the existing contract best | §4.4 | (b) is the fallback and is also viable |
| A7 | `wasserstein_distance_nd` at n=230, d=10 costs ~1.9 s and scales badly beyond | §5.2 | Only strengthens the rejection |
| A8 | Classifier #1's state 2 (one 71-month episode) is the likeliest criterion-3 failure | §5.1 | The prediction is wrong and criterion 3 exonerates everything — which is itself a recordable result |

All other claims carry `[VERIFIED: …]` or `[CITED: …]` inline.

---

## 13. Sources

### Primary (in-repo, read this session — HIGH)

- `src/trading_crab_lib/platform/prediction/nowcaster.py` (lines 11-13, 54-137, 140-177)
- `src/trading_crab_lib/platform/backtest/driver.py` (lines 66-82, 157-173, 290-305, 400-530)
- `src/trading_crab_lib/platform/backtest/joint_driver.py` (lines 97-98, 241-268, 400-520; one-hot at 251)
- `src/trading_crab_lib/platform/allocation/hysteresis.py` (full)
- `src/trading_crab_lib/platform/allocation/tilt.py` (full)
- `src/trading_crab_lib/platform/allocation/joint_tilt.py` (lines 7-45, 139-360)
- `src/trading_crab_lib/platform/labeling/jump_model.py` (full)
- `src/trading_crab_lib/platform/labeling/classifier2.py` (lines 1-330)
- `src/trading_crab_lib/platform/labeling/diagnostics.py` (lines 1-80, function index)
- `src/trading_crab_lib/platform/honesty/cv.py` (full)
- `src/trading_crab_lib/platform/honesty/gating.py` (full)
- `src/trading_crab_lib/platform/honesty/walkforward.py` (lines 38-108)
- `src/trading_crab_lib/platform/report/weekly.py` (lines 95-300)
- `src/trading_crab_lib/platform/evaluation/report.py` (`_reference_label_columns`)
- `scripts/joint_lift_diagnostics.py` (lines 90-175)
- `tests/unit/test_platform_joint_diagnostics_record.py` (full)
- `tests/unit/test_platform_legacy_import_ratchet.py` (lines 1-45)
- `config/platform_settings.yaml` (`labeling`, `labeling_2`, `allocation`, `backtest`)
- `platform_design/platform_design.md` §4.1–4.4 incl. the 2026-09-18 AMENDMENT, §5.1–5.4

### Primary (measured this session from tracked artifacts — HIGH)

- `outputs/reports/platform/joint_lift/joint_lift_{joint,baseline}_{l1only,l2}.parquet`
- `outputs/reports/platform/joint_lift/diagnostics_l1only.json`
- `data/checkpoints/platform/regime_labels{,_2}.parquet`,
  `regime_confidences{,_2}.parquet`
- Distance-estimator benchmark (20 replications × 5 sample sizes × 3 estimators, d=10)
- Live import/version check: scipy 1.17.1, numpy 2.4.6, sklearn 1.9.1, pandas 3.0.5

### Secondary (web — MEDIUM)

- [Politis & White (2004), *Automatic Block-Length Selection for the Dependent Bootstrap*](https://public.econ.duke.edu/~ap172/Politis_White_2004.pdf) — and the [Econometric Reviews entry](https://www.tandfonline.com/doi/abs/10.1081/ETC-120028836); correction by Patton, Politis & White
- [Purged & Embargoed Cross-Validation, Explained](https://quantmemo.com/concepts/purged-embargoed-cv) — the embargo-targets-feature-autocorrelation point
- [Purged cross-validation (overview)](https://en.wikipedia.org/wiki/Purged_cross-validation) — López de Prado, *Advances in Financial Machine Learning* ch. 7
- [Scheduled Sampling in Sequence Prediction](https://www.emergentmind.com/topics/scheduled-sampling) — exposure bias, the ε schedule, and its documented failure modes
- [Hungarian Matching Algorithm](https://www.emergentmind.com/topics/hungarian-matching) and [Hungarian Algorithm for State-to-Regime Matching](https://rpubs.com/Annice/Hungarian) — label switching and minimum-cost bipartite alignment of latent states

### Where the literature does not settle it, and judgement is required

- **Which honest option** (A / B / F) to take. The sequence-modelling literature addresses
  exposure bias in *long* rollouts with a fixed vocabulary and abundant data; none of those
  conditions hold at 588 monthly steps with K=6 and a 44-trial registry ceiling. There is no
  paper to defer to. Q-1.
- **Cold start.** No analogue exists. Q-2.
- **A persistence threshold for criterion 3.** The design gives none, deliberately; §5.5 gives
  yardsticks instead of a number, and the only binding pass/fail is the design's own
  AMENDMENT condition (i).
- **Block length for a *temporally penalized* model.** Politis–White targets variance
  estimation. No source found addresses block bootstrap under a jump-penalty objective, where
  block seams fire the penalty on artefacts. Report across a range and say so.

---

## Metadata

**Confidence breakdown**

- **F-1 / F-2 / F-3 / F-4 (the measured findings):** HIGH. Each is re-derivable in one line
  from a git-tracked artifact at `b1c3519`; the commands are given inline.
- **Option costs (§2.2):** HIGH for A and F (arithmetic on a measured 0.109 s/fit); HIGH for B's
  ~5.2 h (same arithmetic, Σ ≈ 588²/2).
- **Leakage signatures (§3):** HIGH on S-1 and S-2 (they follow from the non-causality of the DP
  decode, quoted verbatim); MEDIUM on S-3's thresholds, which are directional, not numeric.
- **Distance choice (§5.2):** HIGH — measured, 20 replications, at this project's actual d and n.
- **Subsample schemes (§5.6):** MEDIUM. The block-length ladder is `[ASSUMED]`; the objection to
  block bootstrap under a jump penalty is structural and HIGH.
- **§5.3 mechanisms (§4.4):** HIGH on what each *breaks* (measured); MEDIUM on the
  recommendation between them, which is a judgement.

**Research date:** 2026-09-21
**Valid until:** 2026-10-21 for the literature; the in-repo findings are valid until the next
L1 refit or λ/K re-pin, at which point every measured number above must be re-derived.
