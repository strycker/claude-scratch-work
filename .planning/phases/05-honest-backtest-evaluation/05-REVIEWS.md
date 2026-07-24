---
phase: 5
reviewers: [claude]
reviewed_at: 2026-07-24T15:41:04Z
plans_reviewed: [05-01-PLAN.md, 05-02-PLAN.md, 05-03-PLAN.md, 05-04-PLAN.md, 05-05-PLAN.md, 05-06-PLAN.md, 05-07-PLAN.md]
reviewer_note: >-
  Only the Claude CLI was available in this environment (gemini, codex, cursor,
  opencode, qwen, antigravity, coderabbit, and local model servers all absent).
  The review was run as a fresh, separate Claude session (no shared context with
  the orchestrating session) with full read access to the git working tree, so
  its findings are source-grounded. Cross-vendor independence was not achievable;
  treat this as a single independent review, not a multi-model consensus.
---

# Cross-AI Plan Review — Phase 5 (Honest Backtest & Evaluation)

> **Independence caveat:** No third-party AI CLI was installed in this environment.
> The review below comes from one independent Claude session (fresh context, repo
> file access). It is genuinely separate from the planning session, but it is not a
> multi-vendor consensus — weight it as a single grounded reviewer.

## Claude Review

## Summary

This is a well-organized, dependency-correct 7-plan set that mostly does what Phase 5 needs: wire the frozen Phase 1–4 primitives into an honest walk-forward backtest without inventing new math. The plans correctly reuse `expanding_steps()` rather than `run_walkforward()`, correctly apply `split_by_holdout_boundary()` before constructing the loop, correctly avoid importing the salvage file's `FoldReport`/incumbent dependency (with grep gates to enforce it), and correctly build the no-regime ablation as a flag-off call into the same driver rather than a forked implementation. However, tracing the actual data flow between the walk-forward driver's per-step outputs and the two downstream honesty modules (`sojourn_lag.py` and `model_metrics.py`) surfaces a real architectural gap: neither plan specifies how the driver's per-step *multiclass* probability vectors get converted into the single-series inputs that `gap_lag.compute_detection_lag()` and the Brier/calibration accumulator actually require. Given this project's stated goal is a backtest that is "never fooled by its own backtest," and EVAL-03's sojourn/lag ratio is explicitly the phase's headline number, this is the kind of gap that could pass every specified unit test (which only exercise pre-built synthetic single-series inputs) while still producing a meaningless real-world result for the one number the whole phase is built to report honestly.

## Strengths

- Correct identification that `run_walkforward()` (a single-estimator `.fit/.predict` wrapper, `src/trading_crab_lib/platform/honesty/walkforward.py:52-106`) cannot be reused verbatim for a 4-layer (L1–L4) per-step loop, while its inner `expanding_steps()` generator (`walkforward.py:38-49`) is reused unchanged — this is verified against source, not just asserted.
- Holdout discipline is applied correctly and early: `split_by_holdout_boundary(monthly_features, cutoff=DEFAULT_HOLDOUT_CUTOFF)` (`holdout.py:42-54`, cutoff `"2020-12-31"` at `holdout.py:31`) is applied *before* constructing `expanding_steps`, per Plan 02 Task 3, and the `crisis_windows` config default in Plan 01 is hard-bounded with an explicit `ValueError` guard (Plan 03 Task 1) — a concrete, testable mitigation for Pitfall 4 rather than just a comment.
- EVAL-04's salvage adaptation is disciplined: the plans extract only the three pure functions (`_compute_brier_multiclass`, `_calibration_bins`, `_confusion_tidy` — verified present at `ideas/gsd-salvage/prediction/model_metrics_artifacts.py:26-117`) and enforce non-import of `FoldReport`/the salvage file via grep gates in both Plan 04 tasks — a real, mechanically-checked boundary, not just a stated intention.
- The no-regime ablation (Plan 05 Task 3) is built as `run_backtest(..., use_regime_tilt=False)` — a one-line delegation with a grep gate against `vol_targeted_tilt(` / `regime_tilt_weights(` appearing in `baselines.py` — correctly avoiding the anti-pattern RESEARCH.md explicitly warns against (a hand-rolled parallel equal-weight implementation).
- 1972 first-decision arithmetic checks out: `monthly_features` starts 1962-01 (`platform_settings.yaml:10`), `min_train_months: 120` (Plan 01) = 10 years, landing the first `expanding_steps` yield at 1972-01 — verified correct, not just plausible.
- The `backtest:`/`evaluation:` config sections correctly follow the established defensive `.get()` pattern and are explicitly kept out of `_REQUIRED_PLATFORM_SECTIONS` (`config.py:26-33`), matching every prior phase's convention exactly.
- D-04 (no DSR, no holdout eval) is enforced with a grep gate for `"deflated"|"dsr"` in `report.py` (Plan 06 Task 2) — a real mechanical check rather than a documentation-only promise.

## Concerns

- **[HIGH] EVAL-03's headline sojourn/lag ratio has no specified construction path from multiclass probabilities to the single-series input `gap_lag.compute_detection_lag()` requires.** `compute_detection_lag(transitions, filtered_probs: pd.Series, threshold)` (`gap_lag.py:44-92`) applies **one** probability series uniformly to check *every* transition in `transitions` (`gap_lag.py:82-88`: `tail = probs.iloc[t:]`). But the driver's per-step nowcaster output is a K=5-class probability vector (`regime_probs`), and different transitions go to *different* target states — there is no single scalar series that is simultaneously "the right confidence to check" for a transition into state 2 and a transition into state 4. RESEARCH.md's own Pattern 3 code example (lines 358-374) papers over this by passing one static series to all transitions, and neither Plan 03 (`compute_sojourn_lag_headline` signature, `sojourn_lag.py`) nor Plan 06 Task 3 (which calls it with `filtered_probs` built from `per_step_metrics`) specifies how the conversion happens. A naive-but-plausible construction (max-probability-across-classes, regardless of which class) would systematically *understate* detection lag — the exact "fooled by its own backtest" failure mode this project's honesty framework exists to prevent — and Plan 03's own test (`TestSojournLagHeadline`) only proves the orchestration wiring against a **pre-built** synthetic single series, so it cannot catch this gap even if implemented wrong.
- **[MEDIUM-HIGH] EVAL-04's per-step `y_true` accumulator has no consistent, achievable definition of "the label at t."** Plan 02 Task 3's must-haves say per-step accumulators use "y_true = the canonical label at t," but `train_index` in `expanding_steps` is **strictly before** `t` (`walkforward.py:38-49`) — the L1 refit only ever produces labels for `train_index` positions, never for `t` itself. RESEARCH.md's Pattern 1 code example resolves this differently and, I'd argue, incorrectly: `y_true_log.append(labels.iloc[-1])` (RESEARCH.md line 306) uses the **last training-window position's** label (effectively `t-1`'s label under that step's batch fit) as ground truth for a prediction targeting `t` — an off-by-one mismatch between what's being predicted and what's being scored. Neither plan reconciles this contradiction, and no test (`TestEquityCurveCompounding`, `TestRegistryLogging`, `TestRefitFromTrainWindowOnly`, `TestHoldoutBoundary` in Plan 02) checks date alignment between `y_true` and the target month, so a wrong implementation would sail through the specified suite while quietly corrupting the Brier/calibration/confusion artifacts EVAL-04 exists to produce.
- **[MEDIUM] Per-step `classes_` sets may vary in width across the 1972–2020 loop, but Plan 04's stacking step assumes a single consistent class list.** `fit_nowcaster`'s `CalibratedClassifierCV` derives `model.classes_` from whatever labels are actually present in each step's (embargoed) training window (`nowcaster.py:83-120`); in the earliest history it's plausible not all K=5 canonicalized regimes have occurred yet, so `predict_proba` could return fewer than K columns in early steps and K columns later. Plan 04 Task 2's action text ("Stack the driver's per-step accumulators (y_true list, proba list-of-rows -> np.ndarray, classes)...") does not address reconciling a ragged/varying-width `proba` sequence into one `(n_steps, K)` array before calling `compute_brier_multiclass`/`calibration_bins`, which assume a single `classes` list applied to a rectangular `proba` array (`ideas/gsd-salvage/.../model_metrics_artifacts.py:26-41`, adapted verbatim per the plan).
- **[MEDIUM] Cash-return convention for the strategy leg's residual (`tilt["cash"]`) is unspecified and may be asymmetric with the baseline legs.** `vol_targeted_tilt()` returns `"cash": 1.0 - scale` as an un-invested residual (`tilt.py:134-173`) with no attached return series. Plan 02's `_realized_return(tilt["weights"], tilt["cash"], asset_returns.loc[test_index[0]])` doesn't specify whether the cash residual earns 0% or the actual `cash` research series return, while Plan 05's `sixty_forty`/`faber_sma` baselines explicitly use `cash_ret` (the `splice.py` `yield_as_return` series, `platform_settings.yaml:140-144`) for their non-invested legs. If the strategy's cash sits at 0% while Faber's out-of-market position earns the actual T-bill yield, the §23.1 "beats Faber on log wealth" comparison (D-01a's standing target) is quietly cost-asymmetric in the strategy's favor over a 1972–2020 span that includes double-digit T-bill yields — undermining exactly the apples-to-apples framing A5 in RESEARCH.md calls out for the *cost* convention but doesn't address for the *cash-return* convention.
- **[MEDIUM] The no-regime ablation likely doubles the real 1972–2020 run's wall-clock cost without an explicit skip, compounding an already-unverified performance risk (A2).** Plan 05 Task 3's `no_regime_ablation` delegates to `run_backtest(..., use_regime_tilt=False)`, and Plan 02's driver text describes L1/L2 refit happening unconditionally each step with the constant-label override applied only when feeding L3/L4 ("When use_regime_tilt is False, feed a constant-single-state label Series... so the SAME tilt code runs regime-agnostically"). Nothing in the plans states that L1 (jump-model, `n_restarts=10` per `platform_settings.yaml:255-257`) and L2 (nowcaster CV fit) are skipped for the ablation path when their output is discarded anyway. Since RESEARCH.md's own Assumption A2 already flags "588 refits... computationally feasible... not benchmarked" as a medium-risk unverified claim, and Plan 07 is a real, human-verified execution against live 1962+ data, silently doubling that unverified cost is a concrete schedule/feasibility risk for the one plan (07) that cannot be iterated on cheaply.
- **[LOW]** ROADMAP.md places Plan 05 in "Wave 3 *(blocked on Wave 2 completion)*" even though Plan 05's own frontmatter only lists `depends_on: [05-02]` — Plan 05 doesn't actually need 05-03/05-04 to finish first. Not a correctness bug, just unnecessary serialization.
- **[LOW/informational]** RESEARCH.md's Open Question 2 (whether the 3 price-only baselines should log trial-registry rows) is resolved by the plans (only strategy + ablation log trials) but was never explicitly confirmed as a locked decision in CONTEXT.md — a reasonable default, but worth the planner flagging as a judgment call rather than settled fact.

## Suggestions

- Before Plan 03/06 execution, resolve explicitly (in the PLAN text, not left to the executor) how `filtered_probs_series` is constructed per-transition from the driver's multiclass `per_step_metrics`, and add a test that exercises the *real* multiclass-to-single-series conversion, not just `compute_sojourn_lag_headline`'s wiring against pre-built synthetic inputs.
- Resolve the `y_true`-at-`t` contradiction directly: either (a) accumulate `(t, regime_probs)` pairs during the loop and retroactively join against the Plan 06 smoothed full-sample reference labeling once it's built (which already exists and is independently computed — reuse it rather than trying to source ground truth from the still-in-progress walk-forward loop), or (b) explicitly document why `t-1`'s label is an acceptable proxy for `t`'s ground truth (I don't think it is, given the whole point is honest real-time evaluation).
- Add an explicit union-of-classes reconciliation step (fixed K=5 by construction of `canonicalize_states`, so pad missing columns with 0 probability) before stacking `proba` for `compute_brier_multiclass`/`calibration_bins`, and add a test with a synthetic early-history step that has fewer observed classes than K.
- Document the cash-return convention for `tilt["cash"]` explicitly in the report (the same way turnover/cost conventions are documented per Pitfall 3), and make sure it uses the same `cash_ret` series the baselines use for their non-invested legs.
- Add a `backtest.skip_l1l2_for_ablation` (or just hard-code the skip) so the no-regime ablation doesn't redundantly refit an expensive jump-model + nowcaster 588 times for output that's discarded.

## Risk Assessment

**MEDIUM.** The plan set's mechanical scaffolding (config, costs, holdout boundary, registry logging, no-forked-ablation, salvage-import grep gates) is solid and well-verified against source. But the two gaps in EVAL-03 and EVAL-04 sit exactly at the phase's stated purpose — producing honest, unfoolable headline metrics — and neither is caught by the specified test suite, since those tests validate orchestration wiring against synthetic pre-built inputs rather than the real multiclass-to-single-series and label-timing conversions the driver must perform. Given this phase's explicit self-description as a diagnostic whose "verification passes on it ran and reported correctly" (D-01), a metric that runs without crashing but reports a wrong or meaningless number would satisfy the letter of every task's acceptance criteria while failing the phase's actual goal. I'd recommend resolving the EVAL-03 construction question before Plan 03/06 execution, since it's the phase's advertised headline number.

---

## Consensus Summary

Only one independent reviewer ran, so this is a single-reviewer synthesis rather
than a cross-model consensus. The review is strongly source-grounded (every finding
cites `file:line` and traces the actual mechanism), which makes its two HIGH/MEDIUM-HIGH
findings worth treating as action items before execution.

### Agreed Strengths
- **Correct primitive reuse.** Plans reuse `expanding_steps()` (not the single-estimator
  `run_walkforward()`), apply `split_by_holdout_boundary()` *before* loop construction,
  and build the no-regime ablation as `run_backtest(..., use_regime_tilt=False)` rather
  than a forked implementation — all verified against source.
- **Mechanically-enforced boundaries.** grep gates enforce D-04 (no `deflated`/`dsr` in
  `report.py`), non-import of the salvage `FoldReport`, and no hand-rolled tilt in
  `baselines.py`. The 1972 first-decision arithmetic (1962-01 start + `min_train_months: 120`)
  checks out.
- **Config discipline.** New `backtest:`/`evaluation:` sections follow the defensive `.get()`
  pattern and stay out of `_REQUIRED_PLATFORM_SECTIONS`, matching prior phases.

### Priority Concerns (single reviewer, but source-verified — treat as pre-execution blockers)
1. **[HIGH] EVAL-03 headline sojourn/lag ratio has no specified multiclass→single-series
   construction.** `gap_lag.compute_detection_lag()` takes one `filtered_probs` series applied
   to *every* transition, but the driver emits K=5-class vectors and different transitions
   target different states. No plan (03 or 06) specifies the conversion; a naive
   max-probability construction would understate detection lag — the exact "fooled by its own
   backtest" failure the phase exists to prevent. Existing tests only exercise pre-built
   synthetic single series, so they can't catch a wrong conversion. **This is the phase's
   advertised headline number — resolve in PLAN text before Plan 03/06 execution.**
2. **[MEDIUM-HIGH] EVAL-04 per-step `y_true` label-at-`t` is contradictory/off-by-one.**
   `train_index` in `expanding_steps` is strictly before `t`, so the L1 refit never labels
   `t` itself; RESEARCH.md's example scores against `t-1`'s label. No test checks date
   alignment between `y_true` and the target month, so a wrong impl corrupts the
   Brier/calibration/confusion artifacts silently. Recommended fix: retroactively join the
   accumulated `(t, regime_probs)` pairs against the Plan 06 smoothed full-sample reference
   labeling rather than sourcing ground truth from the in-progress loop.
3. **[MEDIUM] Ragged `classes_` width across the 1972–2020 loop.** Early steps may see fewer
   than K=5 regimes; Plan 04's stacking assumes a rectangular `(n_steps, K)` proba array. Add
   an explicit union-of-classes reconciliation (pad missing columns with 0) plus an
   early-history test.
4. **[MEDIUM] Cash-return convention asymmetry.** The strategy leg's `tilt["cash"]` residual
   has no specified return series, while the Faber/60-40 baselines use the actual `cash_ret`
   T-bill yield. Over a 1972–2020 span with double-digit yields, a 0%-cash strategy vs
   yield-earning baselines quietly biases the §23.1 Faber comparison. Document and unify the
   cash convention.
5. **[MEDIUM] No-regime ablation doubles the unverified run cost.** Nothing skips the L1
   jump-model (`n_restarts=10`) + L2 nowcaster refits for the ablation path when their output
   is discarded. Combined with RESEARCH.md's unbenchmarked A2 (588 refits), this is a real
   feasibility risk for Plan 07 (the one plan that can't be cheaply iterated). Add a
   `skip_l1l2_for_ablation` shortcut.

### Divergent Views
Not applicable — a single reviewer ran. Two LOW findings are worth a planner glance:
Plan 05's ROADMAP "Wave 3" placement over-serializes (its frontmatter only `depends_on: [05-02]`),
and RESEARCH.md Open Question 2 (price-only baselines logging trial rows) was resolved by the
plans but never locked in CONTEXT.md.
