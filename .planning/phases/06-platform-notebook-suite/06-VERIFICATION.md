---
phase: 06-platform-notebook-suite
verified: 2026-09-10T00:00:00Z
status: human_needed
score: 5/5 criteria met (with stated evidence-shape limitations); 2 open items awaiting human action (not phase failures)
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Open each of P1–P6 in notebooks/platform/ locally and Run All, top to bottom, against real checkpoints."
    expected: "No exception in any cell; figures render as they do in the already-committed outputs."
    why_human: "D-17 deliberately rejected execution-based CI (no nbmake/papermill). The committed notebooks prove one successful run happened in the executor's environment on 2026-09-09/10, not that a fresh run today reproduces it. This is criterion 1's evidence-shape limit, not a phase defect."
  - test: "Operator reads P3_regime_labeling.ipynb's rendered regime timeline, contingency table, and A13 side-by-side, then fills in the Cold-Start Sign-Off cell (Date / Verdict / Reasoning) and re-saves the notebook."
    expected: "A recorded verdict — accept, reject, or accept-with-caveats. Per D-16, a negative verdict is valid and does not block."
    why_human: "The verdict is inherently a human judgement (D-15/D-16); no agent may record it. Currently blank by design."
---

# Phase 6: Platform Notebook Suite Verification Report

**Phase Goal:** Every platform layer (L0–L4) and the Phase 5 evaluation have a notebook that
shows the data and renders the diagnostics, plus one cold-start selection gate at P3.

**Verified:** 2026-09-10
**Status:** human_needed (all automated checks pass; two known, deliberate open items require
a human operator — see `open_items` below, which are NOT phase failures)
**Re-verification:** No — initial verification

---

## Independent evidence gathered this pass

All of the following were re-run or re-inspected directly against the repository, not taken
from SUMMARY.md claims:

- `git log`, `git status`, `git diff 47e47fa..HEAD --stat` — confirmed the six notebooks exist,
  the diff is confined to `platform/plotting/`, `notebooks/platform/`, tests, and one additive
  32-line/2-deletion change to `platform/evaluation/report.py`. `backtest/`, `allocation/`,
  `labeling/`, `prediction/`, `registry/` all show **zero diff** vs phase-start commit `47e47fa`.
- `pytest tests/ -q` → **1705 passed, 0 skipped, 0 failures** (91s), matching the orchestrator's
  measured facts exactly.
- `pytest tests/unit/test_platform_notebooks.py -v` → 34/34 passed, including all three
  `test_a13_discipline_notebooks_mention_audit_item[...]` parameterizations (P3, P4, P6) — none
  skipped.
- Ran the two named regression tests directly:
  `test_historical_regression_111_raises` and `test_historical_regression_sixty_forty_raises`
  — both PASS. Read their bodies and `drift.py`'s implementation directly (not just the test
  names) to confirm the assertion is `match="universal"` for 111.06 and `match="domain band"`
  for −0.0227 — i.e. the −0.0227 case is proven to raise specifically because it clears the
  universal `[-1,0]` bound and fails only the per-leg domain band `[-0.80, -0.10]` for
  `sixty_forty`. This is the literal lesson the validation doc names, not a restatement of it.
- Read `_DOMAIN_MAX_DRAWDOWN_BANDS` and `assert_max_drawdown_plausible`/
  `assert_terminal_log_wealth_plausible` source: the bounds are named economic constants
  (`spy_buy_hold: (-0.90,-0.30)`, `sixty_forty: (-0.80,-0.10)`, etc.), not re-derived from the
  data they check — a real value-shape check, not existence/shape.
- Confirmed criterion-3 enforcement is live, not decorative, by independently reproducing the
  forbidden-token detection logic against a scratch copy of P1 with an injected
  `import matplotlib.pyplot as plt` cell — the offense list is non-empty, i.e. the check would
  fail the gate. (Reproduced the mechanism directly rather than trusting the SUMMARY's claim of
  having done this.)
- Grepped all six notebooks and `platform/plotting/*.py` for "A13" + "resolved" — the only hit
  is `backtest.py:35`, "...makes A13 *inspectable*" (a negation, in context). No notebook or
  library file claims A13 is settled.
- Grepped all six notebooks for the four void numbers (`111.06`, `99.7%`, `-2.3%`, `-2.27%`).
  Zero true hits. One false-positive substring match for `111.06` in `P5_assets_allocation.ipynb`
  traced to a base64-encoded PNG byte sequence (`...111106//zz...`), confirmed by decoding the
  notebook's cell sources/outputs directly — not a displayed value anywhere in source or text
  output.
- Confirmed `core.A13_CAVEAT`'s exact wording: "NOT INTERPRETABLE (audit item A13) ... no
  plausibility band around this number resolves that until A13 is settled" — a single-sourced
  string used identically by P3, P4, P6 per grep.
- Confirmed the P3 sign-off cell (`Cold-Start Sign-Off`) has literally blank `Date:` /
  `Verdict:` / `Reasoning:` fields, and that P2/P4/P5/P6 each carry an explicit
  "carries no sign-off cell" statement rather than silence.
- Decoded every notebook's cells programmatically: P1 (3 code cells), P2 (8), P3 (10), P4 (10),
  P5 (10), P6 (14) — all cells carry outputs, 0 error outputs across all six notebooks.
- Confirmed `get_holdout_checkpoint_manager`/`load_full_span` usage appears in P2, P4, P5, P6
  (the full-span drift panels); P1 and P3 use `load_platform_checkpoint`/`compute_regime_labeling`
  against the dev-fenced tree, consistent with D-06's "fence is on fitting, not looking."
- No new dependency in `pyproject.toml` / library `pyproject.toml`; no `sign_off(` function
  anywhere in `src/`; the only report-writer touched is the sanctioned additive extension.

---

## Per-Criterion Verdicts

### Criterion 1 — Six notebooks exist and each runs top-to-bottom without error

**Verdict: MET, with a stated and important limitation.**

**Evidence shape:** *existence + committed execution artifact* (each notebook's `.ipynb` file
contains every code cell already executed, in order, with zero error outputs, committed to the
repo). This is stronger than bare file-existence — I independently decoded all six notebooks'
cell/output structure and confirmed 0 error outputs across 55 total code cells — but it is not
proof that the code is *currently* runnable.

**Could this evidence shape fail on a wrong value?** No — this is exactly the "existence/shape"
trap the audit warns about, and the phase's own `06-VALIDATION.md` / D-17 acknowledge it
explicitly: CI does static checks only, deliberately rejecting `nbmake`/`papermill` execution.
A notebook that ran successfully once, was hand-edited afterward without re-execution, or that
depends on a checkpoint state that has since drifted, would show identically (committed outputs,
0 error cells) whether or not it would still run today. Nothing in the automated suite re-proves
"runs top-to-bottom" on a fresh invocation.

**What would and would not be caught:** A renamed function, a broken import, or a notebook that
never ran at all with fabricated outputs, would likely be caught only by the human-verification
step (Run All) — the static gate (`test_platform_notebooks.py`) checks nbformat validity, banned
tokens, and text content, not executability. A regression introduced in `platform/plotting/`
*after* a notebook's last execution (e.g. a signature change) would not be caught by anything
in this repository's automated suite today. This is an honestly-disclosed, deliberate scope
tradeoff (D-17), not an oversight — but it is the correct place to flag it as the weakest
evidence shape backing any of the five criteria, exactly as the audit predicts.
**Routed to human verification** (see frontmatter) rather than assumed.

### Criterion 2 — P3 carries the sign-off cell; the other five carry none

**Verdict: MET.**

**Evidence shape:** *text-content assertion* (a real check that a specific structure is present
or absent in specific files) — stronger than bare existence, weaker than a numeric value check,
but it is not gameable by "any string will do": I read the actual cell content, not just a
grep count. P3's sign-off cell literally contains blank `Date:`/`Verdict:`/`Reasoning:` fields
under a `## Cold-Start Sign-Off` heading, with D-15/D-16 language explaining why the cell is
plain markdown and why a negative verdict does not block. P2/P4/P5/P6 each carry an explicit
prose statement ("P5 carries no sign-off cell and no per-run gate...") rather than a silent
absence, which is itself independently checked by `TestP5NotebookSource::test_carries_no_sign_off_cell`
-style unit tests per plan.

**Could this fail on a wrong value?** Partially. A test asserting "no sign-off heading present"
can genuinely fail if a later edit reintroduces one — this is a real negative assertion, not
mere shape. It cannot, however, detect a sign-off cell that exists but says nothing meaningful;
that remains a human read.

**Open item (not a phase failure):** the P3 verdict itself is blank, awaiting the operator, per
D-15/D-16 by design. Recorded in the frontmatter `human_verification` list.

### Criterion 2b — Full-span reads via the explicit holdout opt-in; fitting stays fenced

**Verdict: MET.**

**Evidence shape:** *call-site presence + configuration-boundary test*. `load_full_span_checkpoint`
(wrapping `honesty.holdout.load_full_span`) is called in P2, P4, P5, P6's drift panels; P1/P3
use the dev-fenced `load_platform_checkpoint`/`compute_regime_labeling` path. The underlying
`honesty/holdout.py` guarantee ("no fallback code path from the default manager to the holdout
tree") is a Phase 2 artifact, not newly asserted here, but its correct *use* in this phase is
independently observable: the plausibility/quality gates that can stop a notebook (P2's
`assert_feature_ranges_plausible`, etc.) run on the DEV frame only, while the full-span read is
confined to a separate drift-only panel — confirmed by reading `features.py`'s structure
directly (the gate call site precedes and is separate from the full-span load).

**Could this fail on a wrong value?** Yes, partially — if a gate that can halt the notebook were
accidentally evaluated against post-2020 data, that would be a real, catchable defect (a
gate firing or not firing based on data it shouldn't see). This was checked by inspection, not
by a dedicated automated test asserting "the gate call happens before the full-span load" —
that specific ordering constraint is enforced by code structure and manual review, not by a
test that would fail if violated. This is a soft spot: worth a note, not a blocker, since manual
inspection of `features.py`, `nowcaster.py`, `allocation.py`, `backtest.py` all confirm the
same pattern (gate on dev frame, drift on full-span frame) consistently across all four notebooks
that touch full-span data.

### Criterion 3 — All plotting logic lives in `platform/plotting/`, never inline

**Verdict: MET, and this is a real, falsifiable check.**

**Evidence shape:** *AST/substring-based negative assertion, independently reproduced to
confirm it actually fires.* I did not just read the test file and take its self-description
at face value — I extracted the forbidden-token detection logic and ran it against a scratch
copy of P1 with an injected `import matplotlib.pyplot as plt; plt.plot(...)` cell. The
detector correctly flagged both `import matplotlib` and `plt.` as offenses. This is exactly
the class of check the audit's four-shapes framework calls out as *not* automatically
trustworthy (a "placement" check can be inert if never exercised) — here it was exercised and
shown to actually fire.

**Could this fail on a wrong value?** Yes — a notebook cell containing any of `import matplotlib`,
`import seaborn`, `plt.`, or `sns.` fails the test outright; this is a hard, non-negotiable gate,
confirmed live. The one gap: it is a substring/AST match, not a semantic one — a cell could
theoretically construct a `Figure` by calling into an aliased import that bypasses the four
literal tokens (e.g. `from matplotlib import pyplot as mpl_plt`). No notebook in the suite does
this (grep confirms all plotting calls route through `pplot.<submodule>.plot_x(...)`), but the
check's coverage is token-based, not import-graph-based, for the notebook side (the *library*
side does use a stricter AST import-closure check per D-01, confirmed present in
`test_platform_plotting.py`'s `TestFreshPackageBoundary`).

### Criterion 4 — P3 shows the 5 regimes against dated economic history

**Verdict: MET.**

**Evidence shape:** *data-flow + rendered content, partly numeric (contingency table),
partly requiring human economic judgement.* The regime timeline overlays a live FRED `USREC`
fetch (confirmed: 776 months, 8 recession periods, in the executed notebook's own printed
output) plus six dated economic eras (`ECONOMIC_EVENTS` module constant). The regime×era
contingency table is a real, bidirectional descriptive statistic — not an existence check —
and its rows are asserted to sum to 1.0 (era-conditional direction), a numeric invariant that
could fail on a bug in the underlying masking logic. The reported contingency (1973 oil shock
100% state 1, 1987 crash 100% state 4, GFC 58% state 0 vs 2% baseline occupancy) is a
substantive, checkable finding, not a shape claim.

**Could this fail on a wrong value?** The row-sums-to-1.0 and value-in-[0,1] invariants can fail
on a real bug (confirmed via `TestRegimeEraContingency`/`TestRegimeEraMarginals` unit tests, not
re-run live here but structurally identical to the pattern already verified for the KPI table).
Whether the contingency *pattern itself* is economically meaningful is inherently a human
judgement — which is exactly why this criterion is paired with the (open, blank) sign-off cell
rather than an automated pass/fail. The D-10 network-degradation path (USREC fetch failing)
was exercised only by unit tests in this environment, since the live fetch succeeded — an
honestly-disclosed gap in end-to-end coverage of the degraded path, not a defect.

### Criterion 5 — P6 renders equity curves, baseline gauntlet, ablation delta, calibration,
and the sojourn/lag headline with its resolved-transition count

**Verdict: MET.**

**Evidence shape:** *rendered figures decoded and visually inspected, plus live-computed
numeric text output, independently re-checked this pass.* I confirmed directly (not just via
the SUMMARY's claim) that:
- `wealth_delta`/`dd_delta` in cell 13 are printed as live output from
  `compute_ablation_delta(kpi_table)` — a computed value, not a hardcoded string.
- Cell 20's stream output contains the literal string
  `resolved       : n_resolved=4 of n_transitions=6 transitions (the ratio is a median over 4 observations)`,
  confirmed via direct grep of the notebook JSON.
- `core.A13_CAVEAT` is rendered verbatim in P6 (grep-confirmed identical string to P3/P4/`core.py`).
- No void number (111.06, 99.7%, −2.3%, −2.27%) appears as a current value anywhere in P6 —
  grep returns 0 real hits (the one false-positive substring elsewhere is in P5's PNG binary
  data, unrelated to P6 and not a displayed value in either notebook).

**Could this fail on a wrong value?** Yes, for the numeric parts: `compute_ablation_delta` is
a genuine computation over the live KPI table, and the sojourn/lag headline's `n_resolved`/
`n_transitions`/`ratio` are recomputed from the persisted `backtest_full_sample_states.parquet`
/ `backtest_filtered_state_probs.parquet` artifacts (verified in plan 06-02's own byte-identity
proof against the live backtest run) rather than typed literals. The equity-curve and
calibration panels are visual/legibility judgements (five-leg line distinguishability, marker
occlusion) that were checked by rendering and eyeballing the PNGs during execution — a category
that presence/shape checks alone cannot certify, and which this phase's own retrospective
(06-03 through 06-06 each shipped a rendering defect that passed `<verify>` before visual
inspection caught it) treats as a first-class risk. I did not re-decode and re-eyeball the PNG
bytes myself this pass (that would require rendering infrastructure beyond this verification's
scope); I relied on the plan's own documented visual-audit table plus the fact that the same
executor caught and fixed three real rendering defects earlier in the phase (P3's invisible
recession shading, P3's overdrawn disagreement strip, P2's colliding annotations) — evidence
that the visual-audit step is a real practice being applied here, not a rubber stamp.

---

## Plausibility contract (VALIDATION.md) — verified live

| Regression case | Band checked | Result |
|---|---|---|
| 60/40-shaped leg, max DD = −2.27% | universal `[-1,0]` (passes) then domain `[-0.80,-0.10]` for `sixty_forty` (fails) | **PASS** — raises with `match="domain band"`, confirmed by reading the raised message text, not just the exception type |
| Terminal log wealth = 111.06 | universal `abs(x) < 10` | **PASS** — raises with `match="universal"` |

Both non-negotiable regression cases from `06-VALIDATION.md` are proven to raise, and to raise
**for the specific reason the validation doc requires** (domain vs. universal), not merely "an
exception was thrown somewhere." This is the strongest evidence in the phase: a plausibility
band that is a genuine value check, backed by named economic constants, independently confirmed
against the source.

---

## Scope fences — confirmed held

- **No tuning:** no cell or function in the diff fits any (K, λ, n_restarts) other than the
  shipped config values (D-14). Confirmed by reading `regime.py`'s `plot_occupancy_and_sojourn`
  call sites and P3's notebook source — the only fit anywhere is the single shipped
  `label_regimes()` call.
- **No report wiring:** `git diff --stat` for `*report.py` shows only the one sanctioned
  additive change to `platform/evaluation/report.py` (32 insertions / 2 deletions, both
  deletions are lines being extended, not removed). `backtest_report.md`/`weekly_report.md`
  content generation is untouched.
- **No package restructuring:** `platform/backtest/`, `platform/allocation/`,
  `platform/labeling/`, `platform/prediction/`, `registry/` all show **empty diffs** vs.
  phase-start commit `47e47fa` — independently confirmed via `git diff --stat`.
- **No execution-based notebook CI:** confirmed no `nbmake`/`papermill` in dependencies or test
  infrastructure; `test_platform_notebooks.py` has a dedicated
  `test_this_module_uses_no_execution_based_testing` guard, which passes.
- **No `sign_off()` helper or machine-readable ledger:** `grep -rn "def sign_off" src/` returns
  nothing.
- **No new dependencies:** `git diff` on both `pyproject.toml` files and `requirements*.txt`
  vs. phase-start is empty.

---

## Open Items (recorded, not phase failures)

1. **P3's cold-start sign-off is blank**, awaiting the human operator. Correct per D-16 — the
   verdict is not an agent's to record.
2. **A13 is not resolved** — by design, this phase's job was to make it inspectable. Measured
   disagreement: 389/470 = 82.8%, no diagonal structure in the confusion table, three
   independent contributing causes documented (differing feature sets, 470/588 coverage gap,
   independent canonicalization).
3. **118 of 588 walk-forward steps (20.1%) are degraded**; the filtered path starts 1974-02,
   ~2 years after the 1972-01 backtest start. Surfaced consistently in P3 and 06-02's summary;
   this coverage gap compounds the A13 finding rather than being a separate defect.
4. **Drift-flagged columns have no `POST-2020-OBSERVATIONS.md` entries.** Three executors
   (06-04, 06-05, 06-06) independently declined to write one, judging that "is this observation
   decision-changing" is the operator's call, not theirs — consistent with the amended honesty
   framing (D-06/D-07).
5. **Criterion 1's execution-based re-proof does not exist by design (D-17).** Recorded above
   as the weakest evidence shape in the phase; routed to human verification.

None of these block phase completion — they are exactly the kind of finding this phase exists
to surface (D-16: "a negative verdict is recorded and does not block").

---

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| NB-01 | SATISFIED (per criteria above) | Six notebooks, plotting library, drift/plausibility checks, A13 discipline, sign-off gate — all independently confirmed present and functioning as designed. |

No orphaned requirements found for Phase 6 in REQUIREMENTS.md beyond NB-01.

---

## Anti-Patterns Found

None blocking. No `TBD`/`FIXME`/`XXX` markers found in the diffed files during this pass
(spot-checked `platform/plotting/*.py` and `platform/evaluation/report.py`). No stub returns,
no hardcoded empty data feeding a chart — every plotting function traced back to a real
checkpoint or persisted artifact load.

---

## Gaps Summary

No gaps found that block phase completion. The phase's automated half (plotting library
correctness, drift/plausibility math, static notebook discipline, the two non-negotiable
regression cases, scope-fence compliance) is fully green and independently re-verified in this
pass, not merely accepted on the SUMMARY.md's word. The phase's human half — Criterion 1's
fresh-run confirmation and Criterion 2's P3 verdict — is open by design (D-17/D-15/D-16) and is
routed to human verification rather than either passed or failed automatically.

---

*Verified: 2026-09-10*
*Verifier: Claude (gsd-verifier)*
