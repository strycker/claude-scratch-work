---
phase: 07-regime-representation
verified: 2026-09-15T14:58:59Z
status: human_needed
score: 4/4 wave-1 criteria verified (criteria 1-4); criterion 8 verified-scoped-to-wave-1-touch, flagged blanket claim unsupported
scope: phase-wave 1 ONLY (07-CONTEXT.md D-09); criteria 5, 6, 7 and INV-01 are wave 2 and correctly absent
behavior_unverified: 0
overrides_applied: 0
human_verification:
  - test: "Confirm the 'criterion 8 is currently TRUE' / 'platform imports nothing from the legacy library' claim in 07-CONTEXT.md and ADR-0001 against MIGRATION-PLAN.md's own same-week admission of 4 unvendored seams (checkpoints, ingestion x2, email)"
    expected: "A human decides whether this is (a) an acceptable pre-existing/out-of-wave-1-scope inaccuracy that should be corrected in ADR-0001's wording, or (b) something that must block phase sign-off"
    why_human: "This is a judgment call about how strictly to read a blanket claim repeated in a phase artifact (ADR-0001) versus the phase's actual scoped deliverable (no new legacy imports in wave-1-touched files, which IS verified)"
---

# Phase 7 (Wave 1): Regime Representation — Verification Report

**Phase goal (ROADMAP.md):** Decide what the regime labeler should see, prove that decision
walk-forward, and add a second, independent labeler on relative/leadership features so the
platform can inform allocation during ordinary markets — not only crisis avoidance.

**Scope of this verification pass:** **phase-wave 1 ONLY**, per `07-CONTEXT.md` D-09. Criteria
5, 6, 7 (classifier #2, disjointness, dependence, joint lift) and INV-01 are wave 2 and are
**correctly absent** from this pass's deliverables — their absence is not scored as a gap.
Confirmed no wave-2 code exists (`grep` for classifier-#2/relative-feature modules under
`platform/` returns nothing; `REQUIREMENTS.md` still lists REG-01/INV-01 as "Pending").

**Verified:** 2026-09-15T14:58:59Z
**Status:** `human_needed` (see the one flagged item below — everything else is `verified`)

---

## Verbatim Criteria (ROADMAP.md, "Phase 7: Regime Representation")

> 1. Driver and report label under **one documented feature policy**; a test fails if they
>    diverge. The choice and its rejected alternatives are recorded as an ADR.
>
> 2. The §5.4 sojourn/lag ratio is **interpretable** — computed between two labelings fit on
>    the same feature space, shown with its resolved-transition count. The A13 caveat is
>    removed only because the cause is fixed, never because the wording was softened.
>
> 3. Post-fix labeling disagreement is measured and reported against the 82.8% pre-fix
>    baseline. No target is set — a number, not a goal, to avoid fitting to it.
>
> 4. Classifier #1's ablation delta is re-measured on **both** axes (`wealth_delta` and
>    `dd_delta`, currently +0.379267 / −0.014364), each within its `06-VALIDATION.md` band.
>
> 5. Classifier #2 exists, is fit **unsupervised** on a feature set disjoint from classifier
>    #1's 13 (a test asserts disjointness), with occupancy summing to 1.0 and no state below
>    the §4.4 5% floor left unmarked. — **WAVE 2, correctly absent.**
>
> 6. Statistical dependence between the two labelings is measured and reported. High
>    dependence is a **failure to add an axis** and is recorded as such. — **WAVE 2, correctly
>    absent.**
>
> 7. Joint (#1 × #2) allocation lift is measured walk-forward against #1 alone, every
>    configuration logged to the trial registry, deflated-Sharpe applied for the full count. —
>    **WAVE 2, correctly absent.**
>
> 8. `platform/` still imports nothing from the legacy library — the import-guard test is
>    extended to the new modules. (Verified 2026-09-10: platform is currently fully
>    decoupled, so the relative-strength algorithms must be **ported**, not imported.) —
>    **judged only for wave-1's own touched files, per the verification-context scope note.**

> **Requirement coverage this pass**: REG-01 **partial** — the driver/reference feature policy,
> §5.4 interpretability, and the ablation re-measurement clauses. **INV-01 is entirely deferred
> to wave 2** by `07-CONTEXT.md` D-09.

**REG-01 = expected PARTIAL. INV-01 = expected DEFERRED, D-09 cited.** Both match the actual
state found below.

---

## Per-Criterion Verdicts

### Criterion 1 — one documented feature policy, test-enforced, ADR-recorded — VERIFIED

**Re-derived live**, did not merely trust the SUMMARY:

```
$ source .venv/bin/activate
$ python -c "
from trading_crab_lib.platform.evaluation.report import _reference_label_columns
from trading_crab_lib.platform.taxonomy import lean_feature_set
from trading_crab_lib.platform.config import load_platform_config
cfg = load_platform_config()
feat = pd.read_parquet('data/checkpoints/platform/monthly_features.parquet')
lean_cols = sorted(lean_feature_set(cfg) & set(feat.columns))   # 13 cols
ref_cols = _reference_label_columns(feat, lean_cols, feat.index[120])
"
→ n ref cols: 10
→ ref cols: ['cape_shiller', 'credit_spread_baa_aaa', 'curve_10y3m', 'div_yield', 'oil',
             'real_rate_level', 'realized_vol_1m', 'realized_vol_3m', 'trailing_return_1m',
             'trailing_return_3m']
→ oil present: True
```

This is the exact frozen set the ADR and CONTEXT.md claim — **re-derived independently, not
copied from the SUMMARY.** A value that would have failed this check: any count other than 10,
or `oil` absent (which is exactly what the pre-fix stale checkpoint produced — 9 columns,
`oil` excluded).

The equivalence test exists and passes against the real checkpoint (not a synthetic fixture,
not a mock):

```
$ pytest tests/unit/test_platform_backtest_driver.py -k FrozenPolicyEquivalence -v
6 passed, including test_real_checkpoint_driver_and_reference_resolve_identical_columns
```

Read the mechanism at `src/trading_crab_lib/platform/evaluation/report.py:815-823` —
`_reference_label_columns` is computed exactly once, before any `run_backtest` call, and the
resulting `ref_cols` is threaded as `frozen_l1_features` into both L1 legs and step (d)'s
full-sample fit — a shared computation, not two independent implementations a test polices from
outside.

ADR exists at `platform_design/adr/0001-l1-feature-policy.md` (382 lines) with rejected
alternatives (see Criterion 3/D-03 discussion below for the evidence-backed rejection). **A
value that would have failed this criterion:** an ADR that only argued rejection without citing
the D-03 trial's real numbers — checked and found present (see below).

**Verdict: VERIFIED** (re-derived, not existence-only).

### Criterion 2 — §5.4 ratio interpretable, one feature space, resolved-transition count shown — VERIFIED

`n_transitions`/`n_resolved` fields exist in `sojourn_lag.py:185-193` and are threaded through
to the rendered report — confirmed directly in `outputs/reports/platform/backtest_report.md`:

```
- sample: median lag over 7 resolved of 7 transitions (a transition resolves only when
  P(target) reaches the 70% action threshold).
```

The A13 caveat constant (`platform/plotting/core.py:90-103`) was read directly, not quoted from
the SUMMARY. It no longer claims uninterpretability; it names both D-08 licensing artifacts by
name (`TestFrozenPolicyEquivalence`, the resolved-transition denominator) and points at the ADR.
This is a **resolution narrative**, not softened wording — a value that would have failed this
check: the constant still containing the phrase "not interpretable" or the retired
4→6→8→9→10→12→13 progression (checked — absent; that progression now lives only in the ADR's
Context section as historical record, confirmed by reading it there).

```
$ pytest tests/unit/test_platform_plotting.py -k "A13CaveatResolution or a13_caveat" -v
5 passed
```

**Verdict: VERIFIED.**

### Criterion 3 — post-fix disagreement measured and reported against the 82.8% baseline, no target set — VERIFIED (with the window-narrowing limitation honestly foregrounded, per the human-signoff binding condition)

Pre-fix baseline reproduced from persisted artifacts (not quoted from `UAT-AUDIT`), per
`07-PREFIX-EVIDENCE.md`: `n_compared=470`, `n_disagree=389`, `pct_disagree=0.82766`.

Post-fix: `pct_disagree=0.80899` (288/356). **This is not a clean two-point improvement** — the
frozen policy's walk-forward produced 232/588 L2-degraded steps (vs. 118/588 pre-fix), leaving a
smaller, differently-dated (1974-02→2017-05, not →2020-12) comparison window. This is the
"Known-open item #1" the verification context asks to confirm is honestly recorded, **not
re-litigated**.

**Checked live in the published artifact** (`outputs/reports/platform/backtest_report.md:38`,
re-read directly, not via SUMMARY):

```
| Labeling disagreement (`pct_disagree`, `n_compared`) | 82.77% (389/470; 1974-02 -> 2020-12) | 80.90% (288/356; 1974-02 -> 2017-05) |
```

**Confirmed: both denominators and both end dates appear in the same table cell**, as required
by the 07-03 human-sign-off's binding condition ("state the differing sample size and end date
INLINE, at the point `pct_disagree` and the §5.4 ratio numbers are shown ... not deferred to a
later discussion paragraph"). This is not a footnote — it is the literal cell text. The same
inline treatment is confirmed present in ADR-0001's Consequences table (lines 176-203), with the
"Named limitation, foregrounded here (not a footnote)" heading immediately below the table.

The §5.4 ratio row gets identical treatment: `0.5910 (4 of 6; 1974-02 -> 2020-12)` vs. `0.6014
(7 of 7, resolved within 1974-02 -> 2017-05)` — "7 of 7" is correctly qualified as
within-the-narrower-window, not over the full 588-step range.

A value that would have failed this check: a table showing only `82.77% → 80.90%` with no
denominator or date qualifier, or the qualifier relegated to prose below the table instead of
inside the cell. Neither occurred.

**Verdict: VERIFIED as worded.** The human sign-off (07-03-SUMMARY.md, relayed via orchestrator)
explicitly settled that criterion 3 does not require identical-population sampling, only that
the same methodology produced both numbers — confirmed methodologically identical by
`measure_label_disagreement()` delegating to the located `label_disagreement()` (not
re-implemented) and reproducing the pre-fix baseline to `1e-12` in a regression test that reads
the pinned pre-fix artifacts via `git show <sha>`, not the live working tree (so it survives the
recompute that overwrote those artifacts).

### Criterion 4 — ablation delta re-measured on both axes, each in-band — VERIFIED

Re-derived directly from the published parquet, not copied from a document:

```
$ python -c "import pandas as pd; print(pd.read_parquet('outputs/reports/platform/backtest_kpi_table.parquet'))"
strategy             4.025085   -0.264192
no_regime_ablation   3.647238   -0.198068
```

`wealth_delta = 4.025085 - 3.647238 = +0.377847` — matches the claimed `+0.3778473581475139`.
`dd_delta = -0.264192 - (-0.198068) = -0.066124` — matches the claimed `-0.0661242048614149`.

Both bands are `[ASSUMED]` per `07-VALIDATION.md` (`abs(wealth_delta) < 5`, `abs(dd_delta) <
0.5`) — both pass. A value that would have failed: `wealth_delta` outside `±5` or `dd_delta`
outside `±0.5`; neither occurred. `dd_delta` came back **more negative** than the pre-fix
compound baseline (−0.0661 vs. −0.0144) — this is reported honestly in both `07-MEASUREMENTS.md`
§7 and the ADR's Wave-2 gate section, and correctly **not** used to reopen the policy (D-04/D-06
— see the anti-fitting check below).

**Verdict: VERIFIED** (re-derived, not existence-only).

### Criterion 8 (import-guard, scoped to wave-1's touched files) — VERIFIED for wave-1's own scope; the BROADER claim in 07-CONTEXT.md/ADR-0001 is FLAGGED, not verified

Per the verification-context's explicit scoping instruction ("+ 8's import-guard where wave 1
touched it"), I checked whether wave 1's own modified/created files introduced any new
non-platform `trading_crab_lib` imports:

```
$ git blame -L <line>,<line> for every `from trading_crab_lib import ...` line in
  driver.py, baselines.py, report.py, plotting/core.py, disagreement.py
→ report.py:67  "from trading_crab_lib import OUTPUT_DIR"  — Claude, 2026-09-09 (pre-Phase-7)
→ core.py:30    "from trading_crab_lib import OUTPUT_DIR"  — Claude, 2026-09-09 (pre-Phase-7)
```

Neither predates nor postdates wave 1 in a way that implicates it: both `OUTPUT_DIR` imports
were already present before phase 7 started. **No new legacy import was introduced by wave 1.**
`disagreement.py` (new file, this phase) imports only `trading_crab_lib.platform.plotting.regime`
— fully in-platform.

**However**, re-deriving the *blanket* claim repeated in `07-CONTEXT.md` ("Re-verified this
session: grep over all 59 platform/*.py modules returns zero legacy imports") and ADR-0001
("`platform/` still imports nothing from the legacy library... criterion 8 is currently TRUE")
turned up a direct contradiction with a **same-week, later-committed** document in this same
repo, `MIGRATION-PLAN.md` (rewritten 2026-09-14, commit `1ed9bc2`), which states as its own
"Coupling Surface" section:

```
platform/  →  legacy lib :  4 narrow seams   (M1 checkpoints.CheckpointManager,
                                                M2/M3 ingestion scraper helpers,
                                                M4 email helpers)
```

and lists P0 ("Ends with `platform/` importing nothing from the legacy library") as a **not-yet-
executed** future migration step, with its own exit-criterion test
(`tests/unit/test_platform_standalone.py`) **not present in the repo** (`ls` confirms missing).

**I re-derived this directly** via AST-based import scan (not the naive/broken grep pattern
`grep -r "from trading_crab_lib\." ... | grep -v platform`, which is itself broken — every line
already contains the literal substring "platform" from the file path, so that exact command,
as written in `MIGRATION-PLAN.md`, silently returns nothing regardless of what is imported):

```
31 hits of `from trading_crab_lib.X import ...` / `import trading_crab_lib` where X is NOT
`platform.*`, across: config.py, snapshots.py, checkpoints.py, report.py, model_metrics.py,
assets/returns.py, assets/vol.py, report/weekly.py, report/holdings.py,
allocation/hysteresis.py, honesty/holdout.py, honesty/registry.py, honesty/gap_lag.py,
plotting/core.py, plotting/loaders.py, ingestion/tiingo.py, ingestion/prices_daily.py,
ingestion/macro_monthly.py, ingestion/macro_daily.py, labeling/diagnostics.py.
```

Most of these are the bare `OUTPUT_DIR`/`ROOT` re-export style import (arguably a package-level
constant, not "legacy pipeline logic"), but several are the genuine, MIGRATION-PLAN-documented
seams: `trading_crab_lib.checkpoints.CheckpointManager` (M1), `trading_crab_lib.email.*` (M4),
and `trading_crab_lib.ingestion.{http,browser,assets}` (M2/M3-adjacent). The only existing
per-module import-guard tests I could find (`test_platform_plotting*.py::TestFreshPackageBoundary`)
check a **narrower** claim — that `platform/plotting` doesn't transitively reach the *legacy
plotting* module specifically — not a blanket "no legacy imports anywhere in platform/" guard.
There is no test enforcing the blanket claim ADR-0001/07-CONTEXT.md make.

**This is not a wave-1 regression** — every seam found predates Phase 7 (confirmed by
`git blame`), and wave-1's own new/touched code does not add to it. But the blanket claim
("criterion 8 is currently TRUE," "zero legacy imports") that ADR-0001 and 07-CONTEXT.md assert
and rely on for wave-2 planning ("relative-strength algorithms must be ported, not imported")
is **not supported by direct re-derivation** — it conflicts with the project's own
`MIGRATION-PLAN.md`, written the same week. This is exactly the kind of unre-derived claim this
verification is supposed to catch rather than wave through.

**Verdict:** wave-1's own scope — **VERIFIED** (no regression). The blanket claim repeated in
the phase's artifacts — **FLAGGED, human decision requested** (not a BLOCKER for wave 1's actual
deliverables, since none of criteria 1-4 depend on it, but material to wave-2 planning, which
explicitly relies on "platform is fully decoupled" to justify porting rather than importing).

---

## Live Re-Derivations (independent of SUMMARY claims), per the verification-context checklist

| Check | Claimed | Re-derived live | Match |
|---|---|---|---|
| Frozen set is 10 cols, `oil` present | 10, `oil` present | `_reference_label_columns()` called directly against the live checkpoint → 10 cols, `oil` present | ✅ |
| Dev checkpoint shape | (708, 53) | `pd.read_parquet(...).shape` → `(708, 53)` | ✅ |
| `oil` non-NaN, first valid | 708 months, 1962-01-31 | `feat['oil'].notna().sum()` → 708; `.first_valid_index()` → 1962-01-31 | ✅ |
| Dev index end | 2020-12-31 | `feat.index.max()` → 2020-12-31 | ✅ |
| Holdout carve | 68 rows, all > 2020-12-31 | `pd.read_parquet('data/holdout/monthly_features.parquet').shape` → `(68, 53)`; `(hold.index > '2020-12-31').all()` → `True`; min 2021-01-31 | ✅ |
| Full suite | 1741 passed, 0 skipped | `pytest tests/ -q` → **1741 passed, 5 warnings, 0 skipped** in 69.98s | ✅ exact match |
| A13 pinned test — exact list equality | `assert observed == EXPECTED_CHANGE_POINTS` | Read the source directly (`test_platform_plotting_regime.py:249`) — exact list-equality assertion, dated D-02-A comment naming root cause, explicit instruction not to weaken it | ✅ |
| D-03 rejected trial's real numbers | wealth 4.048185, DD −17.91%, occupancy 1.29% | `pd.read_parquet(.../trials/impute-13col/backtest_kpi_table.parquet)` → `4.048185`, `-0.179079` (=−17.91%); state-0 occupancy from `backtest_full_sample_states.parquet` → `0.012950` (1.295%) | ✅ exact match, all three figures |
| Registry contamination fix | archived 42 rows, reset with `prior_genuine_trials=38`, `append_trial` refuses untagged, `NO_REGISTRY` exists, CLI requires `--trial-tag`/`--smoke` | `registry/archive/trials-pre-P7W1-reset.jsonl` has 42 rows; `registry/trials.jsonl` has 1 provenance-header row stating `prior_genuine_trials=38`, `discarded_smoke_rows=4`; `append_trial()` raises `ValueError` on missing/blank `trial_tag` unless `path=NO_REGISTRY`; `report.py`'s CLI uses `add_mutually_exclusive_group(required=True)` for `--trial-tag`/`--smoke` | ✅ exact match |
| Window-narrowing inline (binding condition) | both denominators + dates in the same cell, in `backtest_report.md` and ADR-0001 | confirmed in both, verbatim, quoted above | ✅ |
| D-08 caveat retirement — both artifacts named, not softened wording | — | read `A13_CAVEAT` constant directly — names `TestFrozenPolicyEquivalence` and the resolved-transition denominator; no uninterpretability claim remains | ✅ |
| D-04 anti-fitting — no task branched on a measured outcome | — | single commit (`5edef3e`) ran both variants together and recorded measurements; no subsequent commit reruns or revises the policy after seeing `dd_delta`; `dd_delta` came back worse than baseline and was reported, not acted on | ✅ |
| ADR-0001's rejected-13-col trial cites unflattering-to-accepted-variant numbers honestly | wealth 4.048185 > 4.025085 (accepted); dd_delta +0.019 > −0.066 (accepted) | present verbatim in ADR-0001 Considered Options #2, explicitly flagged "looks nominally better ... rejection does not rest on these numbers" | ✅ |
| Criterion 8 import-guard (wave-1 scope) | no new legacy imports in wave-1-touched files | git-blamed every `from trading_crab_lib import ...` line in the 5 touched/created files — both hits predate Phase 7 | ✅ scoped |
| Criterion 8 blanket claim ("platform imports nothing from legacy") | TRUE, per 07-CONTEXT.md/ADR-0001 | AST scan found 31 non-platform `trading_crab_lib` import sites across `platform/`, several matching MIGRATION-PLAN.md's own documented, un-vendored seams (checkpoints, email, ingestion) | ❌ blanket claim not supported — flagged above |

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `07-01-SUMMARY.md` … `07-04-SUMMARY.md` | plan-completion records | ✅ present | All 4 files present, `ls -la` confirmed (34794/19119/33738/17832/36846/17675/43098/22698 bytes) |
| `07-MEASUREMENTS.md` | measured numbers, D-05 pre/post table | ✅ present, substantive | 242 lines, cross-checked against live artifacts above |
| `07-PREFIX-EVIDENCE.md` | reproduced pre-fix baseline | ✅ present, substantive | 154 lines, reproduction recipe verifiable |
| `07-CONTEXT.md` | D-01…D-17, D-02-A | ✅ present | All decisions read; D-09 (wave-1-only scoping) followed correctly |
| `07-VALIDATION.md` | bands, 4 `[ASSUMED]`/unlocked | ✅ present | Confirmed `[ASSUMED]` markers on `pct_disagree` suspicious-threshold, §5.4 ratio target, `n_transitions` implausibility, `wealth_delta`/`dd_delta` bands — none treated as a gate anywhere in the measured record (D-07 honored) |
| `platform_design/adr/0001-l1-feature-policy.md` | the ADR | ✅ present, substantive | 382 lines; `ls platform_design/adr/` confirms the file exists (a prior checker's "not found" claim on this exact file, per the task brief, is directly contradicted by `ls`) |
| `.planning/UAT-AUDIT-2026-09-09.md` | evidence-shape standard | ✅ present, read | Applied its standard throughout this report — re-derivation over existence, per-value failure conditions stated |

**Note on the prior checker's false-absence claim:** both `07-MEASUREMENTS.md` and
`platform_design/adr/0001-l1-feature-policy.md` were confirmed present via direct `ls` at the
start of this verification and were read in full. No file named in this phase's scope was
found missing.

---

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| REG-01 | **Partial** (as declared) | Criteria 1-4 verified above; criteria 5-7's clauses explicitly deferred to wave 2 by D-09, restated in ADR-0001's "Deferrals and open items" |
| INV-01 | **Deferred to wave 2** (as declared) | `07-CONTEXT.md` D-09, D-12; ADR-0001 restates the deferral; no M2/credit ingestion code exists yet (confirmed: no new ingestion module for M2/TOTALSL found) |

`.planning/REQUIREMENTS.md` still lists both as "Pending" (line 164-165) — consistent with wave
1 being a partial pass, not a completion.

---

## Anti-Patterns Scan

Scanned wave-1's touched/created files (`driver.py`, `baselines.py`, `report.py`,
`plotting/core.py`, `disagreement.py`, `scripts/run_policy_trials.py`,
`scripts/recompute_monthly_features.py`) for `TBD`/`FIXME`/`XXX`/`TODO`/`HACK`/`PLACEHOLDER`:

```
$ grep -n "TBD\|FIXME\|XXX\|TODO\|HACK\|PLACEHOLDER" <those files>
(no matches)
```

No debt markers found. No stub patterns (`return null`, hardcoded empty returns feeding
rendering) found in the reviewed diffs — all values traced to real computation against real
checkpoints.

---

## Behavioral Spot-Checks / Probe Execution

No dedicated `scripts/*/tests/probe-*.sh` convention exists in this project; this phase's
"probe" equivalent is the two real `run_full_backtest_evaluation()` calls
(`scripts/run_policy_trials.py --variant frozen|impute`), which were **re-verified by direct
artifact inspection** above (KPI table values, occupancy fractions) rather than re-run (each
takes ~2 minutes; the persisted parquet outputs are the trustworthy record and match every
claimed number byte-for-byte). This satisfies the "re-derive or re-run something" bar without
an unnecessary 4-minute re-execution, since the artifacts are content-addressable checkpoints
that were themselves independently re-read here, not merely cited.

The one genuine re-run performed live in this verification: the full pytest suite (`pytest
tests/ -q`, once, per the "at most once" constraint), and the single named A13-pin test in
isolation before the full run.

---

## Gaps Summary

**No BLOCKER-level gap found.** All four wave-1 success criteria (1-4) are VERIFIED against
live re-derivation, not existence-only inspection. The window-narrowing limitation (criterion
3) and the D-03 rejected-alternative's nominally-better numbers (criterion 1's ADR) are both
honestly recorded, inline, exactly as the human sign-off's binding condition required — checked
directly in the rendered artifacts, not assumed from the SUMMARY's own description of itself.

**One WARNING-level item, routed to human verification:** ADR-0001 and `07-CONTEXT.md` assert a
blanket "platform imports nothing from the legacy library" claim that does not survive direct
re-derivation — `MIGRATION-PLAN.md`, committed the same week, documents 4 concrete unvendored
seams, and my own AST scan confirms real (pre-existing, non-wave-1) legacy imports exist in
`platform/`. This does not block wave 1's own deliverables (none of criteria 1-4 depend on it),
but it is material to wave 2's planning premise ("relative-strength algorithms must be ported,
not imported" — a claim that assumes full decoupling that presently does not hold) and should be
either corrected in the ADR's wording or resolved (MIGRATION-PLAN's own P0 step) before wave 2
relies on it.

---

*Verified: 2026-09-15T14:58:59Z*
*Verifier: Claude (gsd-verifier)*
