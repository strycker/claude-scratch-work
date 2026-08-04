# Status Review & Forward Plan — 2026-08-04

A full survey of the repo after the Phase 5 merge: what's done, what's genuinely left,
what can be simplified, and a prioritized path forward.

---

## 1. Where We Actually Are

**Milestone v1.0 is 5/6 phases complete (28/28 plans).** Every phase carries a
`status: passed` VERIFICATION.md. The full suite is **1157 passing**.

| Phase | Plans | Verification | Notes |
|---|---|---|---|
| 1. Monthly Data Layer | 7/7 | passed | One open *demonstration* item (below) |
| 2. Honesty Infrastructure | 5/5 | passed | Holdout, registry, purged CV, causal gating |
| 3. Regime Labeling & Prediction | 4/4 | passed | Jump model + calibrated nowcaster |
| 4. Asset Prediction & Allocation | 5/5 | passed | Returns-by-regime, EWMA vol, tilt, tripwire |
| 5. Honest Backtest & Evaluation | 7/7 | passed | Human-verified real 1972–2020 run |
| 6. Migration to Public Repo | 0/TBD | — | **Not started** |

### Nothing is silently in-progress

I checked every phase's VERIFICATION.md front-matter. All five say `status: passed`.
There is exactly **one** outstanding human item, and it is already effectively closed:

> **Phase 1 human-verification:** "the end-to-end `build_monthly_spine()` live run
> demonstrating a real ~1962-start monthly_features checkpoint has not yet been executed
> *here*." — You have since run this locally (it's what produced the Phase 5 real-run
> numbers). It needs a one-line status update in `01-VERIFICATION.md`, not new work.

One informational note carried in STATE.md: `holdout.py` / `registry.py` hardcode
constants that duplicate `config/platform_settings.yaml` rather than reading it. Pure
future-divergence risk; no current bug.

---

## 2. The Three Findings That Should Drive What We Do Next

### Finding A — `MIGRATION-PLAN.md` migrates the *wrong codebase*

This is the most consequential thing in the review.

`MIGRATION-PLAN.md` (last touched 2026-04-21) lays out Q0–Q9 to migrate the **legacy
quarterly 9-step pipeline** — ingest → features → clustering → regime → predict → assets
→ diagnostics → tactics → CLI. It predates the platform entirely.

But `.planning/ROADMAP.md` Phase 6 says the thing to migrate is "**the new L0–L4
modules**" — the regime-conditional platform.

These are two different migrations of two different codebases. Following MIGRATION-PLAN.md
as written would port 10,935 lines of superseded quarterly code and none of the 6,977
lines that the last five phases actually built and verified.

**MIGRATION-PLAN.md must be rewritten against the platform before Phase 6 starts.**

### Finding B — The platform is *nearly* self-contained, so migration is far cheaper than the plan implies

I traced the dependency direction:

```
platform/ → incumbent lib:   3 narrow reuse points
incumbent lib → platform/:   ZERO
```

The only things `platform/` borrows from the legacy library:

| Reuse point | Used by | Size |
|---|---|---|
| `checkpoints.CheckpointManager` | 5 modules | one class |
| `ingestion.multpl._scrape_raw_rows`, `_SUFFIX_MAP`, `RATE_LIMIT_SECONDS` | `macro_monthly.py` | ~3 helpers |
| `ingestion.macrotrends._extract_json_data`, `HEADERS` | `macro_monthly.py` | ~2 helpers |
| `email.{load_email_config, build_weekly_email_body, send_weekly_email}` | `report/weekly.py` | 3 functions |

That's the entire coupling surface. **`platform/` + those four seams is a shippable
package.** Migration is "lift `platform/`, vendor four small things," not a ten-phase
re-derivation.

### Finding C — The "too many parameters" concern is already solved *in the platform*, and unsolved *in the legacy*

| | Feature count | Reduction |
|---|---|---|
| **Platform L1/L2 lean set** | **13 features** | none needed — curated by hand |
| Legacy `clustering_features` | 101 features | → PCA(5) |

The platform's regime model runs on 13 named, economically-meaningful features
(`cape_shiller`, `credit_spread_baa_aaa`, `curve_10y2y`, `curve_10y3m`, `div_yield`,
`fred_vix`, `gold`, `oil`, `real_rate_level`, `realized_vol_1m`, `realized_vol_3m`,
`trailing_return_1m`, `trailing_return_3m`) with K=5, λ=52.

So the parameter bloat you're worried about lives in the **legacy quarterly pipeline**,
which is exactly the code we should stop carrying — not in the platform we're migrating.

---

## 3. On Dimensionality Reduction & "Invariants" — a real tension to resolve

Your instinct (find conserved quantities that carry more signal than any single ticker)
is well-motivated and I want it in the plan. But it collides head-on with a **locked
design decision** that should be re-opened deliberately rather than by accident.

`platform_design.md` §11 row **R4**:

> | R4 | PCA(5) before clustering | **Marginal**: PCA on mixed-unit features obscures
> interpretability and the semantic skeleton | **Prefer standardized curated features,
> no PCA** (or PCA only as diagnostic); skeleton constraints need named dimensions |

The design rejected PCA for L1 because the jump model's regime *skeleton constraints*
(§4.4 acceptance criteria) are stated in terms of named economic dimensions. Anonymous
principal components break the ability to say "regime 3 is the one with inverted curve
and widening credit."

**These are reconcilable, and the distinction matters:**

- ❌ What R4 rejects: replacing 13 named features with 5 anonymous PCs in the L1 labeler.
- ✅ What you're actually describing: *discovering* interpretable low-dimensional
  invariants — M2/GDP, total market cap/GDP, real rate level — that are themselves
  economically nameable, then admitting them as **named features** or as a **fair-value
  gap** signal.

That second framing doesn't violate R4 at all. It's closer to the design's own §16.1
factor-projection layer and §15's cointegration/VECM error-correction terms. It's also
the physics framing you used: a conserved quantity is *interesting because you can name
it*, not because it's the top eigenvector.

**Recommendation:** run this as a **research track producing named candidate features**,
with PCA/SVD/Lasso as the *discovery tool* and the deliverable being 2–5 new named,
interpretable series — never an anonymous PC block wired into L1. Gate admission on the
existing honesty rails (trial registry + walk-forward), so a candidate has to earn its
place. If the research shows anonymous components genuinely beat named ones, *that's* the
moment to formally revisit R4 with evidence.

---

## 4. Simplification & Cleanup Opportunities

### 4.1 The big one: retire the legacy quarterly pipeline

Currently carried, and entirely superseded by `platform/`:

| Asset | Size |
|---|---|
| Legacy library modules (`src/trading_crab_lib/*.py`, excl. `platform/`) | 10,935 LOC |
| Legacy test files | 35 files |
| Legacy pipeline scripts (`pipelines/01–09`) | 9 scripts |
| Legacy notebooks (`notebooks/01–12`) | 6.0 MB |
| `run_pipeline.py`, `src/trading_crab/pipeline.py` (9-step orchestrator) | 1,527 LOC |

The platform reuses **four small seams** from all of that (Finding B). CLAUDE.md calls
the quarterly pipeline "the incumbent baseline, frozen" — but Phase 5's baseline gauntlet
is SPY / 60-40 / Faber, not the quarterly pipeline. **It isn't actually the baseline for
anything we measure.**

I would not delete it here. I'd **stop migrating it** — let it stay frozen in
`claude-scratch-work` as reference (same status as `legacy/`), and migrate only the
platform. That's simplification-by-not-carrying-forward, which costs nothing and risks
nothing.

### 4.2 Root markdown: 16 files, 6 already self-marked SUPERSEDED

| File | State |
|---|---|
| `META_PLAN.md`, `NEXT_STEPS.md`, `PHASE_B/C/D/E_PLAN.md` | ⚠️ self-marked **SUPERSEDED** |
| `MONITORING_EXPANSION_PLAN.md`, `RENAME_PLAN.md` | Fully executed (D30–D39, D15) |
| `REBUILD-FROM-SCRATCH-GUIDE.md` (942 lines) | Describes rebuilding the *legacy* pipeline |
| `MIGRATION-PLAN.md` | **Wrong target** (Finding A) |
| `STATE.md` (root) | Last real update 2026-04-02; `.planning/STATE.md` is the live one |
| `ROADMAP.md` (root) vs `.planning/ROADMAP.md` | Two roadmaps, divergent |

**Proposal:** `docs/archive/` for the 8 superseded/executed ones, leaving `README.md`,
`CLAUDE.md`, `ROADMAP.md`, `MIGRATION-PLAN.md` (rewritten), `LESSONS_LEARNED.md`,
`DISTRIBUTION.md`. Removes a real source of "which doc is true?" confusion.

### 4.3 CLAUDE.md is 1,983 lines and mostly describes the legacy pipeline

It documents the quarterly pipeline in exhaustive detail (ADRs #1–#12, pitfalls P1–P27,
decisions D1–D50) and barely mentions `platform/`. For a migration target it's actively
misleading. Needs a platform-first rewrite — the legacy detail moves to the archive.

### 4.4 Dead/duplicated code worth a look

- `prediction/__init__.py` (flat API) vs `prediction/classifier.py` (bundle API) — ADR #12
  keeps both alive purely so two test files can inspect per-fold CV. If the legacy
  pipeline isn't migrated, **neither migrates**, and the duplication evaporates.
- Three empty submodules (`gsd-scratch-work`, `trading-crab-lib`, `trading-crab`) are
  gitlinks not checked out; CLAUDE.md calls two "frozen archives." Candidates for removal.

---

## 5. The Notebook Gap — and why it's the right thing to do first

You asked whether notebooks should come first. **Yes, and the reason is stronger than
convenience.**

```
notebooks/01–12  →  ALL legacy quarterly pipeline
platform/ L0–L4  →  ZERO notebooks
```

Every one of the 12 notebooks drives the quarterly pipeline. The platform — the thing
that passed five phases of verification, the thing we're migrating — has **no EDA
surface and no human-in-the-loop validation surface at all.**

That matters for three compounding reasons:

1. **You can't validate the migration without them.** The migration checklist in
   MIGRATION-PLAN.md is literally "run the notebook, verify plots." For the platform
   there's nothing to run.
2. **The Phase 5 numbers are currently unexaminable.** Log wealth 111.06, max DD −66.2%,
   Brier 0.20, ratio 0.52 over 2-of-6 resolved transitions — all real, all verified, and
   all only inspectable as a markdown report. There's no way to *look at* the regime
   timeline, the calibration, or the allocation path.
3. **The invariants research needs an EDA surface to even begin.** You can't hunt for
   conserved quantities without a place to plot candidates against regimes.

Notebooks first is not a detour — it's the prerequisite for both the migration and the
research track.

---

## 6. Recommended Path — four tranches

Ordered by dependency, not ambition.

### ▸ Tranche 1 — Platform Notebook Suite *(do first)*

Six notebooks mirroring L0–L4 + evaluation, each with EDA and an explicit human sign-off
cell. Built on the existing `plotting/` conventions (ADR #11: logic lives in library
functions, notebooks call them).

| Notebook | Covers | Human validates |
|---|---|---|
| `P1_data_spine.ipynb` | L0: splices, ALFRED vintages, coverage | Do 1962+ histories look right? Splice joins clean? |
| `P2_features_taxonomy.ipynb` | Lean 13, fast/slow/agency, causal-vs-centered | Are features economically sane? Any look-ahead? |
| `P3_regime_labeling.ipynb` | L1: jump model, sojourns, churn, (K,λ) | **Do the 5 regimes correspond to real history?** |
| `P4_nowcaster.ipynb` | L2: calibration, transition-window acc, detection lag | Is it calibrated? Does it beat persistence? |
| `P5_assets_allocation.ipynb` | L3/L4: returns-by-regime, vol, tilt, hysteresis | Are the tilts defensible? |
| `P6_backtest_evaluation.ipynb` | Phase 5: equity curves, gauntlet, ablation, KPIs | **Does the strategy earn its complexity?** |

P3 and P6 are the load-bearing ones — they're where "the regime layer doesn't pay rent
yet" either gets confirmed or diagnosed.

**Why first:** unblocks Tranches 2 and 4; makes five phases of verified-but-unseen work
actually visible.

### ▸ Tranche 2 — Migration Readiness

1. **Rewrite `MIGRATION-PLAN.md`** against the platform (Finding A). New P0–P6 steps
   mirroring L0–L4 + honesty + evaluation, each with its Tranche-1 notebook as the
   validation gate.
2. **Cut the four seams** (Finding B): vendor `CheckpointManager`, the multpl/macrotrends
   scraper helpers, and the email functions into `platform/` so it stands alone. Small,
   mechanical, well-tested.
3. **Platform-first `CLAUDE.md`**; archive superseded docs (4.2).
4. Close the Phase 1 demonstration note; wire `holdout.py`/`registry.py` to config.

**Exit:** `platform/` imports nothing from the legacy library, and every module has a
notebook that renders.

### ▸ Tranche 3 — Execute Phase 6 (Migration)

Lift the now-standalone platform into `strycker/trading-crab`, notebooks included, CI
green. This is the roadmap's actual Phase 6 and it becomes nearly mechanical after
Tranche 2.

### ▸ Tranche 4 — Invariants & Dimensional Reduction *(the research track)*

Runs in the new repo, or here in parallel after Tranche 1. Framed per §3 above:

- **4a — Candidate construction.** Build the named ratio/invariant candidates: M2/GDP,
  total market cap/GDP (Buffett indicator), credit/GDP, real M2 growth, energy-cost share.
  Each economically nameable by construction.
- **4b — Discovery.** PCA/SVD/sparse-PCA/Lasso *as tools* over the wide feature space to
  find which combinations are stable across eras — "history rhymes" made testable by
  checking loading stability across 1962–1985 / 1985–2005 / 2005–2020.
- **4c — Admission.** Any survivor competes on walk-forward through the trial registry.
  Admitted as a **named feature**, never an anonymous PC block. R4 stays intact unless
  evidence says otherwise.
- **4d — Interpretation.** The deliverable is as much understanding as accuracy: which
  conserved quantities actually govern regime transitions?

**Why last:** it's the only tranche that is genuinely research (uncertain payoff), and it
needs Tranche 1's EDA surface to be tractable.

---

## 7. Suggested Immediate Next Step

**Start Tranche 1, and start with `P3_regime_labeling.ipynb`.**

Not P1. P3 is where the open question lives — Phase 5 concluded "the regime layer does
not pay rent yet," and the sojourn/lag headline resolves on only 2 of 6 transitions. If
the 5 regimes don't map onto recognizable economic history when you look at them, that
finding reshapes Tranches 2–4. Better to learn it before migrating.

P1 and P2 are then quick follow-ons (mostly coverage/QA plots), and P4–P6 complete the set.
