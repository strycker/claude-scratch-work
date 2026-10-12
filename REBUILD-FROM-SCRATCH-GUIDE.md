# REBUILD-FROM-SCRATCH-GUIDE.md: the regime-conditional platform, planned without regret

**Written:** 2026-09-29, after Phases 1–8 of the platform (plus 8's gap closure).
**Updated:** 2026-10-12 (08.5): learnings from 8.1–8.5, the Phase 9.1–9.8 order, sizes re-measured.
**For:** rebuilding the platform in `strycker/trading-crab` from zero, or simplifying it in place.
The two are the same exercise.
**Not this:** `docs/archive/REBUILD-FROM-SCRATCH-GUIDE.md` rebuilds the *legacy quarterly
pipeline* (March 2026). That pipeline is superseded.
**Companion:** `platform_design/DECISIONS.md`, one row per decision, human-editable. Read and
edit it before planning anything. This guide says *in what order* to build; the register says
*what was decided*.

This is not a history. It is the path we would take knowing what we know now.

---

## 1. What we know now that we wish we had known at Phase 1

**1. The regime layer has never paid rent.** At every measurement, the regime-tilted strategy
failed to beat its own no-regime ablation:

- Phase 5: ablation delta −2.83 log-wealth.
- Phase 7 criterion 7: joint lift −0.123 nats.
- Phase 8 re-measure: −0.125.
- The DSR gate fails on both legs.

Everything built around the regime signal is scaffolding around a signal we have not yet shown
exists: persistence filters, hysteresis, stability metrics, a second classifier, the joint
driver. **Lesson:** the one number that matters, *regime-tilt vs no-regime ablation, net of
costs, walk-forward*, should be the first thing built and the gate on every later module.
Don't build machinery to stabilise a signal before the signal beats its baseline.

**2. Planning grew until it became coding.** Planning-document lines per phase, against source
lines changed:

| Phase | Planning lines | Note |
|---|---|---|
| 1–5 | 2.9k–4.2k each | roughly 1:1 with code |
| 6 (notebooks) | 8.3k | |
| 7 | 16.2k | |
| 8 | 17.8k | vs **3.2k** source lines changed, about 5.5:1 |

Most of Phase 7–8's planning went into ruling records, measurement amendments,
trial-budget arithmetic and confirm-only-check tallies. **Lesson:** cap planning per module
(§5). Rigor belongs in the few numbers that drive decisions, not in every number.

**3. Point-in-time alignment must be a data-layer invariant on day 1.** A Phase 8 code review
found that `fred_m2sl` (about 1 month of publication lag), `fred_totalsl` (about 2) and multpl's
`div_yield` (2–3) enter features at their reference month. `div_yield` has sat in classifier
#1's feature set since Phase 1. One table and one test would have prevented it:

- a `publication_lag_months` entry per raw series;
- a test that fails if any feature at date t uses a value published after t.

**4. The live-scoring path must exist in the tracer bullet, end to end, on real data.** The
weekly report loaded a nowcaster that no supported command produced. Every weekly test used a
fake model, so nobody noticed until a UAT run by hand in Phase 8. **Lesson:** "real data →
report" is a smoke test that runs from the first milestone.

**5. Every model output needs a variability check.** At full history the served nowcaster
returns **one identical posterior** for all 231 scorable months. The feature-admission rule let
late-starting features cut training to 153 rows (2007–2019, 3 of 6 regimes), and sigmoid
calibration flattened what remained. **Lesson:** after any fit, count the distinct outputs.
A constant model is a bug until proven otherwise.

**6. State ids are not stable across refits.** Walk-forward refits renumber regimes. Any
temporal machinery that carries state across steps (filters, hysteresis, churn counts) is
comparing unrelated ids unless the labeler aligns its states to the previous fit's centroids
(Hungarian matching). **Lesson:** alignment is part of the labeler's API, not an afterthought.

**7. Compare floats with a relative tolerance from the start.** macOS (Accelerate) and Linux
(OpenBLAS) differ by 1–32 ULP on identical inputs. Two UAT failures came from bit-exact
assertions. Use rel 1e-9 and abs 0 everywhere.

**8. Build the new package clean, where it will live.** Nesting the platform inside the legacy
library left 31 legacy-import sites to unpick before migration. Start in the target repo, or
a clean sibling package, with no imports from the legacy code.

**9. Notebooks are the human gate, so build them with each module.** The P1–P6 notebooks were
written after the fact in Phase 6. Phases 7–8 added the classifier #2, the Bayes filter, the
no-trade band and serving, and no notebook covers any of them yet.

**10. Build honesty rails early, but make them cheap.** The holdout carve, walk-forward runner
and append-only registry are cheap and worth having from day 1. Deflated-Sharpe gating, trial
ceilings, per-number ruling records and mutation proofs for every test are worth their cost
**only once a candidate beats its baseline and you are choosing between candidates**. Before
that point they are overhead.

### Learnings from 8.1–8.5

Added in 08.5 (2026-10-12). One line each; the row in `platform_design/DECISIONS.md` has the detail.

- **Month-end P&L:** P&L uses month-end prices; monthly averages are model inputs only (E-07, E-08).
- **Fail-loud build:** a missing source stops the build before any write, unless explicitly allowed (D-08).
- **Gold:** the World Bank monthly series is the research source (D-09).
- **Point-in-time lags belong in M1:** one measured lag per raw series, applied once at ingestion (08.1, D-05).
- **Live state is untracked:** the operator's book and serving state stay machine-local (G-11).
- **One small state tool:** `platform/state.py` archives, resets and restores (G-12).
- **Freeze by tag, not by a promoted archive:** the pre-reset tree is tag `model/pre-reset-2026-10` (G-14).
- **Nothing generated in git:** `data/`, `outputs/` and notebook outputs are ignored or stripped (G-14).
- **The `realdata` marker:** tests that need a build skip without one and run on the Mac after a build (G-14).
- **Registry true-zero epoch:** 48 trials archived; early hurdles understate the burden, the 2021+ holdout guards (H-12, ADR-0005).
- **Page identity compares run 2 with run 3:** run 1 has no previous book (D-06 as amended 2026-10-12).

---

## 2. What the finished MVP must do (the usable product)

Every Friday, one command produces a report Glenn can act on:

1. **Data is current:** the report states its as-of date and which series lag, and nothing is
   used before it was published.
2. **Baseline allocation:** a vol-targeted core mix (SPY / TLT / IAU / USO / cash) that works
   **without** any regime model. It has to be usable on its own.
3. **Regime view (advisory until it earns weight):** the current regime, its profile in one
   line, and how confident the model is.
4. **Trades implied:** target vs current per account, after a no-trade band.
5. **Crash tripwire:** three signals, red or green.
6. **Honest scoreboard:** regime-tilt vs no-regime vs SPY vs 60/40, walk-forward 1972–2020,
   net of costs, all in one table.

The regime tilt gets weight in item 4 **only** once item 6 shows it beats the ablation.

---

## 3. Module map: each module is code plus a notebook plus a gate

A **module** is the unit of planning, building, migrating and human review. It has one small
interface, one notebook that shows its data and outputs to a human, and one gate.

| # | Module | Interface (in → out) | Notebook | Gate (human-checkable) | Current code |
|---|---|---|---|---|---|
| M0 | **Skeleton** | config, checkpoint manager, CLI, CI | — | `tradingcrab --help` runs; CI green | `platform/config.py`, `checkpoints.py` |
| M1 | **Data spine** | raw sources → `monthly_raw` (with publication lags applied) | N1 data | coverage plot; **point-in-time test passes**; splices continuous | `ingestion/`, `splice.py`, `transforms_monthly.py` (P1) |
| M2 | **Features** | `monthly_raw` → `monthly_features` (causal only) | N2 features | taxonomy table; no centred smoothing in causal features | `taxonomy.py` (P2) |
| M3 | **Baseline allocator + backtest** | returns → vol-targeted weights → walk-forward curve vs SPY / 60-40 / Faber | N3 baseline | scoreboard table renders; costs applied | `assets/`, `allocation/tilt.py`, `backtest/{driver,baselines,costs}.py` (P5–P6) |
| M4 | **Weekly report (serving)** | latest data + weights + holdings → markdown | N4 report | run on real data twice → identical report; as-of date shown | `report/{weekly,serving,holdings}.py` |
| **MVP-1** | *Usable product, no regime model* | | | Glenn trades from it | |
| M5 | **Regime labeler** | features → states (+ profiles), **aligned across refits** | N5 regimes | timeline vs recessions; occupancy; profiles make sense | `labeling/jump_model.py`, `diagnostics.py` (P3) |
| M6 | **Nowcaster + filter** | causal features → P(regime now) → filtered belief | N6 nowcast | **distinct-output count > 1**; truncation (no look-ahead) test | `prediction/` (P4, partly) |
| M7 | **Regime tilt + ablation** | belief → tilted weights; scoreboard adds tilt vs ablation | N7 does-regime-pay | tilt beats ablation net of costs, or it stays advisory | `allocation/`, `backtest/driver.py` |
| **MVP-2** | *Regime-aware product* | | | | |
| M8 | **Honesty upgrade** | registry budgets, DSR, holdout evaluation | N8 honesty | DSR computed against the registry | `honesty/`, `evaluation/deflated_sharpe.py` |
| M9+ | Tripwire, stability, second classifier, mixture of experts, tactics | as designed | per module | per module | `tripwire/`, `evaluation/*` (the parked stability, classifier #2 and joint-driver code was deleted in G-13) |

Current file mapping: `platform_design/MODULE-MAP.md` (08.2-02).

**Phases:** M0–M7 are ROADMAP Phases 9.1–9.8 (M0 = 9.1 … M7 = 9.8), one module each. M8 has no phase of its
own: its code moves with M1 (9.2) and M3 (9.4). M9+ is backlog until MVP-2 (DECISIONS G-15).

**Notebook coverage today:** P1–P9 cover M1–M7 (P7 = N4 serving, P8 = N6 filtered belief, P9 = N7 "does the
regime layer pay rent?", added in 08.2). Since 08.5 they are committed without outputs; run them after a build.
P6, P8 and P9 also need the first budgeted run (9.4), because they read its outputs.
**Gaps:** M0 and M8 have no notebook.

---

## 4. Build order: usable first, regime second, rigor when it pays

Steps 1–8 are ROADMAP Phases 9.1–9.8 (G-15). Every phase starts from a fresh clone with nothing generated
in git: the build makes the data, and nothing it makes is committed (G-14).

1. **M0 skeleton, in the target repo (9.1).** Two-package layout, config, checkpoints, the state
   tool, CI. Import nothing from the legacy library.
2. **M1 data spine, with publication lags applied from the start (9.2).**
   - A `publication_lag_months` entry for every series in config, plus the point-in-time test.
   - A build that fails loud on a missing source (D-08).
   - Physical 2021+ holdout carve and the walk-forward runner (cheap; from M8, G-15).
   - Notebook N1.
3. **M2 features (9.3).** Causal only. The lean named taxonomy, not 100+ derived columns. Notebook N2.
4. **M3 baseline allocator and backtest harness (9.4).**
   - Vol-targeted core mix, the vol target on month-end returns (E-11); SPY / 60-40 / Faber
     baselines; 10 bps costs; P&L on month-end prices (E-08).
   - Append-only registry, budgets and DSR, reported, not gating (from M8, G-15). The first
     budgeted run sets the new baseline and its DSR line carries the ADR-0005 epoch note.
   - Relative float tolerances in every test.
   - Notebook N3 with the scoreboard.
5. **M4 weekly report on real data (9.5).**
   - One command; runs 2 and 3 identical apart from the run date (run 1 has no previous book).
   - Latest complete month scored, with a staleness cap.
   - Notebook N4.
   - **→ MVP-1: Glenn can use it.**
6. **M5 regime labeler (9.6).**
   - Jump model with K and λ from the design protocol.
   - States aligned across refits (CR-02).
   - Human-readable profiles.
   - Notebook N5, with a human sign-off.
7. **M6 nowcaster and filter (9.7).**
   - Fixed long-history feature set, the same freeze rule as L1 (the L2 recipe).
   - Calibrated; distinct-output check.
   - Bayes filter using the model's own training prior.
   - Truncation-invariance test.
   - Notebook N6.
8. **M7 regime tilt vs ablation (9.8).**
   - The decisive experiment.
   - If the tilt doesn't beat the ablation it stays **advisory** (shown, not weighted) and
     MVP-1's allocation stands.
   - Notebook N7.
   - **→ MVP-2.**
9. **M8 honesty upgrade**, once there are candidates to choose between. Its code already moved
   with M1 and M3 (G-15); this step turns the reports into gates:
   - trial budgets per phase (ADR-0004);
   - the DSR hurdle (ADR-0003);
   - the single 2021+ holdout evaluation at design freeze.
10. **M9+**, each gated on M7's scoreboard:
    - tripwire;
    - stability and persistence tuning (hysteresis, no-trade band);
    - second classifier;
    - mixture-of-experts L3;
    - tactics.

**The order differs from the history:**
- Serving (M4) and the baseline scoreboard (M3) come **before** any regime model.
- Heavy honesty ceremony (M8) comes **after** a candidate exists.
- Point-in-time lags are part of M1, not a later audit phase.

---

## 5. How to plan in lean mode (GSD or not)

- **KISS first (P-07).** If a plan needs a new config switch, a bespoke verification layer or a new abstraction, ask
  whether a plainer version would do. One person must be able to read every module and notebook.
- **One module per phase.** Two or three plans at most.
- **Discussion:** at most 5 questions. Decisions go straight into `DECISIONS.md` as rows, not
  prose documents.
- **Plan length:** about 150 lines per PLAN. If a plan needs more, the module is too big; split it.
- **Every phase ends with the same three gates:**
  1. the module's notebook runs top to bottom and Glenn signs its one sign-off cell;
  2. the real-data → report smoke passes and the module's `realdata` tests pass on the Mac after a build;
  3. the full suite is green on Mac and Linux.
- **Rigor budget:** mutation proofs, ruling records and measurement amendments only for
  **decision-bearing** numbers, which is essentially M7's scoreboard. Every other number is a
  diagnostic and gets reported once.
- **Trial budget:** one line in the phase context (ADR-0004), not an arithmetic section.
- **Deviations:** the executor fixes and notes them; it does not stop for a ruling unless a
  number that drives a decision changes.

---

## 6. What carries over from the current code, and what to leave behind

| Keep, port as is | Simplify while porting | Leave behind / defer |
|---|---|---|
| `splice.py`, `ingestion/` (free sources), `taxonomy.py`, holdout carve, registry append, walk-forward runner, purged CV, jump model, EWMA vol, vol-targeted tilt, costs, baselines, weekly report + serving, holdings YAML, plotting core, P1–P9 notebooks (committed without outputs) | Nowcaster feature admission (L2-02, fixed long-history set); filter (add cross-refit alignment); DSR and budgets (report, don't gate, before MVP-2); evaluation modules (keep KPIs, sojourn/lag, churn; drop one-off diagnostics scripts) | Classifier #2 and the joint driver (no lift), stability sub-sampling suite, per-number ruling apparatus, paid-provider adapters (placeholders), the legacy quarterly pipeline |

Measured size today (2026-10-12, after the G-13 deletions and the 08.5 reset): `platform/` is 17.5k
source lines (17,487) across 11 subpackages and 20.6k lines of platform tests (20,550; 62 files).
- `evaluation/` (3.1k), `plotting/` (3.7k) and `backtest/` (1.3k) hold most of the code.
- The regime-signal core (`labeling/jump_model.py`, `prediction/`, `allocation/tilt.py`) is
  about 1.1k lines.
- Measured with `cat $(find src/trading_crab_lib/platform -name '*.py') | wc -l`,
  `ls tests/unit/test_platform_*.py | wc -l` and `cat tests/unit/test_platform_*.py | wc -l`.

A lean MVP-2 port is roughly 6–8k source lines.

---

## 7. Open items a rebuild must resolve (not decided yet)

- **CR-02:** a cross-refit state-alignment method. Hungarian matching on centroids is the default.
  → M5 (Phase 9.6).
- ~~**CR-03:** the exact publication-lag table for every raw series (measured, not assumed).~~
  **Resolved** by 08.1's measured table (`publication_lags` in config; DECISIONS D-05).
- **L2 recipe:** a fixed long-history feature set, and whether calibration is needed at all.
  → M6 (Phase 9.7).
- **Hysteresis thresholds:** revisit once the belief is final (DECISIONS A-03). → M9+, after MVP-2.
- **REG-01:** does a second independent axis exist once data is point-in-time? Answer this only
  after MVP-2. → M9+.
