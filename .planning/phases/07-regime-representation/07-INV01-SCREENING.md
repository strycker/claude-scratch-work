# Phase 7 Plan 07 — INV-01 Invariant Screening Record

**Measured:** 2026-09-17, `screen_invariant_candidates()` run once against the live,
holdout-carved `monthly_raw` checkpoint (`data/checkpoints/platform/monthly_raw.parquet`,
776 × 47 rows on disk, `fred_m2sl`/`fred_totalsl` added by the targeted FRED-only fetch
`0ea91ec`), `trial_tag="07-07-inv01-screen"`, against the **real** default trial registry
(`registry/trials.jsonl`) — not `NO_REGISTRY`. This is a genuine, evaluated screen, not a
smoke run.

**Read this record with R4 in view:** PCA appears in this screen only to *look at* how the
named candidates relate to one another (their loadings on a discovered axis). No principal
component is ever admitted as a feature — see §2.

---

## 1. Candidate table — every candidate, survivors and rejects alike

Decision window: post-holdout dev data (≤ 2020-12-31), first admissible month evaluated at
the 1972-01 decision date derived from `cfg["backtest"]["min_train_months"]=120` applied to
the dev-side `monthly_raw` index (`working.index[120]` = **1972-01-31**), exactly mirroring
classifier #1's own `_reference_label_columns` freeze rule (D-11). Era-stability assessed via
`loading_stability_across_eras()` (`expanding_steps`, `min_train=120`, `step=60` months) —
**10 eras**, ending 1972-01-31 through 2017-01-31 (the same 356-vs-470-month window-narrowing
caveat wave 1 documented does not apply here: this screen assesses eras over the raw
`monthly_raw` index directly, not L2's admission path).

| Candidate | `monthly_raw` source columns | First admissible month (own natural first-valid date) | Loading on PC1, every era (1972→2017, 10 eras) | Stability verdict (tolerance = 0.15) | Final verdict | Reason |
|---|---|---|---|---|---|---|
| `m2_gdp` | `fred_m2sl`, `fred_gdp` | **1962-02-28** (measured on dev data ≤2020-12-31) | **0.70711** in every one of the 10 eras (range = 4.4e-16, machine-precision only) | stable | **survive** | non-NaN for every month from 1972-01-31 onward (D-11 freeze passed) |
| `credit_gdp` | `fred_totalsl`, `fred_gdp` | **1962-02-28** (measured on dev data ≤2020-12-31) | **0.70711** in every one of the 10 eras (range = 2.2e-16, machine-precision only) | stable | **survive** | non-NaN for every month from 1972-01-31 onward (D-11 freeze passed) |

Both candidates survive. There is no reject row in this table because
`INVARIANT_CANDIDATES` currently has exactly two members and both are fully computable and
common-support-complete over the entire dev-side decision range. **§3 below records three
additional rejections that never reached this table at all** — they were rejected before
ingestion and have no `monthly_raw` column to screen.

Per-era detail (`m2_gdp` PC1 loading, identical pattern for `credit_gdp` to 4 decimal
places):

| Era end | Loading |
|---|---|
| 1972-01-31 | 0.7071067811865476 |
| 1977-01-31 | 0.7071067811865476 |
| 1982-01-31 | 0.7071067811865475 |
| 1987-01-31 | 0.7071067811865476 |
| 1992-01-31 | 0.7071067811865475 |
| 1997-01-31 | 0.7071067811865475 |
| 2002-01-31 | 0.7071067811865475 |
| 2007-01-31 | 0.7071067811865475 |
| 2012-01-31 | 0.7071067811865474 |
| 2017-01-31 | 0.7071067811865475 |

**§4's suspicion section explains why this value is fixed at exactly 1/√2 — this is a
mathematical property of screening exactly two standardized candidates, not evidence, by
itself, of genuine five-decade economic stability. Read it before treating "stable" here as
a strong finding.**

---

## 2. Named survivors (R4)

**`["m2_gdp", "credit_gdp"]`** — the exact, ordered list classifier #2's candidate feature
set may draw from (plan 07-08). Both are **named features**: `m2_gdp = fred_m2sl / fred_gdp`
and `credit_gdp = fred_totalsl / fred_gdp`, each a documented economic ratio a human can
read and reason about. **No principal component was admitted as a feature anywhere in this
screen** — `compute_candidate_loadings()` reads only `pca.components_` (the loadings
matrix); `pca.transform()` is never called anywhere in `platform/features/invariants.py`,
so no component SCORE was ever computed, let alone handed to a caller who could mistake it
for a feature. This is design decision R4, held structurally rather than by convention.

---

## 3. Rejections, with reasons

Three invariant candidates were considered and rejected **before ingestion** — they were
never added to `INVARIANT_CANDIDATES` and have no `monthly_raw` column, so they do not
appear in §1's table. Recording them here (not silently) is what makes this screening record
auditable rather than a record of only what happened to work.

| Candidate | Reason for rejection | Evidence |
|---|---|---|
| `BCNSDODNS` (Nonfinancial Corporate Business; Debt Securities and Loans, Liability, Level — a Federal Reserve Z.1 Flow-of-Funds series) | **Quarterly-native**, not monthly. 305 observations from 1945-10-01 to 2026-04-01 (≈3.8 points/year) — using it would require the same `fred_gdp`-style quarterly-repeat forward-fill treatment `TOTALSL` avoids entirely, adding an alignment cost neither chosen series carries. | Live-verified via `fredapi` during plan 07-05's research session (`07-RESEARCH.md` Pitfall 6, `07-PATTERNS.md` §6); restated (not re-verified) here per the plan's own instruction that this rejection is recorded, not re-litigated. |
| `TOTBKCR` (Total Bank Credit, All Commercial Banks) | **Starts 1973-01-03**, after the platform's 1962-01 spine start — would leave an 11-year gap at the front of the decision range, failing the freeze rule by construction for the entire pre-1973 span. | Live-verified via `fredapi` during plan 07-05's research session (same sources as above). |
| Market-cap/GDP ("Buffett indicator") | **No free 1962+ market-cap source exists in current ingestion.** FRED's Wilshire total-market-cap series starts ≈1970, still eight years short of the spine start, and no other free source was found. D-12's decision: leave it out rather than approximate it with a shorter or paid series. | `config/platform_settings.yaml`'s `buffett_indicator` comment (line ~248, added prior to Phase 7); restated in `platform/features/invariants.py`'s module docstring and in `INVARIANT_CANDIDATES`'s own comment. |

These three rejections are recorded so ADR-0002 (plan 07-08/07-11) can restate them without
re-deriving the reasoning.

---

## 4. Trial arithmetic

**Formula:** `rows_added = len(INVARIANT_CANDIDATES) = 2` — one `append_trial` row per
candidate, survivors and rejects alike (here, both survive, so both still logged).

| Reading | Value | Timestamp (UTC, live) |
|---|---|---|
| `total_trial_count()` **before** the screen | **38** | 2026-09-17T15:09:17.306103+00:00 |
| `total_trial_count()` **after** the screen | **40** | 2026-09-17T15:10:37.532107+00:00 |
| Difference | **+2** | matches the formula above exactly (`len(INVARIANT_CANDIDATES) == 2`) |

Both new registry rows are visible in `registry/trials.jsonl`, each carrying
`config["candidate_name"]` (`m2_gdp` / `credit_gdp`), `config["verdict"]="survive"`, and
`config["trial_tag"]="07-07-inv01-screen"`, timestamped `2026-09-17T15:09:27` (both rows
written back-to-back within the same `screen_invariant_candidates()` call). This count feeds
D-16's deflated-Sharpe denominator (`total_trial_count()`) and must appear in ADR-0002's
trial ceiling **alongside** the evaluation-run formula from `platform_design/adr/0001-l1-
feature-policy.md`'s "Trial ceiling" section — never folded into it, since these two rows are
a screening activity, not an evaluation run.

**No count in this record is quoted from a planning document.** Both `total_trial_count()`
readings above were read live, immediately before and immediately after the screen executed
in this session — not copied from `07-CONTEXT.md`'s "30 trials as of 2026-09-10" or any other
prior figure, all of which are stale by this project's own repeated observation.

---

## 5. Suspicion section

Per `07-VALIDATION.md`'s third check class: naming what in these results would indicate a
wiring bug rather than a genuine finding, and what was actually checked.

**(a) Both candidates have IDENTICAL loadings (0.70711) in every single era.**

*What this would indicate if unexplained:* the two candidates are secretly the same data
(e.g. a copy-paste bug feeding `fred_m2sl` into both ratio computations), or
`compute_candidate_loadings` is broken and returning a constant regardless of input.

*What was checked:* (1) `m2_gdp` and `credit_gdp` are built from different source columns
(`fred_m2sl`/`fred_gdp` vs. `fred_totalsl`/`fred_gdp`) — confirmed by reading
`INVARIANT_CANDIDATES`' own `source_columns` field. (2) The two series' actual values are NOT
identical or even the same order of magnitude — `m2_gdp` ranges ≈0.09–1.02, `credit_gdp`
ranges ≈17–227 (`working[["m2_gdp","credit_gdp"]].describe()`, live). (3) Their correlation
is high but **not** 1.0 and **does vary** across the sample: 0.9572 (window ending 1972-01),
0.9690 (window ending 1997-01), 0.9674 (window ending 2020-12) — genuinely different numbers,
not a fixed constant.

*What was concluded:* the identical-loading value is a **mathematical property of PCA on
exactly two standardized features**, not a wiring bug. For any 2×2 covariance matrix with
equal diagonal entries (guaranteed here because `standardize_features` scales every column to
unit variance before PCA), the eigenvectors are fixed at exactly `(±1/√2, ±1/√2)` for **any**
positive off-diagonal correlation ρ — only the corresponding eigenVALUE (`1+ρ`) carries the
correlation-strength information, and this screen's `compute_candidate_loadings()` does not
currently surface `explained_variance_ratio_`. **This is a genuine methodological limitation
of the current era-stability assessment, given only two named candidates today**: with
exactly two candidates whose correlation stays positive, the PC1 loading magnitude is
*mathematically guaranteed* to read "stable" regardless of whether the true relationship
strengthened, weakened, or stayed flat — it can only ever detect a **sign flip** (ρ crossing
zero) or a genuine third candidate joining the mix. This is recorded here as a limitation for
ADR-0002 to carry forward, not corrected in this plan (correcting it — e.g. by also tracking
`explained_variance_ratio_` — would be new scope beyond what plan 07-07 specifies).

**(b) `first_admissible_month` (1962-02-28) is ONE MONTH AFTER the spine start (1962-01-31),
even though both `fred_m2sl` and `fred_totalsl` individually start exactly at the spine
start.**

*What this would indicate if unexplained:* a plan-cited suspicion signature is "a candidate
whose first admissible month equals the spine start exactly when its source series starts
later" — the mirror-image version of that bug (a candidate's first-valid month lagging the
spine when its OWN sources don't) would indicate a stray off-by-one in the ratio computation
or an unintended forward-fill gap.

*What was checked:* read `monthly_raw`'s own per-column `first_valid_index()` directly:
`fred_m2sl` = 1962-01-31 (776 obs eventually, 775 non-null on the merged frame), `fred_totalsl`
= 1962-01-31, but **`fred_gdp` = 1962-02-28** — one month later than either credit/money
series.

*What was concluded:* the one-month lag is caused entirely by `fred_gdp`'s own limiting first
valid month (both ratios divide by GDP), not by any defect in the M2SL/TOTALSL ingestion
(plan 07-05) — both of those series are actually available from the very first month of the
spine. This is consistent with GDP's documented publication-lag/quarterly-native treatment
elsewhere in this codebase (the legacy library's ADR #7 shifts GDP +1 quarter for the same
underlying reason: GDP data has structural release-timing quirks other monthly FRED series do
not). Not a bug; a correctly-propagated limiting denominator.

**(c) Perfectly stable loadings across every era is unusually clean for real macro data over
five decades.**

*What this would indicate if unexplained:* a frozen or memoized computation that isn't
actually re-fitting PCA per era, or an era loop that isn't really varying its training window.

*What was checked:* the era-end dates themselves visibly advance (1972-01-31 through
2017-01-31 in 5-year steps, 10 distinct eras — see §1's per-era table), and the raw loading
values differ at the 14th-16th decimal place between eras (e.g. `0.7071067811865476` at
1972-01 vs. `0.7071067811865474` at 2012-01) — proving PCA IS being independently refit each
time on a growing window, not memoized or short-circuited; a truly static/frozen computation
would return bit-identical floats every time, not floating-point noise at the precision floor.

*What was concluded:* this is the SAME finding as (a) restated at the era level — the
apparent "perfect" stability is the mathematical 1/√2-loading property holding independently
in every era (because the two-candidate correlation stays positive throughout, per the actual
correlation values in (a)), not independent evidence that the underlying economic
relationship literally never moved. The genuinely informative number here is the *raw
correlation* trend in (a) (0.9572 → 0.9690 → 0.9674 — essentially flat, itself a real and
plausible finding for two closely related monetary/credit aggregates over 1972-2020), not the
loading magnitude.

**No fourth suspicion item was needed to reach three, but one additional observation is
recorded for completeness:** the two `total_trial_count()` readings in §4 were taken roughly
80 seconds apart in wall-clock time (15:09:17 → 15:10:37) even though the registry rows
themselves are timestamped 15:09:27 — the gap is this session's own Python startup/import
overhead between the two separate `python3 -c` invocations used to read the counter, not a
sign of a delayed or retried write. Verified: only 2 rows appended total (checked via
`wc -l`/tail on `registry/trials.jsonl` before and after), not 4 from an accidental double-run.

---

## Conclusion

Both of INV-01's currently-ingested named candidates — `m2_gdp` and `credit_gdp` — survive
the D-11 common-support freeze over the full 1972+ decision range and report as era-stable
under the stated tolerance, with the important caveat in §5(a)/(c) that "stable" here is
partly a mathematical artifact of screening exactly two candidates and should not be
over-read as strong five-decade economic evidence on its own — the raw correlation trend
(≈0.96-0.97, essentially flat) is the more informative underlying number. Three additional
candidates (`BCNSDODNS`, `TOTBKCR`, market-cap/GDP) were rejected before ever reaching this
screen, each for a documented, source-verifiable reason. The registry moved by exactly the
stated formula (+2, from 38 to 40), both rows tagged `07-07-inv01-screen`.
