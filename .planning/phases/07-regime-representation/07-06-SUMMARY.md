---
phase: 07-regime-representation
plan: 06
subsystem: honesty-framework
tags: [deflated-sharpe, trial-registry, bailey-lopez-de-prado, scipy, pandas, honesty]

requires:
  - phase: 07-regime-representation (wave 1)
    provides: "the trial registry ledger reset with a provenance header (registry/trials.jsonl, prior_genuine_trials=38) and append_trial's mandatory trial_tag contract"
provides:
  - "total_trial_count() in platform/honesty/registry.py — reads the provenance header's prior_genuine_trials plus post-header rows, the correct D-16 denominator (38 today, never 1 or 39)"
  - "platform/evaluation/deflated_sharpe.py — expected_max_sharpe, deflated_sharpe_ratio, registry_sharpe_variance, format_dsr_verdict"
  - "07-DSR-ESTIMATOR-NOTE.md — the written, ADR-0002-quotable sharpe_variance estimator decision"
affects: [07-08 (ADR-0002 drafting), 07-10 (joint-lift run consuming total_trial_count() + deflated_sharpe_ratio), 07-11 (ADR-0002 acceptance)]

actuals:
  tokens: 13025
  tasks: 3
  commits: 3

tech-stack:
  added: []
  patterns:
    - "Provenance-header discrimination in an append-only JSONL ledger: a reserved config.record_type value marks accounting rows so a wrapper function (total_trial_count) can subtract them from the row count while adding back their carried prior — read_trials itself stays header-unaware."
    - "Degenerate-case variance fallback that is never zero: a variance-based multiplicative correction (expected_max_sharpe) must never fall back to 0.0 on missing data, because 0.0 variance is mathematically indistinguishable from 'no correction needed' and silently disables the safeguard it exists to provide."
    - "Primary-source PDF text extraction via a minimal zlib/regex script when poppler-utils is unavailable and PDF Python libraries cannot be installed — recovers plain-text streams and any embedded code listings verbatim, but not equations rendered as vector/image objects (name the gap, don't paper over it)."

key-files:
  created:
    - .planning/phases/07-regime-representation/07-DSR-ESTIMATOR-NOTE.md
    - src/trading_crab_lib/platform/evaluation/deflated_sharpe.py
    - tests/unit/test_platform_honesty_registry.py
    - tests/unit/test_platform_evaluation_deflated_sharpe.py
  modified:
    - src/trading_crab_lib/platform/honesty/registry.py

key-decisions:
  - "sharpe_variance estimator: sample variance of the registry's own Sharpe-bearing trial rows (non-header), read live via registry_sharpe_variance(), because design §22 declares the registry the true DSR denominator and a fixed constant divorced from it would never reflect what this project actually searched."
  - "Degenerate-case fallback is 1.0, never 0.0 — verified live that the registry currently holds zero usable Sharpe observations (0 of 42 archived rows, 0 of 1 post-reset row), so this fallback is load-bearing today, not a theoretical edge case; 0.0 would silently zero out expected_max_sharpe for every trial count, the exact under-penalization T-07-05 names as this project's closest analog to a security defect."
  - "Claim boundary: 'a deflated Sharpe ratio,' not 'the deflated Sharpe design §22 specifies' — the DSR equation's rendered glyphs (Eq. 2) could not be extracted from the primary-source PDF as text (embedded image), and the variance input rests on the 1.0 placeholder today, not a measured registry population."
  - "kurtosis parameter uses the RAW (non-excess) convention, i.e. 3.0 for a Normal distribution — matches the standard Mertens (2002) PSR variance approximation, the RESEARCH.md sketch already in this repo, and Prado's own book/mlfinlab implementation."

patterns-established:
  - "Pattern: a written specification note (07-DSR-ESTIMATOR-NOTE.md) precedes and gates implementation of any formula whose estimator choice is a genuine, consequential design decision — Task 3 was explicitly instructed to implement against the note, not re-derive it."

requirements-completed: [REG-01]

coverage:
  - id: D1
    description: "total_trial_count() reads the trial registry's provenance header (prior_genuine_trials) plus post-header rows as D-16's whole-registry denominator, never the raw row count"
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_honesty_registry.py::TestTotalTrialCount (11 tests: header-only=38, +1 row=39, two-headers-sum=18, no-header=7, missing/empty=0, malformed-config-degrades, missing-prior-key-defaults-zero, floor invariant, live-ledger floor>=38, read_trials unchanged)"
        status: pass
    human_judgment: false
  - id: D2
    description: "expected_max_sharpe and deflated_sharpe_ratio implement the Bailey-Lopez de Prado (2014) DSR formula, proven against independently hand-typed scipy.stats.norm oracles (not range checks, not values copied from the implementation)"
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_deflated_sharpe.py::TestExpectedMaxSharpe, ::TestDeflatedSharpeRatio (two independent oracles: n_trials=1 normal case, n_trials=50/nonzero-skew case; strict 3-point monotonicity chains for both functions; negative-SR<0.5; non-positive-denominator raises named ValueError)"
        status: pass
    human_judgment: false
  - id: D3
    description: "registry_sharpe_variance() and format_dsr_verdict() apply the estimator note's degenerate-case policy and report an unsoftened multiple-testing verdict"
    requirement: REG-01
    verification:
      - kind: unit
        ref: "tests/unit/test_platform_evaluation_deflated_sharpe.py::TestRegistrySharpeVariance, ::TestFormatDsrVerdict (degenerate on missing/header-only/single-observation ledgers, real sample variance at >=2 observations, never raises on malformed metrics; verdict wording identical at 0.5 and 0.49, no softening qualifiers)"
        status: pass
    human_judgment: false
  - id: D4
    description: "07-DSR-ESTIMATOR-NOTE.md records the sharpe_variance estimator choice, its rejected alternative, the usable-observation count, the degenerate-case policy, and the claim boundary — read from the primary source, not guessed"
    requirement: REG-01
    verification:
      - kind: unit
        ref: "verify command: python3 keyword-presence check over 07-DSR-ESTIMATOR-NOTE.md (euler, sharpe_variance, kurtosis, degenerate, limitation; length >= 1500 chars)"
        status: pass
    human_judgment: true
    rationale: "The note's substantive content — whether the estimator choice is actually well-justified, whether the primary-source reading claim is honest, whether the claim-boundary sentence genuinely commits — is a judgment call about the quality of written reasoning, not something a keyword-presence script can certify. The verify command only confirms structural completeness."

duration: 20min
completed: 2026-09-17
status: complete
---

# Phase 7 Plan 06: Deflated Sharpe Ratio & Whole-Registry Trial Count Summary

**Reads the Bailey–López de Prado (2014) primary paper directly (PDF, hand-extracted via a stdlib zlib script when poppler-utils was unavailable), writes the sharpe_variance estimator decision to disk before implementing, then ships `total_trial_count()` and `platform/evaluation/deflated_sharpe.py` proven against independently hand-typed scipy oracles — not range checks.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-09-17T14:18:00Z (approx.)
- **Completed:** 2026-09-17T14:36:31Z
- **Tasks:** 3/3 completed
- **Files modified:** 5 (4 created, 1 modified)

## Accomplishments

- Retrieved and read the full text of the primary DSR paper (SSRN 2460551, mirrored on
  co-author David H. Bailey's own site since SSRN itself returned HTTP 403 to this
  environment) via a custom zlib/regex PDF-stream extractor, since `poppler-utils`
  could not be installed (`apt-get` 404'd on this sandbox's mirror). Confirmed the
  `expected_max_sharpe` formula (Eq. 5/6) verbatim against the paper's own embedded
  Python snippet — not re-derived from memory or a secondary paraphrase.
- Discovered, by reading the actual registry contents (not assuming), that the
  registry currently has **zero** usable Sharpe-bearing observations: 0 of 42
  archived rows (which carry only `n_steps`/`terminal_log_wealth`) and 0 of the 1
  post-reset row (the provenance header itself). This made the degenerate-case
  fallback policy load-bearing today, not a theoretical edge case, and is recorded
  explicitly in the estimator note.
- `total_trial_count()` reads the live ledger correctly: **38**, not 1 (the raw
  row-count undercount T-07-05 names as this project's closest analog to a security
  defect) and not 39 (double-counting the header as a trial).
- `deflated_sharpe_ratio()` proven against two independent scipy-typed oracles (not
  copied from the implementation): the `n_trials=1` normal case and a second
  `n_trials=50`/nonzero-skew case, both matching to `1e-12`. Strict 3-point
  monotonicity chains assert `expected_max_sharpe` increases and `deflated_sharpe_ratio`
  decreases as trial count grows — not 2-point or non-strict comparisons.
- `format_dsr_verdict()` reports identically unsoftened wording at DSR=0.5 and
  DSR=0.49 — no "close," "borderline," or "nearly" language at any distance from the
  hurdle, verified by an explicit qualifier-absence test.

## Task Commits

Each task was committed atomically:

1. **Task 1: Read the primary source and fix the DSR estimator choice in writing** —
   `b3dba93` (docs)
2. **Task 2: total_trial_count() — read the provenance header, never the raw row
   count** — `a039b9d` (feat, TDD)
3. **Task 3: platform/evaluation/deflated_sharpe.py — the DSR, with an oracle test**
   — `fdd4c38` (feat, TDD)

_Note: Tasks 2 and 3 each wrote their test file first (RED), confirmed the import
failure, then implemented (GREEN) in the same commit — both were committed as a
single `feat` commit per task since the RED state was a collection error, not a
separately-committable passing-test-suite state._

**Plan metadata:** this commit (docs: complete plan)

## Files Created/Modified

- `.planning/phases/07-regime-representation/07-DSR-ESTIMATOR-NOTE.md` — the written,
  ADR-0002-quotable estimator decision: registry-population sample variance,
  1.0-never-0.0 degenerate fallback, "a deflated Sharpe ratio" claim boundary
- `src/trading_crab_lib/platform/honesty/registry.py` — adds `PROVENANCE_RECORD_TYPE`
  constant and `total_trial_count()`; `read_trials()` body unchanged (verified via
  `git diff`, purely additive)
- `src/trading_crab_lib/platform/evaluation/deflated_sharpe.py` — new module:
  `expected_max_sharpe`, `deflated_sharpe_ratio`, `registry_sharpe_variance`,
  `format_dsr_verdict`, `DEGENERATE_SHARPE_VARIANCE`
- `tests/unit/test_platform_honesty_registry.py` — new file, `TestTotalTrialCount`
  (11 tests)
- `tests/unit/test_platform_evaluation_deflated_sharpe.py` — new file, 24 tests
  across `TestExpectedMaxSharpe`, `TestDeflatedSharpeRatio`,
  `TestFormatDsrVerdict`, `TestRegistrySharpeVariance`

## Decisions Made

- **sharpe_variance estimator: registry-population sample variance, 1.0-never-0.0
  degenerate fallback.** Full rationale, rejected alternative (assumed-iid proxy),
  and named limitation (heterogeneous strategy legs; zero usable observations today)
  are in `07-DSR-ESTIMATOR-NOTE.md` §3–5 and its closing ADR-0002-quotable
  paragraph.
- **kurtosis parameter is RAW (non-excess), 3.0 for Normal** — matches the standard
  Mertens (2002) PSR variance-approximation convention already sketched in
  `07-RESEARCH.md`/`07-PATTERNS.md`, and matches Bailey & López de Prado's own
  book/mlfinlab reference implementation.
- **Claim boundary: "a deflated Sharpe ratio," not "the deflated Sharpe design §22
  specifies."** Two independent reasons: Eq. 2's literal rendered glyphs could not
  be extracted as text from the primary-source PDF (embedded image, not a secondary-source
  substitution), and the variance input rests on the 1.0 placeholder given zero
  live usable observations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] `poppler-utils` unavailable for PDF text extraction**
- **Found during:** Task 1
- **Issue:** Reading the primary paper requires extracting text from a 22-page PDF.
  Neither `pdftotext`/`pdftoppm` (poppler-utils) nor any Python PDF library
  (`pypdf`, `PyPDF2`, `pdfminer.six`, `fitz`) was installed, and `apt-get install
  poppler-utils` failed (404 from the configured Ubuntu security mirror in this
  sandbox).
- **Fix:** Wrote a small, self-contained stdlib script
  (`zlib` + `re`, no new dependency) that locates `stream...endstream` blocks,
  inflates FlateDecode-compressed content streams, and extracts `Tj`/`TJ`
  text-showing operators. This recovered the paper's full prose text and its one
  embedded Python code listing (the `getExpMaxSR` snippet) verbatim. It could NOT
  recover the paper's numbered equations (Eq. 1–6, including Eq. 2, the DSR formula
  itself), which are rendered as embedded vector/image objects in this PDF (a
  MathType-in-Word export artifact) rather than as text — this gap is named
  explicitly in `07-DSR-ESTIMATOR-NOTE.md`'s opening section and in §5's claim
  boundary, not silently papered over.
- **Files modified:** none in the repo (extraction script and downloaded PDF live
  only in the session scratchpad, not committed)
- **Verification:** the recovered `getExpMaxSR` Python snippet matches the
  RESEARCH.md sketch's `expected_max_sharpe` implementation character-for-character
  (variable names `emc`, `maxZ` included), confirming the extraction was faithful
  where it succeeded.
- **Committed in:** N/A (extraction tooling was not part of the deliverable; only
  its output — the estimator note — is committed, in `b3dba93`)

---

**Total deviations:** 1 auto-fixed (Rule 3, tooling substitution for a blocked
system-package install — not a "package manager install" in the excluded sense,
since no new project dependency was added or installed into the venv).
**Impact on plan:** None on scope or correctness. The primary source was read in
full text; only its rendered equation glyphs (not needed to confirm the already-
sketched formula) were unreachable, and that specific gap is documented rather than
hidden.

## Issues Encountered

None beyond the tooling substitution above. All acceptance criteria in the plan
were met without needing to weaken any test or amend the note after Task 3's
implementation.

## User Setup Required

None — no external service configuration required. `scipy` and `numpy` were
already installed and already used elsewhere in `platform/`; no new package was
installed into the project environment.

## Next Phase Readiness

- `total_trial_count()` and `deflated_sharpe_ratio()` are ready for plan 07-10's
  joint-lift run to wire together, exactly as the plan's `key_links` describe.
- `07-DSR-ESTIMATOR-NOTE.md`'s closing "For ADR-0002" section is written to be
  lifted verbatim into plan 07-08's ADR-0002 draft.
- **Live re-read before 07-10 runs anything:** `total_trial_count()` currently
  returns 38 and `registry_sharpe_variance()` currently returns the 1.0 degenerate
  fallback (0 usable Sharpe rows in the ledger). Both must be re-read live
  immediately before and after 07-10's runs, per `07-RESEARCH.md` Pitfall 8 and
  Pitfall 4 — neither number should be assumed to still hold by the time 07-10
  executes, especially once real `metrics["sharpe"]` values start being appended.
- No blockers identified for 07-08 (ADR-0002 drafting) or 07-10 (joint-lift run).

---
*Phase: 07-regime-representation*
*Plan: 06*
*Completed: 2026-09-17*
