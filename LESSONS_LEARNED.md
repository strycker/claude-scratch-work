# Lessons Learned — Trading-Crab

A first-person retrospective on pitfalls encountered building this pipeline.
Intended audience: anyone rebuilding this project or building something similar.
For the full implementation guide, see `REBUILD-FROM-SCRATCH-GUIDE.md`.
For formal design decisions, see the ADR section of `CLAUDE.md`.

---

## L1. Look-ahead bias is subtle and catastrophic

**What happened:** The smoothed derivative features (`d1`, `d2`, `d3`) were computed with a centered rolling window (`center=True`) for the entire dataset. When a supervised model was trained on these features, every "current" value implicitly knew 2–3 future quarters — the model looked amazing in CV but would be useless in production.

**What we'd do differently:** Maintain two separate feature files from the start: `features.parquet` (centered, for unsupervised/historical analysis) and `features_supervised.parquet` (causal, backward-only windows, for any model that will score real-time data). Never let them share a file.

**Rule:** If the feature value for quarter Q uses data from Q+1 or later, it cannot be used to predict Q's label.

---

## L2. The `market_code` column was silently poisoning gap-fill

**What happened:** The gap-fill function used `df[[col, "market_code"]].dropna()` to find the valid row range for interpolation. Since `market_code` was sourced from different label sets (`grok`, `clustered`, `predicted`) with different NaN patterns, gap-fill boundaries varied between runs. This made all downstream derivative values non-deterministic — same input data, different features each run.

**What we'd do differently:** Feature functions must never peek at label columns. Gap-fill and derivative calculation should use only the feature column itself (`df[[col]].dropna()`) to determine valid rows.

**Rule:** Labels are metadata. Feature engineering must be a pure function of the feature columns alone.

---

## L3. Mixing two APIs in one package without documentation is a maintenance trap

**What happened:** `prediction/__init__.py` grew a flat API (returns a bare `RandomForestClassifier`) and `prediction/classifier.py` grew a bundle API (returns a dict with per-fold CV metadata). Both co-existed in the same package. Without clear documentation, new code kept accidentally importing the wrong one.

**What we'd do differently:** Document the two-API pattern explicitly in `CLAUDE.md` the moment it appears. Add an import guard or docstring to each module saying which consumers it serves. Better yet, consider whether one module should be in `tests/` if it's only ever used by tests.

---

## L4. `STEPS` dict holds function references, not names — patch accordingly

**What happened:** Tests attempted to mock pipeline step functions by patching their module-level names (`trading_crab.pipeline.step1_ingest`). But the `STEPS` dict was built at module import time with direct references to the original function objects, bypassing the module attribute. Patching the attribute changed the name but not what `STEPS[1]` pointed to.

**What we'd do differently:** Patch `STEPS` dict values directly in tests. Alternatively, make `main()` look up step functions by name at call time rather than storing them at module load. The latter is cleaner but breaks static analysis.

---

## L5. Don't let a `release: published` trigger and a tag-push trigger coexist for the same package

**What happened:** Three of six GitHub Actions workflows overlapped: `publish.yml` (triggered on release), `python-publish.yml` (also on release), and `publish-app.yml` (on `v*` tags). Creating a GitHub Release would trigger two separate PyPI uploads for the same package, causing a version-already-exists error on the second.

**What we'd do differently:** Pick one trigger per package and stick with it. For a two-package monorepo: `lib-v*` tags trigger the library publish; `v*` tags (excluding `lib-v*`) trigger the app publish. Delete the GitHub boilerplate workflows immediately — they're never correct out of the box.

---

## L6. Checkpoint freshness uses wall-clock time, not data staleness

**What happened:** `CheckpointManager.is_fresh("macro_raw", max_age_days=7)` returns `True` if the checkpoint was saved less than 7 days ago — even if FRED released new data the next day. A "fresh" checkpoint can be stale data.

**What we'd do differently:** Add a data-staleness concept: store the latest timestamp of the actual data rows in the checkpoint manifest alongside the creation time. Freshness check should fail if `data_end_date < today - expected_lag`.

---

## L7. Optional dependency noise in pytest is worse than it looks

**What happened:** When `statsmodels` was installed, `test_markov.py` emitted dozens of `RuntimeWarning` and `UserWarning` lines during optimization on short synthetic data. These warnings were harmless but made CI logs unreadable, causing real failures to be missed.

**What we'd do differently:** Suppress known-harmless warnings from optional dependencies at two levels from the start: globally in `[tool.pytest.ini_options] filterwarnings` (using message-pattern filters, not class-path filters, so they don't break when the dep is absent), and with `pytestmark = filterwarnings(...)` in the specific test file.

---

## L8. Pickle files are silent time bombs

**What happened:** `outputs/models/current_regime.pkl` was serialized with `pickle.dump`. On a Python minor version bump (e.g., 3.10 → 3.11) the file could fail to load with a cryptic error. Also, any pickle file from an untrusted source executes arbitrary Python on load.

**What we'd do differently:** Use `joblib.dump` / `joblib.load` for sklearn models from day one (more stable across Python versions). Add a note in `CLAUDE.md` that pickle files are never committed and never loaded from external sources.

---

## L9. Silent partial ingestion produces plausible-looking but wrong outputs

**What happened:** If FRED rate-limited a request or multpl.com returned an empty table, the pipeline caught the exception and continued with whatever data was fetched. The resulting `macro_raw.parquet` had fewer columns than expected, but no error was raised. Downstream steps ran to completion on incomplete data, producing output that looked fine but was wrong.

**What we'd do differently:** Add an ingestion completeness check immediately after step 1. Assert that the parquet file has at least N columns (configurable), and warn loudly if any expected column is missing or has > X% NaN. Log a count of columns by source at every ingestion run.

---

## L10. Don't hardcode `end_date` in config — it will go stale

**What happened:** `config/settings.yaml` had `end_date: "2025-09-30"` hardcoded. The pipeline silently ignored all data after that date. This went unnoticed for months because the data still looked complete — it just didn't include the most recent quarters.

**What we'd do differently:** Set `end_date: null` from day one and handle it in ingestion code as `datetime.today()`. Add a monitoring check that warns if the DataFrame's max date is more than one quarter behind today.

---

## L11. Two CI workflows testing the same thing doubles noise, halves signal

**What happened:** `python-app.yml` (Python 3.10 only) and `python-package.yml` (Python 3.10–3.13 matrix) both ran lint + tests on every push to `main`. Every PR generated duplicate status checks, duplicate artifact uploads, and doubled CI minutes — but only the single-version run was ever actually checked by reviewers.

**What we'd do differently:** Start with one CI workflow. Add the multi-version matrix only when you actually have users on multiple Python versions. When you do add it, delete the single-version workflow immediately.

---

## L12. Bernstein gap-fill must happen after log transform — not before

**What happened:** Early prototype interpolated missing values in raw space (e.g., between S&P 500 values of 1000 and 4000). The midpoint was 2500. In log space the midpoint is exp((ln(1000)+ln(4000))/2) ≈ 2000 — closer to the exponential growth reality of equity prices.

**What we'd do differently:** Establish the canonical feature pipeline order as an invariant in `CLAUDE.md` before writing any code: cross-ratios → log → select → gap-fill → derivatives → select. Never allow gap-fill to run before log transform.

---

## Summary Table

| # | Lesson | Root cause | Fix |
|---|--------|-----------|-----|
| L1 | Look-ahead bias in features | Centered rolling used everywhere | Two separate feature files from day 1 |
| L2 | Label column poisoning gap-fill | market_code in dropna | Feature functions must not read labels |
| L3 | Two-API confusion | Undocumented API split | Document consumers in CLAUDE.md immediately |
| L4 | STEPS dict bypasses mock patches | Direct function refs at module load | Patch STEPS dict entries, not module attrs |
| L5 | Duplicate CI publish triggers | Multiple overlapping workflows | One trigger per package; delete boilerplate immediately |
| L6 | Wall-clock freshness ≠ data freshness | Checkpoint age vs data recency | Store data_end_date in manifest |
| L7 | pytest warning noise | Optional dep warnings | Suppress from day 1 in pyproject.toml |
| L8 | Pickle fragility | pickle.dump for sklearn models | joblib from day 1 |
| L9 | Silent partial ingestion | Caught exceptions + continued | Completeness check after ingestion |
| L10 | Hardcoded end_date goes stale | Config maintenance oversight | null + datetime.today() always |
| L11 | Duplicate CI workflows | Template accumulation | Start with one; matrix only when needed |
| L12 | Gap-fill in wrong space | Pipeline order not enforced | Establish canonical order as invariant early |
